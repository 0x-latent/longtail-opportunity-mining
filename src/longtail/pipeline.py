from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

import logging

from .config import load_yaml
from .ingest import build_metadata, deduplicate_dataframe, ensure_dirs, load_csv, normalize_dataframe, write_processed
from .label_match import LabelMatcher
from .masking import Masker
from .utils import residual_ratio, tokenize_chinese

logger = logging.getLogger(__name__)


def process_csv(
    config_path: str | Path,
    input_path: str | Path | None = None,
    labels_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent  # config/ -> project root
    config = load_yaml(config_path)
    paths = config.get("paths", {})

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    raw_path = _resolve(input_path or paths.get("sample_input"))
    processed_dir = _resolve(paths.get("processed_dir", "data/processed"))
    output_dir = _resolve(paths.get("output_dir", "data/output"))
    ensure_dirs(processed_dir, output_dir)

    ingest_cfg = config.get("ingest", {})
    df = load_csv(
        raw_path,
        column_map=ingest_cfg.get("column_map"),
        encoding=ingest_cfg.get("encoding"),
    )
    df = normalize_dataframe(df, min_text_len=ingest_cfg.get("min_text_len", 10))
    df = deduplicate_dataframe(df, subset=config.get("pipeline", {}).get("dedup_subset"))

    labels_path = labels_override or _resolve(config.get("labels", {}).get("file"))
    matcher = LabelMatcher.from_yaml(
        labels_path,
        dimension_priority=config.get("matching", {}).get("dimension_priority", []),
    )
    masker = Masker(matcher.mask_tokens)
    all_mask_tokens = list(matcher.mask_tokens.values())

    processed_rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        matches = matcher.match(row["text_clean"])
        text_masked = masker.soft_mask(row["text_clean"], matches)
        text_hard_filtered = masker.hard_filter(row["text_clean"], matches)
        processed_rows.append(
            {
                **row,
                "text_masked": text_masked,
                "text_hard_filtered": text_hard_filtered,
                "matched_labels": json.dumps([m.to_dict() for m in matches], ensure_ascii=False),
                "label_counts": json.dumps(matcher.count_by_dimension(matches), ensure_ascii=False),
                "residual_ratio": residual_ratio(text_masked, all_mask_tokens),
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    files = write_processed(processed_df, processed_dir)
    metadata = build_metadata(processed_df)
    metadata_path = output_dir / "phase1_summary.json"
    metadata_path.write_text(json.dumps({**metadata, "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"files": files, "summary": str(metadata_path), "rows": len(processed_df)}


def process_clusters(config_path: str | Path) -> dict[str, Any]:
    """Phase 2: filter → embed → UMAP → HDBSCAN → c-TF-IDF → output."""
    import numpy as np

    from .clustering import build_cluster_summaries, cluster_embeddings, compute_ctfidf, reduce_embeddings
    from .embedding import embed_texts

    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    p2 = config.get("phase2", {})

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    # 1. Load Phase 1 output
    processed_dir = _resolve(paths.get("processed_dir", "data/processed"))
    input_path = processed_dir / "processed_posts.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 1 output not found: {input_path}. Run Phase 1 first.")

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d posts from %s", len(df), input_path)

    # 2. Filter by residual_ratio and min token count
    threshold = p2.get("residual_ratio_threshold",
                       config.get("pipeline", {}).get("residual_ratio_threshold", 0.15))
    min_tokens = p2.get("min_text_tokens", 3)

    mask_eligible = df["residual_ratio"] >= threshold
    token_counts = df["text_masked"].map(lambda t: len(tokenize_chinese(t)))
    mask_eligible &= token_counts >= min_tokens

    df_cluster = df[mask_eligible].reset_index(drop=True)
    n_filtered = len(df) - len(df_cluster)
    print(f"  Phase 2 过滤: {len(df_cluster)} 条帖子参与聚类 "
          f"({n_filtered} 条被过滤, residual_ratio < {threshold} 或 token 数 < {min_tokens})")

    # 3. Embed
    emb_cfg = p2.get("embedding", {})
    cache_path = _resolve(emb_cfg.get("cache_file", "data/processed/embeddings.npy"))
    embeddings = embed_texts(
        texts=df_cluster["text_masked"].tolist(),
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
        batch_size=emb_cfg.get("batch_size", 256),
        normalize=emb_cfg.get("normalize", True),
        cache_path=cache_path,
    )

    # 4. UMAP
    umap_cfg = p2.get("umap", {})
    reduced = reduce_embeddings(
        embeddings,
        n_components=umap_cfg.get("n_components", 10),
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.0),
        metric=umap_cfg.get("metric", "cosine"),
        random_state=umap_cfg.get("random_state", 42),
    )

    # 5. HDBSCAN
    hdb_cfg = p2.get("hdbscan", {})
    labels, _clusterer = cluster_embeddings(
        reduced,
        min_cluster_size=hdb_cfg.get("min_cluster_size", 10),
        min_samples=hdb_cfg.get("min_samples", 5),
        cluster_selection_method=hdb_cfg.get("cluster_selection_method", "eom"),
        metric=hdb_cfg.get("metric", "euclidean"),
    )

    # 6. c-TF-IDF
    ctfidf_cfg = p2.get("ctfidf", {})
    ngram_range = tuple(ctfidf_cfg.get("ngram_range", [1, 3]))
    ctfidf_terms = compute_ctfidf(
        texts=df_cluster["text_masked"].tolist(),
        labels=labels,
        ngram_range=ngram_range,
        top_n=ctfidf_cfg.get("top_n_terms", 15),
        min_df=ctfidf_cfg.get("min_df", 2),
    )

    # 7. Build cluster summaries
    summaries = build_cluster_summaries(
        post_ids=df_cluster["post_id"].tolist(),
        labels=labels,
        embeddings=embeddings,
        ctfidf_terms=ctfidf_terms,
        label_counts_json=df_cluster["label_counts"].tolist(),
    )

    # 8. Save outputs
    output_cfg = p2.get("output", {})
    output_dir = _resolve(paths.get("output_dir", "data/output"))
    ensure_dirs(output_dir, processed_dir)

    # 8a. Cluster assignments parquet
    assignments_path = _resolve(output_cfg.get("assignments_file", "data/processed/cluster_assignments.parquet"))
    df_assignments = df_cluster[["post_id"]].copy()
    df_assignments["cluster_id"] = labels
    df_assignments.to_parquet(assignments_path, index=False)

    # 8b. Cluster details JSON (truncate post_ids for readability)
    clusters_path = _resolve(output_cfg.get("clusters_file", "data/output/clusters.json"))
    summaries_for_json = []
    for s in summaries:
        s_copy = {**s}
        s_copy["post_ids"] = s_copy["post_ids"][:5]
        s_copy["post_ids_truncated"] = len(s["post_ids"]) > 5
        summaries_for_json.append(s_copy)
    clusters_path.write_text(
        json.dumps(summaries_for_json, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 8c. Phase 2 summary
    n_clusters = len(summaries)
    noise_count = int((labels == -1).sum())
    noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0
    mean_coherence = float(np.mean([s["coherence_score"] for s in summaries])) if summaries else 0

    summary_data = {
        "total_posts": len(df),
        "eligible_posts": len(df_cluster),
        "filtered_posts": n_filtered,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": round(noise_ratio, 4),
        "mean_coherence": round(mean_coherence, 4),
        "cluster_sizes": {s["cluster_id"]: s["size"] for s in summaries},
        "files": {
            "clusters": str(clusters_path),
            "assignments": str(assignments_path),
            "embeddings_cache": str(cache_path),
        },
    }
    summary_path = _resolve(output_cfg.get("summary_file", "data/output/phase2_summary.json"))
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Phase 2 完成: {n_clusters} 个聚类, noise ratio={noise_ratio:.1%}, "
          f"mean coherence={mean_coherence:.4f}")
    return summary_data
