from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MASK_PATTERN = re.compile(r"\[[A-Z_]+\]")

# 中文停用词 + 标点 + 残留 mask 碎片
_STOPWORDS = frozenset(
    # 标点
    list("，。！？、；：""''（）【】《》—…·~,.!?;:\"'()[]<>-/\\@#$%^&*+={}|`")
    # 高频虚词
    + "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
      "自己 这 他 她 它 吗 吧 呢 啊 哦 嗯 呀 么 但 还 而 或 把 被 让 给 对 从 向 往 比 "
      "这个 那个 什么 怎么 可以 因为 所以 如果 虽然 但是 而且 或者 以及 "
      "这样 那样 这种 那种 那么 其实 已经 然后 之后 之前 "
      "起来 出来 下去 过来 回来 知道 觉得 感觉 时候".split()
)


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """UMAP dimensionality reduction."""
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("缺少 umap-learn，请运行: pip install -e '.[phase2]'") from exc

    logger.info("UMAP: %dD -> %dD (n_neighbors=%d)", embeddings.shape[1], n_components, n_neighbors)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    reduced: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
) -> tuple[np.ndarray, Any]:
    """HDBSCAN clustering. Returns (labels, clusterer)."""
    try:
        import hdbscan
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
            logger.info("Using sklearn.cluster.HDBSCAN as fallback")
            clusterer = SklearnHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                metric=metric,
            )
            labels = clusterer.fit_predict(reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int((labels == -1).sum())
            logger.info("Found %d clusters, %d noise points (%.1f%%)",
                        n_clusters, noise_count, 100 * noise_count / len(labels))
            return labels, clusterer
        except ImportError:
            raise RuntimeError("缺少 hdbscan，请运行: pip install -e '.[phase2]'")

    logger.info("HDBSCAN: min_cluster_size=%d, min_samples=%d", min_cluster_size, min_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
    )
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    logger.info("Found %d clusters, %d noise points (%.1f%%)",
                n_clusters, noise_count, 100 * noise_count / len(labels))
    return labels, clusterer


def compute_ctfidf(
    texts: list[str],
    labels: np.ndarray,
    ngram_range: tuple[int, int] = (1, 3),
    top_n: int = 15,
    min_df: int = 2,
    tokenizer_fn=None,
) -> dict[int, list[tuple[str, float]]]:
    """
    Compute c-TF-IDF: per-cluster term importance.

    Returns {cluster_id: [(term, score), ...]} sorted descending.
    Excludes noise cluster (-1) and mask tokens.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError("缺少 scikit-learn，请运行: pip install -e '.[phase2]'") from exc

    if tokenizer_fn is None:
        from .utils import tokenize_chinese
        tokenizer_fn = tokenize_chinese

    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        return {}

    # Build per-cluster concatenated documents
    cluster_docs = []
    for cid in cluster_ids:
        mask = labels == cid
        joined = " ".join(t for t, m in zip(texts, mask) if m)
        cluster_docs.append(joined)

    def _tokenize(text: str) -> list[str]:
        tokens = tokenizer_fn(text)
        return [
            t for t in tokens
            if not MASK_PATTERN.match(t)
            and t not in _STOPWORDS
            and len(t.strip()) > 0
        ]

    vectorizer = CountVectorizer(
        tokenizer=_tokenize,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=None,
    )
    tf_matrix = vectorizer.fit_transform(cluster_docs)
    feature_names = vectorizer.get_feature_names_out()

    # c-TF-IDF: tf(t,c) * log(1 + N / n_t)
    tf = tf_matrix.toarray().astype(float)
    tf = tf / (tf.sum(axis=1, keepdims=True) + 1e-9)

    n_clusters_total = len(cluster_ids)
    df = (tf_matrix.toarray() > 0).sum(axis=0)
    idf = np.log(1 + n_clusters_total / (df + 1e-9))

    ctfidf = tf * idf

    result = {}
    for i, cid in enumerate(cluster_ids):
        scores = ctfidf[i]
        top_indices = scores.argsort()[::-1][:top_n]
        result[cid] = [(feature_names[j], float(scores[j])) for j in top_indices if scores[j] > 0]

    return result


def build_cluster_summaries(
    post_ids: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    ctfidf_terms: dict[int, list[tuple[str, float]]],
    label_counts_json: list[str],
) -> list[dict[str, Any]]:
    """Build per-cluster summary records."""
    cluster_ids = sorted(set(labels) - {-1})
    summaries = []

    for cid in cluster_ids:
        mask = labels == cid
        indices = np.where(mask)[0]
        member_ids = [post_ids[i] for i in indices]
        member_embs = embeddings[indices]

        # Centroid and coherence (mean cosine similarity to centroid)
        centroid = member_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        norms = np.linalg.norm(member_embs, axis=1, keepdims=True) + 1e-9
        normed = member_embs / norms
        coherence = float(np.mean(normed @ centroid_norm))

        # Aggregate label_counts across cluster members
        label_dist: dict[str, int] = {}
        for idx in indices:
            try:
                counts = json.loads(label_counts_json[idx])
            except (json.JSONDecodeError, IndexError):
                continue
            for dim, count in counts.items():
                label_dist[dim] = label_dist.get(dim, 0) + count

        summaries.append({
            "cluster_id": int(cid),
            "size": len(member_ids),
            "post_ids": member_ids,
            "top_terms": ctfidf_terms.get(cid, []),
            "coherence_score": round(coherence, 4),
            "label_distribution": label_dist,
        })

    return summaries
