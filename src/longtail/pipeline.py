from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from .config import load_yaml
from .ingest import build_metadata, deduplicate_dataframe, ensure_dirs, load_csv, normalize_dataframe, write_processed
from .label_match import LabelMatcher
from .masking import Masker
from .utils import residual_ratio


def process_csv(config_path: str | Path, input_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    raw_path = Path(input_path or paths.get("sample_input"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    output_dir = Path(paths.get("output_dir", "data/output"))
    ensure_dirs(processed_dir, output_dir)

    df = deduplicate_dataframe(normalize_dataframe(load_csv(raw_path)), subset=config.get("pipeline", {}).get("dedup_subset"))

    matcher = LabelMatcher.from_yaml(
        config.get("labels", {}).get("file"),
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
