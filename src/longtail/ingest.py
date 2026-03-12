from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import normalize_text

REQUIRED_COLUMNS = ["post_id", "text"]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["post_id"] = result["post_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    result["text_clean"] = result["text"].map(normalize_text)
    return result


def deduplicate_dataframe(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    subset = subset or ["text_clean"]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_processed(df: pd.DataFrame, processed_dir: str | Path, stem: str = "processed_posts") -> dict[str, str]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / f"{stem}.parquet"
    csv_path = processed_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return {"parquet": str(parquet_path), "csv": str(csv_path)}


def build_metadata(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
