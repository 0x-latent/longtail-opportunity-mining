from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import normalize_text

# 默认列名映射：原始 CSV 列名 → 标准列名
DEFAULT_COLUMN_MAP = {
    "ID": "post_id",
    "id": "post_id",
    "文本": "text",
    "标签": "labels_raw",
    "平台": "platform",
}

REQUIRED_COLUMNS = ["post_id", "text"]

# ---------- 社媒文本去噪 ----------

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HASHTAG = re.compile(r"#[^#\s]+#?")          # #话题标签# 或 #话题标签
_RE_MENTION = re.compile(r"@[\w\u4e00-\u9fff]+")  # @用户名
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002640-\U00002642"
    "]+",
    flags=re.UNICODE,
)
_RE_PLATFORM_NOISE = re.compile(
    r"DOU\+小助手|@DOU\+小助手|小红书号[：:]\s*\d+|我的小红书号\S*"
)


def clean_social_text(text: str) -> str:
    """去除社媒噪声：URL、#话题、@提及、emoji、平台水印、残留符号。"""
    if not isinstance(text, str):
        return ""
    text = _RE_URL.sub("", text)
    text = _RE_PLATFORM_NOISE.sub("", text)
    text = _RE_HASHTAG.sub("", text)
    text = _RE_MENTION.sub("", text)
    text = _RE_EMOJI.sub("", text)
    # 清除残留的孤立 # 号
    text = re.sub(r"#", "", text)
    return text


def load_csv(
    path: str | Path,
    column_map: dict[str, str] | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """读取 CSV，自动探测编码，并按 column_map 重命名列。"""
    path = Path(path)
    column_map = column_map or DEFAULT_COLUMN_MAP

    # 自动探测编码
    if encoding is None:
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                df = pd.read_csv(path, nrows=5, encoding=enc)
                encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"无法自动识别 {path} 的编码")

    df = pd.read_csv(path, encoding=encoding)

    # 列名映射
    rename = {k: v for k, v in column_map.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}（当前列: {list(df.columns)}）")
    return df


def normalize_dataframe(
    df: pd.DataFrame,
    min_text_len: int = 10,
) -> pd.DataFrame:
    """标准化文本 + 社媒去噪 + 短文本过滤。"""
    result = df.copy()
    result["post_id"] = result["post_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    # 社媒去噪 → Unicode 标准化
    result["text_clean"] = (
        result["text"]
        .map(clean_social_text)
        .map(normalize_text)
    )
    # 过滤短文本
    before = len(result)
    result = result[result["text_clean"].str.len() >= min_text_len].reset_index(drop=True)
    dropped = before - len(result)
    if dropped:
        print(f"  短文本过滤: 移除 {dropped} 行 (< {min_text_len} 字)")
    return result


def deduplicate_dataframe(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    subset = subset or ["text_clean"]
    before = len(df)
    result = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dropped = before - len(result)
    if dropped:
        print(f"  精确去重: 移除 {dropped} 行")
    return result


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
