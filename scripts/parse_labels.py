#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_labels.py — 解析原始 CSV 中的多层级标签列

标签格式示例:
  品牌认知.产品特点.服务.客服态度.客服态度.客服态度|场景.人物.基本属性_年龄.0-3岁.4个月.4个月
  ^^^^^^^^ ^^^^^^^^ ^^^^ ^^^^^^^^ --------------------
  top_dim   sub1   sub2  canonical   surface_forms (L4+, 去重)

多条标签以 | 分隔，每条用 . 分隔层级 (3-6 层)。

输出:
  data/processed/labels_flat.csv   — 每行 = (post_id, 一条标签路径) + 各层级拆分
  data/processed/labels_by_dim.csv — 每帖一行，每个 top_dim 一列，汇总所有 canonical_value

用法:
  .venv/Scripts/python parse_labels.py                          # 使用默认文件
  .venv/Scripts/python parse_labels.py -i data/raw/other.csv    # 指定文件
  .venv/Scripts/python parse_labels.py --encoding gbk           # 指定编码
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# 自动探测标签列时尝试的候选列名
LABEL_COL_CANDIDATES = ["标签", "label", "tags", "标注"]

# 尝试的文件编码顺序
ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "gbk", "gb18030"]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def detect_encoding(filepath: str | Path) -> str:
    """依次尝试候选编码，返回第一个能成功读取首行的编码。"""
    for enc in ENCODING_CANDIDATES:
        try:
            with open(filepath, encoding=enc) as f:
                f.readline()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 兜底: latin-1 永远不会失败
    return "latin-1"


def find_label_column(df: pd.DataFrame) -> str:
    """在 DataFrame 列名中匹配标签列，忽略大小写。"""
    col_map = {c.strip().lower(): c for c in df.columns}
    for candidate in LABEL_COL_CANDIDATES:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    # 没找到已知名称，退而求其次: 找第一个字符串列且包含 '.' 和 '|' 的
    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str)
        if sample.str.contains(r"\|").any() and sample.str.contains(r"\.").any():
            return col
    raise ValueError(
        f"无法自动识别标签列。列名: {list(df.columns)}。"
        f"请确保列名为 {LABEL_COL_CANDIDATES} 之一。"
    )


def find_id_column(df: pd.DataFrame) -> str:
    """在 DataFrame 中查找 ID 列。"""
    for name in ["ID", "id", "post_id", "帖子ID"]:
        if name in df.columns:
            return name
    # 未找到则使用 DataFrame 索引
    return ""


def parse_single_path(path_str: str) -> dict:
    """
    解析单条标签路径 (dot-separated) 为结构化字段。

    品牌认知.产品特点.服务.客服态度.客服态度.客服态度
    L0       L1       L2   L3       L4       L5
    top_dim  sub1     sub2 canonical ---- surface_forms (去重) ----
    """
    parts = [p.strip() for p in path_str.split(".") if p.strip()]
    result = {
        "raw_path": path_str,
        "depth": len(parts),
        "top_dim": parts[0] if len(parts) > 0 else "",
        "sub1": parts[1] if len(parts) > 1 else "",
        "sub2": parts[2] if len(parts) > 2 else "",
        "canonical_value": parts[3] if len(parts) > 3 else "",
    }
    # L4+ 是 surface forms（用户原文表述），去重后以 ; 拼接
    if len(parts) > 4:
        # 去重但保持顺序
        seen = set()
        unique = []
        for sf in parts[4:]:
            if sf not in seen:
                seen.add(sf)
                unique.append(sf)
        result["surface_forms"] = ";".join(unique)
    else:
        result["surface_forms"] = ""
    return result


def parse_label_string(label_str: str) -> list[dict]:
    """将整条标签字符串 (|分隔) 拆成多条解析后的路径记录。"""
    if not isinstance(label_str, str) or not label_str.strip():
        return []
    paths = [p.strip() for p in label_str.split("|") if p.strip()]
    return [parse_single_path(p) for p in paths]


# ---------------------------------------------------------------------------
# 核心处理
# ---------------------------------------------------------------------------

def process_file(filepath: str | Path, encoding: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    处理单个 CSV 文件，返回 (labels_flat_df, labels_by_dim_df)。
    """
    filepath = Path(filepath)
    if encoding is None:
        encoding = detect_encoding(filepath)
    print(f"  读取: {filepath.name}  (encoding={encoding})")

    df = pd.read_csv(filepath, encoding=encoding)
    print(f"  行数: {len(df):,}  列: {list(df.columns)}")

    label_col = find_label_column(df)
    print(f"  标签列: '{label_col}'")

    id_col = find_id_column(df)

    # --- 1) labels_flat: 每行一条标签路径 ---
    flat_rows: list[dict] = []
    # --- 2) labels_by_dim: 每帖每个维度的 canonical values ---
    dim_map: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for idx, row in df.iterrows():
        post_id = row[id_col] if id_col else idx
        label_str = row[label_col]
        parsed_paths = parse_label_string(str(label_str) if pd.notna(label_str) else "")

        for p in parsed_paths:
            p["post_id"] = post_id
            p["source_file"] = filepath.name
            flat_rows.append(p)
            # 收集维度汇总
            if p["canonical_value"]:
                dim_map[idx][p["top_dim"]].append(p["canonical_value"])

    flat_df = pd.DataFrame(flat_rows)
    if not flat_df.empty:
        # 调整列顺序
        cols = ["post_id", "source_file", "top_dim", "sub1", "sub2",
                "canonical_value", "surface_forms", "depth", "raw_path"]
        flat_df = flat_df[[c for c in cols if c in flat_df.columns]]

    # --- 构建 labels_by_dim pivot ---
    # 收集所有 top_dim
    all_dims = sorted({p["top_dim"] for p in flat_rows if p["top_dim"]})
    pivot_rows: list[dict] = []
    for idx, row in df.iterrows():
        post_id = row[id_col] if id_col else idx
        pivot_row: dict = {"post_id": post_id, "source_file": filepath.name}
        for dim in all_dims:
            vals = dim_map[idx].get(dim, [])
            # 去重并用 ; 拼接
            seen = set()
            unique_vals = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
            pivot_row[dim] = ";".join(unique_vals) if unique_vals else ""
        pivot_rows.append(pivot_row)

    pivot_df = pd.DataFrame(pivot_rows)
    return flat_df, pivot_df


def print_summary(flat_df: pd.DataFrame) -> None:
    """打印统计摘要。"""
    if flat_df.empty:
        print("\n  [!] 没有解析到任何标签。")
        return

    n_posts = flat_df["post_id"].nunique()
    n_labels = len(flat_df)
    top_dims = flat_df["top_dim"].unique()

    print("\n" + "=" * 60)
    print("  摘要 (Summary)")
    print("=" * 60)
    print(f"  帖子总数 (posts):           {n_posts:,}")
    print(f"  标签路径总数 (label paths):  {n_labels:,}")
    print(f"  顶层维度数 (top dimensions): {len(top_dims)}")
    print(f"  顶层维度:  {', '.join(sorted(top_dims))}")
    print()

    # 每个维度的 Top-10 canonical values
    for dim in sorted(top_dims):
        subset = flat_df[flat_df["top_dim"] == dim]
        counts = subset["canonical_value"].value_counts().head(10)
        print(f"  【{dim}】 — 共 {len(subset):,} 条路径")
        for val, cnt in counts.items():
            print(f"    {val:30s}  {cnt:>6,}")
        print()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="解析原始 CSV 标签列，输出结构化标签文件"
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="输入 CSV 文件路径。默认扫描 data/raw/*.csv 全部文件。"
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="CSV 文件编码 (默认自动检测，常见: utf-8, gbk, gb18030)"
    )
    parser.add_argument(
        "-o", "--outdir",
        default=str(OUT_DIR),
        help=f"输出目录 (默认: {OUT_DIR})"
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 确定要处理的文件列表
    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(RAW_DIR.glob("*.csv"))
        if not files:
            print(f"[ERROR] 在 {RAW_DIR} 下未找到 CSV 文件。", file=sys.stderr)
            sys.exit(1)

    print(f"待处理文件: {len(files)} 个")
    print("-" * 60)

    all_flat: list[pd.DataFrame] = []
    all_pivot: list[pd.DataFrame] = []

    for f in files:
        flat_df, pivot_df = process_file(f, encoding=args.encoding)
        all_flat.append(flat_df)
        all_pivot.append(pivot_df)

    # 合并所有文件的结果
    flat_combined = pd.concat(all_flat, ignore_index=True) if all_flat else pd.DataFrame()
    pivot_combined = pd.concat(all_pivot, ignore_index=True) if all_pivot else pd.DataFrame()

    # 写出
    flat_path = outdir / "labels_flat.csv"
    pivot_path = outdir / "labels_by_dim.csv"

    flat_combined.to_csv(flat_path, index=False, encoding="utf-8-sig")
    pivot_combined.to_csv(pivot_path, index=False, encoding="utf-8-sig")

    print(f"\n  输出: {flat_path}  ({len(flat_combined):,} 行)")
    print(f"  输出: {pivot_path}  ({len(pivot_combined):,} 行)")

    # 打印摘要
    print_summary(flat_combined)


if __name__ == "__main__":
    main()
