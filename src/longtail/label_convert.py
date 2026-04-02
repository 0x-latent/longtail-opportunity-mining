"""将标签 CSV (label_path + keywords) 转换为 pipeline 使用的 YAML 词表。

CSV 格式:
    label_path,keywords
    症状/咽喉/咳嗽,咳嗽;一直咳;干咳;痰咳

YAML 输出格式 (与 LabelMatcher 兼容):
    version: "1.0"
    dimensions:
      症状:
        mask_token: "[症状]"
        match_mode: phrase
        entries:
          - key: 咽喉/咳嗽
            surface_forms: ["咳嗽", "一直咳", "干咳", "痰咳"]
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_labels_csv(path: str | Path) -> list[dict[str, Any]]:
    """读取标签 CSV，返回解析后的条目列表。"""
    path = Path(path)
    rows = []

    # 尝试多种编码
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            text = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"无法识别 {path} 的编码")

    reader = csv.DictReader(text.splitlines())
    for row in reader:
        label_path = row.get("label_path", "").strip()
        keywords = row.get("keywords", "").strip()
        if not label_path or not keywords:
            continue

        parts = [p.strip() for p in label_path.split("/") if p.strip()]
        if len(parts) < 2:
            continue

        dimension = parts[0]
        label_key = "/".join(parts[1:])
        surface_forms = [k.strip() for k in keywords.split(";") if k.strip()]

        rows.append({
            "dimension": dimension,
            "label_key": label_key,
            "surface_forms": surface_forms,
        })

    return rows


def build_yaml_config(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """将解析后的条目构建为 YAML 词表结构。"""
    dimensions: dict[str, list[dict]] = defaultdict(list)

    for entry in entries:
        dimensions[entry["dimension"]].append({
            "key": entry["label_key"],
            "surface_forms": entry["surface_forms"],
        })

    yaml_config: dict[str, Any] = {
        "version": "1.0",
        "language": "zh",
        "dimensions": {},
    }

    for dim, dim_entries in dimensions.items():
        yaml_config["dimensions"][dim] = {
            "mask_token": f"[{dim}]",
            "match_mode": "phrase",
            "entries": dim_entries,
        }

    return yaml_config


def convert_labels_csv_to_yaml(
    csv_path: str | Path,
    yaml_path: str | Path | None = None,
) -> Path:
    """
    将标签 CSV 转换为 YAML 词表文件。

    如果 yaml_path 为 None，则输出到 csv 同目录下同名 .yaml 文件。
    """
    csv_path = Path(csv_path)
    if yaml_path is None:
        yaml_path = csv_path.with_suffix(".yaml")
    else:
        yaml_path = Path(yaml_path)

    entries = parse_labels_csv(csv_path)
    if not entries:
        raise ValueError(f"标签 CSV 为空或格式不正确: {csv_path}")

    config = build_yaml_config(entries)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    dim_count = len(config["dimensions"])
    entry_count = sum(len(d["entries"]) for d in config["dimensions"].values())
    print(f"  标签转换: {csv_path.name} → {yaml_path.name} ({dim_count} 个维度, {entry_count} 个标签)")

    return yaml_path
