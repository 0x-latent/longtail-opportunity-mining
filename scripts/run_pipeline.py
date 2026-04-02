from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.config import load_yaml
from longtail.pipeline import process_clusters, process_csv


def _load_env():
    """从 .env 文件加载环境变量（如果存在）。"""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    import os
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _sync_from_cos(config_path: str, project: str) -> dict[str, Path]:
    """从 COS 下载项目数据到 data/raw/。"""
    from longtail.cos_client import sync_project

    config = load_yaml(config_path)
    cos_config = config.get("cos")
    if not cos_config:
        raise ValueError("config 中未配置 cos.bucket 和 cos.region")

    config_path_resolved = Path(config_path).resolve()
    project_root = config_path_resolved.parent.parent
    raw_dir = project_root / config.get("paths", {}).get("raw_dir", "data/raw")

    return sync_project(cos_config, project, raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run longtail opportunity mining pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--input", default=None, help="Input CSV (Phase 1 only)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Pipeline phase: 1=preprocess, 2=clustering")
    parser.add_argument("--sync", metavar="PROJECT",
                        help="Sync data from COS before running (e.g. --sync cold_cough)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    _load_env()

    # 从 COS 同步数据
    labels_yaml = None
    if args.sync:
        downloaded = _sync_from_cos(args.config, args.sync)
        print(f"  COS 同步完成: {downloaded}")
        # 如果没有指定 --input，自动使用下载的 posts 文件
        if args.input is None and "posts" in downloaded:
            args.input = str(downloaded["posts"])
        # 自动将 labels CSV 转换为 YAML 词表
        if "labels" in downloaded:
            from longtail.label_convert import convert_labels_csv_to_yaml
            labels_yaml = convert_labels_csv_to_yaml(
                downloaded["labels"],
                PROJECT_ROOT / "config" / "labels" / f"{args.sync}.yaml",
            )

    if args.phase == 1:
        result = process_csv(args.config, args.input, labels_override=labels_yaml)
    elif args.phase == 2:
        result = process_clusters(args.config)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
