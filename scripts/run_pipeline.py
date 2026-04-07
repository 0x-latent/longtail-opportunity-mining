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


def _sync_from_cos(config_path: str, vendor: str, project_filter: str | None = None) -> list[dict]:
    """从 COS 同步供应商数据到 data/raw/。"""
    from longtail.cos_client import sync_vendor

    config = load_yaml(config_path)
    cos_config = config.get("cos")
    if not cos_config:
        raise ValueError("config 中未配置 cos.bucket 和 cos.region")

    config_path_resolved = Path(config_path).resolve()
    project_root = config_path_resolved.parent.parent
    raw_dir = project_root / config.get("paths", {}).get("raw_dir", "data/raw")

    return sync_vendor(cos_config, vendor, raw_dir, project_filter=project_filter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run longtail opportunity mining pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--input", default=None, help="Input CSV (Phase 1 only)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Pipeline phase: 1=preprocess, 2=clustering")
    parser.add_argument("--sync", metavar="VENDOR",
                        help="Sync data from COS vendor folder (e.g. --sync hongyuan)")
    parser.add_argument("--project", default=None,
                        help="Project name (auto-detected from COS file, or specify for --phase 2)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto, up to 16)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    _load_env()

    # 从 COS 同步数据
    if args.sync:
        synced = _sync_from_cos(args.config, args.sync, project_filter=args.project)
        for item in synced:
            print(f"  COS 同步完成: {item['vendor']}/{item['project']} → {item['files']}")

        # 逐个项目跑 Phase 1
        if args.phase == 1:
            for item in synced:
                input_path = args.input or str(item["files"].get("posts", ""))
                labels_yaml = None
                if "labels" in item["files"]:
                    from longtail.label_convert import convert_labels_csv_to_yaml
                    labels_yaml = convert_labels_csv_to_yaml(
                        item["files"]["labels"],
                        PROJECT_ROOT / "config" / "labels" / f"{item['project']}.yaml",
                    )
                print(f"\n{'='*60}")
                print(f"  运行 Phase 1: {item['project']}")
                print(f"{'='*60}")
                result = process_csv(
                    args.config, input_path,
                    labels_override=labels_yaml,
                    n_workers=args.workers,
                    project=item["project"],
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.phase == 2:
            for item in synced:
                print(f"\n{'='*60}")
                print(f"  运行 Phase 2: {item['project']}")
                print(f"{'='*60}")
                result = process_clusters(args.config, project=item["project"])
                print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 不用 COS，手动指定 input
        if args.phase == 1:
            result = process_csv(
                args.config, args.input,
                n_workers=args.workers,
                project=args.project,
            )
        elif args.phase == 2:
            result = process_clusters(args.config, project=args.project)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
