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

from longtail.pipeline import process_clusters, process_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run longtail opportunity mining pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--input", default=None, help="Input CSV (Phase 1 only)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Pipeline phase: 1=preprocess, 2=clustering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    if args.phase == 1:
        result = process_csv(args.config, args.input)
    elif args.phase == 2:
        result = process_clusters(args.config)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
