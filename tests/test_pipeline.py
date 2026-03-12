from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.pipeline import process_csv


def test_pipeline_generates_outputs(tmp_path):
    config_path = PROJECT_ROOT / "config/default.yaml"
    result = process_csv(config_path, PROJECT_ROOT / "data/raw/sample_posts.csv")
    assert result["rows"] >= 1
    processed_csv = Path(result["files"]["csv"])
    assert processed_csv.exists()
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    assert "files" in summary
