from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.pipeline import process_csv


def _make_sample_csv(tmp_path: Path) -> Path:
    """创建一个符合三九数据格式的测试 CSV。"""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "ID,文本,标签,平台\n"
        "abc123,晚上咳嗽睡不着怎么办 #宝妈日常 @DOU+小助手,需求.症状.咳嗽,小红书\n"
        "def456,宝宝三岁反复咳嗽有痰吃什么药好,需求.症状.咳痰,宝宝树\n"
        "ghi789,短文本,需求.其他,抖音\n",
        encoding="utf-8",
    )
    return csv_path


def test_pipeline_generates_outputs(tmp_path):
    config_path = PROJECT_ROOT / "config/default.yaml"
    sample_csv = _make_sample_csv(tmp_path)
    result = process_csv(config_path, sample_csv)
    assert result["rows"] >= 1
    processed_csv = Path(result["files"]["csv"])
    assert processed_csv.exists()
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    assert "files" in summary
