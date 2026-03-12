from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.label_match import LabelMatcher


def test_label_matcher_finds_chinese_labels():
    matcher = LabelMatcher.from_yaml(PROJECT_ROOT / "config/labels/cold_cough_zh_v1.yaml")
    text = "晚上咳嗽，润喉糖不太管用"
    matches = matcher.match(text)
    dims = [m.dimension for m in matches]
    assert "scenario" in dims
    assert "symptom" in dims
    assert "solution" in dims


def test_longest_match_first_without_overlap():
    config = {
        "dimensions": {
            "symptom": {
                "mask_token": "[SYMPTOM]",
                "entries": [
                    {"key": "a", "surface_forms": ["喉咙"]},
                    {"key": "b", "surface_forms": ["喉咙痛"]},
                ],
            }
        }
    }
    matcher = LabelMatcher(config)
    matches = matcher.match("喉咙痛厉害")
    assert len(matches) == 1
    assert matches[0].label_text == "喉咙痛"
