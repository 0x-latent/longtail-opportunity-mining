from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.label_match import MatchedLabel
from longtail.masking import Masker
from longtail.utils import residual_ratio


def test_soft_mask_and_hard_filter():
    masker = Masker({"symptom": "[SYMPTOM]", "scenario": "[SCENE]"})
    text = "晚上咳嗽睡不着"
    matches = [
        MatchedLabel("scenario", "night", "晚上", 0, 2),
        MatchedLabel("symptom", "cough", "咳嗽", 2, 4),
    ]
    assert masker.soft_mask(text, matches) == "[SCENE][SYMPTOM]睡不着"
    assert masker.hard_filter(text, matches) == "睡不着"


def test_residual_ratio_is_bounded():
    ratio = residual_ratio("[SCENE][SYMPTOM]睡不着", ["[SCENE]", "[SYMPTOM]"])
    assert 0 <= ratio <= 1
