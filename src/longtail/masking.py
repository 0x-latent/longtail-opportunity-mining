from __future__ import annotations

import re
from typing import Iterable

from .label_match import MatchedLabel


class Masker:
    def __init__(self, dimension_to_mask: dict[str, str]):
        self.dimension_to_mask = dimension_to_mask

    def soft_mask(self, text: str, matches: Iterable[MatchedLabel]) -> str:
        return self._apply(text, list(matches), mode="soft")

    def hard_filter(self, text: str, matches: Iterable[MatchedLabel]) -> str:
        return self._apply(text, list(matches), mode="hard")

    def _apply(self, text: str, matches: list[MatchedLabel], mode: str) -> str:
        if not matches:
            return self._clean(text)
        chunks: list[str] = []
        cursor = 0
        for match in sorted(matches, key=lambda x: x.span_start):
            chunks.append(text[cursor:match.span_start])
            if mode == "soft":
                chunks.append(self.dimension_to_mask[match.dimension])
            cursor = match.span_end
        chunks.append(text[cursor:])
        return self._clean("".join(chunks))

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([，。！？,.!?:;；])", r"\1", text)
        return text
