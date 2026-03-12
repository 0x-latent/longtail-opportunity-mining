from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from .config import load_yaml


@dataclass
class MatchedLabel:
    dimension: str
    label_key: str
    label_text: str
    span_start: int
    span_end: int
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurfacePattern:
    dimension: str
    label_key: str
    surface_form: str
    mask_token: str
    priority: int


class LabelMatcher:
    def __init__(self, label_config: dict[str, Any], dimension_priority: list[str] | None = None):
        self.label_config = label_config
        self.dimension_priority = dimension_priority or []
        self.patterns = self._build_patterns()
        self.mask_tokens = {
            dim: cfg.get("mask_token", f"[{dim.upper()}]")
            for dim, cfg in self.label_config.get("dimensions", {}).items()
        }

    @classmethod
    def from_yaml(cls, path: str | Path, dimension_priority: list[str] | None = None) -> "LabelMatcher":
        return cls(load_yaml(path), dimension_priority=dimension_priority)

    def _priority_of(self, dimension: str) -> int:
        if dimension in self.dimension_priority:
            return self.dimension_priority.index(dimension)
        return len(self.dimension_priority) + 100

    def _build_patterns(self) -> list[SurfacePattern]:
        patterns: list[SurfacePattern] = []
        dimensions = self.label_config.get("dimensions", {})
        for dim, cfg in dimensions.items():
            mask_token = cfg.get("mask_token", f"[{dim.upper()}]")
            for entry in cfg.get("entries", []):
                for surface in entry.get("surface_forms", []):
                    patterns.append(
                        SurfacePattern(
                            dimension=dim,
                            label_key=entry["key"],
                            surface_form=surface,
                            mask_token=mask_token,
                            priority=self._priority_of(dim),
                        )
                    )
        patterns.sort(key=lambda p: (-len(p.surface_form), p.priority, p.surface_form))
        return patterns

    def match(self, text: str) -> list[MatchedLabel]:
        occupied: list[tuple[int, int]] = []
        matches: list[MatchedLabel] = []
        for pattern in self.patterns:
            for m in re.finditer(re.escape(pattern.surface_form), text):
                start, end = m.span()
                if self._overlaps((start, end), occupied):
                    continue
                occupied.append((start, end))
                matches.append(
                    MatchedLabel(
                        dimension=pattern.dimension,
                        label_key=pattern.label_key,
                        label_text=m.group(0),
                        span_start=start,
                        span_end=end,
                        confidence=1.0,
                    )
                )
        matches.sort(key=lambda x: x.span_start)
        return matches

    @staticmethod
    def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
        start, end = span
        for s, e in spans:
            if start < e and end > s:
                return True
        return False

    def count_by_dimension(self, matches: list[MatchedLabel]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in matches:
            counts[item.dimension] = counts.get(item.dimension, 0) + 1
        return counts
