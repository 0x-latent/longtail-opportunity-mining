# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese social media text mining pipeline for discovering niche ("longtail") opportunities in the cold/cough domain. Uses a **mask-then-cluster** approach: tag known concepts via a YAML label taxonomy, soft-mask them, then cluster the residual text to surface novel signals.

**Implementation status**: Phase 1 (ingest → label match → masking) is complete. Phase 2 (embedding + clustering) has stubs. Phases 3–5 (scoring, insight generation, experiments, dashboard) are not started. See `DESIGN_PLAN.md` for the full spec.

## Commands

```bash
# Install (editable, from project root)
pip install -e .

# Run Phase 1 pipeline
python scripts/run_pipeline.py

# Run with custom input
python scripts/run_pipeline.py --input data/raw/other.csv

# Parse raw label columns from CSV into structured label files
python scripts/parse_labels.py
python scripts/parse_labels.py -i data/raw/other.csv --encoding gbk

# Run all tests
pytest

# Run a single test file
pytest tests/test_label_match.py

# Run a single test by name
pytest tests/test_label_match.py::test_longest_match_first -v
```

No linter or formatter is configured.

## Architecture

**5-stage pipeline DAG** (only stages 1–3 implemented):

1. **Ingest** (`ingest.py`) — Load CSV (`post_id`, `text`), normalize Unicode (NFKC), clean social media noise (URLs, hashtags, @mentions, emoji, platform watermarks), collapse whitespace, filter short texts, deduplicate
2. **Label Match** (`label_match.py`) — Regex-based matching against `config/labels/*.yaml`; longest-match-first, no overlapping spans, dimension-priority tiebreaker
3. **Mask** (`masking.py`) — **Soft mask**: replace matched spans with dimension tokens like `[SYMPTOM]`. **Hard filter**: remove matched spans entirely. Both produce cleaned text for downstream use
4. **Cluster** (`clustering.py`) — *Stub*: UMAP + HDBSCAN on embedded residual text
5. **Insight** — *Not started*: opportunity scoring + Claude API insight cards

Entry point: `scripts/run_pipeline.py` → `src/longtail/pipeline.py:process_csv()` orchestrates the DAG.

**Configuration**: `config/default.yaml` sets language, label file path, dimension priority list, and residual ratio threshold (0.15). Label taxonomy lives in `config/labels/cold_cough_zh_v1.yaml` (8 dimensions: scenario, symptom, solution, audience, emotion, efficacy, dosage_form, concern).

**Key data flow**: CSV → `load_csv` (auto-detect encoding, column rename via `column_map`) → `normalize_dataframe` (social text cleaning + NFKC + short text filter) → `deduplicate_dataframe` → `LabelMatcher.match()` per row → `Masker.soft_mask()` / `Masker.hard_filter()` → `residual_ratio` computed via jieba tokenization → Parquet + CSV under `data/processed/`, JSON summary under `data/output/`.

**Label matching internals**: `LabelMatcher._build_patterns()` sorts all surface forms by descending length then ascending dimension priority. `match()` iterates patterns against text via `re.finditer(re.escape(...))`, rejecting any span that overlaps an already-matched region. This guarantees longest-match-first with priority tiebreaking.

## Conventions

- Python ≥3.11; `from __future__ import annotations` everywhere
- Pathlib for all file I/O
- Optional heavy dependencies (sentence-transformers, hdbscan, anthropic) guarded by `ImportError` with helpful messages
- Chinese text processing: jieba for tokenization, UTF-8 encoding throughout
- CSV encoding auto-detection tries utf-8-sig → utf-8 → gbk → gb18030
- Default column mapping: `ID→post_id`, `文本→text`, `标签→labels_raw`, `平台→platform` (configurable via `ingest.column_map` in config)
- `residual_ratio = (non-mask tokens) / (total tokens)` via jieba segmentation — posts below threshold (0.15) are mostly known labels
- `data/` is gitignored; raw CSVs go in `data/raw/`, processed outputs in `data/processed/`, summaries in `data/output/`
- Tests use `pytest`; test files manually prepend `src/` to `sys.path` so imports work without installation
