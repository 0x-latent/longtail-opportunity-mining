# Longtail Opportunity Mining — Design Plan

## 1. Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
│  Ingestion   │───>│  Preprocess   │───>│  Label Match  │───>│  Masking   │
│  & Dedup     │    │  & Normalize  │    │  & Tag        │    │  (Soft)    │
└─────────────┘    └──────────────┘    └───────────────┘    └─────┬──────┘
                                                                  │
                   ┌──────────────┐    ┌───────────────┐    ┌─────▼──────┐
                   │  Insight Card │<───│  Cluster      │<───│  Residual  │
                   │  Generation   │    │  Scoring      │    │  Embedding │
                   └──────┬───────┘    └───────────────┘    └────────────┘
                          │
              ┌───────────▼────────────┐
              │  Opportunity Ranking   │
              │  & Export              │
              └────────────────────────┘
```

Pipeline runs as a DAG of stages. Each stage reads from / writes to a shared data store (Parquet files on disk for V1). Stages are independently re-runnable.

---

## 2. Project Structure

```
longtail-opportunity-mining/
├── PLAN_BRIEF.md
├── DESIGN_PLAN.md
├── pyproject.toml              # project metadata, deps
├── config/
│   ├── default.yaml            # pipeline config (paths, model, params)
│   └── labels/
│       └── cold_cough_v1.yaml  # label taxonomy for cold/cough domain
├── src/
│   └── longtail/
│       ├── __init__.py
│       ├── pipeline.py         # orchestrator / DAG runner
│       ├── ingest.py           # data loading, dedup, normalization
│       ├── label_match.py      # label matching engine
│       ├── masking.py          # soft filtering / hard filtering
│       ├── embedding.py        # residual text embedding
│       ├── clustering.py       # clustering + c-TF-IDF
│       ├── scoring.py          # opportunity scoring & ranking
│       ├── insight_cards.py    # card generation (LLM-assisted)
│       └── utils.py            # shared helpers
├── experiments/
│   ├── exp_soft_vs_hard.py
│   ├── exp_cluster_vs_none.py
│   ├── exp_unigram_vs_phrase.py
│   └── exp_topic_vs_combo.py
├── notebooks/
│   └── exploration.ipynb       # manual review / EDA
├── data/                       # gitignored
│   ├── raw/
│   ├── processed/
│   └── output/
├── tests/
│   ├── test_label_match.py
│   ├── test_masking.py
│   └── test_clustering.py
└── scripts/
    └── run_pipeline.py         # CLI entry point
```

---

## 3. Data Schema

### 3.1 Raw Post Record

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | str | Unique identifier |
| `text` | str | Original post text |
| `platform` | str | Source platform |
| `created_at` | datetime | Post timestamp |
| `author_id` | str (optional) | For dedup |
| `metadata` | dict | Platform-specific fields |

### 3.2 Processed Record (after label matching + masking)

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | str | FK to raw |
| `text_clean` | str | Normalized text |
| `text_masked` | str | Text with known labels replaced by mask tokens |
| `text_hard_filtered` | str | Text with matched spans fully removed (for comparison) |
| `matched_labels` | list[MatchedLabel] | All label matches with positions |
| `label_counts` | dict[str, int] | Per-dimension match count |
| `residual_ratio` | float | len(non-mask tokens) / len(all tokens) — to filter posts that become near-empty |

### 3.3 MatchedLabel

```python
@dataclass
class MatchedLabel:
    dimension: str      # "scenario", "symptom", etc.
    label_key: str      # canonical label identifier
    label_text: str     # matched surface form
    span_start: int
    span_end: int
    confidence: float   # 1.0 for exact match, <1 for fuzzy
```

### 3.4 Cluster Record

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | int | Cluster identifier |
| `post_ids` | list[str] | Member posts |
| `size` | int | Number of posts |
| `top_terms` | list[tuple[str, float]] | c-TF-IDF ranked terms/phrases |
| `label_distribution` | dict[str, Counter] | Per-dimension label frequency in cluster |
| `centroid_embedding` | ndarray | Cluster centroid |
| `coherence_score` | float | Intra-cluster similarity |
| `topic_label` | str | Auto-generated cluster name |

### 3.5 Insight Card

```python
@dataclass
class InsightCard:
    card_id: str
    opportunity_name: str
    why_it_matters: str
    representative_posts: list[str]       # 3-5 original posts
    label_distribution: dict[str, Counter]
    top_residual_terms: list[str]
    business_interpretation: str
    follow_up_worthy: bool
    score: OpportunityScore
    output_type: str                      # "topic_group" | "structured_combo"
    structured_combo: dict | None         # e.g. {"scenario": X, "symptom": Y, "need": Z}
```

### 3.6 OpportunityScore

| Factor | Weight (default) | Computation |
|--------|-------------------|-------------|
| `novelty` | 0.35 | Inverse of label-coverage ratio — higher when residual content is dominant |
| `coherence` | 0.25 | Intra-cluster cosine similarity |
| `business_relevance` | 0.25 | Heuristic: presence of need/solution/barrier terms |
| `support_size` | 0.15 | log-scaled cluster size, capped to avoid domination by large clusters |

---

## 4. Label Configuration Format

```yaml
# config/labels/cold_cough_v1.yaml
version: "1.0"
dimensions:
  scenario:
    mask_token: "[SCENE]"
    match_mode: "phrase"          # "exact" | "phrase" | "fuzzy"
    fuzzy_threshold: 0.85         # only used if fuzzy
    entries:
      - key: "nighttime"
        surface_forms: ["at night", "nighttime", "before bed", "sleeping"]
      - key: "workplace"
        surface_forms: ["at work", "office", "meeting"]
      # ...

  symptom:
    mask_token: "[SYMPTOM]"
    match_mode: "phrase"
    entries:
      - key: "sore_throat"
        surface_forms: ["sore throat", "throat pain", "scratchy throat"]
      # ...

  solution:
    mask_token: "[PRODUCT]"
    match_mode: "phrase"
    entries:
      - key: "honey_lemon"
        surface_forms: ["honey and lemon", "honey lemon", "lemon honey tea"]
      # ...

  audience:
    mask_token: "[AUDIENCE]"
    # ...

  emotion:
    mask_token: "[EMOTION]"
    # ...

  efficacy:
    mask_token: "[EFFICACY]"
    # ...

  dosage_form:
    mask_token: "[FORM]"
    # ...

  concern:
    mask_token: "[CONCERN]"
    # ...
```

New dimensions or entries can be added without code changes. The matching engine reads this config at runtime.

---

## 5. Masking Strategy Details

### 5.1 Matching Pipeline

1. **Normalize** text (lowercase, collapse whitespace, expand contractions).
2. **Longest-match-first**: sort all surface forms by length descending, match greedily to avoid partial overlaps.
3. **Overlapping spans**: if two labels from different dimensions overlap, keep the longer match. If equal length, prefer the dimension with higher priority (configurable).
4. **Record** all MatchedLabel objects with span positions.

### 5.2 Soft Filtering (default)

Replace each matched span with its dimension's mask token:

```
Original:  "My sore throat kept me up all night, tried honey lemon tea"
Masked:    "My [SYMPTOM] kept me up all [SCENE], tried [PRODUCT]"
```

The mask tokens are **semantically neutral placeholders** — they preserve sentence structure but remove known-label signal from the embedding space.

### 5.3 Hard Filtering (comparison baseline)

Remove matched spans entirely and collapse whitespace:

```
Hard:      "My kept me up all tried"
```

### 5.4 Residual Ratio Filter

Posts where `residual_ratio < 0.15` (tunable) are flagged as "fully explained by known labels" and excluded from clustering but still counted in label distributions.

---

## 6. Residual Clustering Design

### 6.1 Embedding

- **Default model**: `text-embedding-3-small` (OpenAI) or `BAAI/bge-small-en-v1.5` (local, free).
  - **Recommended default**: `bge-small-en-v1.5` via `sentence-transformers` — no API cost, fast, good quality.
- Input: `text_masked` (soft-filtered text).
- Batch embed all posts; store as numpy memmap or Parquet binary column.

### 6.2 Dimensionality Reduction

- UMAP: `n_components=10`, `n_neighbors=15`, `min_dist=0.0`, `metric="cosine"`.
- 10D output feeds into clustering; 2D projection saved separately for visualization.

### 6.3 Clustering Algorithm

- **HDBSCAN** (default): `min_cluster_size=10`, `min_samples=5`, `cluster_selection_method="eom"`.
  - Naturally produces a noise class (-1) for posts that don't fit any cluster — these are reviewed separately as potential single-post anomalies.
- **Alternative**: KMeans with silhouette-based K selection — simpler but less suited for variable-density longtail data.

### 6.4 c-TF-IDF (Class-based TF-IDF)

Per-cluster term importance to surface what makes each cluster distinctive:

```
c-TF-IDF(t, c) = (f(t,c) / |c|) * log(1 + N / n(t))
```

Where:
- `f(t,c)` = frequency of term `t` in cluster `c`
- `|c|` = total terms in cluster `c`
- `N` = total number of clusters
- `n(t)` = number of clusters containing term `t`

**Tokenization options** (for experiment comparison):
- **Unigram**: standard word tokenization
- **Phrase-level**: use `CountVectorizer(ngram_range=(1,3))` with optional keyphrase extraction via `KeyBERT` or simple n-gram filtering

Mask tokens (`[SYMPTOM]`, etc.) are **excluded** from c-TF-IDF computation — they carry no residual signal.

---

## 7. Two Output Modes

### 7.1 Topic-Group Output

Direct output of clustering: each cluster becomes an insight card. The cluster's top c-TF-IDF terms define the topic. Label distributions provide context about what known dimensions are associated.

**Pros**: discovers truly unknown themes. **Cons**: may produce vague or noisy clusters.

### 7.2 Structured-Combination Output

Cross-tabulate `matched_labels` across dimensions to find interesting co-occurrence patterns:

1. For each post, extract its `(scenario, symptom, need)` tuple from `matched_labels`.
2. Build a co-occurrence matrix / contingency table.
3. Score each combination by:
   - **Frequency**: raw count
   - **Residual richness**: average residual text length for posts in this combo — longer residual = more undiscovered context
   - **Residual cluster diversity**: how many distinct residual clusters does this combo span — more = richer sub-themes

4. For top-ranked combos, generate cards with the residual text clustered within that combo's posts.

**Priority for V1**: `scenario x symptom x need` (where "need" is extracted from residual text, not a pre-defined label).

### 7.3 Hybrid Approach (Recommended Default)

Run both. Use topic-group output for pure discovery. Use structured-combination output to ground discoveries in actionable business coordinates. The insight card schema supports both via the `output_type` field.

---

## 8. Insight Card Generation

Use an LLM (Claude API) to synthesize each cluster into a readable insight card:

**Input to LLM**:
- Top 10 c-TF-IDF terms
- 5-10 representative posts (closest to centroid)
- Label distribution summary
- Cluster size and coherence

**Prompt template** asks the LLM to produce:
- `opportunity_name`: concise label (5-8 words)
- `why_it_matters`: 2-3 sentences on business significance
- `business_interpretation`: suggested marketing angle or product hypothesis
- `follow_up_worthy`: boolean judgment

This is a **generation step, not a classification step** — the LLM adds interpretive value on top of the statistical clustering.

---

## 9. Evaluation Framework

### 9.1 Automated Metrics (Sanity Checks)

| Metric | Target |
|--------|--------|
| Cluster count | 15-100 (tunable; too few = too coarse, too many = noise) |
| Noise ratio (HDBSCAN -1) | < 40% of posts |
| Mean cluster coherence | > 0.3 cosine similarity |
| Residual ratio distribution | Median > 0.3 (masking isn't destroying all signal) |
| c-TF-IDF top terms overlap across clusters | Low (clusters should be distinctive) |

### 9.2 Human Evaluation (Primary)

- **Top-20 review**: manually assess top 20 ranked insight cards.
  - Tag each as: `actionable` / `interesting_but_vague` / `noise` / `already_known`
  - **Success**: >= 5 of top 20 are `actionable`
- **Novelty check**: compare top insights against a baseline of "just show top-frequency labels" — are the longtail results genuinely different?
- **Translation test**: can at least 2-3 cards be turned into a concrete marketing brief or product hypothesis in < 10 minutes?

### 9.3 Experiment Comparison Protocol

For each experiment pair (e.g., soft vs hard filtering):
1. Run both variants on the same data.
2. Compare automated metrics.
3. Compare top-20 human evaluation scores.
4. Document in a comparison table.

---

## 10. Experiment Plan

| # | Experiment | Variable | Metric |
|---|-----------|----------|--------|
| 1 | Soft filtering vs Hard filtering | `masking.mode = soft \| hard` | Cluster quality, top-20 human eval |
| 2 | Clustering + c-TF-IDF vs No clustering (LLM-only summarization) | `pipeline.use_clustering = true \| false` | Actionability of outputs |
| 3 | Unigram vs Phrase-level c-TF-IDF | `ctfidf.ngram_range = (1,1) \| (1,3)` | Term interpretability, human preference |
| 4 | Topic-group vs Structured-combo output | `output.mode = topic \| combo \| hybrid` | Actionability, novelty |

Run experiments sequentially after the MVP pipeline works end-to-end.

---

## 11. Phased Implementation

### Phase 1: Foundation (implement first)
- Project scaffolding, config loading
- Data ingestion, dedup, normalization (`ingest.py`)
- Label config parser + matching engine (`label_match.py`)
- Soft/hard masking (`masking.py`)
- Unit tests for matching and masking
- **Deliverable**: processed Parquet with `text_masked`, `matched_labels`, `residual_ratio`

### Phase 2: Clustering Core
- Embedding generation (`embedding.py`)
- UMAP + HDBSCAN clustering (`clustering.py`)
- c-TF-IDF computation
- Basic cluster inspection notebook
- **Deliverable**: cluster assignments + top terms per cluster

### Phase 3: Output & Scoring
- Opportunity scoring (`scoring.py`)
- Insight card generation via Claude API (`insight_cards.py`)
- Structured-combination cross-tabulation
- Export to JSON / Markdown
- **Deliverable**: ranked insight cards, exportable

### Phase 4: Experiments & Iteration
- Implement experiment scripts for the 4 comparison pairs
- Human evaluation workflow (simple Markdown/CSV review format)
- Iterate on parameters based on results
- **Deliverable**: experiment comparison report

### Phase 5: Polish (optional for V1)
- Interactive dashboard (Streamlit or similar)
- Batch processing for larger datasets
- Label taxonomy expansion tooling

---

## 12. Dependencies

```
# Core
pandas >= 2.0
pyarrow                 # Parquet I/O
pyyaml                  # Config
sentence-transformers   # Local embeddings
umap-learn
hdbscan
scikit-learn            # CountVectorizer, metrics
numpy

# LLM
anthropic               # Claude API for insight generation

# Optional
keybert                 # Phrase extraction
plotly                  # Visualization
streamlit               # Dashboard (Phase 5)
```

---

## 13. Recommended Default Path

For fastest path to useful results:

1. **Embedding**: `bge-small-en-v1.5` (local, no API cost)
2. **Masking**: soft filtering (preserves structure)
3. **Clustering**: HDBSCAN (handles variable density)
4. **c-TF-IDF**: phrase-level `(1,3)` ngrams (more interpretable)
5. **Output**: hybrid (both topic-group and structured-combo)
6. **Insight generation**: Claude API (`claude-sonnet-4-6` for cost/quality balance)
7. **Scoring weights**: novelty-heavy (`novelty=0.35`) to bias toward longtail

---

## 14. Open Questions (Need User Confirmation)

| # | Question | Default Assumption | Impact |
|---|----------|-------------------|--------|
| 1 | **Data format**: What format is the raw ~100k posts in? CSV, JSON, database dump? | CSV/JSON files in `data/raw/` | Ingest module design |
| 2 | **Language**: Chinese, English, or mixed? | Chinese (given domain context) — affects tokenizer, embedding model choice | If Chinese: use `BAAI/bge-small-zh-v1.5` instead, add `jieba` for tokenization |
| 3 | **Label taxonomy**: Do you have an existing label list or should we build one from scratch? | Partial list exists; we'll design config format to accommodate | Phase 1 timeline |
| 4 | **"Need" dimension**: This isn't in the label list but is key for structured combos. Should it be a label dimension, or extracted from residual text? | Extracted from residual text via clustering | Output design |
| 5 | **LLM budget**: Is Claude API usage acceptable for insight card generation (~$0.5-2 per full run)? | Yes | Could fall back to local LLM if not |
| 6 | **Embedding model preference**: Local (free, ~384d) vs API (paid, ~1536d)? | Local | Cost vs quality tradeoff |
| 7 | **Dedup strategy**: Exact dedup only, or near-dedup (MinHash/SimHash)? | Near-dedup with MinHash at 0.8 threshold | Preprocessing complexity |
| 8 | **How to handle very short posts** (< 5 tokens after masking)? | Exclude from clustering, keep in label stats | May lose some signal |

---

## 15. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Masking removes too much signal, residual text is uninformative | Medium | Monitor `residual_ratio` distribution; tune label granularity; experiment with partial masking (mask only high-confidence matches) |
| Clusters are too noisy / vague to interpret | Medium | Tune HDBSCAN `min_cluster_size`; try hierarchical merging; use LLM to filter low-quality clusters |
| Label taxonomy incomplete, misses important known concepts | High | Iterative: run pipeline, inspect "known" terms appearing in residual clusters, add them to taxonomy |
| 100k posts may be too few for fine-grained longtail discovery | Low-Medium | Start with coarser clusters; expand dataset if needed |
| Chinese text requires different NLP tooling than English | High (if Chinese) | Use Chinese-specific models, tokenizers; confirm language before Phase 1 |

---

## 16. Summary

This plan designs a **mask-then-cluster** pipeline that:
1. Tags known concepts in social media posts using a configurable label taxonomy
2. Replaces known concepts with mask tokens to reveal **residual, undiscovered signal**
3. Embeds and clusters the residual text to find **niche opportunity themes**
4. Scores and ranks clusters by novelty, coherence, and business relevance
5. Generates human-readable **insight cards** with LLM assistance

The recommended default path uses local embeddings, soft masking, HDBSCAN clustering, phrase-level c-TF-IDF, and Claude API for card generation. Implementation is split into 4 core phases, with the first deliverable (processed + masked data) achievable quickly, and full end-to-end results by end of Phase 3.

**Next step**: confirm the open questions in Section 14 (especially language and data format), then begin Phase 1 implementation.
