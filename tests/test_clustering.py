from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longtail.clustering import MASK_PATTERN, build_cluster_summaries, compute_ctfidf

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import hdbscan  # noqa: F401
    HAS_HDBSCAN = True
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN  # noqa: F401
        HAS_HDBSCAN = True
    except ImportError:
        HAS_HDBSCAN = False

try:
    import umap  # noqa: F401
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

needs_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
phase2 = pytest.mark.skipif(not (HAS_HDBSCAN and HAS_UMAP), reason="Phase 2 deps not installed")


# ---------- c-TF-IDF tests ----------

@needs_sklearn
def test_ctfidf_basic():
    """c-TF-IDF should rank cluster-specific terms higher."""
    texts = ["苹果 苹果 橙子", "苹果 苹果 橙子", "香蕉 香蕉 西瓜", "香蕉 香蕉 西瓜"]
    labels = np.array([0, 0, 1, 1])

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    result = compute_ctfidf(texts, labels, ngram_range=(1, 1), top_n=5, min_df=1,
                            tokenizer_fn=simple_tokenizer)
    assert 0 in result
    assert 1 in result
    # Cluster 0 should have 苹果 as top term
    top_terms_0 = [t for t, _ in result[0]]
    assert "苹果" in top_terms_0[:2]
    # Cluster 1 should have 香蕉 as top term
    top_terms_1 = [t for t, _ in result[1]]
    assert "香蕉" in top_terms_1[:2]


@needs_sklearn
def test_ctfidf_excludes_mask_tokens():
    """Mask tokens like [SYMPTOM] should not appear in c-TF-IDF output."""
    texts = ["[SYMPTOM] 咳嗽 厉害", "[SYMPTOM] 咳嗽 厉害", "发烧 [PRODUCT] 有效", "发烧 [PRODUCT] 有效"]
    labels = np.array([0, 0, 1, 1])

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    result = compute_ctfidf(texts, labels, ngram_range=(1, 1), top_n=5, min_df=1,
                            tokenizer_fn=simple_tokenizer)
    for cid, terms in result.items():
        for term, _ in terms:
            assert not MASK_PATTERN.match(term), f"Mask token '{term}' should be excluded"


@needs_sklearn
def test_ctfidf_skips_noise():
    """Noise cluster (-1) should not appear in c-TF-IDF output."""
    texts = ["foo bar", "baz qux", "hello world"]
    labels = np.array([-1, 0, 0])

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    result = compute_ctfidf(texts, labels, ngram_range=(1, 1), top_n=5, min_df=1,
                            tokenizer_fn=simple_tokenizer)
    assert -1 not in result


# ---------- Cluster summary tests ----------

def test_build_cluster_summaries_structure():
    """Summary output should have all required fields."""
    post_ids = ["a", "b", "c", "d"]
    labels = np.array([0, 0, 1, 1])
    embeddings = np.random.randn(4, 8).astype(np.float32)
    ctfidf_terms = {0: [("term1", 0.5)], 1: [("term2", 0.3)]}
    label_counts = [
        json.dumps({"symptom": 2}),
        json.dumps({"symptom": 1, "scenario": 1}),
        json.dumps({"solution": 1}),
        json.dumps({}),
    ]

    summaries = build_cluster_summaries(post_ids, labels, embeddings, ctfidf_terms, label_counts)
    assert len(summaries) == 2
    for s in summaries:
        assert "cluster_id" in s
        assert "size" in s
        assert "post_ids" in s
        assert "top_terms" in s
        assert "coherence_score" in s
        assert "label_distribution" in s
        assert 0 < s["coherence_score"] <= 1.0


def test_coherence_identical_embeddings():
    """Identical embeddings should give coherence close to 1.0."""
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.tile(vec, (5, 1))
    labels = np.array([0, 0, 0, 0, 0])

    summaries = build_cluster_summaries(
        post_ids=["a", "b", "c", "d", "e"],
        labels=labels,
        embeddings=embeddings,
        ctfidf_terms={0: []},
        label_counts_json=[json.dumps({}) for _ in range(5)],
    )
    assert summaries[0]["coherence_score"] > 0.99


# ---------- Integration tests ----------

@phase2
def test_reduce_and_cluster():
    """UMAP + HDBSCAN on synthetic separable data."""
    from longtail.clustering import cluster_embeddings, reduce_embeddings

    rng = np.random.RandomState(42)
    # 3 well-separated clusters in 50D
    data = np.vstack([
        rng.randn(50, 50) + np.array([10, 0, 0] + [0] * 47),
        rng.randn(50, 50) + np.array([0, 10, 0] + [0] * 47),
        rng.randn(50, 50) + np.array([0, 0, 10] + [0] * 47),
    ]).astype(np.float32)

    reduced = reduce_embeddings(data, n_components=5, n_neighbors=10)
    assert reduced.shape == (150, 5)

    labels, _ = cluster_embeddings(reduced, min_cluster_size=10, min_samples=3)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2  # should find at least 2 of the 3 clusters
