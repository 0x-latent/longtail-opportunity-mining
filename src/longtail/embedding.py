from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def embed_texts(
    texts: Sequence[str],
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 256,
    normalize: bool = True,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Embed texts with sentence-transformers, with batching and optional disk cache."""
    # Return cached if valid
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            logger.info("Loaded cached embeddings from %s (%d rows)", cache_path, cached.shape[0])
            return cached
        logger.warning("Cache shape mismatch (%d vs %d), re-embedding", cached.shape[0], len(texts))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers，请运行: pip install -e '.[phase2]'"
        ) from exc

    model = SentenceTransformer(model_name)
    texts_list = list(texts)
    logger.info("Embedding %d texts with %s (batch_size=%d)", len(texts_list), model_name, batch_size)

    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Saved embeddings cache to %s", cache_path)

    return embeddings
