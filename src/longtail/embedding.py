from __future__ import annotations

from typing import Iterable


def embed_texts(texts: Iterable[str], model_name: str = "BAAI/bge-small-zh-v1.5"):
    """Phase 2 可运行骨架：如安装 sentence-transformers，则返回 embeddings。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 sentence-transformers，请在 Phase 2 前安装相关依赖") from exc

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), normalize_embeddings=True)
