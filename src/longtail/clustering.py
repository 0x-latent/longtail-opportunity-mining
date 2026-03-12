from __future__ import annotations


def cluster_embeddings(embeddings, min_cluster_size: int = 8, min_samples: int = 3):
    """Phase 2 可运行骨架：如安装 umap/hdbscan，则返回 cluster labels。"""
    try:
        import umap
        import hdbscan
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 umap-learn / hdbscan，请在 Phase 2 前安装相关依赖") from exc

    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(reduced)
    return reduced, labels
