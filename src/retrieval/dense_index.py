"""Dense vector index over chunk embeddings.

Search runs against an HNSW graph (hnswlib) by default. The exact numpy scan is
kept as a second backend, both as a fallback when hnswlib is not installed and
as the reference used to measure the recall of the approximate path.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    hnswlib = None
    HNSWLIB_AVAILABLE = False


DEFAULT_VECTOR_SEARCH = {
    "backend": "hnsw",
    "metric": "cosine",
    "hnsw": {
        "M": 16,
        "ef_construction": 200,
        "ef_search": 100,
        "index_dir": "./data/cache/hnsw",
    },
}

# hnswlib returns a distance. For these two spaces the vectors are unit length,
# so distance = 1 - cosine similarity and the conversion back is exact.
SIMILARITY_SPACES = ("cosine", "ip")


class DenseIndex:
    """Nearest neighbour search over a matrix of L2-normalised chunk embeddings.

    With backend "hnsw" queries run against a persisted HNSW graph, with backend
    "exact" they run against the full matrix with a dot product. Both return
    (row index, cosine similarity) pairs, so the two paths are interchangeable
    from the retriever's point of view.
    """

    def __init__(
        self,
        backend: str = "hnsw",
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        index_dir: str = "./data/cache/hnsw"
    ):
        if metric not in SIMILARITY_SPACES:
            raise ValueError(
                f"Unsupported metric {metric!r}, expected one of {SIMILARITY_SPACES}"
            )
        if backend not in ("hnsw", "exact"):
            raise ValueError(f"Unsupported backend {backend!r}, expected 'hnsw' or 'exact'")

        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_dir = Path(index_dir)

        self.backend = backend
        if backend == "hnsw" and not HNSWLIB_AVAILABLE:
            print("hnswlib is not installed, falling back to exact dense search")
            self.backend = "exact"

        self.embeddings: Optional[np.ndarray] = None
        self.index = None

    @classmethod
    def from_config(cls, vector_search: Optional[Dict] = None) -> "DenseIndex":
        """Build from the retrieval.vector_search block of configs/config.yaml."""
        vector_search = vector_search or {}
        hnsw = dict(DEFAULT_VECTOR_SEARCH["hnsw"])
        hnsw.update(vector_search.get("hnsw") or {})

        return cls(
            backend=vector_search.get("backend", DEFAULT_VECTOR_SEARCH["backend"]),
            metric=vector_search.get("metric", DEFAULT_VECTOR_SEARCH["metric"]),
            M=hnsw["M"],
            ef_construction=hnsw["ef_construction"],
            ef_search=hnsw["ef_search"],
            index_dir=hnsw["index_dir"]
        )

    def build(self, embeddings: np.ndarray):
        """Index an embedding matrix, loading a persisted HNSW graph if one matches.

        The graph is keyed by a fingerprint of the embeddings and the build
        parameters, so a changed corpus or a changed M / ef_construction gets a
        fresh index instead of a stale one.
        """
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = None

        if self.backend != "hnsw":
            return

        fingerprint = self._fingerprint()
        index_path = self.index_dir / f"{fingerprint}.bin"
        meta_path = self.index_dir / f"{fingerprint}.json"

        if self._load_index(index_path, meta_path):
            return

        self._build_index()
        self._save_index(index_path, meta_path, fingerprint)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return the top_k (index, similarity) pairs using the active backend."""
        if self.embeddings is None:
            raise ValueError("No embeddings indexed. Call build first.")

        if self.index is None:
            return self.search_exact(query_embedding, top_k)

        k = min(top_k, len(self.embeddings))
        # ef bounds the search frontier and has to be at least k.
        self.index.set_ef(max(self.ef_search, k))

        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        labels, distances = self.index.knn_query(query, k=k)

        return [
            (int(idx), float(1.0 - dist))
            for idx, dist in zip(labels[0], distances[0])
        ]

    def search_exact(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Exhaustive dot product over every embedding, used as the ground truth."""
        if self.embeddings is None:
            raise ValueError("No embeddings indexed. Call build first.")

        k = min(top_k, len(self.embeddings))
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        similarities = np.dot(self.embeddings, query)
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _build_index(self):
        num_elements, dim = self.embeddings.shape
        print(
            f"Building HNSW index over {num_elements} vectors "
            f"(M={self.M}, ef_construction={self.ef_construction})..."
        )

        index = hnswlib.Index(space=self.metric, dim=dim)
        index.init_index(
            max_elements=num_elements,
            M=self.M,
            ef_construction=self.ef_construction
        )
        index.add_items(self.embeddings, np.arange(num_elements))
        index.set_ef(self.ef_search)

        self.index = index

    def _load_index(self, index_path: Path, meta_path: Path) -> bool:
        if not (index_path.exists() and meta_path.exists()):
            return False

        num_elements, dim = self.embeddings.shape

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if meta.get('num_elements') != num_elements or meta.get('dim') != dim:
                return False

            index = hnswlib.Index(space=self.metric, dim=dim)
            index.load_index(str(index_path), max_elements=num_elements)
        except Exception as e:
            print(f"Could not load HNSW index from {index_path} ({e}), rebuilding")
            return False

        index.set_ef(self.ef_search)
        self.index = index

        print(f"Loaded HNSW index for {num_elements} vectors from {index_path}")
        return True

    def _save_index(self, index_path: Path, meta_path: Path, fingerprint: str):
        num_elements, dim = self.embeddings.shape

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(index_path))

        meta = {
            'fingerprint': fingerprint,
            'num_elements': num_elements,
            'dim': dim,
            'metric': self.metric,
            'M': self.M,
            'ef_construction': self.ef_construction
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved HNSW index to {index_path}")

    def _fingerprint(self) -> str:
        """Hash the embeddings and the parameters that change the built graph.

        ef_search is left out on purpose. It only widens the search at query
        time, so changing it does not invalidate a persisted index.
        """
        digest = hashlib.md5()
        digest.update(self.embeddings.tobytes())
        digest.update(
            f"{self.metric}|{self.M}|{self.ef_construction}".encode()
        )
        return digest.hexdigest()
