"""Microbenchmark of the HNSW index against the exact numpy scan.

This measures the retrieval layer in isolation, on synthetic vectors shaped like
the real corpus (4,498 chunks by 1024 dimensions). It deliberately does not load
BAAI/bge-large-en-v1.5, so it says nothing about end to end question latency,
only about how long a dense lookup takes once a query vector exists.

    python scripts/benchmark_hnsw.py
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.retrieval.dense_index import DenseIndex

# The real corpus: 4,498 chunks embedded by BGE-large into 1024 dimensions.
NUM_VECTORS = 4498
DIM = 1024


def normalize(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)).astype(np.float32)


def synthetic_embeddings(
    num_vectors: int = NUM_VECTORS,
    dim: int = DIM,
    rank: int = 64,
    num_clusters: int = 60,
    seed: int = 0
) -> np.ndarray:
    """Unit vectors with the coarse structure sentence embeddings tend to have.

    Real embeddings are not uniform on the sphere. They sit near a low rank
    subspace with a decaying spectrum, they cluster by topic, and they share a
    common direction. This generator reproduces those three properties, which
    are the ones approximate search is sensitive to. It is a stand in for real
    BGE vectors, not a copy of them.
    """
    rng = np.random.default_rng(seed)

    basis = normalize(rng.normal(size=(rank, dim)))
    common_direction = normalize(rng.normal(size=dim))
    spectrum = 1.0 / np.sqrt(np.arange(1, rank + 1))

    cluster_centers = rng.normal(size=(num_clusters, rank)) * spectrum
    assignments = rng.integers(0, num_clusters, size=num_vectors)
    coefficients = cluster_centers[assignments] + rng.normal(size=(num_vectors, rank)) * spectrum

    return normalize(0.5 * common_direction + coefficients @ basis)


def synthetic_queries(
    embeddings: np.ndarray,
    num_queries: int = 200,
    noise: float = 0.15,
    seed: int = 1
) -> np.ndarray:
    """Query vectors sitting near corpus vectors, as a paraphrase would."""
    rng = np.random.default_rng(seed)

    picks = rng.integers(0, len(embeddings), size=num_queries)
    perturbed = embeddings[picks] + noise * rng.normal(size=(num_queries, embeddings.shape[1]))

    return normalize(perturbed)


def time_search(index: DenseIndex, queries: np.ndarray, top_k: int, exact: bool):
    search = index.search_exact if exact else index.search

    # One untimed pass so neither path pays for a cold cache.
    search(queries[0], top_k)

    latencies = []
    results = []
    for query in queries:
        start = time.perf_counter()
        hits = search(query, top_k)
        latencies.append((time.perf_counter() - start) * 1000.0)
        results.append(hits)

    return latencies, results


def recall_at_k(approximate, exact) -> float:
    hits = 0
    for approx_hits, exact_hits in zip(approximate, exact):
        approx_ids = {idx for idx, _ in approx_hits}
        exact_ids = {idx for idx, _ in exact_hits}
        hits += len(approx_ids & exact_ids)

    return hits / sum(len(exact_hits) for exact_hits in exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-vectors", type=int, default=NUM_VECTORS)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    args = parser.parse_args()

    print(f"Generating {args.num_vectors} synthetic vectors of dimension {args.dim}...")
    embeddings = synthetic_embeddings(args.num_vectors, args.dim)
    queries = synthetic_queries(embeddings, args.num_queries)

    with tempfile.TemporaryDirectory() as index_dir:
        hnsw = DenseIndex(
            backend="hnsw",
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            index_dir=index_dir
        )

        start = time.perf_counter()
        hnsw.build(embeddings)
        build_seconds = time.perf_counter() - start

        if hnsw.backend != "hnsw":
            print("hnswlib is not installed, nothing to compare against")
            return

        index_bytes = sum(path.stat().st_size for path in Path(index_dir).glob("*.bin"))

        exact_latencies, exact_results = time_search(hnsw, queries, args.top_k, exact=True)
        hnsw_latencies, hnsw_results = time_search(hnsw, queries, args.top_k, exact=False)

    recall = recall_at_k(hnsw_results, exact_results)

    print()
    print(f"vectors            {args.num_vectors} x {args.dim}")
    print(f"queries            {args.num_queries}, top_k={args.top_k}")
    print(f"parameters         M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    print(f"build time         {build_seconds:.2f} s")
    print(f"index on disk      {index_bytes / 1e6:.1f} MB")
    print()
    print(f"exact  mean {statistics.mean(exact_latencies):.3f} ms   median {statistics.median(exact_latencies):.3f} ms")
    print(f"hnsw   mean {statistics.mean(hnsw_latencies):.3f} ms   median {statistics.median(hnsw_latencies):.3f} ms")
    print(f"speedup (median)   {statistics.median(exact_latencies) / statistics.median(hnsw_latencies):.1f}x")
    print(f"recall@{args.top_k}          {recall:.4f}")


if __name__ == "__main__":
    main()
