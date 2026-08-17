"""Tests for the HNSW dense index.

These run on synthetic vectors shaped like the real corpus, 4,498 vectors by
1024 dimensions, so they need neither the BGE model nor the embedding cache.
The file is self contained and does not import the rest of the test suite:

    pytest tests/test_dense_index.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.dense_index import DenseIndex, HNSWLIB_AVAILABLE
from scripts.benchmark_hnsw import (
    DIM,
    NUM_VECTORS,
    synthetic_embeddings,
    synthetic_queries
)

requires_hnswlib = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE,
    reason="hnswlib is not installed"
)


@pytest.fixture(scope="module")
def embeddings():
    return synthetic_embeddings()


@pytest.fixture(scope="module")
def queries(embeddings):
    return synthetic_queries(embeddings, num_queries=200)


def recall_at_k(index: DenseIndex, queries: np.ndarray, top_k: int) -> float:
    """Fraction of the exact top-k that the approximate search also returns."""
    hits = 0

    for query in queries:
        approximate = {idx for idx, _ in index.search(query, top_k)}
        exact = {idx for idx, _ in index.search_exact(query, top_k)}
        hits += len(approximate & exact)

    return hits / (len(queries) * top_k)


@requires_hnswlib
@pytest.mark.parametrize("seed", [0, 1])
def test_recall_at_10_against_exact_search(tmp_path, seed):
    """HNSW has to agree with the exhaustive scan on at least 95% of the top 10.

    Recall depends on the vector distribution, and this measures it on the
    synthetic vectors described in scripts/benchmark_hnsw.py, not on real BGE
    embeddings.
    """
    corpus = synthetic_embeddings(NUM_VECTORS, DIM, seed=seed)
    probes = synthetic_queries(corpus, num_queries=200, seed=seed + 100)

    index = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    index.build(corpus)
    assert index.backend == "hnsw"

    assert recall_at_k(index, probes, top_k=10) >= 0.95


@requires_hnswlib
def test_hnsw_scores_are_cosine_similarities(tmp_path, embeddings, queries):
    index = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    index.build(embeddings)

    for query in queries[:20]:
        for idx, score in index.search(query, top_k=10):
            assert score == pytest.approx(float(np.dot(embeddings[idx], query)), abs=1e-5)


@requires_hnswlib
def test_index_is_persisted_and_reused(tmp_path, monkeypatch, embeddings, queries):
    first = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    first.build(embeddings)

    assert len(list(tmp_path.glob("*.bin"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    def fail_on_rebuild(self):
        raise AssertionError("index was rebuilt instead of loaded from disk")

    monkeypatch.setattr(DenseIndex, "_build_index", fail_on_rebuild)

    second = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    second.build(embeddings)

    for query in queries[:20]:
        assert second.search(query, top_k=10) == first.search(query, top_k=10)


@requires_hnswlib
def test_changed_corpus_rebuilds_the_index(tmp_path, embeddings):
    DenseIndex(backend="hnsw", index_dir=str(tmp_path)).build(embeddings)

    changed = embeddings.copy()
    changed[0] = -changed[0]

    rebuilt = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    rebuilt.build(changed)

    assert len(list(tmp_path.glob("*.bin"))) == 2

    top_hit = rebuilt.search(changed[0], top_k=1)[0]
    assert top_hit[0] == 0


@requires_hnswlib
def test_changed_build_parameters_rebuild_the_index(tmp_path, embeddings):
    DenseIndex(backend="hnsw", M=16, index_dir=str(tmp_path)).build(embeddings)
    DenseIndex(backend="hnsw", M=32, index_dir=str(tmp_path)).build(embeddings)

    assert len(list(tmp_path.glob("*.bin"))) == 2


@requires_hnswlib
def test_ef_search_does_not_invalidate_the_index(tmp_path, embeddings):
    DenseIndex(backend="hnsw", ef_search=100, index_dir=str(tmp_path)).build(embeddings)
    DenseIndex(backend="hnsw", ef_search=40, index_dir=str(tmp_path)).build(embeddings)

    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_exact_backend_matches_numpy(tmp_path, embeddings, queries):
    index = DenseIndex(backend="exact", index_dir=str(tmp_path))
    index.build(embeddings)

    for query in queries[:20]:
        similarities = np.dot(embeddings, query)
        expected = np.argsort(similarities)[-10:][::-1]

        results = index.search(query, top_k=10)
        assert [idx for idx, _ in results] == [int(idx) for idx in expected]
        assert [score for _, score in results] == pytest.approx(
            [float(similarities[idx]) for idx in expected]
        )

    assert list(tmp_path.glob("*")) == []


def test_falls_back_to_exact_without_hnswlib(tmp_path, monkeypatch, embeddings, queries):
    monkeypatch.setattr("src.retrieval.dense_index.HNSWLIB_AVAILABLE", False)

    index = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    assert index.backend == "exact"

    index.build(embeddings)
    assert index.search(queries[0], top_k=10) == index.search_exact(queries[0], top_k=10)


def test_top_k_larger_than_corpus_is_clamped(tmp_path, embeddings):
    small = embeddings[:5]

    index = DenseIndex(backend="hnsw", index_dir=str(tmp_path))
    index.build(small)

    assert len(index.search(small[0], top_k=50)) == 5


def test_search_before_build_raises(tmp_path, embeddings):
    index = DenseIndex(backend="hnsw", index_dir=str(tmp_path))

    with pytest.raises(ValueError):
        index.search(embeddings[0], top_k=10)


def test_rejects_unsupported_settings(tmp_path):
    with pytest.raises(ValueError):
        DenseIndex(metric="l2", index_dir=str(tmp_path))

    with pytest.raises(ValueError):
        DenseIndex(backend="qdrant", index_dir=str(tmp_path))


def test_from_config_reads_the_config_block(tmp_path):
    index = DenseIndex.from_config({
        "backend": "exact",
        "metric": "cosine",
        "hnsw": {"M": 32, "ef_search": 64, "index_dir": str(tmp_path)}
    })

    assert index.backend == "exact"
    assert index.M == 32
    assert index.ef_search == 64
    # Left out of the config block, so the module default survives.
    assert index.ef_construction == 200


def test_from_config_defaults_to_hnsw():
    index = DenseIndex.from_config(None)

    assert index.M == 16
    assert index.ef_construction == 200
    assert index.ef_search == 100
    assert index.backend == ("hnsw" if HNSWLIB_AVAILABLE else "exact")
