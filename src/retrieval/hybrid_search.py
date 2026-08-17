import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re


@dataclass
class Chunk:
    """Data class for document chunks"""
    text: str
    chunk_id: str
    doc_id: str
    page_num: Optional[int] = None
    section: Optional[str] = None
    contains_table: bool = False
    contains_numeric: bool = False
    metadata: Optional[Dict] = None


class HybridRetriever:
    """
    Combines dense embeddings with BM25 sparse retrieval
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_stemming: bool = True
    ):
        print(f"Initializing Hybrid Retriever...")

        self.embedding_model = SentenceTransformer(embedding_model)

        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        self.tokenized_chunks = []

        self.use_stemming = use_stemming
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                self.stemmer = PorterStemmer()
            except ImportError:
                print("NLTK not found, disabling stemming")
                self.use_stemming = False
                self.stemmer = None

    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def index_documents(self, chunks: List[Chunk]):
        print(f"Indexing {len(chunks)} chunks for hybrid search...")

        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]
        self.embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.tokenized_chunks = [
            self.preprocess_text(chunk.text)
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        print(f"✅ Indexed {len(chunks)} chunks for hybrid search")

    def search_dense(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results

    def search_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        tokenized_query = self.preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def reciprocal_rank_fusion(
        self,
        results_lists: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        rrf_scores = {}

        for results in results_lists:
            for rank, (idx, _) in enumerate(results):
                if idx not in rrf_scores:
                    rrf_scores[idx] = 0
                rrf_scores[idx] += 1 / (k + rank + 1)

        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        method: str = "rrf"
    ) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            raise ValueError("No documents indexed!")

        dense_results = self.search_dense(query, top_k=top_k*2)
        sparse_results = self.search_sparse(query, top_k=top_k*2)

        if method == "rrf":
            combined = self.reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=60
            )

        elif method == "weighted":
            combined_scores = {}

            if dense_results:
                max_dense = max(score for _, score in dense_results)
                min_dense = min(score for _, score in dense_results)
                range_dense = max_dense - min_dense if max_dense != min_dense else 1

                for idx, score in dense_results:
                    norm_score = (score - min_dense) / range_dense
                    combined_scores[idx] = alpha * norm_score

            if sparse_results:
                max_sparse = max(score for _, score in sparse_results)
                min_sparse = min(score for _, score in sparse_results)
                range_sparse = max_sparse - min_sparse if max_sparse != min_sparse else 1

                for idx, score in sparse_results:
                    norm_score = (score - min_sparse) / range_sparse
                    if idx in combined_scores:
                        combined_scores[idx] += (1 - alpha) * norm_score
                    else:
                        combined_scores[idx] = (1 - alpha) * norm_score

            combined = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

        else:
            combined_scores = {}

            for idx, score in dense_results:
                combined_scores[idx] = alpha * score

            for idx, score in sparse_results:
                if idx in combined_scores:
                    combined_scores[idx] += (1 - alpha) * score
                else:
                    combined_scores[idx] = (1 - alpha) * score

            combined = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

        results = []
        for idx, score in combined[:top_k]:
            results.append((self.chunks[idx], score))

        return results

    def rerank_with_keywords(
        self,
        query: str,
        initial_results: List[Tuple[Chunk, float]],
        boost_keywords: List[str] = None
    ) -> List[Tuple[Chunk, float]]:
        if boost_keywords is None:
            boost_keywords = [
                'co2', 'carbon', 'emission', 'energy', 'kwh', 'mwh',
                'pue', 'gpu', 'training', 'bert', 'gpt', 'efficiency'
            ]

        reranked = []
        query_lower = query.lower()

        for chunk, score in initial_results:
            boost = 1.0
            chunk_lower = chunk.text.lower()

            for keyword in boost_keywords:
                if keyword in query_lower and keyword in chunk_lower:
                    boost *= 1.2

            if chunk.contains_numeric and any(
                q in query_lower for q in ['how much', 'how many', 'what is the']
            ):
                boost *= 1.3

            if chunk.section and chunk.section.lower() in query_lower:
                boost *= 1.1

            reranked.append((chunk, score * boost))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
