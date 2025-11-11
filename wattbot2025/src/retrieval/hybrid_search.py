import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from collections import Counter
import math

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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_stemming: bool = True
    ):
        """
        Initialize hybrid retriever

        Args:
            embedding_model: Model for dense embeddings
            use_stemming: Whether to use stemming for BM25
        """
        print(f"Initializing Hybrid Retriever...")

        # Dense retriever (embeddings)
        self.embedding_model = SentenceTransformer(embedding_model)

        # Storage
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        self.tokenized_chunks = []

        # Preprocessing
        self.use_stemming = use_stemming
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                self.stemmer = PorterStemmer()
            except ImportError:
                print("NLTK not found, disabling stemming")
                self.use_stemming = False
                self.stemmer = None

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Simple tokenization (can be improved with NLTK)
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Apply stemming if available
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def index_documents(self, chunks: List[Chunk]):
        """
        Index documents for both dense and sparse retrieval

        Args:
            chunks: List of document chunks
        """
        print(f"Indexing {len(chunks)} chunks for hybrid search...")

        self.chunks = chunks

        # 1. Create dense embeddings
        texts = [chunk.text for chunk in chunks]
        self.embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # 2. Create BM25 index
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
        """
        Dense embedding search

        Returns:
            List of (chunk_index, score) tuples
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Cosine similarity (dot product for normalized embeddings)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results

    def search_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        BM25 sparse search

        Returns:
            List of (chunk_index, score) tuples
        """
        tokenized_query = self.preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def reciprocal_rank_fusion(
        self,
        results_lists: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion to combine multiple ranked lists

        Args:
            results_lists: List of result lists, each containing (index, score) tuples
            k: RRF parameter (typically 60)

        Returns:
            Combined ranked list of (index, score) tuples
        """
        rrf_scores = {}

        for results in results_lists:
            for rank, (idx, _) in enumerate(results):
                if idx not in rrf_scores:
                    rrf_scores[idx] = 0
                # RRF formula: 1 / (k + rank)
                rrf_scores[idx] += 1 / (k + rank + 1)

        # Sort by RRF score
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
        method: str = "weighted"
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform hybrid search combining dense and sparse methods

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for sparse)
            method: Combination method ("weighted", "rrf", or "linear")

        Returns:
            List of (chunk, score) tuples
        """
        if not self.chunks:
            raise ValueError("No documents indexed!")

        # Get results from both methods
        dense_results = self.search_dense(query, top_k=top_k*2)
        sparse_results = self.search_sparse(query, top_k=top_k*2)

        if method == "rrf":
            # Reciprocal Rank Fusion
            combined = self.reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=60
            )

        elif method == "weighted":
            # Weighted combination with normalization
            combined_scores = {}

            # Normalize dense scores
            if dense_results:
                max_dense = max(score for _, score in dense_results)
                min_dense = min(score for _, score in dense_results)
                range_dense = max_dense - min_dense if max_dense != min_dense else 1

                for idx, score in dense_results:
                    norm_score = (score - min_dense) / range_dense
                    combined_scores[idx] = alpha * norm_score

            # Normalize sparse scores
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

        else:  # linear
            # Simple linear combination
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

        # Convert to chunk objects
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
        """
        Rerank results based on keyword presence

        Args:
            query: Original query
            initial_results: Initial search results
            boost_keywords: Keywords to boost (e.g., ['CO2', 'emissions'])

        Returns:
            Reranked results
        """
        if boost_keywords is None:
            # Default keywords for environmental AI domain
            boost_keywords = [
                'co2', 'carbon', 'emission', 'energy', 'kwh', 'mwh',
                'pue', 'gpu', 'training', 'bert', 'gpt', 'efficiency'
            ]

        reranked = []
        query_lower = query.lower()

        for chunk, score in initial_results:
            boost = 1.0
            chunk_lower = chunk.text.lower()

            # Check for keyword matches
            for keyword in boost_keywords:
                if keyword in query_lower and keyword in chunk_lower:
                    boost *= 1.2

            # Boost numeric chunks for quantitative questions
            if chunk.contains_numeric and any(
                q in query_lower for q in ['how much', 'how many', 'what is the']
            ):
                boost *= 1.3

            # Boost if section matches query intent
            if chunk.section and chunk.section.lower() in query_lower:
                boost *= 1.1

            reranked.append((chunk, score * boost))

        # Sort by boosted scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


# Test implementation
def test_hybrid_search():
    """Test the hybrid search functionality"""

    # Initialize retriever
    retriever = HybridRetriever()

    # Create test chunks
    test_chunks = [
        Chunk(
            text="Training BERT-base produced 1438 lbs of CO2 emissions according to Strubell et al. 2019 study on NLP model carbon footprint.",
            chunk_id="c1",
            doc_id="strubell2019",
            contains_numeric=True
        ),
        Chunk(
            text="The Power Usage Effectiveness (PUE) of modern data centers typically ranges from 1.2 to 1.8, with best-in-class facilities achieving 1.1.",
            chunk_id="c2",
            doc_id="patterson2022",
            contains_numeric=True
        ),
        Chunk(
            text="GPT-3 training consumed approximately 1287 MWh of electricity, equivalent to the annual consumption of 126 US homes.",
            chunk_id="c3",
            doc_id="brown2020",
            contains_numeric=True
        ),
        Chunk(
            text="Renewable energy adoption and carbon-neutral computing are essential for sustainable AI development in the future.",
            chunk_id="c4",
            doc_id="wu2022",
            contains_numeric=False
        ),
        Chunk(
            text="Model compression techniques like quantization and pruning can reduce inference energy consumption by up to 90%.",
            chunk_id="c5",
            doc_id="han2023",
            contains_numeric=True
        )
    ]

    # Index documents
    retriever.index_documents(test_chunks)

    # Test queries
    test_queries = [
        ("What is the CO2 emission of BERT?", "weighted"),
        ("energy consumption GPT-3", "rrf"),
        ("PUE data center efficiency", "linear")
    ]

    print("\n" + "="*60)
    print("HYBRID SEARCH RESULTS")
    print("="*60)

    for query, method in test_queries:
        print(f"\nQuery: {query}")
        print(f"Method: {method}")
        print("-"*40)

        results = retriever.hybrid_search(
            query,
            top_k=3,
            alpha=0.6,  # Slightly favor dense search
            method=method
        )

        for i, (chunk, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f} | Doc: {chunk.doc_id}")
            print(f"   {chunk.text[:80]}...")

        # Also test with reranking
        reranked = retriever.rerank_with_keywords(query, results)
        print("\n   After keyword boosting:")
        for i, (chunk, score) in enumerate(reranked[:2], 1):
            print(f"   {i}. Boosted Score: {score:.4f} | Doc: {chunk.doc_id}")

    print("\n✅ Hybrid search test complete!")


if __name__ == "__main__":
    # Install required package if needed
    try:
        import rank_bm25
    except ImportError:
        print("Installing rank-bm25...")
        import subprocess
        subprocess.check_call(["pip", "install", "rank-bm25"])

    test_hybrid_search()