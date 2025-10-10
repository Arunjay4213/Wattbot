import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
import hashlib
from pathlib import Path


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


class EmbeddingRetriever:
    """
    Handles embedding generation and similarity search for document retrieval
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-large-en-v1.5",
            cache_dir: str = "./data/cache/embeddings",
            device: str = None
    ):
        """
        Initialize the embedding retriever

        Args:
            model_name: HuggingFace model name for embeddings
            cache_dir: Directory to cache embeddings
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            #u should probably do this on colab after choosing the a100 gpu
        print(f"Initializing {model_name} on {self.device}...")

        # Load the model
        self.model = SentenceTransformer(model_name, device=self.device)

        # For BGE models, add instruction prefix
        self.use_instruction = "bge" in model_name.lower()
        self.query_instruction = "Represent this sentence for searching relevant passages: "

        # Storage for chunks and embeddings
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_map: Dict[str, int] = {}  # chunk_id to index mapping

    def encode_chunks(
            self,
            chunks: List[Chunk],
            batch_size: int = 32,
            show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of chunks into embeddings

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        # Extract texts
        texts = []
        for chunk in chunks:
            text = chunk.text

            # Add metadata to text if available (helps with retrieval)
            if chunk.section:
                text = f"Section: {chunk.section}\n{text}"
            if chunk.doc_id:
                text = f"Document: {chunk.doc_id}\n{text}"

            texts.append(text)

        # Check cache first
        cache_key = self._get_cache_key(texts)
        cached_embeddings = self._load_from_cache(cache_key)

        if cached_embeddings is not None:
            print(f"Loaded {len(texts)} embeddings from cache")
            return cached_embeddings

        # Generate embeddings
        print(f"Encoding {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for dot product similarity
        )

        # Save to cache
        self._save_to_cache(cache_key, embeddings)

        return embeddings

    def index_documents(
            self,
            chunks: List[Chunk],
            batch_size: int = 32
    ):
        """
        Index a collection of document chunks

        Args:
            chunks: List of Chunk objects to index
            batch_size: Batch size for encoding
        """
        print(f"Indexing {len(chunks)} chunks...")

        # Store chunks
        self.chunks = chunks

        # Create chunk_id to index mapping
        self.chunk_map = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

        # Generate embeddings
        self.embeddings = self.encode_chunks(chunks, batch_size)

        print(f"Successfully indexed {len(chunks)} chunks")
        print(f"Embedding shape: {self.embeddings.shape}")

    def search(
            self,
            query: str,
            top_k: int = 5,
            filter_doc_ids: Optional[List[str]] = None,
            min_score: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks

        Args:
            query: Query string
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc_ids to search within
            min_score: Minimum similarity score threshold

        Returns:
            List of (chunk, score) tuples
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        # Encode query with instruction for BGE models
        if self.use_instruction:
            query = self.query_instruction + query

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate similarities (dot product since embeddings are normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Apply doc_id filter if specified
        if filter_doc_ids:
            mask = np.array([
                chunk.doc_id in filter_doc_ids
                for chunk in self.chunks
            ])
            similarities = similarities * mask - (1 - mask)  # Set filtered to -1

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Filter by minimum score and create results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))

        return results

    def hybrid_search(
            self,
            query: str,
            top_k: int = 5,
            alpha: float = 0.5
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid search combining dense and keyword matching

        Args:
            query: Query string
            top_k: Number of results
            alpha: Weight for dense search (1-alpha for keyword)

        Returns:
            List of (chunk, score) tuples
        """
        # Dense search
        dense_results = self.search(query, top_k=top_k * 2)

        # Simple keyword matching (BM25 alternative)
        keyword_scores = self._keyword_score(query, self.chunks)

        # Combine scores
        combined_scores = {}

        # Add dense scores
        for chunk, score in dense_results:
            combined_scores[chunk.chunk_id] = alpha * score

        # Add keyword scores
        for chunk, score in keyword_scores[:top_k * 2]:
            chunk_id = chunk.chunk_id
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (1 - alpha) * score
            else:
                combined_scores[chunk_id] = (1 - alpha) * score

        # Sort and get top k
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Convert back to chunk objects
        results = []
        for chunk_id, score in sorted_chunks:
            idx = self.chunk_map[chunk_id]
            results.append((self.chunks[idx], score))

        return results

    def _keyword_score(
            self,
            query: str,
            chunks: List[Chunk]
    ) -> List[Tuple[Chunk, float]]:
        """
        Simple keyword scoring (TF-IDF approximation)

        Args:
            query: Query string
            chunks: List of chunks

        Returns:
            List of (chunk, score) tuples
        """
        query_tokens = set(query.lower().split())
        scores = []

        for chunk in chunks:
            chunk_tokens = set(chunk.text.lower().split())

            # Calculate overlap
            overlap = len(query_tokens & chunk_tokens)

            # Normalize by query length
            score = overlap / len(query_tokens) if query_tokens else 0

            # Boost if contains numbers/units (important for WattBot)
            if chunk.contains_numeric and any(
                    token in query.lower()
                    for token in ['co2', 'emission', 'kwh', 'pue', 'energy']
            ):
                score *= 1.5

            scores.append((chunk, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def find_similar_chunks(
            self,
            chunk_id: str,
            top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Find chunks similar to a given chunk

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return

        Returns:
            List of (chunk, score) tuples
        """
        if chunk_id not in self.chunk_map:
            raise ValueError(f"Chunk {chunk_id} not found in index")

        idx = self.chunk_map[chunk_id]
        chunk_embedding = self.embeddings[idx]

        # Calculate similarities
        similarities = np.dot(self.embeddings, chunk_embedding)

        # Exclude the chunk itself
        similarities[idx] = -1

        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def save_index(self, path: str):
        """Save the indexed chunks and embeddings"""
        index_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'chunk_map': self.chunk_map,
            'model_name': self.model_name
        }

        with open(path, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"Index saved to {path}")

    def load_index(self, path: str):
        """Load a saved index"""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        self.chunks = index_data['chunks']
        self.embeddings = index_data['embeddings']
        self.chunk_map = index_data['chunk_map']

        if index_data['model_name'] != self.model_name:
            print(f"Warning: Loaded index was created with {index_data['model_name']}")

        print(f"Loaded index with {len(self.chunks)} chunks")

    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts"""
        text_str = "".join(texts)
        return hashlib.md5(text_str.encode()).hexdigest()

    def _save_to_cache(self, key: str, embeddings: np.ndarray):
        """Save embeddings to cache"""
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, embeddings)

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None


# Specialized retriever for scientific papers
class ScientificPaperRetriever(EmbeddingRetriever):
    """
    Specialized retriever for scientific papers with domain-specific enhancements
    """

    def __init__(self, cache_dir: str = "./data/cache/embeddings"):
        # Use SciBERT for better scientific text understanding
        super().__init__(
            model_name="allenai/scibert_scivocab_uncased",
            cache_dir=cache_dir
        )

        # Domain-specific keywords for boosting
        self.boost_keywords = {
            'emission', 'co2', 'carbon', 'energy', 'kwh', 'mwh',
            'pue', 'wue', 'efficiency', 'consumption', 'footprint',
            'training', 'inference', 'gpu', 'tpu', 'bert', 'gpt'
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[Chunk, float]]:
        """Enhanced search with domain-specific boosting"""
        results = super().search(query, top_k=top_k * 2, **kwargs)

        # Boost scores based on domain relevance
        boosted_results = []
        query_lower = query.lower()

        for chunk, score in results:
            boost = 1.0

            # Check for keyword presence
            chunk_lower = chunk.text.lower()
            for keyword in self.boost_keywords:
                if keyword in query_lower and keyword in chunk_lower:
                    boost *= 1.2

            # Boost if chunk contains numbers and query asks for numbers
            if any(word in query_lower for word in ['how much', 'how many', 'what is the']):
                if chunk.contains_numeric:
                    boost *= 1.3

            boosted_results.append((chunk, score * boost))

        # Resort and return top k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]


# Example usage and testing
if __name__ == "__main__":
    # Test the retriever
    retriever = EmbeddingRetriever()

    # Create sample chunks
    sample_chunks = [
        Chunk(
            text="Training BERT-base produced 1438 lbs of CO2 emissions.",
            chunk_id="chunk_1",
            doc_id="strubell2019",
            contains_numeric=True
        ),
        Chunk(
            text="The PUE of modern data centers ranges from 1.2 to 1.8.",
            chunk_id="chunk_2",
            doc_id="patterson2022",
            contains_numeric=True
        ),
        Chunk(
            text="Renewable energy adoption is crucial for sustainable AI.",
            chunk_id="chunk_3",
            doc_id="wu2022",
            contains_numeric=False
        )
    ]

    # Index the chunks
    retriever.index_documents(sample_chunks)

    # Test search
    query = "What is the CO2 emission of BERT training?"
    results = retriever.search(query, top_k=2)

    print(f"\nQuery: {query}")
    print("\nResults:")
    for chunk, score in results:
        print(f"Score: {score:.4f} | Doc: {chunk.doc_id}")
        print(f"Text: {chunk.text[:100]}...")
        print()