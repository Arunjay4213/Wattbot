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
    type: str = "text"
    contains_numeric: bool = False
    contains_table: bool = False
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
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing {model_name} on {self.device}...")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.use_instruction = "bge" in model_name.lower()
        self.query_instruction = "Represent this sentence for searching relevant passages: "

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_map: Dict[str, int] = {}

    def encode_chunks(
            self,
            chunks: List[Chunk],
            batch_size: int = 32,
            show_progress: bool = True
    ) -> np.ndarray:
        texts = []
        for chunk in chunks:
            text = chunk.text

            if chunk.section:
                text = f"Section: {chunk.section}\n{text}"
            if chunk.doc_id:
                text = f"Document: {chunk.doc_id}\n{text}"

            texts.append(text)

        cache_key = self._get_cache_key(texts)
        cached_embeddings = self._load_from_cache(cache_key)

        if cached_embeddings is not None:
            print(f"Loaded {len(texts)} embeddings from cache")
            return cached_embeddings

        print(f"Encoding {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self._save_to_cache(cache_key, embeddings)

        return embeddings

    def index_documents(
            self,
            chunks: List[Chunk],
            batch_size: int = 32
    ):
        print(f"Indexing {len(chunks)} chunks...")

        self.chunks = chunks
        self.chunk_map = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
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
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        if self.use_instruction:
            query = self.query_instruction + query

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(self.embeddings, query_embedding)

        if filter_doc_ids:
            mask = np.array([
                chunk.doc_id in filter_doc_ids
                for chunk in self.chunks
            ])
            similarities = similarities * mask - (1 - mask)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

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
        dense_results = self.search(query, top_k=top_k * 2)
        keyword_scores = self._keyword_score(query, self.chunks)

        combined_scores = {}

        for chunk, score in dense_results:
            combined_scores[chunk.chunk_id] = alpha * score

        for chunk, score in keyword_scores[:top_k * 2]:
            chunk_id = chunk.chunk_id
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (1 - alpha) * score
            else:
                combined_scores[chunk_id] = (1 - alpha) * score

        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

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
        query_tokens = set(query.lower().split())
        scores = []

        for chunk in chunks:
            chunk_tokens = set(chunk.text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / len(query_tokens) if query_tokens else 0

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
        if chunk_id not in self.chunk_map:
            raise ValueError(f"Chunk {chunk_id} not found in index")

        idx = self.chunk_map[chunk_id]
        chunk_embedding = self.embeddings[idx]

        similarities = np.dot(self.embeddings, chunk_embedding)
        similarities[idx] = -1

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def save_index(self, path: str):
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
        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        self.chunks = index_data['chunks']
        self.embeddings = index_data['embeddings']
        self.chunk_map = index_data['chunk_map']

        if index_data['model_name'] != self.model_name:
            print(f"Warning: Loaded index was created with {index_data['model_name']}")

        print(f"Loaded index with {len(self.chunks)} chunks")

    def _get_cache_key(self, texts: List[str]) -> str:
        text_str = "".join(texts)
        return hashlib.md5(text_str.encode()).hexdigest()

    def _save_to_cache(self, key: str, embeddings: np.ndarray):
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, embeddings)

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None
