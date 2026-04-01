from rank_bm25 import BM25Okapi
import json
from pathlib import Path


class BM25Reranker:
    def __init__(self, chunks):
        self.chunks = chunks
        self.documents = [chunk["content"].split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.documents)

    def rerank(self, query, top_n=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return [
            {
                "rank": i + 1,
                "score": float(scores[idx]),
                "content": self.chunks[idx]["content"][:400],
                "type": self.chunks[idx]["type"]
            }
            for i, idx in enumerate(ranked_indices[:top_n])
        ]


if __name__ == "__main__":
    chunks_dir = Path("data/chunks")
    ranked_dir = Path("data/ranked")

    ranked_dir.mkdir(parents=True, exist_ok=True)

    query = "carbon emissions reduction goals"

    for file_path in chunks_dir.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        reranker = BM25Reranker(chunks)
        results = reranker.rerank(query, top_n=20)

        output_file = ranked_dir / f"{file_path.stem}_ranked.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved ranked results to {output_file}")
