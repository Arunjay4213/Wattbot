import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from retrieval.embeddings import EmbeddingRetriever, Chunk
from llm.answer_generator import AnswerGenerator


class VectorDBPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("Initializing Vector DB Pipeline...")

        self.retriever = EmbeddingRetriever(
            model_name=self.config['models']['embedding']['model']
        )
        self.generator = AnswerGenerator(config_path)

        self.chunks_dir = Path(self.config['paths']['chunks_dir'])
        self.index_path = Path(self.config['paths']['index_path'])

    def load_chunks_from_json(self) -> List[Chunk]:
        """Load chunks from JSON files"""
        all_chunks = []

        json_files = list(self.chunks_dir.glob("*_chunks.json"))
        if not json_files:
            raise ValueError(f"No chunk files found in {self.chunks_dir}. Run chunker.py first!")

        for json_file in json_files:
            doc_id = json_file.stem.replace("_chunks", "")

            with open(json_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)

            for i, chunk_dict in enumerate(chunk_data):
                text = chunk_dict['content']

                # Check for numeric content
                contains_numeric = any(
                    word in text.lower()
                    for word in ['kwh', 'mwh', 'co2', 'tco2e', 'gpu', 'percent', 'emissions', 'energy']
                ) or any(char.isdigit() for char in text)

                chunk = Chunk(
                    text=text,
                    chunk_id=f"{doc_id}_chunk_{i}",
                    doc_id=doc_id,
                    contains_table=(chunk_dict['type'] == 'table'),
                    contains_numeric=contains_numeric
                )
                all_chunks.append(chunk)

        print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} documents")
        return all_chunks

    def build_or_load_index(self):
        """Build or load vector index"""
        if self.index_path.exists():
            print(f"Loading existing index from {self.index_path}")
            self.retriever.load_index(str(self.index_path))
        else:
            print("Building new vector index...")
            chunks = self.load_chunks_from_json()

            self.retriever.index_documents(chunks, batch_size=32)

            # Save index
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.retriever.save_index(str(self.index_path))
            print(f"Index saved to {self.index_path}")

    def answer_question(self, question: str) -> Dict:
        """Answer a single question"""
        # Retrieve relevant chunks
        results = self.retriever.search(
            question,
            top_k=self.config['retrieval']['top_k']
        )

        if not results:
            return self.generator._get_fallback_response()

        # Build context and metadata
        context_parts = []
        doc_ids = set()

        for chunk, score in results:
            context_parts.append(f"[Document: {chunk.doc_id}]\n{chunk.text}")
            doc_ids.add(chunk.doc_id)

        context = "\n\n---\n\n".join(context_parts)
        metadata = {"source_docs": list(doc_ids)}

        # Generate answer
        answer = self.generator.generate(question, context, metadata)

        return answer

    def test_on_training_data(self):
        """Test pipeline on training questions"""
        train_df = pd.read_csv("data/raw/train_QA.csv")

        print("\n" + "=" * 60)
        print("Testing on Training Data")
        print("=" * 60)

        correct = 0
        for i in range(min(5, len(train_df))):  # Test first 5
            row = train_df.iloc[i]
            answer = self.answer_question(row['question'])

            print(f"\n[{i + 1}] Question: {row['question'][:100]}...")
            print(f"Generated: {answer['answer_value']} {answer['answer_unit']}")
            print(f"Expected:  {row['answer_value']} {row['answer_unit']}")

            # Check if answer matches
            if str(answer['answer_value']).strip() == str(row['answer_value']).strip():
                correct += 1
                print("Correct!")
            else:
                print("Different")

        print(f"\n Accuracy: {correct}/5 = {correct / 5 * 100:.0f}%")

    def process_test_questions(self, save_every=10):
        """Process all test questions"""
        test_df = pd.read_csv("data/raw/test_Q.csv")

        print("\n" + "=" * 60)
        print(f"Processing {len(test_df)} Test Questions")
        print("=" * 60)

        results = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
            try:
                answer = self.answer_question(row['question'])
                answer['id'] = row['id']
                results.append(answer)

                # Save periodically
                if (idx + 1) % save_every == 0:
                    self._save_results(results, f'checkpoint_{idx + 1}.csv')

            except Exception as e:
                print(f"\nError on {row['id']}: {e}")
                # Add fallback answer
                answer = self.generator._get_fallback_response()
                answer['id'] = row['id']
                results.append(answer)

        # Final save
        self._save_results(results, 'submission.csv')
        print(f"\n Saved final submission with {len(results)} answers")
        return results

    def _save_results(self, results: List[Dict], filename: str):
        """Save results to CSV"""
        df = pd.DataFrame(results)

        # Ensure all required columns
        required_columns = ['id', 'answer', 'answer_value', 'answer_unit',
                            'ref_id', 'ref_url', 'supporting_materials', 'explanation']

        for col in required_columns:
            if col not in df.columns:
                df[col] = 'is_blank'

        # Reorder columns
        df = df[required_columns]

        # Save
        output_path = Path("data/processed") / filename
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")