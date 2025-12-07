import pandas as pd
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from retrieval.embeddings import EmbeddingRetriever, Chunk
from retrieval.hybrid_search import HybridRetriever
from llm.answer_generator import AnswerGenerator


class VectorDBPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("🚀 Initializing Vector DB Pipeline...")

        # Use hybrid retriever for better results
        self.retriever = HybridRetriever(
            embedding_model=self.config['models']['embedding']['model']
        )
        self.generator = AnswerGenerator(config_path)

        self.chunks_dir = Path(self.config['paths']['chunks_dir'])
        self.index_path = Path(self.config['paths']['index_path'])

        # Load training examples for few-shot prompting
        self.train_df = pd.read_csv("data/raw/train_QA.csv")
        self.few_shot_examples = self._prepare_few_shot_examples()

    def _prepare_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Organize training examples by question type."""
        examples = {
            "numeric": [],
            "true_false": [],
            "named_entity": [],
            "calculation": [],
        }

        for _, row in self.train_df.iterrows():
            q = row['question'].lower()
            example = {
                "question": row['question'],
                "answer": row['answer'],
                "answer_value": row['answer_value'],
                "answer_unit": row['answer_unit'],
            }

            if q.startswith("true or false"):
                examples["true_false"].append(example)
            elif "name of" in q or ("what is" in q and "name" in q):
                examples["named_entity"].append(example)
            elif "difference" in q or "compare" in q or "factor" in q:
                examples["calculation"].append(example)
            else:
                examples["numeric"].append(example)

        return examples

    def classify_question(self, question: str) -> str:
        """Classify question type for specialized handling."""
        q_lower = question.lower()

        if q_lower.startswith("true or false"):
            return "true_false"
        elif "name of" in q_lower or ("what is" in q_lower and "name" in q_lower):
            return "named_entity"
        elif any(w in q_lower for w in ["difference between", "compare", "by what factor", "how many times"]):
            return "calculation"
        else:
            return "numeric"

    def expand_query(self, question: str) -> List[str]:
        """Generate multiple query variants for better retrieval."""
        queries = [question]

        q_lower = question.lower()

        # Simplified version
        simplified = re.sub(r'^(what|how|which|when|where|who|true or false:?)\s+', '', q_lower)
        simplified = re.sub(r'\?$', '', simplified)
        queries.append(simplified)

        # Extract entities and metrics
        entities = re.findall(r'\b(BERT|GPT-\d|LLaMA|T5|BLOOM|OPT|Gemini|Claude)\b', question, re.I)
        metrics = re.findall(r'\b(CO2|carbon|emission|kWh|MWh|GWh|PUE|WUE|energy|water)\b', question, re.I)

        if entities and metrics:
            queries.append(f"{' '.join(entities)} {' '.join(metrics)}")

        # For numeric questions, add table variant
        if any(w in q_lower for w in ["how much", "how many", "what is the"]):
            queries.append(f"{simplified} table data")

        return queries[:3]

    def get_few_shot_prompt(self, question_type: str, n_examples: int = 2) -> str:
        """Get relevant few-shot examples."""
        examples = self.few_shot_examples.get(question_type, [])[:n_examples]

        if not examples:
            return ""

        prompt_parts = ["Here are examples of similar questions and their correct answers:\n"]

        for ex in examples:
            prompt_parts.append(f"""
Question: {ex['question']}
Answer: {ex['answer']}
Value: {ex['answer_value']}
Unit: {ex['answer_unit']}
---""")

        return "\n".join(prompt_parts)

    def build_prompt(self, question: str, context: str, q_type: str) -> str:
        """Build prompt specialized for question type."""

        few_shot = self.get_few_shot_prompt(q_type, n_examples=2)

        base = """You are an expert at extracting precise answers from environmental AI research papers.
Your task is to find the EXACT answer to the question from the provided context."""

        type_instructions = {
            "true_false": """
IMPORTANT: This is a True/False question.
- Answer MUST be either "TRUE" or "FALSE"
- answer_value MUST be "1" for TRUE or "0" for FALSE
- Look for explicit statements that confirm or deny the claim""",

            "numeric": """
IMPORTANT: This is a numeric question.
- Extract the EXACT number from the context
- Include the correct unit (e.g., "tCO2e", "MWh", "percent", "days")
- If a range is given, use the format "[min,max]"
- Pay attention to tables which often contain precise values""",

            "named_entity": """
IMPORTANT: This question asks for a specific name or identifier.
- Extract the EXACT name as stated in the document
- answer_value should be "is_blank" for named entities
- The answer field should contain the name""",

            "calculation": """
IMPORTANT: This question may require calculation.
- Show your calculation steps in the explanation
- Extract all relevant numbers from context first
- Provide the final computed value in answer_value"""
        }

        return f"""{base}

{type_instructions.get(q_type, type_instructions['numeric'])}

{few_shot}

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {question}

Respond with ONLY valid JSON in this exact format:
{{
    "answer": "Your natural language answer here",
    "answer_value": "The specific value (number, TRUE/FALSE indicator, or 'is_blank')",
    "answer_unit": "The unit of measurement or 'is_blank'",
    "ref_id": "Document ID(s) or 'is_blank'",
    "supporting_materials": "Quote or table reference or 'is_blank'",
    "explanation": "Brief reasoning or 'is_blank'"
}}"""

    def post_process_answer(self, answer: Dict, q_type: str) -> Dict:
        """Post-process answer based on question type."""

        # Clean up None/empty values
        for key in answer:
            if answer[key] in [None, "", "null", "none", "None", "N/A", "n/a"]:
                answer[key] = "is_blank"

        if q_type == "true_false":
            ans_lower = str(answer.get('answer', '')).lower()
            if 'true' in ans_lower:
                answer['answer_value'] = "1"
            elif 'false' in ans_lower:
                answer['answer_value'] = "0"

        elif q_type == "numeric":
            val = answer.get('answer_value', '')
            if val != 'is_blank' and '[' not in str(val):
                numbers = re.findall(r'[\d.]+', str(val))
                if numbers:
                    answer['answer_value'] = numbers[0]

        return answer

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

                contains_numeric = chunk_dict.get('contains_numeric', False)
                if not contains_numeric:
                    contains_numeric = bool(re.search(
                        r'\b\d+\.?\d*\s*(kwh|mwh|co2|tco2e?|percent|%|lbs?|kg|gpu)',
                        text.lower()
                    ))

                chunk = Chunk(
                    text=text,
                    chunk_id=f"{doc_id}_chunk_{i}",
                    doc_id=doc_id,
                    section=chunk_dict.get('section'),
                    contains_table=(chunk_dict.get('type') == 'table'),
                    contains_numeric=contains_numeric
                )
                all_chunks.append(chunk)

        print(f"📚 Loaded {len(all_chunks)} chunks from {len(json_files)} documents")
        return all_chunks

    def build_or_load_index(self):
        """Build or load vector index"""
        # Always rebuild for now to ensure fresh index
        print("📦 Building vector index...")
        chunks = self.load_chunks_from_json()
        self.retriever.index_documents(chunks)
        print("✅ Index built successfully")

    def answer_question(self, question: str) -> Dict:
        """Answer a single question with query expansion and specialized prompting."""

        # Classify question type
        q_type = self.classify_question(question)

        # Expand query and retrieve
        queries = self.expand_query(question)
        all_results = {}

        for query in queries:
            results = self.retriever.hybrid_search(
                query,
                top_k=self.config['retrieval']['top_k'],
                alpha=0.6,
                method="rrf"
            )

            for chunk, score in results:
                if chunk.chunk_id not in all_results:
                    all_results[chunk.chunk_id] = (chunk, score)
                else:
                    # Boost if found by multiple queries
                    existing = all_results[chunk.chunk_id][1]
                    all_results[chunk.chunk_id] = (chunk, max(score, existing) * 1.1)

        # Sort and get top results
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:self.config['retrieval']['top_k']]

        if not top_results:
            return self._get_fallback_response()

        # Build context
        context_parts = []
        doc_ids = set()

        for chunk, score in top_results:
            source_info = f"[Source: {chunk.doc_id}"
            if chunk.section:
                source_info += f", Section: {chunk.section}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{chunk.text}")
            doc_ids.add(chunk.doc_id)

        context = "\n\n---\n\n".join(context_parts)

        # Build specialized prompt
        prompt = self.build_prompt(question, context, q_type)

        # Generate answer
        try:
            if self.generator.provider == "anthropic":
                response = self.generator.client.messages.create(
                    model=self.generator.model,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    answer = json.loads(json_match.group())
                else:
                    answer = self._get_fallback_response()
            else:
                response = self.generator.client.chat.completions.create(
                    model=self.generator.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                answer = json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = self._get_fallback_response()

        # Post-process
        answer = self.post_process_answer(answer, q_type)

        # Add ref_id if missing
        if answer.get('ref_id') == 'is_blank' and doc_ids:
            answer['ref_id'] = ", ".join(list(doc_ids)[:3])

        return answer

    def _get_fallback_response(self) -> Dict:
        """Return a properly formatted fallback response"""
        return {
            "answer": "Unable to answer with confidence based on the provided documents.",
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank"
        }

    def test_on_training_data(self, n_samples: int = 10):
        """Test pipeline on training questions"""
        print(f"\n📊 Testing on {n_samples} training samples...")

        correct = 0
        for i in range(min(n_samples, len(self.train_df))):
            row = self.train_df.iloc[i]
            answer = self.answer_question(row['question'])

            pred_val = str(answer['answer_value']).strip().lower()
            true_val = str(row['answer_value']).strip().lower()

            # Flexible matching
            match = (pred_val == true_val or pred_val in true_val or true_val in pred_val)

            status = "✓" if match else "✗"
            print(f"{status} Q{i+1}: Pred={pred_val} | Actual={true_val}")

            if match:
                correct += 1

        accuracy = correct / min(n_samples, len(self.train_df))
        print(f"\n🎯 Accuracy: {correct}/{n_samples} = {accuracy:.1%}")
        return accuracy

    def process_test_questions(self, save_every=10):
        """Process all test questions"""
        test_df = pd.read_csv("data/raw/test_Q.csv")

        print(f"\n🔄 Processing {len(test_df)} Test Questions...")

        results = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
            try:
                answer = self.answer_question(row['question'])
                answer['id'] = row['id']
                results.append(answer)

                if (idx + 1) % save_every == 0:
                    self._save_results(results, f'checkpoint_{idx + 1}.csv')

            except Exception as e:
                print(f"\n⚠️ Error on {row['id']}: {e}")
                answer = self._get_fallback_response()
                answer['id'] = row['id']
                results.append(answer)

        self._save_results(results, 'submission.csv')
        print(f"\n💾 Saved final submission with {len(results)} answers")
        return results

    def _save_results(self, results: List[Dict], filename: str):
        """Save results to CSV"""
        df = pd.DataFrame(results)

        required_columns = ['id', 'answer', 'answer_value', 'answer_unit',
                            'ref_id', 'ref_url', 'supporting_materials', 'explanation']

        for col in required_columns:
            if col not in df.columns:
                df[col] = 'is_blank'

        df = df[required_columns]

        output_path = Path("data/processed") / filename
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
