"""
SOTA Answer Generator with Advanced RAG Techniques
Implements:
- Query expansion
- Few-shot prompting with training examples
- Self-consistency voting
- Answer verification
- Chain-of-thought reasoning
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import yaml
from pathlib import Path
import pandas as pd

load_dotenv()


class SOTAAnswerGenerator:
    """State-of-the-art answer generator with advanced techniques"""

    def __init__(self, config_path: str = None):
        """Initialize with config file"""
        if config_path is None:
            possible_paths = [
                "configs/config.yaml",
                "../configs/config.yaml",
                "../../configs/config.yaml",
                Path(__file__).parent.parent.parent / "configs" / "config.yaml"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    config_path = str(path)
                    break

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.provider = self.config['models']['llm'].get('provider', 'gemini')
        self.model = self.config['models']['llm']['primary']
        self.temperature = self.config['models']['llm']['temperature']

        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

        # Load training examples for few-shot learning
        self.train_examples = self._load_training_examples()

    def _load_training_examples(self) -> Dict[str, List[Dict]]:
        """Load and organize training examples by question type"""
        examples = {
            "numeric": [],
            "true_false": [],
            "comparison": [],
            "factual": []
        }

        try:
            train_df = pd.read_csv("data/raw/train_QA.csv")

            for _, row in train_df.head(20).iterrows():  # Use top 20 examples
                q_lower = row['question'].lower()
                example = {
                    "question": row['question'],
                    "answer": row['answer'],
                    "answer_value": row['answer_value'],
                    "answer_unit": row['answer_unit'],
                    "context": "..."  # Shortened for token efficiency
                }

                if "true or false" in q_lower:
                    examples["true_false"].append(example)
                elif any(w in q_lower for w in ["how much", "how many", "what is the"]):
                    examples["numeric"].append(example)
                elif any(w in q_lower for w in ["difference", "compare", "factor"]):
                    examples["comparison"].append(example)
                else:
                    examples["factual"].append(example)

        except Exception as e:
            print(f"Warning: Could not load training examples: {e}")

        return examples

    def classify_question(self, question: str) -> str:
        """Classify question type for appropriate prompting"""
        q_lower = question.lower()

        if "true or false" in q_lower:
            return "true_false"
        elif any(w in q_lower for w in ["how much", "how many", "what is the"]):
            return "numeric"
        elif any(w in q_lower for w in ["difference", "compare", "factor", "percentage"]):
            return "comparison"
        else:
            return "factual"

    def expand_query(self, question: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = [question]

        # Add simplified version
        simplified = question.replace("?", "").strip()
        variations.append(simplified)

        # Add query with context hints
        q_type = self.classify_question(question)
        if q_type == "numeric":
            variations.append(f"{simplified} measurement value")
        elif q_type == "comparison":
            variations.append(f"{simplified} difference calculation")

        return variations

    def get_few_shot_examples(self, question: str, n: int = 3) -> List[Dict]:
        """Get relevant few-shot examples based on question type"""
        q_type = self.classify_question(question)
        examples = self.train_examples.get(q_type, [])
        return examples[:n]

    def generate_with_chain_of_thought(
        self,
        question: str,
        context: str,
        chunk_metadata: Optional[Dict] = None
    ) -> Dict:
        """Generate answer using chain-of-thought reasoning"""

        # Get few-shot examples
        examples = self.get_few_shot_examples(question)
        examples_text = self._format_few_shot_examples(examples)

        prompt = f"""You are an expert at analyzing environmental AI research papers.
Use step-by-step reasoning to answer questions accurately.

{examples_text}

Now, answer this question using the same format:

CONTEXT:
{context}

QUESTION: {question}

Think step by step:
1. First, identify the key information needed to answer the question
2. Locate relevant data in the context
3. Extract the specific value and unit
4. Verify the answer makes sense
5. Provide the final answer in JSON format

Return your answer as valid JSON with these fields:
{{
    "reasoning": "Your step-by-step thought process",
    "answer": "Natural language answer",
    "answer_value": "The specific value or 'is_blank'",
    "answer_unit": "The unit or 'is_blank'",
    "ref_id": "Document ID(s) or 'is_blank'",
    "supporting_materials": "Direct quote/reference or 'is_blank'",
    "explanation": "Brief explanation or 'is_blank'",
    "confidence": "high/medium/low"
}}

Important:
- Use 'is_blank' for fields without values
- Be precise with numbers and units
- If unsure, answer should be "Unable to answer based on the provided documents"
- Include your reasoning process
"""

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1500,
                )
            )

            content = response.text

            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                answer_json = json.loads(json_match.group())
            else:
                answer_json = json.loads(content)

            return self._format_answer(answer_json, chunk_metadata)

        except Exception as e:
            print(f"Error in CoT generation: {e}")
            return self._get_fallback_response()

    def generate_with_self_consistency(
        self,
        question: str,
        context: str,
        chunk_metadata: Optional[Dict] = None,
        n_samples: int = 3
    ) -> Dict:
        """Generate multiple answers and vote for consistency"""

        answers = []
        for i in range(n_samples):
            # Use slightly different temperature for diversity
            temp_config = genai.types.GenerationConfig(
                temperature=min(self.temperature + i * 0.1, 0.4),
                max_output_tokens=1500,
            )

            try:
                answer = self.generate_with_chain_of_thought(
                    question, context, chunk_metadata
                )
                answers.append(answer)
            except:
                continue

        if not answers:
            return self._get_fallback_response()

        # Vote on most consistent answer
        return self._vote_best_answer(answers)

    def _vote_best_answer(self, answers: List[Dict]) -> Dict:
        """Vote for the most consistent answer"""
        # Count answer_value occurrences
        value_counts = {}
        for ans in answers:
            val = ans.get('answer_value', 'is_blank')
            if val not in value_counts:
                value_counts[val] = []
            value_counts[val].append(ans)

        # Return the most common answer
        if not value_counts:
            return answers[0]

        most_common = max(value_counts.items(), key=lambda x: len(x[1]))
        return most_common[1][0]  # Return first answer with most common value

    def _format_few_shot_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples for prompt"""
        if not examples:
            return ""

        examples_text = "Here are some example questions and answers:\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Q: {ex['question']}\n"
            examples_text += f"A: {ex['answer']}\n"
            examples_text += f"Value: {ex['answer_value']}\n"
            examples_text += f"Unit: {ex['answer_unit']}\n\n"

        return examples_text

    def _format_answer(self, answer_json: Dict, chunk_metadata: Optional[Dict]) -> Dict:
        """Ensure answer follows competition format"""
        formatted = {
            "answer": str(answer_json.get("answer", "Unable to answer based on the provided documents")),
            "answer_value": str(answer_json.get("answer_value", "is_blank")),
            "answer_unit": str(answer_json.get("answer_unit", "is_blank")),
            "ref_id": str(answer_json.get("ref_id", "is_blank")),
            "supporting_materials": str(answer_json.get("supporting_materials", "is_blank")),
            "explanation": str(answer_json.get("explanation", "is_blank"))
        }

        # Clean up
        for key, value in formatted.items():
            if value in [None, "", "null", "none", "NaN", "None"]:
                formatted[key] = "is_blank"
            elif isinstance(value, str):
                formatted[key] = value.strip()

        # Add metadata ref_id if not present
        if formatted["ref_id"] == "is_blank" and chunk_metadata:
            if "source_docs" in chunk_metadata:
                formatted["ref_id"] = ", ".join(chunk_metadata["source_docs"])

        return formatted

    def _get_fallback_response(self) -> Dict:
        """Return fallback response"""
        return {
            "answer": "Unable to answer based on the provided documents",
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank"
        }


def test_sota_generator():
    """Test the SOTA answer generator"""
    generator = SOTAAnswerGenerator()

    test_question = "What is the CO2 emission of training BERT?"
    test_context = """
    According to Strubell et al. (2019), training a BERT base model produces
    approximately 1,438 lbs of CO2 emissions, which is roughly equivalent to
    a trans-American flight for one person.
    """
    test_metadata = {"source_docs": ["strubell2019"]}

    print("Testing SOTA Answer Generator with Chain-of-Thought...")
    result = generator.generate_with_chain_of_thought(test_question, test_context, test_metadata)

    print("\nQuestion:", test_question)
    print("\nGenerated Answer:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


if __name__ == "__main__":
    test_sota_generator()
