# src/llm/answer_generator.py
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai
import yaml
from pathlib import Path

load_dotenv()


class AnswerGenerator:
    def __init__(self, config_path: str = None):
        """Initialize with config file"""
        # Find config file
        if config_path is None:
            # Try different paths depending on where script is run from
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

            if config_path is None:
                raise FileNotFoundError("Could not find config.yaml")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.provider = self.config['models']['llm'].get('provider', 'openai')
        self.model = self.config['models']['llm']['primary']
        self.temperature = self.config['models']['llm']['temperature']

        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in .env file")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")
            self.client = openai.OpenAI(api_key=api_key)

    def generate(self, question: str, context: str, chunk_metadata: Optional[Dict] = None) -> Dict:
        """Generate answer from question and context"""

        prompt = self._build_prompt(question, context, chunk_metadata)

        try:
            if self.provider == "anthropic":
                # Claude API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": f"""You are an expert at analyzing environmental AI research papers.
                        Always provide accurate, evidence-based answers.

                        {prompt}

                        Remember to return valid JSON only, no additional text."""
                    }]
                )

                # Extract text from Claude response
                content = response.content[0].text

                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        answer_json = json.loads(json_match.group())
                    else:
                        answer_json = json.loads(content)
                except:
                    # If JSON parsing fails, create structured response from text
                    answer_json = self._parse_text_response(content)

            elif self.provider == "gemini":
                # Gemini API call
                full_prompt = f"""You are an expert at analyzing environmental AI research papers.
                Always provide accurate, evidence-based answers.

                {prompt}

                Remember to return valid JSON only, no additional text."""

                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=1000,
                    )
                )

                content = response.text

                # Try to extract JSON from the response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        answer_json = json.loads(json_match.group())
                    else:
                        answer_json = json.loads(content)
                except:
                    # If JSON parsing fails, create structured response from text
                    answer_json = self._parse_text_response(content)

            else:
                # OpenAI API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing environmental AI research papers. Always provide accurate, evidence-based answers."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}  # Only works with OpenAI
                )

                answer_json = json.loads(response.choices[0].message.content)

            # Validate and format
            return self._format_answer(answer_json, chunk_metadata)

        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._get_fallback_response()

    def _build_prompt(self, question: str, context: str, chunk_metadata: Optional[Dict]) -> str:
        """Build the prompt for the LLM"""

        prompt = f"""
        Based on the following context from environmental AI research papers, answer the question.

        CONTEXT:
        {context}

        QUESTION: {question}

        Provide your answer in valid JSON format with these exact fields:
        {{
            "answer": "Natural language answer to the question. If you cannot answer based on the context, say 'Unable to answer based on the provided documents'",
            "answer_value": "The specific numeric or categorical value that answers the question, or 'is_blank' if not applicable",
            "answer_unit": "The unit of measurement for answer_value, or 'is_blank' if not applicable",
            "ref_id": "The document ID(s) that support your answer, or 'is_blank' if not found",
            "supporting_materials": "Direct quote or reference (e.g., 'Table 3', 'Figure 2') that supports your answer, or 'is_blank'",
            "explanation": "Brief explanation of how you arrived at the answer, or 'is_blank'"
        }}

        Important rules:
        1. If the context doesn't contain enough information to answer the question, the answer field should be "Unable to answer based on the provided documents"
        2. Use 'is_blank' for any field that doesn't have a value, never use null, None, or empty string
        3. For ref_id, use comma-separated values if multiple documents are referenced
        4. Be precise with numeric values and units
        5. Return ONLY valid JSON, no other text
        """

        # Add metadata if available
        if chunk_metadata and 'source_docs' in chunk_metadata:
            prompt += f"\n\nSource documents: {', '.join(chunk_metadata['source_docs'])}"

        return prompt

    def _parse_text_response(self, text: str) -> Dict:
        """Parse text response when JSON parsing fails"""
        # Basic parsing logic for non-JSON responses
        result = {
            "answer": "Unable to answer based on the provided documents",
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank"
        }

        # Try to extract answer from text
        if text and len(text) > 10:
            result["answer"] = text.strip()

            # Look for numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                result["answer_value"] = numbers[0]

            # Look for units
            units = re.findall(r'\b(kg|lbs|CO2|tCO2e|MWh|kWh|GPU|percent|%)\b', text, re.IGNORECASE)
            if units:
                result["answer_unit"] = units[0]

        return result

    def _format_answer(self, answer_json: Dict, chunk_metadata: Optional[Dict]) -> Dict:
        """Ensure answer follows competition format exactly"""

        formatted = {
            "answer": str(answer_json.get("answer", "Unable to answer based on the provided documents")),
            "answer_value": str(answer_json.get("answer_value", "is_blank")),
            "answer_unit": str(answer_json.get("answer_unit", "is_blank")),
            "ref_id": str(answer_json.get("ref_id", "is_blank")),
            "supporting_materials": str(answer_json.get("supporting_materials", "is_blank")),
            "explanation": str(answer_json.get("explanation", "is_blank"))
        }

        # Clean up any None or empty values
        for key, value in formatted.items():
            if value in [None, "", "null", "none", "NaN", "nan", "None"]:
                formatted[key] = "is_blank"
            elif isinstance(value, str):
                formatted[key] = value.strip()

        # Add metadata ref_id if not present
        if formatted["ref_id"] == "is_blank" and chunk_metadata:
            if "source_docs" in chunk_metadata:
                formatted["ref_id"] = ", ".join(chunk_metadata["source_docs"])

        return formatted

    def _get_fallback_response(self) -> Dict:
        """Return a properly formatted fallback response"""
        return {
            "answer": "Unable to answer based on the provided documents",
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank"
        }

    def generate_with_fallback_detection(self, question: str, context: str,
                                         chunk_metadata: Optional[Dict] = None) -> Dict:
        """Generate answer with automatic fallback detection"""

        # First check if context is empty or too short
        if not context or len(context.strip()) < 50:
            return self._get_fallback_response()

        # For Claude and Gemini, we can't easily do a quick relevance check due to API differences
        # So we'll skip the relevance check and go straight to generation
        if self.provider in ["anthropic", "gemini"]:
            return self.generate(question, context, chunk_metadata)

        # OpenAI relevance check
        relevance_prompt = f"""
        Can the following context answer this question? Reply with only 'yes' or 'no'.

        Question: {question}
        Context: {context[:1000]}
        """

        try:
            relevance_check = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0,
                max_tokens=10
            )

            if "no" in relevance_check.choices[0].message.content.lower():
                return self._get_fallback_response()

        except:
            pass

            # Proceed with normal generation
        return self.generate(question, context, chunk_metadata)

    def batch_generate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Generate answers for multiple questions"""
        results = []

        for i, pair in enumerate(qa_pairs):
            print(f"Processing question {i + 1}/{len(qa_pairs)}")

            question = pair.get("question", "")
            context = pair.get("context", "")
            metadata = pair.get("metadata", None)

            answer = self.generate_with_fallback_detection(question, context, metadata)
            answer["question_id"] = pair.get("id", f"q_{i}")

            results.append(answer)

        return results


def test_answer_generator(custom_question=None, custom_context=None):
    """Test the answer generator"""

    # Initialize generator (will use config file)
    generator = AnswerGenerator()

    # Use custom or default test data
    if custom_question and custom_context:
        test_question = custom_question
        test_context = custom_context
        test_metadata = {}
    else:
        test_question = "What is the CO2 emission of training BERT?"
        test_context = """
        According to Strubell et al. (2019), training a BERT base model produces 
        approximately 1,438 lbs of CO2 emissions, which is roughly equivalent to 
        a trans-American flight for one person.
        """
        test_metadata = {"source_docs": ["strubell2019"]}

    # Generate answer
    result = generator.generate(test_question, test_context, test_metadata)

    print("Question:", test_question)
    print("\nGenerated Answer:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


if __name__ == "__main__":
    print("Testing Answer Generator...")

    try:
        result = test_answer_generator()
        print("Answer Generator working successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Create a .env file with your OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
        print("2. Install required packages: pip install openai anthropic google-generativeai python-dotenv pyyaml")
        print("3. Check that config.yaml exists and is properly configured")