import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()


class AnswerGenerator:
    def __init__(self, model: str = "gpt-4"):
        """Initialize the answer generator with OpenAI API"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        openai.api_key = self.api_key
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, question: str, context: str, chunk_metadata: Optional[Dict] = None) -> Dict:
        """
        Generate answer from question and context
        
        Args:
            question: The user's question
            context: Retrieved context from documents
            chunk_metadata: Optional metadata about source chunks
            
        Returns:
            Dictionary with required competition format
        """

        # Build the prompt
        prompt = self._build_prompt(question, context, chunk_metadata)

        try:
            # Call OpenAI API
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
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            # Parse the response
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
        """

        # Add metadata if available
        if chunk_metadata and 'source_docs' in chunk_metadata:
            prompt += f"\n\nSource documents: {', '.join(chunk_metadata['source_docs'])}"

        return prompt

    def _format_answer(self, answer_json: Dict, chunk_metadata: Optional[Dict]) -> Dict:
        """Ensure answer follows competition format exactly"""

        formatted = {
            "answer": answer_json.get("answer", "Unable to answer based on the provided documents"),
            "answer_value": answer_json.get("answer_value", "is_blank"),
            "answer_unit": answer_json.get("answer_unit", "is_blank"),
            "ref_id": answer_json.get("ref_id", "is_blank"),
            "supporting_materials": answer_json.get("supporting_materials", "is_blank"),
            "explanation": answer_json.get("explanation", "is_blank")
        }

        # Clean up any None or empty values
        for key, value in formatted.items():
            if value in [None, "", "null", "none", "NaN", "nan"]:
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
        """
        Generate answer with automatic fallback detection

        This method first checks if the context is sufficient to answer the question
        """

        # First check if context is empty or too short
        if not context or len(context.strip()) < 50:
            return self._get_fallback_response()

        # Check relevance using a quick LLM call
        relevance_prompt = f"""
            Can the following context answer this question? Reply with only 'yes' or 'no'.

            Question: {question}
            Context: {context[:1000]}  # Use first 1000 chars for speed
            """

        try:
            relevance_check = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use faster model for this check
                messages=[
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0,
                max_tokens=10
            )

            if "no" in relevance_check.choices[0].message.content.lower():
                return self._get_fallback_response()

        except:
            pass  # If check fails, continue with generation

        # Proceed with normal generation
        return self.generate(question, context, chunk_metadata)

    def batch_generate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Generate answers for multiple questions

        Args:
            qa_pairs: List of dicts with 'question' and 'context' keys

        Returns:
            List of formatted answers
        """
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

    def test_answer_generator(model="gpt-3.5-turbo", custom_question=None, custom_context=None):
        # so to use this you can do it in the following ways
        # No parameters (uses defaults)
        # test_answer_generator()
        #
        # With different model
        # test_answer_generator(model="gpt-4")
        #
        # With custom question
        # test_answer_generator(custom_question="What is PUE?", custom_context="PUE stands for...")

        
        # Initialize generator
        generator = AnswerGenerator(model=model)

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
        # Test the module
        print("Testing Answer Generator...")
        print("=" * 50)

        try:
            result = test_answer_generator()
            print("\n Answer Generator working successfully!")
        except Exception as e:
            print(f"\n Error: {e}")
            print("\nMake sure to:")
            print("1. Create a .env file with your OPENAI_API_KEY")
            print("2. Install required packages: pip install openai python-dotenv")
