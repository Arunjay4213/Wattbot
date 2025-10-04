"""Answer generation module"""
import openai
from typing import Dict, List

class AnswerGenerator:
    def __init__(self, model="gpt-4"):
        self.model = model

    def generate(self, question: str, context: str) -> Dict:
        """Generate answer from question and context"""
        # TODO: Implement answer generation
        return {
            "answer": "",
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank"
        }
