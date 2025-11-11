import re
from typing import Dict, List, Tuple
from enum import Enum


class QuestionType(Enum):
    NUMERIC_FACT = "numeric_fact"  # "What is the CO2 emission of..."
    COMPARISON = "comparison"  # "Compare X and Y"
    DEFINITION = "definition"  # "What is..."
    CALCULATION = "calculation"  # "Calculate the total..."
    TEMPORAL = "temporal"  # "How has X changed over time"
    CAUSAL = "causal"  # "Why does..."
    LISTING = "listing"  # "List all models that..."
    UNKNOWN = "unknown"


class QuestionRouter:
    """Routes questions to appropriate handlers"""

    def __init__(self):
        # Define patterns for each question type
        self.patterns = {
            QuestionType.NUMERIC_FACT: [
                r'what (?:is|are) the .* (?:of|for)',
                r'how (?:much|many)',
                r'(?:emission|consumption|usage|footprint) of',
            ],
            QuestionType.COMPARISON: [
                r'compare',
                r'difference between',
                r'which is (?:better|worse|more|less)',
                r'versus|vs\.?',
            ],
            QuestionType.CALCULATION: [
                r'calculate',
                r'total',
                r'sum',
                r'average',
                r'ratio',
            ],
            QuestionType.TEMPORAL: [
                r'over (?:time|years)',
                r'trend',
                r'changed? (?:since|from|between)',
                r'evolution',
            ],
            QuestionType.DEFINITION: [
                r'^what (?:is|are) \w+\??$',
                r'define',
                r'meaning of',
            ],
            QuestionType.CAUSAL: [
                r'why',
                r'cause',
                r'reason',
                r'because',
            ],
            QuestionType.LISTING: [
                r'list (?:all)?',
                r'which \w+ (?:have|has|are)',
                r'examples of',
            ],
        }

    def classify_question(self, question: str) -> QuestionType:
        """Classify question type"""
        question_lower = question.lower()

        for q_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type

        return QuestionType.UNKNOWN

    def extract_query_components(self, question: str) -> Dict:
        """Extract key components from question"""
        components = {
            'entities': [],
            'metrics': [],
            'operations': [],
            'constraints': []
        }

        # Extract entities (models, hardware, etc.)
        entity_patterns = [
            r'\b(BERT|GPT-\d|LLaMA|T5)\b',
            r'\b(V100|A100|TPU)\b',
        ]
        for pattern in entity_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            components['entities'].extend(matches)

        # Extract metrics
        metric_patterns = [
            r'\b(CO2|carbon|emission)\b',
            r'\b(energy|power|electricity)\b',
            r'\b(PUE|efficiency)\b',
        ]
        for pattern in metric_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            components['metrics'].extend(matches)

        # Extract operations
        if 'compare' in question.lower():
            components['operations'].append('compare')
        if 'calculate' in question.lower():
            components['operations'].append('calculate')

        return components