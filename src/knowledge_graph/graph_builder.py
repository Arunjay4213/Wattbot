import networkx as nx
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import spacy


@dataclass
class Entity:
    """Entity in the knowledge graph"""
    id: str
    type: str  # MODEL, METRIC, VALUE, HARDWARE, METHOD
    text: str
    properties: Dict


@dataclass
class Relation:
    """Relationship between entities"""
    source: str
    target: str
    type: str  # MEASURES, USES, PRODUCES, REQUIRES
    properties: Dict


class KnowledgeGraphBuilder:
    """
    Builds and queries a knowledge graph from research papers
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = []

        # Load NLP model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Define entity patterns for environmental AI domain
        self.patterns = {
            'MODEL': r'\b(BERT|GPT-\d|LLaMA|T5|BLOOM|OPT|PaLM|Claude|Gemini)\b',
            'METRIC': r'\b(CO2|carbon|emission|kWh|MWh|GWh|PUE|WUE|FLOPS|joules?)\b',
            'VALUE': r'\b(\d+\.?\d*)\s*(lbs?|kg|tons?|MW[hH]|kW[hH]|GW[hH]|CO2e?)\b',
            'HARDWARE': r'\b(V100|A100|H100|TPU|GPU|CPU|RTX\s?\d+)\b',
            'METHOD': r'\b(training|inference|fine-tuning|pre-training|distillation)\b'
        }

    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []

        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    id=f"{doc_id}_{entity_type}_{len(entities)}",
                    type=entity_type,
                    text=match.group(),
                    properties={'doc_id': doc_id, 'span': match.span()}
                )
                entities.append(entity)

        return entities

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relationships between entities"""
        relations = []

        # Pattern-based relation extraction
        relation_patterns = [
            (r'(\w+)\s+(?:training|trained)\s+(?:on|using)\s+(\w+)', 'TRAINED_ON'),
            (r'(\w+)\s+(?:produces?|emits?|generates?)\s+(\d+\.?\d*\s*\w+)', 'PRODUCES'),
            (r'(\w+)\s+(?:requires?|needs?|consumes?)\s+(\d+\.?\d*\s*\w+)', 'REQUIRES'),
            (r'(\w+)\s+(?:achieves?|has|with)\s+(?:a\s+)?PUE\s+of\s+(\d+\.?\d*)', 'HAS_PUE'),
        ]

        for pattern, rel_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find matching entities
                source_text = match.group(1)
                target_text = match.group(2)

                source_entity = self._find_entity(source_text, entities)
                target_entity = self._find_entity(target_text, entities)

                if source_entity and target_entity:
                    relations.append(Relation(
                        source=source_entity.id,
                        target=target_entity.id,
                        type=rel_type,
                        properties={'evidence': match.group()}
                    ))

        return relations

    def build_graph(self, chunks: List[Dict]):
        """Build knowledge graph from document chunks"""
        print("Building knowledge graph...")

        for chunk in chunks:
            text = chunk['text']
            doc_id = chunk['doc_id']

            # Extract entities
            entities = self.extract_entities(text, doc_id)

            # Add entities to graph
            for entity in entities:
                self.graph.add_node(
                    entity.id,
                    type=entity.type,
                    text=entity.text,
                    **entity.properties
                )
                self.entities[entity.id] = entity

            # Extract and add relations
            relations = self.extract_relations(text, entities)
            for relation in relations:
                self.graph.add_edge(
                    relation.source,
                    relation.target,
                    type=relation.type,
                    **relation.properties
                )
                self.relations.append(relation)

        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def query_numeric_facts(self, query: str) -> List[Dict]:
        """
        Query graph for numeric facts

        Example: "What is the CO2 emission of BERT?"
        """
        results = []

        # Extract entities from query
        query_entities = []
        for pattern in self.patterns.values():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            query_entities.extend([match.group().lower() for match in matches])

        # Find relevant nodes
        for node_id, node_data in self.graph.nodes(data=True):
            if any(qe in node_data.get('text', '').lower() for qe in query_entities):
                # Get connected numeric values
                for neighbor in self.graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('type') == 'VALUE':
                        edge_data = self.graph.get_edge_data(node_id, neighbor)
                        results.append({
                            'entity': node_data.get('text'),
                            'value': neighbor_data.get('text'),
                            'relation': edge_data[0].get('type') if edge_data else None,
                            'evidence': edge_data[0].get('evidence') if edge_data else None,
                            'doc_id': node_data.get('doc_id')
                        })

        return results

    def get_entity_context(self, entity_text: str, hop_distance: int = 2) -> Dict:
        """Get graph context around an entity"""
        context = {'nodes': [], 'edges': []}

        # Find matching nodes
        start_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if entity_text.lower() in node_data.get('text', '').lower():
                start_nodes.append(node_id)

        # Get k-hop neighborhood
        for start_node in start_nodes:
            neighborhood = nx.single_source_shortest_path_length(
                self.graph, start_node, cutoff=hop_distance
            )

            for node in neighborhood:
                node_data = self.graph.nodes[node]
                context['nodes'].append({
                    'id': node,
                    'type': node_data.get('type'),
                    'text': node_data.get('text')
                })

            # Get edges in neighborhood
            for u, v, data in self.graph.edges(data=True):
                if u in neighborhood and v in neighborhood:
                    context['edges'].append({
                        'source': u,
                        'target': v,
                        'type': data.get('type')
                    })

        return context

    def _find_entity(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity matching text"""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        return None