from typing import Dict, List, Optional
import json
from ..routing.question_router import QuestionType

class HybridRAGPipeline:
    """
    Combines RAG, GraphRAG, and specialized handlers
    """

    def __init__(
            self,
            retriever,  # Your hybrid retriever
            graph_builder,  # Knowledge graph
            generator,  # LLM generator
            router  # Question router
    ):
        self.retriever = retriever
        self.graph = graph_builder
        self.generator = generator
        self.router = router

    def answer_question(self, question: str) -> Dict:
        """
        Main entry point for question answering
        """
        # 1. Classify question
        q_type = self.router.classify_question(question)
        components = self.router.extract_query_components(question)

        print(f"Question type: {q_type.value}")
        print(f"Components: {components}")

        # 2. Route to appropriate handler
        if q_type == QuestionType.NUMERIC_FACT:
            return self._handle_numeric_question(question, components)
        elif q_type == QuestionType.COMPARISON:
            return self._handle_comparison(question, components)
        elif q_type == QuestionType.CALCULATION:
            return self._handle_calculation(question, components)
        else:
            return self._handle_semantic_question(question, components)

    def _handle_numeric_question(self, question: str, components: Dict) -> Dict:
        """
        Handle numeric fact questions using GraphRAG
        """
        # 1. Query knowledge graph first
        graph_results = self.graph.query_numeric_facts(question)

        if graph_results:
            # Found in graph - high confidence
            best_result = graph_results[0]

            # 2. Get supporting context from RAG
            search_query = f"{best_result['entity']} {best_result['value']}"
            rag_results = self.retriever.search(search_query, top_k=3)

            # 3. Generate answer with both graph and RAG context
            context = self._combine_contexts(graph_results, rag_results)

            answer = self.generator.generate(
                question,
                context,
                metadata={'source': 'graph+rag'}
            )

            # Ensure numeric value is correctly extracted
            answer['answer_value'] = best_result['value']

        else:
            # Fall back to pure RAG
            rag_results = self.retriever.hybrid_search(question, top_k=5)
            context = "\n\n".join([chunk.text for chunk, _ in rag_results])

            answer = self.generator.generate(
                question,
                context,
                metadata={'source': 'rag_only'}
            )

        return answer

    def _handle_comparison(self, question: str, components: Dict) -> Dict:
        """
        Handle comparison questions
        """
        entities = components['entities']

        if len(entities) >= 2:
            # Get graph context for both entities
            contexts = []
            for entity in entities[:2]:
                entity_context = self.graph.get_entity_context(entity)
                contexts.append(entity_context)

            # Get RAG results for comparison
            rag_results = self.retriever.hybrid_search(question, top_k=5)

            # Combine and generate
            combined = self._format_comparison_context(contexts, rag_results)
            answer = self.generator.generate(question, combined)

        else:
            # Standard RAG approach
            answer = self._handle_semantic_question(question, components)

        return answer

    def _handle_calculation(self, question: str, components: Dict) -> Dict:
        """
        Handle calculation questions
        """
        # 1. Get all relevant numeric facts from graph
        numeric_facts = []
        for entity in components['entities']:
            facts = self.graph.query_numeric_facts(f"{entity} {' '.join(components['metrics'])}")
            numeric_facts.extend(facts)

        # 2. Get RAG context
        rag_results = self.retriever.hybrid_search(question, top_k=5)

        # 3. Create calculation prompt
        calc_context = self._format_calculation_context(numeric_facts, rag_results)

        # 4. Generate with calculation instructions
        calc_prompt = f"""
        {calc_context}

        Question: {question}

        Instructions:
        1. Extract all relevant numeric values
        2. Show calculation steps
        3. Provide final answer with units
        """

        answer = self.generator.generate(question, calc_prompt)
        return answer

    def _handle_semantic_question(self, question: str, components: Dict) -> Dict:
        """
        Handle general semantic questions with standard RAG
        """
        # Enhanced retrieval with entity boost
        results = self.retriever.hybrid_search(
            question,
            top_k=5,
            alpha=0.6  # Favor semantic search for these
        )

        # Check if any entities have graph context
        graph_contexts = []
        for entity in components['entities']:
            ctx = self.graph.get_entity_context(entity, hop_distance=1)
            if ctx['nodes']:
                graph_contexts.append(ctx)

        # Combine contexts
        rag_context = "\n\n".join([chunk.text for chunk, _ in results])

        if graph_contexts:
            graph_summary = self._summarize_graph_context(graph_contexts)
            final_context = f"{graph_summary}\n\n{rag_context}"
        else:
            final_context = rag_context

        answer = self.generator.generate(question, final_context)
        return answer

    def _combine_contexts(self, graph_results: List, rag_results: List) -> str:
        """Combine graph and RAG contexts"""
        context_parts = []

        # Add graph facts
        context_parts.append("EXTRACTED FACTS:")
        for fact in graph_results[:3]:
            context_parts.append(
                f"- {fact['entity']} {fact['relation']} {fact['value']} "
                f"(Source: {fact['doc_id']})"
            )

        # Add RAG context
        context_parts.append("\nSUPPORTING TEXT:")
        for chunk, score in rag_results[:3]:
            context_parts.append(chunk.text)

        return "\n".join(context_parts)

    def _format_comparison_context(self, graph_contexts: List, rag_results: List) -> str:
        """Format context for comparison questions"""
        # Implementation here
        pass

    def _format_calculation_context(self, facts: List, rag_results: List) -> str:
        """Format context for calculations"""
        # Implementation here
        pass

    def _summarize_graph_context(self, contexts: List) -> str:
        """Summarize graph context"""
        # Implementation here
        pass