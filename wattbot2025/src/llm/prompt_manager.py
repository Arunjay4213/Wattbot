class FewShotPromptManager:
    def __init__(self, train_qa_df):
        self.examples = self.organize_by_type(train_qa_df)
        
    def organize_by_type(self, df):
        """
        Group training examples by:
        - Numeric vs categorical
        - Single vs multi-document
        - Has calculation vs direct extraction
        """
        
    def get_relevant_examples(self, question, k=3):
        """
        Use semantic similarity to find best examples
        from the 41 training samples
        """
        
    def build_prompt(self, question, context, examples):
        prompt = f"""
        You are analyzing environmental AI research papers.
        
        Examples of similar questions and answers:
        {self.format_examples(examples)}
        
        Context from papers:
        {context}
        
        Question: {question}
        
        Provide answer in this EXACT format:
        - answer: [natural language response]
        - answer_value: [numeric/categorical or "is_blank"]
        - answer_unit: [unit or "is_blank"]
        - supporting_materials: [quotes/table refs or "is_blank"]
        - explanation: [reasoning or "is_blank"]
        """