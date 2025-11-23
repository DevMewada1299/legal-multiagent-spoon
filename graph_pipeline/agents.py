from spoon_ai.agents.base import BaseAgent
import os

class ClauseExtractionAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(name="ClauseExtractionAgent", llm=llm)

    def execute(self, legal_text: str) -> list[str]:
        # In a real-world scenario, this would involve more sophisticated logic.
        # For this example, we'll just split the text by newline.
        return legal_text.split('\n')

class SummarizationAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(name="SummarizationAgent", llm=llm)
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            # The ChatBot class now handles initialization, so we may not need this explicit initialize method.
            # However, we'll keep it for now to ensure the API key is checked.
            self._initialized = True

    async def execute(self, clauses: list[str]) -> str:
        if not self._initialized:
            await self.initialize()
        text_to_summarize = "\n".join(clauses)
        prompt = f"Please summarize the following legal text in one sentence:\n\n{text_to_summarize}"
        response = await self.llm.ask(messages=[{"role": "user", "content": prompt}])
        return response

class RiskAnalysisAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(name="RiskAnalysisAgent", llm=llm)

    async def execute(self, clauses: list[str]) -> list[str]:
        clauses_text = "\n".join(clauses)
        prompt = f'''Analyze each of the following legal clauses for potential risks, liabilities, or unfavorable terms. Return your analysis for each clause as a new line. If a clause has no risks, return 'No risks found' for that line.

{clauses_text}'''
        response = await self.llm.ask(messages=[{"role": "user", "content": prompt}])
        return response.split('\n')

class ObligationExtractionAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(name="ObligationExtractionAgent", llm=llm)

    async def execute(self, clauses: list[str]) -> list[str]:
        clauses_text = "\n".join(clauses)
        prompt = f'''For each of the following legal clauses, extract the specific duties, responsibilities, and obligations. Return the obligations for each clause as a new line. If a clause has no obligations, return 'No obligations found' for that line.

{clauses_text}'''
        response = await self.llm.ask(messages=[{"role": "user", "content": prompt}])
        return response.split('\n')