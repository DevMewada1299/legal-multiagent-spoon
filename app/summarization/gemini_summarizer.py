import os
from typing import List, Dict, Any
from spoon_ai.llm.factory import LLMFactory
from spoon_ai.llm.providers import GeminiProvider
from .base import Summarizer

class GeminiSummarizer(Summarizer):
    def __init__(self):
        if "gemini" not in LLMFactory._providers:
            LLMFactory.register("gemini")(GeminiProvider)
        self.llm = LLMFactory.create(provider="gemini")

    async def initialize(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        await self.llm.initialize(config={"api_key": api_key})

    async def summarize(self, clauses: List[Dict[str, Any]]) -> str:
        text_to_summarize = "\n".join([c['text'] for c in clauses])
        prompt = f"Please summarize the following legal text in one sentence:\n\n{text_to_summarize}"
        
        response = await self.llm.completion(prompt=prompt)
        return response.content