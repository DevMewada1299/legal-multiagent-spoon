from typing import List, Dict, Any
import google.generativeai as genai
from .base import Summarizer

class GeminiSummarizer(Summarizer):
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self._model = None

    async def initialize(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key required")
        genai.configure(api_key=api_key)
        try:
            available = list(genai.list_models())
        except Exception:
            available = []
        preferred = [
            "gemini-2.5-pro-latest",
            "gemini-2.5-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        chosen = None
        if available:
            gen_methods = "generateContent"
            by_name = {m.name: m for m in available if gen_methods in getattr(m, "supported_generation_methods", [])}
            for name in preferred:
                if name in by_name:
                    chosen = name
                    break
            if not chosen:
                for m in available:
                    if gen_methods in getattr(m, "supported_generation_methods", []):
                        chosen = m.name
                        break
        self.model_name = chosen or self.model_name
        self._model = genai.GenerativeModel(self.model_name)

    async def summarize(self, clauses: List[Dict[str, Any]]) -> str:
        """
        Summarize legal clauses using Gemini AI as a Summarization Agent.

        Args:
            clauses: List of clause dictionaries with 'text', 'label', and 'has_obligation' fields

        Returns:
            A comprehensive summary of the legal document
        """
        if not self._model:
            raise ValueError("Model not initialized. Call initialize() first.")

        # Organize clauses by category
        categorized = {}
        obligation_clauses = []

        for clause in clauses:
            category = clause.get('label', 'General')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(clause['text'])

            if clause.get('has_obligation', False):
                obligation_clauses.append(clause)

        # Create a structured summary prompt
        prompt = f"""You are a Legal Document Summarization Agent. Your task is to create a comprehensive yet concise summary of a legal document.

Document Analysis:
- Total Clauses: {len(clauses)}
- Clauses with Obligations: {len(obligation_clauses)}

Clauses by Category:
"""
        for category, texts in categorized.items():
            prompt += f"\n{category} ({len(texts)} clauses):\n"
            for text in texts[:3]:  # Limit to first 3 per category to avoid token limits
                prompt += f"  - {text[:200]}...\n"

        if obligation_clauses:
            prompt += f"\n\nKey Obligations ({len(obligation_clauses)} found):\n"
            for clause in obligation_clauses[:5]:  # Top 5 obligations
                obligation_detail = clause.get('obligation_details', clause['text'][:200])
                prompt += f"  - {obligation_detail}\n"

        prompt += """

Please provide a structured summary with the following sections:

1. **Document Overview**: Brief description of the document type and purpose (2-3 sentences)
2. **Key Provisions**: Main categories covered and their significance (3-4 bullet points)
3. **Critical Obligations**: Most important binding requirements identified (3-4 bullet points)
4. **Risk Highlights**: Any notable liability, termination, or dispute resolution clauses (2-3 bullet points)

Format your response in markdown with clear section headers."""

        response = await self._model.generate_content_async(prompt)
        return getattr(response, "text", "")
