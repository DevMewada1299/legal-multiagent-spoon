from typing import List, Dict, Any
import google.generativeai as genai
import json
from .base import Extractor

class GeminiExtractor(Extractor):
    """
    Gemini-powered Obligation Extraction Agent
    Identifies binding obligations and legal requirements in clauses using AI.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self._model = None

    async def initialize(self, api_key: str):
        """Initialize the Gemini model with API key."""
        if not api_key:
            raise ValueError("Gemini API key required")
        genai.configure(api_key=api_key)

        # Try to find the best available model
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

    async def extract(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract obligations from legal clauses using Gemini AI.

        Args:
            clauses: List of clause dictionaries with 'text' field

        Returns:
            List of clauses with added 'has_obligation' field and 'obligations' details
        """
        if not self._model:
            raise ValueError("Model not initialized. Call initialize() first.")

        # Process clauses in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]

            # Create prompt for extraction agent
            prompt = self._create_extraction_prompt(batch)

            # Get response from Gemini
            response = await self._model.generate_content_async(prompt)
            response_text = getattr(response, "text", "")

            # Parse the response and update clauses
            self._parse_and_update_clauses(batch, response_text)

        return clauses

    def _create_extraction_prompt(self, clauses: List[Dict[str, Any]]) -> str:
        """Create a structured prompt for the obligation extraction agent."""
        clauses_text = ""
        for idx, clause in enumerate(clauses):
            clauses_text += f"\n\nClause {idx}:\n{clause['text']}"

        prompt = f"""You are a Legal Obligation Extraction Agent. Your task is to identify binding obligations, requirements, and duties in legal clauses.

An obligation exists when:
- The clause contains mandatory language (shall, must, will, required to, obligated to, etc.)
- The clause imposes a duty or requirement on a party
- The clause creates a binding commitment or responsibility
- The clause specifies actions that must be taken or avoided

For each clause below, determine:
1. Whether it contains a legal obligation (true/false)
2. If true, extract the specific obligation(s) and the party responsible

Clauses to analyze:{clauses_text}

Respond with a JSON array containing objects with:
- "index": the clause number
- "has_obligation": true or false
- "obligation_details": (if has_obligation is true) a brief description of the obligation and who must perform it

Example format:
[
  {{"index": 0, "has_obligation": true, "obligation_details": "The Buyer shall pay the purchase price within 30 days"}},
  {{"index": 1, "has_obligation": false, "obligation_details": ""}}
]

Only return the JSON array, nothing else."""

        return prompt

    def _parse_and_update_clauses(self, clauses: List[Dict[str, Any]], response_text: str):
        """Parse Gemini's response and update clause obligation information."""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            extractions = json.loads(response_text)

            # Update clauses with obligation information
            for item in extractions:
                idx = item.get("index", -1)
                has_obligation = item.get("has_obligation", False)
                obligation_details = item.get("obligation_details", "")

                if 0 <= idx < len(clauses):
                    clauses[idx]["has_obligation"] = has_obligation
                    if has_obligation and obligation_details:
                        clauses[idx]["obligation_details"] = obligation_details

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: set all to no obligation if parsing fails
            print(f"Warning: Failed to parse extraction response: {e}")
            for clause in clauses:
                if "has_obligation" not in clause:
                    clause["has_obligation"] = False
