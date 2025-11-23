from typing import List, Dict, Any
import google.generativeai as genai
import json
from .base import Classifier

class GeminiClassifier(Classifier):
    """
    Gemini-powered Classification Agent
    Analyzes legal clauses and classifies them into categories using AI.
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

        preferred = ["gemini-2.0-flash-lite"
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

    async def classify(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify legal clauses using Gemini AI.

        Args:
            clauses: List of clause dictionaries with 'text' field

        Returns:
            List of clauses with added 'label' field containing the classification
        """
        if not self._model:
            raise ValueError("Model not initialized. Call initialize() first.")

        # Process clauses in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]

            # Create prompt for classification agent
            prompt = self._create_classification_prompt(batch)

            # Get response from Gemini
            response = await self._model.generate_content_async(prompt)
            response_text = getattr(response, "text", "")

            # Parse the response and update clauses
            self._parse_and_update_clauses(batch, response_text)

        return clauses

    def _create_classification_prompt(self, clauses: List[Dict[str, Any]]) -> str:
        """Create a structured prompt for the classification agent."""
        clauses_text = ""
        for idx, clause in enumerate(clauses):
            clauses_text += f"\n\nClause {idx}:\n{clause['text']}"

        prompt = f"""You are a Legal Classification Agent. Your task is to analyze legal clauses and categorize them.

Classify each of the following legal clauses into ONE of these categories:
- Termination: Clauses about ending agreements or contracts
- Indemnification: Clauses about liability, compensation, or holding harmless
- Confidentiality: Clauses about non-disclosure or confidential information
- Payment: Clauses about fees, payment terms, or compensation
- Intellectual Property: Clauses about IP rights, ownership, or licensing
- Liability: Clauses about limitations of liability or warranties
- Dispute Resolution: Clauses about arbitration, mediation, or legal jurisdiction
- Term: Clauses about contract duration or renewal
- General: Any other type of clause

Clauses to classify:{clauses_text}

Respond with a JSON array containing objects with "index" (the clause number) and "category" (the classification).
Example format: [{{"index": 0, "category": "Termination"}}, {{"index": 1, "category": "Confidentiality"}}]

Only return the JSON array, nothing else."""

        return prompt

    def _parse_and_update_clauses(self, clauses: List[Dict[str, Any]], response_text: str):
        """Parse Gemini's response and update clause labels."""
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
            classifications = json.loads(response_text)

            # Update clauses with classifications
            for item in classifications:
                idx = item.get("index", -1)
                category = item.get("category", "General")
                if 0 <= idx < len(clauses):
                    clauses[idx]["label"] = category

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: set all to "General" if parsing fails
            print(f"Warning: Failed to parse classification response: {e}")
            for clause in clauses:
                if "label" not in clause:
                    clause["label"] = "General"
