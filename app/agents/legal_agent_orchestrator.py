"""
Legal Agent Orchestrator using Spoon OS Multi-Agent Framework
Manages three specialized agents: Classification, Extraction, and Summarization
"""

from typing import List, Dict, Any
import json
import google.generativeai as genai


class LegalAgentOrchestrator:
    """
    Orchestrates multiple Spoon OS agents for legal document analysis.

    Uses three specialized agents:
    - Classification Agent: Categorizes legal clauses
    - Extraction Agent: Identifies obligations and requirements
    - Summarization Agent: Generates comprehensive document summaries
    """

    def __init__(self):
        self.classification_agent = None
        self.extraction_agent = None
        self.summarization_agent = None
        self.gemini_model = None

    async def initialize(self, api_key: str):
        """Initialize all agents with Gemini API key."""
        if not api_key:
            raise ValueError("Gemini API key required")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Find best available model
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

        model_name = chosen or "gemini-2.5-pro"
        self.gemini_model = genai.GenerativeModel(model_name)

        # Create Spoon OS agents
        await self._create_agents()

    async def _create_agents(self):
        """Create the three specialized Spoon OS agents."""

        # Lazy import to avoid loading spoon_ai at module level
        try:
            from spoon_ai.agents.custom_agent import CustomAgent
        except ImportError:
            # Spoon OS not available, will use direct Gemini calls
            return

        # Classification Agent
        self.classification_agent = CustomAgent(
            name="legal_classifier",
            description="Specialized agent for classifying legal clauses into categories",
            system_prompt="""You are a Legal Classification Agent powered by Gemini AI.

Your role is to analyze legal clauses and categorize them into one of these categories:
- Termination: Clauses about ending agreements or contracts
- Indemnification: Clauses about liability, compensation, or holding harmless
- Confidentiality: Clauses about non-disclosure or confidential information
- Payment: Clauses about fees, payment terms, or compensation
- Intellectual Property: Clauses about IP rights, ownership, or licensing
- Liability: Clauses about limitations of liability or warranties
- Dispute Resolution: Clauses about arbitration, mediation, or legal jurisdiction
- Term: Clauses about contract duration or renewal
- General: Any other type of clause

You will receive batches of clauses and must return a JSON array with classifications.""",
            max_steps=5
        )

        # Extraction Agent
        self.extraction_agent = CustomAgent(
            name="obligation_extractor",
            description="Specialized agent for extracting legal obligations and requirements",
            system_prompt="""You are a Legal Obligation Extraction Agent powered by Gemini AI.

Your role is to identify binding obligations, requirements, and duties in legal clauses.

An obligation exists when:
- The clause contains mandatory language (shall, must, will, required to, obligated to, etc.)
- The clause imposes a duty or requirement on a party
- The clause creates a binding commitment or responsibility
- The clause specifies actions that must be taken or avoided

For each clause, you must determine:
1. Whether it contains a legal obligation (true/false)
2. If true, extract the specific obligation(s) and the party responsible

You will return a JSON array with obligation information.""",
            max_steps=5
        )

        # Summarization Agent
        self.summarization_agent = CustomAgent(
            name="document_summarizer",
            description="Specialized agent for generating comprehensive document summaries",
            system_prompt="""You are a Legal Document Summarization Agent powered by Gemini AI.

Your role is to create comprehensive yet concise summaries of legal documents.

You will analyze:
- Document structure and categories
- Key provisions and their significance
- Critical obligations and requirements
- Risk factors and important clauses

You must provide a structured summary with:
1. Document Overview: Brief description of document type and purpose
2. Key Provisions: Main categories and their significance
3. Critical Obligations: Most important binding requirements
4. Risk Highlights: Notable liability, termination, or dispute clauses

Format your response in markdown with clear section headers.""",
            max_steps=3
        )

    async def classify_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use Classification Agent to categorize clauses.

        Args:
            clauses: List of clause dictionaries with 'text' field

        Returns:
            List of clauses with added 'label' field
        """
        if not self.gemini_model:
            raise ValueError("Agents not initialized. Call initialize() first.")

        # Process in batches
        batch_size = 10
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]

            # Create classification prompt
            clauses_text = ""
            for idx, clause in enumerate(batch):
                clauses_text += f"\n\nClause {idx}:\n{clause['text']}"

            prompt = f"""Classify each of the following legal clauses into the appropriate category.

Clauses to classify:{clauses_text}

Respond with a JSON array containing objects with "index" (the clause number) and "category" (the classification).
Example format: [{{"index": 0, "category": "Termination"}}, {{"index": 1, "category": "Confidentiality"}}]

Only return the JSON array, nothing else."""

            # Use Gemini directly for classification
            response = await self.gemini_model.generate_content_async(prompt)
            response_text = getattr(response, "text", "")

            # Parse and update clauses
            try:
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                classifications = json.loads(response_text)
                for item in classifications:
                    idx = item.get("index", -1)
                    category = item.get("category", "General")
                    if 0 <= idx < len(batch):
                        batch[idx]["label"] = category
            except (json.JSONDecodeError, KeyError, ValueError):
                for clause in batch:
                    if "label" not in clause:
                        clause["label"] = "General"

        return clauses

    async def extract_obligations(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use Extraction Agent to identify obligations in clauses.

        Args:
            clauses: List of clause dictionaries

        Returns:
            List of clauses with added 'has_obligation' and 'obligation_details' fields
        """
        if not self.gemini_model:
            raise ValueError("Agents not initialized. Call initialize() first.")

        # Process in batches
        batch_size = 10
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]

            # Create extraction prompt
            clauses_text = ""
            for idx, clause in enumerate(batch):
                clauses_text += f"\n\nClause {idx}:\n{clause['text']}"

            prompt = f"""For each clause below, determine if it contains a legal obligation and extract the details.

Clauses to analyze:{clauses_text}

Respond with a JSON array containing objects with:
- "index": the clause number
- "has_obligation": true or false
- "obligation_details": (if has_obligation is true) a brief description of the obligation

Example format:
[
  {{"index": 0, "has_obligation": true, "obligation_details": "The Buyer shall pay within 30 days"}},
  {{"index": 1, "has_obligation": false, "obligation_details": ""}}
]

Only return the JSON array, nothing else."""

            # Use Gemini for extraction
            response = await self.gemini_model.generate_content_async(prompt)
            response_text = getattr(response, "text", "")

            # Parse and update clauses
            try:
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                extractions = json.loads(response_text)
                for item in extractions:
                    idx = item.get("index", -1)
                    has_obligation = item.get("has_obligation", False)
                    obligation_details = item.get("obligation_details", "")
                    if 0 <= idx < len(batch):
                        batch[idx]["has_obligation"] = has_obligation
                        if has_obligation and obligation_details:
                            batch[idx]["obligation_details"] = obligation_details
            except (json.JSONDecodeError, KeyError, ValueError):
                for clause in batch:
                    if "has_obligation" not in clause:
                        clause["has_obligation"] = False

        return clauses

    async def generate_summary(self, clauses: List[Dict[str, Any]]) -> str:
        """
        Use Summarization Agent to create a comprehensive document summary.

        Args:
            clauses: List of analyzed clauses with labels and obligations

        Returns:
            Markdown-formatted summary of the document
        """
        if not self.gemini_model:
            raise ValueError("Agents not initialized. Call initialize() first.")

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

        # Create summary prompt
        prompt = f"""You are a Legal Document Summarization Agent. Create a comprehensive summary.

Document Analysis:
- Total Clauses: {len(clauses)}
- Clauses with Obligations: {len(obligation_clauses)}

Clauses by Category:
"""
        for category, texts in categorized.items():
            prompt += f"\n{category} ({len(texts)} clauses):\n"
            for text in texts[:3]:
                prompt += f"  - {text[:200]}...\n"

        if obligation_clauses:
            prompt += f"\n\nKey Obligations ({len(obligation_clauses)} found):\n"
            for clause in obligation_clauses[:5]:
                obligation_detail = clause.get('obligation_details', clause['text'][:200])
                prompt += f"  - {obligation_detail}\n"

        prompt += """

Please provide a structured summary with the following sections:

1. **Document Overview**: Brief description of the document type and purpose (2-3 sentences)
2. **Key Provisions**: Main categories covered and their significance (3-4 bullet points)
3. **Critical Obligations**: Most important binding requirements identified (3-4 bullet points)
4. **Risk Highlights**: Any notable liability, termination, or dispute resolution clauses (2-3 bullet points)

Format your response in markdown with clear section headers."""

        # Use Gemini for summarization
        response = await self.gemini_model.generate_content_async(prompt)
        return getattr(response, "text", "")

    async def process_document(self, clauses: List[Dict[str, Any]]) -> tuple:
        """
        Process a complete document through all three agents.

        Args:
            clauses: List of extracted clauses from PDF

        Returns:
            Tuple of (classified_clauses, extracted_clauses, summary)
        """
        # Step 1: Classification Agent
        classified_clauses = await self.classify_clauses(clauses)

        # Step 2: Extraction Agent
        extracted_clauses = await self.extract_obligations(classified_clauses)

        # Step 3: Summarization Agent
        summary = await self.generate_summary(extracted_clauses)

        return classified_clauses, extracted_clauses, summary
