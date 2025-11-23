from spoon_ai.agents.base import BaseAgent
import os
import json
import asyncio
import time
import re
from collections import deque
from spoon_ai.llm.errors import RateLimitError

_call_times = deque()

def _record_call():
    now = time.time()
    _call_times.append(now)
    cutoff = now - 60
    while _call_times and _call_times[0] < cutoff:
        _call_times.popleft()

def get_rpm():
    now = time.time()
    cutoff = now - 60
    return sum(1 for t in _call_times if t >= cutoff)

def get_rpm_limit():
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").lower()
    if "pro" in model:
        return 5
    if "flash-lite" in model:
        return 15
    if "flash" in model:
        return 10
    return 10

class CustomBaseAgent(BaseAgent):
    async def _execute_with_retry(self, prompt):
        max_retries = 0
        retries = 0

        while True:
            try:
                while get_rpm() >= get_rpm_limit():
                    await asyncio.sleep(1)
                _record_call()
                return await self.llm.ask(messages=[{"role": "user", "content": prompt}])
            except RateLimitError as e:
                fallback_flag = os.getenv("GEMINI_FALLBACK_ON_429", "false").lower() in {"true", "1", "yes"}
                if fallback_flag:
                    current = getattr(self.llm, "model_name", "") or os.getenv("GEMINI_MODEL", "")
                    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "")
                    if not fallback_model:
                        if "pro" in current:
                            fallback_model = "gemini-2.0-flash"
                        elif "flash-lite" in current:
                            fallback_model = "gemini-2.0-flash"
                        else:
                            fallback_model = "gemini-2.0-flash-lite"
                    try:
                        self.llm._update_provider_config(provider=self.llm.llm_provider, model_name=fallback_model)
                        _record_call()
                        return await self.llm.ask(messages=[{"role": "user", "content": prompt}])
                    except RateLimitError:
                        pass
                retries += 1
                if retries >= max_retries:
                    raise e

class ClauseExtractionAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(name="ClauseExtractionAgent", llm=llm)

    def execute(self, legal_text: str) -> list[str]:
        blocks = re.split(r"\n\s*\n+", legal_text.strip())
        clauses: list[str] = []
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            paragraph = " ".join(lines)
            if len(paragraph) < 30:
                continue
            if paragraph.lower() in {"page", "date", "signature"}:
                continue
            clauses.append(paragraph)
        return clauses

class ComprehensiveClauseAnalyserAgent(CustomBaseAgent):
    def __init__(self, llm):
        super().__init__(name="ComprehensiveClauseAnalyserAgent", llm=llm)

    def _heuristic_single(self, clause: str) -> dict:
        text = clause.lower()
        obligation_terms = [
            "shall", "must", "required", "agree", "obligated", "will",
            "responsible", "undertakes", "commit", "ensure"
        ]
        risk_terms = [
            "liability", "breach", "penalty", "termination", "damages",
            "indemnify", "default", "failure", "unauthorized", "risk",
            "non-compliance", "fine"
        ]
        obligations_found = any(t in text for t in obligation_terms)
        risks_found = any(t in text for t in risk_terms)
        obligations = []
        risks = []
        if obligations_found:
            obligations.append({"actor": "Unknown", "action": "Prescriptive terms present", "deadline": None})
        if risks_found:
            risks.append({"description": "Liability/penalty/breach terms present", "severity": "medium", "category": "liability"})
        return {"risks": risks, "obligations": obligations}

    async def _execute_single(self, clause: str) -> dict:
        prompt = f'''You are a contract analysis assistant.
Task: Analyze the clause and output one JSON object only with keys "risks" and "obligations".
Rules:
- Output MUST be valid JSON with exactly these two keys.
- No markdown fences or prose.
- risks: array of up to 2 objects with keys {"description","severity","category"}. Use severity: low/medium/high.
- obligations: array of up to 2 objects with keys {"actor","action","deadline"}. actor MUST be one of terms found in text (e.g., CONSULTANT, COMMISSION, Executive Director). If actor is unknown, use null.
- Use concise phrases; if none, use empty arrays.

Clause:
"""
{clause}
"""'''
        try:
            response = await self._execute_with_retry(prompt)
        except RateLimitError:
            return self._heuristic_single(clause)
        except Exception:
            return self._heuristic_single(clause)
        try:
            if response.strip().startswith("```json"):
                response = response.strip()[7:-3].strip()
            data = json.loads(response)
            if isinstance(data, dict) and "risks" in data and "obligations" in data:
                return data
            return {"risks": [], "obligations": []}
        except json.JSONDecodeError:
            return self._heuristic_single(clause)

    async def execute(self, clauses: list[str]) -> list[dict]:
        clauses_str = "\n".join([f"{i+1}. {clause}" for i, clause in enumerate(clauses)])
        prompt = f'''You are a contract analysis assistant.
Task: For each clause, produce an array of JSON objects in the same order, each with keys "risks" and "obligations".
Rules:
- Return ONLY a single valid JSON array.
- No markdown fences, no prose, no trailing commas.
- Array length MUST equal the number of clauses.
- risks: array of up to 2 objects with keys {"description","severity","category"}; severity: low/medium/high; empty array if none.
- obligations: array of up to 2 objects with keys {"actor","action","deadline"}; actor MUST be a term found in text (e.g., CONSULTANT, COMMISSION); empty array if none.

Clauses:
"""
{clauses_str}
"""'''
        try:
            response = await self._execute_with_retry(prompt)
        except RateLimitError:
            return [self._heuristic_single(c) for c in clauses]
        except Exception:
            return [self._heuristic_single(c) for c in clauses]
        try:
            if response.strip().startswith("```json"):
                response = response.strip()[7:-3].strip()
            analysis_results = json.loads(response)
            if isinstance(analysis_results, list) and len(analysis_results) == len(clauses):
                return analysis_results
            else:
                # One retry with stricter instruction to ensure JSON array only
                retry_prompt = (
                    "Return ONLY a valid JSON array of objects, each with keys 'risks' and 'obligations'. "
                    "risks: array of {description,severity,category}; obligations: array of {actor,action,deadline}. "
                    "No markdown, no prose, length must equal number of clauses.\n\nClauses:\n\n"
                    + clauses_str
                )
                try:
                    retry_response = await self._execute_with_retry(retry_prompt)
                except RateLimitError:
                    return [self._heuristic_single(c) for c in clauses]
                try:
                    if retry_response.strip().startswith("```json"):
                        retry_response = retry_response.strip()[7:-3].strip()
                    retry_results = json.loads(retry_response)
                    if isinstance(retry_results, list) and len(retry_results) == len(clauses):
                        return retry_results
                except json.JSONDecodeError:
                    pass
                return [{"risks": [], "obligations": []}] * len(clauses)
        except json.JSONDecodeError:
            # One retry only, then return neutral defaults to avoid excessive API calls
            retry_prompt = (
                "Return ONLY a valid JSON array of objects, each with keys 'risks' and 'obligations'. "
                "risks: array of {description,severity,category}; obligations: array of {actor,action,deadline}. "
                "No markdown, no prose, length must equal number of clauses.\n\nClauses:\n\n"
                + clauses_str
            )
            try:
                retry_response = await self._execute_with_retry(retry_prompt)
            except RateLimitError:
                return [self._heuristic_single(c) for c in clauses]
            try:
                if retry_response.strip().startswith("```json"):
                    retry_response = retry_response.strip()[7:-3].strip()
                retry_results = json.loads(retry_response)
                if isinstance(retry_results, list) and len(retry_results) == len(clauses):
                    return retry_results
            except json.JSONDecodeError:
                pass
            return [{"risks": [], "obligations": []}] * len(clauses)

class SummarizationAgent(CustomBaseAgent):
    def __init__(self, llm):
        super().__init__(name="SummarizationAgent", llm=llm)

    async def execute(self, clauses: list[str], pages: list[int], ids: list[str]) -> str:
        lines = []
        for i, c in enumerate(clauses):
            p = pages[i] if i < len(pages) else None
            cid = ids[i] if i < len(ids) else None
            if p is None:
                if cid:
                    lines.append(f"Clause {i+1} [{cid}]: {c}")
                else:
                    lines.append(f"Clause {i+1}: {c}")
            else:
                if cid:
                    lines.append(f"Page {p} [{cid}]: {c}")
                else:
                    lines.append(f"Page {p}: {c}")
        text_to_summarize = "\n".join(lines)
        prompt = f'''You are a helpful legal assistant.
Please provide a clear and concise executive summary of the following document.
The summary should be well-structured, easy to read, and highlight the key aspects of the agreement.
Do not truncate the summary.

{text_to_summarize}'''
        try:
            response = await self._execute_with_retry(prompt)
            return response
        except RateLimitError:
            if clauses:
                sample = clauses[0][:200]
                return f"Summary (local): {sample}"
            return "Summary (local): No content"