from dataclasses import dataclass, field, asdict
import os

from typing import List, Dict, Any
from spoon_ai.graph.engine import StateGraph, END
from spoon_ai.graph.builder import (
    DeclarativeGraphBuilder,
    GraphTemplate,
    NodeSpec,
    EdgeSpec,
)
from .agents import (
    ClauseExtractionAgent,
    SummarizationAgent,
    ComprehensiveClauseAnalyserAgent,
)
from spoon_ai.chat import ChatBot


@dataclass
class LegalAnalysisState:
    legal_text: str
    clause_items: List[Dict[str, Any]] = field(default_factory=list)
    clauses: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    summary: str = ""
    execution_log: List[str] = field(default_factory=list)


class LegalAnalysisGraph:
    def __init__(self, analysis_model: str | None = None, summary_model: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")

        default_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        analysis_env = os.getenv("GEMINI_MODEL_ANALYSIS")
        summary_env = os.getenv("GEMINI_MODEL_SUMMARY")

        analysis_model = analysis_model or analysis_env or default_model
        summary_model = summary_model or summary_env or default_model

        analysis_bot = ChatBot(llm_provider="gemini", api_key=api_key, model_name=analysis_model)
        summary_bot = ChatBot(llm_provider="gemini", api_key=api_key, model_name=summary_model)

        self.clause_extractor = ClauseExtractionAgent(llm=analysis_bot)
        self.comprehensive_analyzer = ComprehensiveClauseAnalyserAgent(llm=analysis_bot)
        self.summarizer = SummarizationAgent(llm=summary_bot)
        self.graph_template = self._build_template()

    def _build_template(self) -> GraphTemplate:
        nodes = [
            NodeSpec(name="extract_clauses", handler=self.handle_clause_extraction),
            NodeSpec(name="full_analysis", handler=self.handle_full_analysis),
        ]
        edges = [
            EdgeSpec(start="extract_clauses", end="full_analysis"),
            EdgeSpec(start="full_analysis", end=END),
        ]
        return GraphTemplate(entry_point="extract_clauses", nodes=nodes, edges=edges)

    async def handle_clause_extraction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get('clause_items'):
            texts = [item.get('text', '') for item in state['clause_items']]
            pages = [item.get('page') for item in state['clause_items']]
            ids = [item.get('id') for item in state['clause_items']]
            return {
                "clauses": texts,
                "pages": pages,
                "ids": ids,
                "execution_log": state['execution_log'] + ["Loaded pre-segmented clauses"]
            }
        clauses = self.clause_extractor.execute(state['legal_text'])
        return {"clauses": clauses, "pages": [None] * len(clauses), "ids": [None] * len(clauses), "execution_log": state['execution_log'] + ["Extracted clauses"]}

    async def handle_full_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            clauses = state.get('clauses') or []
            pages = state.get('pages') or [None] * len(clauses)
            ids = state.get('ids') or [None] * len(clauses)
            batch_size = max(min(20, len(clauses)), 1)
            analysis_results: List[Dict[str, Any]] = []
            for i in range(0, len(clauses), batch_size):
                batch = clauses[i:i+batch_size]
                try:
                    batch_results = await self.comprehensive_analyzer.execute(batch)
                except Exception:
                    batch_results = [self.comprehensive_analyzer._heuristic_single(c) for c in batch]
                analysis_results.extend(batch_results)

            analysis: List[Dict[str, Any]] = []
            for idx, clause in enumerate(clauses):
                result = analysis_results[idx] if idx < len(analysis_results) else {"risks": [], "obligations": []}
                analysis.append({
                    "index": idx + 1,
                    "clause_excerpt": clause[:200],
                    "page": pages[idx] if idx < len(pages) else None,
                    "id": ids[idx] if idx < len(ids) else None,
                    "risks": result.get("risks", []),
                    "obligations": result.get("obligations", []),
                })

            all_risks = [item["risks"] for item in analysis]
            all_obligations = [item["obligations"] for item in analysis]

            summary = await self.summarizer.execute(state['clauses'], pages, ids)

            return {
                "risks": all_risks,
                "obligations": all_obligations,
                "summary": summary,
                "execution_log": state['execution_log'] + ["Analyzed risks and obligations in batches, then summarized."],
                "analysis": analysis,
            }
        except Exception:
            # Defensive fallback: return empty structured outputs so UI never crashes
            clauses = state.get('clauses', [])
            analysis = [{
                "index": i + 1,
                "clause_excerpt": (clauses[i] if i < len(clauses) else "")[:200],
                "page": None,
                "id": None,
                "risks": [],
                "obligations": [],
            } for i in range(len(clauses))]
            return {
                "risks": [a["risks"] for a in analysis],
                "obligations": [a["obligations"] for a in analysis],
                "summary": "Summary (local): Unable to compute due to provider error",
                "execution_log": state.get('execution_log', []) + ["Full analysis fallback due to error"],
                "analysis": analysis,
            }

    async def run(self, legal_text: str = "", clause_items: List[Dict[str, Any]] = None) -> dict:
        builder = DeclarativeGraphBuilder(state_schema=LegalAnalysisState)
        graph: StateGraph = builder.build(self.graph_template)
        app = graph.compile()

        initial_state = LegalAnalysisState(legal_text=legal_text, clause_items=clause_items or [])

        final_state = await app.invoke(asdict(initial_state))

        return final_state