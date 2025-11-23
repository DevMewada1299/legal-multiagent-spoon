from dataclasses import dataclass, field, asdict
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
    RiskAnalysisAgent,
    ObligationExtractionAgent,
)
from spoon_ai.chat import ChatBot


@dataclass
class LegalAnalysisState:
    legal_text: str
    clauses: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    summary: str = ""
    execution_log: List[str] = field(default_factory=list)


class LegalAnalysisGraph:
    def __init__(self):
        chatbot = ChatBot(llm_provider="gemini")
        self.clause_extractor = ClauseExtractionAgent(llm=chatbot)
        self.risk_analyzer = RiskAnalysisAgent(llm=chatbot)
        self.obligation_extractor = ObligationExtractionAgent(llm=chatbot)
        self.summarizer = SummarizationAgent(llm=chatbot)
        self.graph_template = self._build_template()

    def _build_template(self) -> GraphTemplate:
        nodes = [
            NodeSpec(name="extract_clauses", handler=self.handle_clause_extraction),
            NodeSpec(name="analyze_risks", handler=self.handle_risk_analysis),
            NodeSpec(name="extract_obligations", handler=self.handle_obligation_extraction),
            NodeSpec(name="summarize_clauses", handler=self.handle_summarization),
        ]
        edges = [
            EdgeSpec(start="extract_clauses", end="analyze_risks"),
            EdgeSpec(start="analyze_risks", end="extract_obligations"),
            EdgeSpec(start="extract_obligations", end="summarize_clauses"),
            EdgeSpec(start="summarize_clauses", end=END),
        ]
        return GraphTemplate(entry_point="extract_clauses", nodes=nodes, edges=edges)

    async def handle_clause_extraction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        clauses = self.clause_extractor.execute(state['legal_text'])
        return {"clauses": clauses, "execution_log": state['execution_log'] + ["Extracted clauses"]}

    async def handle_risk_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        risks = await self.risk_analyzer.execute(state['clauses'])
        return {"risks": risks, "execution_log": state['execution_log'] + ["Analyzed risks"]}

    async def handle_obligation_extraction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        obligations = await self.obligation_extractor.execute(state['clauses'])
        return {"obligations": obligations, "execution_log": state['execution_log'] + ["Extracted obligations"]}

    async def handle_summarization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summary = await self.summarizer.execute(state['clauses'])
        return {"summary": summary, "execution_log": state['execution_log'] + ["Summarized clauses"]}

    async def run(self, legal_text: str) -> dict:
        builder = DeclarativeGraphBuilder(state_schema=LegalAnalysisState)
        graph: StateGraph = builder.build(self.graph_template)
        app = graph.compile()

        initial_state = LegalAnalysisState(legal_text=legal_text)

        final_state = await app.invoke(asdict(initial_state))

        return final_state