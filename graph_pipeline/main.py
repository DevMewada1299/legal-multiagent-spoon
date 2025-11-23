import sys
import asyncio
import os
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.pdf_ingestor import PDFIngestor
from .agents import get_rpm
from graph_pipeline.graph import LegalAnalysisGraph

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Run the legal analysis pipeline.")
    parser.add_argument("--file-path", default="data/SampleContract-Shuttle.pdf", help="Path to the PDF file to analyze.")
    parser.add_argument("--analysis-model", default=None, help="LLM model for clause analysis (e.g., gemini-2.5-pro)")
    parser.add_argument("--summary-model", default=None, help="LLM model for summary (e.g., gemini-2.5-pro)")
    args = parser.parse_args()
    pdf_path = args.file_path

    ingestor = PDFIngestor()
    clauses = ingestor.ingest(pdf_path)
    
    # Create and run the graph with page-aware clause items
    analysis_graph = LegalAnalysisGraph(analysis_model=args.analysis_model, summary_model=args.summary_model)
    final_state = await analysis_graph.run(clause_items=clauses)

    print("--- Legal Analysis Results ---")
    print("\n--- Summary ---")
    print(final_state['summary'])
    print("\n--- Clause Analysis ---")
    for item in final_state.get('analysis', []):
        page = item.get('page')
        if page:
            print(f"Clause {item['index']} (p{page}): {item['clause_excerpt']}")
        else:
            print(f"Clause {item['index']}: {item['clause_excerpt']}")
        risks = item.get('risks') or []
        obligations = item.get('obligations') or []
        if risks:
            for r in risks[:2]:
                desc = r.get('description')
                sev = r.get('severity')
                cat = r.get('category')
                print(f"  Risk: {desc} (severity: {sev}, category: {cat})")
        else:
            print("  Risk: none")
        if obligations:
            for o in obligations[:2]:
                actor = o.get('actor')
                action = o.get('action')
                deadline = o.get('deadline')
                if deadline:
                    print(f"  Obligation: {actor} -> {action} (deadline: {deadline})")
                else:
                    print(f"  Obligation: {actor} -> {action}")
        else:
            print("  Obligation: none")
    print("\n--- LLM Requests in Last Minute ---")
    print(get_rpm())

if __name__ == "__main__":
    asyncio.run(main())