import sys
import asyncio
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.pdf_ingestor import PDFIngestor
from graph_pipeline.graph import LegalAnalysisGraph

load_dotenv()

async def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "scripts/SampleContract-Shuttle.pdf" 

    ingestor = PDFIngestor()
    clauses = ingestor.ingest(pdf_path)
    
    # We need to extract the text from the clauses
    clause_texts = [clause['text'] for clause in clauses]

    # Create and run the graph
    analysis_graph = LegalAnalysisGraph()
    final_state = await analysis_graph.run("\n".join(clause_texts))

    print("--- Legal Analysis Results ---")
    print("\n--- Summary ---")
    print(final_state['summary'])
    print("\n--- Risks ---")
    for risk in final_state['risks']:
        print(f"- {risk}")
    print("\n--- Obligations ---")
    for obligation in final_state['obligations']:
        print(f"- {obligation}")

if __name__ == "__main__":
    asyncio.run(main())