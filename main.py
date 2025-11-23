import asyncio
from dotenv import load_dotenv
from app.ingestion import PDFIngestor
from app.classification import LegalClassifier
from app.extraction import ObligationExtractor
from app.retrieval import FaissRetriever
from app.summarization import GeminiSummarizer

load_dotenv()

import argparse

async def main(file_path: str):
    # 1. Ingestion
    ingestor = PDFIngestor()
    clauses = ingestor.ingest(file_path)

    # 2. Classification
    classifier = LegalClassifier()
    classified_clauses = classifier.classify(clauses)

    # 3. Extraction
    extractor = ObligationExtractor()
    extracted_clauses = extractor.extract(classified_clauses)

    # 4. Retrieval
    retriever = FaissRetriever()
    retriever.index(extracted_clauses)
    retrieved_clauses = retriever.retrieve("What are the termination conditions?")

    # 5. Summarization
    summarizer = GeminiSummarizer()
    await summarizer.initialize()
    summary = await summarizer.summarize(extracted_clauses)

    print("--- Classified Clauses ---")
    for clause in extracted_clauses:
        print(f"- {clause['id']}: {clause['label']} (Obligation: {clause['has_obligation']})")

    print("\n--- Retrieved Clauses ---")
    for clause in retrieved_clauses:
        print(f"- {clause['id']}: {clause['text']}")

    print("\n--- Summary ---")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a legal document.")
    parser.add_argument("file_path", help="The path to the PDF file to analyze.")
    args = parser.parse_args()
    asyncio.run(main(args.file_path))