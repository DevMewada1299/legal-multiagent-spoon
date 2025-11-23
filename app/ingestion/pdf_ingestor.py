import pymupdf as fitz  # PyMuPDF
from typing import List, Dict, Any
from .base import Ingestor

class PDFIngestor(Ingestor):
    def ingest(self, file_path: str) -> List[Dict[str, Any]]:
        document = fitz.open(file_path)
        clauses = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")
            # Simple clause segmentation by splitting on double newlines
            for i, clause_text in enumerate(text.split('\n\n')):
                if clause_text.strip():
                    clauses.append({
                        "id": f"page_{page_num+1}_clause_{i+1}",
                        "text": clause_text.strip(),
                        "page": page_num + 1,
                    })
        return clauses