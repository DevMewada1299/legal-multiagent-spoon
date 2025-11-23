import fitz  # PyMuPDF
from typing import List, Dict

class PDFIngestionTool:
    @staticmethod
    def process(file_path: str) -> List[Dict]:
        """
        Extracts text from PDF while preserving page structure.
        Returns the JSON schema you requested.
        """
        doc = fitz.open(file_path)
        pages_data = []

        for i, page in enumerate(doc):
            text = page.get_text("text") # Preserve basic layout with blocks
            
            page_obj = {
                "page_number": i + 1,
                "source_file": file_path.split("/")[-1],
                "character_count": len(text),
                "raw_text": text
            }
            pages_data.append(page_obj)
            
        return pages_data