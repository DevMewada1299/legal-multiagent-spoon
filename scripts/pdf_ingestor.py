import fitz  # PyMuPDF
import json

def ingest_pdf_to_memory(pdf_path):
    # 1. Initialize the in-memory store
    document_memory = []
    
    try:
        # 2. Open the Document
        doc = fitz.open(pdf_path)
        print(f"Successfully opened: {pdf_path}")
        print(f"Total Pages: {len(doc)}\n" + "-"*30)

        # 3. Iterate through pages
        for page_num, page in enumerate(doc, start=1):
            
            # Extract raw text (fastest method)
            text_content = page.get_text()
            
            # Create the data object (The Schema)
            page_data = {
                "page_number": page_num,
                "source_file": pdf_path,
                "character_count": len(text_content),
                "raw_text": text_content
            }
            
            # Append to our in-memory list
            document_memory.append(page_data)
            
            # Print verification to console
            print(f"Processed Page {page_num}: Extracted {len(text_content)} chars.")

        doc.close()
        return document_memory

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

# --- Usage ---
pdf_file = "scripts/SampleContract-Shuttle.pdf" # Replace with your actual file path
memory_store = ingest_pdf_to_memory(pdf_file)

# Verify what is stored in memory
if memory_store:
    print("\n--- INSPECTING MEMORY (Page 1) ---")
    print(json.dumps(memory_store[0], indent=2))