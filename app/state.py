from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel

class Clause(BaseModel):
    id: str
    text: str
    page_number: int
    label: str = "Unclassified"
    obligations: List[str] = []
    metadata: Dict[str, Any] = {}

class ContractState(BaseModel):
    """
    The shared state object that flows through the Graph.
    """
    source_filename: str = ""
    raw_pages: List[Dict] = []  # Input format: {page_number, raw_text, ...}
    
    # Processed Data
    clauses: List[Clause] = []
    
    # Retrieval Data
    index_id: str = ""
    
    # Final Output
    summary: str = ""
    citations: Dict[str, str] = {} # Map Summary ID -> Clause ID

    class Config:
        arbitrary_types_allowed = True