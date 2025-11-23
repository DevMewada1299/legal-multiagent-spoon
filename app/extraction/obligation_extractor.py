from typing import List, Dict, Any
from .base import Extractor

class ObligationExtractor(Extractor):
    def extract(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for clause in clauses:
            text = clause['text'].lower()
            if "shall" in text or "must" in text:
                clause['has_obligation'] = True
            else:
                clause['has_obligation'] = False
        return clauses