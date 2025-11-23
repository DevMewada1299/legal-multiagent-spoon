from typing import List, Dict, Any
from .base import Classifier

class LegalClassifier(Classifier):
    def classify(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for clause in clauses:
            text = clause['text'].lower()
            if "termination" in text:
                clause['label'] = "Termination"
            elif "indemnify" in text or "indemnification" in text:
                clause['label'] = "Indemnification"
            elif "confidentiality" in text:
                clause['label'] = "Confidentiality"
            else:
                clause['label'] = "General"
        return clauses