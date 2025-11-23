from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Classifier(ABC):
    @abstractmethod
    def classify(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass