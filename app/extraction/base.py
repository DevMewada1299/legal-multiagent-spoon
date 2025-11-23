from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Extractor(ABC):
    @abstractmethod
    def extract(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass