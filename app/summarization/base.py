from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Summarizer(ABC):
    @abstractmethod
    def summarize(self, clauses: List[Dict[str, Any]]) -> str:
        pass