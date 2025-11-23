from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Retriever(ABC):
    @abstractmethod
    def index(self, clauses: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        pass