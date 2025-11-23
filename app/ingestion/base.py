from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Ingestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> List[Dict[str, Any]]:
        pass