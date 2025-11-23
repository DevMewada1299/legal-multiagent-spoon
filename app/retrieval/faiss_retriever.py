import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from .base import Retriever

class FaissRetriever(Retriever):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Initializing FaissRetriever")
        self.model = SentenceTransformer(model_name)
        self._index = None
        self.clauses = []

    def index(self, clauses: List[Dict[str, Any]]):
        self.clauses = clauses
        embeddings = self.model.encode([c['text'] for c in clauses])
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._index:
            return []
        query_embedding = self.model.encode([query])
        distances, indices = self._index.search(query_embedding, k)
        return [self.clauses[i] for i in indices[0]]