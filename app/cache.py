# app/cache.py

import numpy as np
from typing import Dict, List, Tuple

class EmbeddingCache:
    """A simple in-memory cache for face embeddings."""
    def __init__(self):
        self.names: List[str] = []
        self.ids: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        print("EmbeddingCache initialized.")

    def is_empty(self) -> bool:
        return len(self.names) == 0

    def get_all(self) -> Tuple[List[str], np.ndarray, List[str]]:
        return self.names, self.embeddings, self.ids

    def update(self, names: List[str], embeddings: np.ndarray, ids: List[str]):
        self.names = names
        self.embeddings = embeddings
        self.ids = ids
        print(f"Cache updated with {len(names)} embeddings.")

    def add_employee(self, name: str, embedding: np.ndarray, emp_id: str):
        self.names.append(name)
        self.ids.append(emp_id)
        if self.embeddings.size == 0:
            self.embeddings = np.expand_dims(embedding, axis=0)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        print(f"Added '{name}' (ID: {emp_id}) to cache.")

# Global cache instance
embedding_cache = EmbeddingCache()