# app/cache.py

import numpy as np
from typing import Dict, List, Tuple

class EmbeddingCache:
    """A simple in-memory cache for face embeddings."""
    def __init__(self):
        self.names: List[str] = []
        self.ids: List[str] = []
        self.member_codes: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        print("EmbeddingCache initialized.")

    def is_empty(self) -> bool:
        return len(self.names) == 0

    def get_all(self) -> Tuple[List[str], np.ndarray, List[str]]:
        return self.names, self.embeddings, self.ids, self.member_codes

    def update(self, names: List[str], embeddings: np.ndarray, ids: List[str], member_codes: List[str]): # <-- ADD member_codes here
        """
        Updates the entire cache with fresh data from the database.
        """
        self.names = names
        self.embeddings = embeddings
        self.ids = ids
        self.member_codes = member_codes
        print(f"Cache updated with {len(names)} embeddings.")

    def update_or_add_employee(self, emp_id: str, name: str, member_code: str, embedding: np.ndarray):
        """
        Updates an existing employee's details in the cache,
        or adds them if they don't exist.
        """
        try:
            idx = self.ids.index(emp_id)
            self.names[idx] = name
            self.member_codes[idx] = member_code 
            self.embeddings[idx] = embedding
            print(f"Updated '{name}' (ID: {emp_id}) in cache.")
        except ValueError:
            self.names.append(name)
            self.ids.append(emp_id)
            self.member_codes.append(member_code)
            if self.embeddings.size == 0:
                self.embeddings = np.expand_dims(embedding, axis=0)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            print(f"Added new employee '{name}' (ID: {emp_id}) to cache.")

    def remove_employee(self, emp_id: str) -> bool:
        """
        Removes an employee from the cache by their ID.
        Returns True if successful, False if the employee was not found.
        """
        try:
            idx = self.ids.index(emp_id)

            self.ids.pop(idx)
            name = self.names.pop(idx)
            self.member_codes.pop(idx)
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            
            print(f"Removed '{name}' (ID: {emp_id}) from cache.")
            return True
        except ValueError:
            print(f"Attempted to remove non-existent employee (ID: {emp_id}) from cache.")
            return False

# Global cache instance
embedding_cache = EmbeddingCache()