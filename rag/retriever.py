#!/usr/bin/env python3
"""
Phase 13 Module: Local Medical Retriever
Wraps LocalVectorDatabase to perform top-k context retrieval for user medical queries.
"""

from typing import List, Dict, Any, Tuple, Optional
from rag.database import LocalVectorDatabase

class MedicalRetriever:
    """
    Retriever class for querying local vector database.
    """
    def __init__(self, db_dir: str = "data/rag_db", vector_db: Optional[LocalVectorDatabase] = None):
        if vector_db is not None:
            self.db = vector_db
        else:
            self.db = LocalVectorDatabase()
            self.db.load(db_dir)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant document chunks for query string.
        """
        results = self.db.search(query=query, top_k=top_k)
        retrieved_chunks = []
        for chunk, score in results:
            item = dict(chunk)
            item["score"] = score
            retrieved_chunks.append(item)
        return retrieved_chunks
