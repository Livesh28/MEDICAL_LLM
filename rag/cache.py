"""
Medical Query Cache Module
Provides high-performance, thread-safe in-memory LRU caching for verified medical RAG responses.
Reduces latency from ~2-3 seconds down to < 1ms for repeated questions.
"""

import re
import time
import threading
from typing import Dict, Any, Optional
from collections import OrderedDict


def normalize_query(query: str) -> str:
    """Normalizes question for consistent cache key generation."""
    if not query:
        return ""
    q = query.lower().strip()
    # Remove surrounding punctuation
    q = re.sub(r'^[^\w]+|[^\w]+$', '', q)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q)
    return q


class MedicalQueryCache:
    """
    Thread-safe LRU cache storing verified RAG answers, sources, and metadata.
    """

    def __init__(self, capacity: int = 500, ttl_seconds: float = 86400.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, query: str, model: str = "openbiollm") -> Optional[Dict[str, Any]]:
        norm_key = f"{model.lower().strip()}::{normalize_query(query)}"
        with self._lock:
            if norm_key not in self._cache:
                return None
            
            entry = self._cache[norm_key]
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[norm_key]
                return None
            
            # Move to MRU position
            self._cache.move_to_end(norm_key)
            result = dict(entry["data"])
            result["cache_hit"] = True
            return result

    def set(self, query: str, data: Dict[str, Any], model: str = "openbiollm"):
        norm_key = f"{model.lower().strip()}::{normalize_query(query)}"
        with self._lock:
            if norm_key in self._cache:
                self._cache.move_to_end(norm_key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)  # Evict oldest LRU item
            
            self._cache[norm_key] = {
                "timestamp": time.time(),
                "data": dict(data)
            }

    def clear(self):
        """Clears cache, e.g. when vector database is updated with new documents."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singleton instance
medical_cache = MedicalQueryCache(capacity=1000)
