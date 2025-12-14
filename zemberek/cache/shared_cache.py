"""
L2 Shared Memory Cache for Zemberek hash functions.

This module provides a process-safe shared cache using multiprocessing.Manager,
allowing all worker processes to share the same cache.
"""

from multiprocessing import Manager, Lock
from typing import Optional, Dict, Any


class SharedCache:
    """
    Singleton shared memory cache for multi-process environments.

    Uses multiprocessing.Manager().dict() for cross-process sharing.
    Thread-safe with Lock protection.
    """

    _instance: Optional['SharedCache'] = None
    _lock = Lock()

    def __new__(cls) -> 'SharedCache':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._manager = Manager()
            self._java_hash_cache: Dict[str, int] = self._manager.dict()
            self._str_hash_cache: Dict[str, int] = self._manager.dict()
            self._access_lock = self._manager.Lock()
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'SharedCache':
        """Get the singleton instance."""
        return cls()

    def get(self, key: str, hash_type: str) -> Optional[int]:
        """
        Get cached hash value.

        Args:
            key: The string key to look up
            hash_type: "java_hash" or "str_hash"

        Returns:
            Cached hash value or None if not found
        """
        cache = self._get_cache(hash_type)
        if cache is None:
            return None

        with self._access_lock:
            return cache.get(key)

    def set(self, key: str, value: int, hash_type: str) -> None:
        """
        Set cached hash value.

        Args:
            key: The string key
            value: The hash value to cache
            hash_type: "java_hash" or "str_hash"
        """
        cache = self._get_cache(hash_type)
        if cache is None:
            return

        with self._access_lock:
            cache[key] = value

    def _get_cache(self, hash_type: str) -> Optional[Dict[str, int]]:
        """Get the appropriate cache dict based on hash_type."""
        if hash_type == "java_hash":
            return self._java_hash_cache
        elif hash_type == "str_hash":
            return self._str_hash_cache
        return None

    def clear(self) -> None:
        """Clear all cached values."""
        with self._access_lock:
            self._java_hash_cache.clear()
            self._str_hash_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._access_lock:
            return {
                'java_hash_count': len(self._java_hash_cache),
                'str_hash_count': len(self._str_hash_cache),
                'total': len(self._java_hash_cache) + len(self._str_hash_cache)
            }

    def __len__(self) -> int:
        """Return total number of cached entries."""
        with self._access_lock:
            return len(self._java_hash_cache) + len(self._str_hash_cache)
