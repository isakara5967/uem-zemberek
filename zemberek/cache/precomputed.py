"""
L0 Pre-computed Cache for Zemberek hash functions.

This module provides pre-computed hash values for frequently used strings,
eliminating the need for runtime computation of java_hash_code and hash_for_str.
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from importlib.resources import files


class PrecomputedCache:
    """
    Singleton class for pre-computed hash values.

    Loads hash values from binary file at startup.
    Falls back to empty dict if file doesn't exist.
    """

    _instance: Optional['PrecomputedCache'] = None
    _initialized: bool = False

    def __new__(cls) -> 'PrecomputedCache':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if PrecomputedCache._initialized:
            return

        self._java_hashes: Dict[str, int] = {}
        self._str_hashes: Dict[Tuple[str, int], int] = {}

        self._load_cache()
        PrecomputedCache._initialized = True

    def _load_cache(self) -> None:
        """Load pre-computed hashes from binary file using importlib.resources."""
        try:
            # Try importlib.resources first (works with installed packages)
            resource = files("zemberek").joinpath("resources", "precomputed_hashes.bin")
            with resource.open('rb') as f:
                data = pickle.load(f)
                self._java_hashes = data.get('java_hashes', {})
                self._str_hashes = data.get('str_hashes', {})
                return
        except (FileNotFoundError, TypeError, AttributeError):
            pass

        # Fallback to direct file path (development mode)
        try:
            cache_path = Path(__file__).parent.parent / "resources" / "precomputed_hashes.bin"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self._java_hashes = data.get('java_hashes', {})
                    self._str_hashes = data.get('str_hashes', {})
        except (pickle.PickleError, IOError, KeyError):
            pass

    def _get_cache_path(self) -> Path:
        """Get the cache file path."""
        return Path(__file__).parent.parent / "resources" / "precomputed_hashes.bin"

    def save_cache(self) -> None:
        """Save current cache to binary file."""
        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'java_hashes': self._java_hashes,
            'str_hashes': self._str_hashes
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_java_hash(self, key: str) -> Optional[int]:
        """
        Get pre-computed java_hash_code value.

        Args:
            key: The string to look up

        Returns:
            Pre-computed hash value or None if not cached
        """
        return self._java_hashes.get(key)

    def set_java_hash(self, key: str, value: int) -> None:
        """Store a java_hash_code value."""
        self._java_hashes[key] = value

    def get_str_hash(self, key: str, seed: int = 0) -> Optional[int]:
        """
        Get pre-computed hash_for_str value.

        Args:
            key: The string to look up
            seed: The seed value used in hashing

        Returns:
            Pre-computed hash value or None if not cached
        """
        return self._str_hashes.get((key, seed))

    def set_str_hash(self, key: str, seed: int, value: int) -> None:
        """Store a hash_for_str value."""
        self._str_hashes[(key, seed)] = value

    def __len__(self) -> int:
        """Return total number of cached entries."""
        return len(self._java_hashes) + len(self._str_hashes)

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'java_hashes': len(self._java_hashes),
            'str_hashes': len(self._str_hashes),
            'total': len(self)
        }
