"""Simple Worker Pool for parallel morphological analysis."""

import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

# Global cache for worker processes
_worker_cache = None


def _init_worker():
    """Initialize worker with PrecomputedCache."""
    global _worker_cache
    from zemberek.cache import PrecomputedCache
    _worker_cache = PrecomputedCache()


def _process_word(word: str) -> Tuple[str, Optional[int]]:
    """Process single word, return (word, hash or None)."""
    global _worker_cache
    if _worker_cache:
        result = _worker_cache.get_str_hash(word)
        return (word, result)
    return (word, None)


class WorkerPool:
    """Multi-process pool for parallel word processing."""

    def __init__(self, workers: int = None):
        """Initialize pool with specified number of workers."""
        self.num_workers = workers or max(1, cpu_count() - 1)
        self._pool: Optional[Pool] = None

    def _ensure_pool(self):
        """Create pool if not exists."""
        if self._pool is None:
            self._pool = Pool(
                processes=self.num_workers,
                initializer=_init_worker
            )

    def process_words(self, words: List[str]) -> dict:
        """Process words in parallel, return {word: hash}."""
        self._ensure_pool()
        results = self._pool.map(_process_word, words)
        return {word: h for word, h in results if h is not None}

    def warm_up(self):
        """Warm up pool by initializing workers."""
        self._ensure_pool()
        self._pool.map(lambda x: x, range(self.num_workers))

    def shutdown(self):
        """Shutdown pool gracefully."""
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        self._ensure_pool()
        return self

    def __exit__(self, *args):
        self.shutdown()
