"""
CacheManager - Coordinates L0, L1, and L2 cache layers with detailed metrics.

Cache Hierarchy:
- L0: PrecomputedCache (pre-loaded from disk, read-only)
- L1: @lru_cache (per-process, in-memory) - tracked externally
- L2: LocalCache (in-process dict, NO IPC overhead)
"""

import time
from typing import Optional, Dict, Any, Callable, List
from statistics import mean, quantiles


class LocalCache:
    """Simple in-process cache using dict. No IPC overhead."""

    def __init__(self, maxsize: int = 100000):
        self._java_hash: Dict[str, int] = {}
        self._str_hash: Dict[str, int] = {}
        self._maxsize = maxsize

    def get(self, key: str, cache_type: str) -> Optional[int]:
        if cache_type == 'java_hash':
            return self._java_hash.get(key)
        return self._str_hash.get(key)

    def set(self, key: str, value: int, cache_type: str) -> None:
        target = self._java_hash if cache_type == 'java_hash' else self._str_hash
        if len(target) < self._maxsize:
            target[key] = value

    def __len__(self) -> int:
        return len(self._java_hash) + len(self._str_hash)


class CacheManager:
    """
    Singleton cache manager coordinating all cache layers.
    Lookup order: L0 -> L2 -> compute -> store in L2
    """

    _instance: Optional['CacheManager'] = None
    _initialized: bool = False

    def __new__(cls) -> 'CacheManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CacheManager._initialized:
            return

        from zemberek.cache.precomputed import PrecomputedCache

        self._l0 = PrecomputedCache()
        self._l2 = LocalCache()

        # Detailed statistics
        self._stats = {
            'l0_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'computes': 0,
            'java_hash_calls': 0,
            'str_hash_calls': 0,
        }

        # Latency tracking (in microseconds)
        self._latencies: Dict[str, List[float]] = {
            'l0_lookup': [],
            'l2_lookup': [],
            'compute': [],
            'total': [],
        }
        self._track_latency = False
        self._max_latency_samples = 10000

        CacheManager._initialized = True

    @classmethod
    def get_instance(cls) -> 'CacheManager':
        return cls()

    def enable_latency_tracking(self, enable: bool = True):
        """Enable/disable latency tracking."""
        self._track_latency = enable
        if enable:
            self._latencies = {k: [] for k in self._latencies}

    def get_java_hash(self, key: str, compute_func: Callable[[str], int]) -> int:
        self._stats['java_hash_calls'] += 1
        total_start = time.perf_counter() if self._track_latency else 0

        # L0: Check pre-computed cache
        l0_start = time.perf_counter() if self._track_latency else 0
        result = self._l0.get_java_hash(key)
        if self._track_latency:
            self._record_latency('l0_lookup', l0_start)

        if result is not None:
            self._stats['l0_hits'] += 1
            if self._track_latency:
                self._record_latency('total', total_start)
            return result

        # L2: Check local cache
        l2_start = time.perf_counter() if self._track_latency else 0
        result = self._l2.get(key, 'java_hash')
        if self._track_latency:
            self._record_latency('l2_lookup', l2_start)

        if result is not None:
            self._stats['l2_hits'] += 1
            if self._track_latency:
                self._record_latency('total', total_start)
            return result

        # MISS: Compute and store
        self._stats['misses'] += 1
        self._stats['computes'] += 1

        compute_start = time.perf_counter() if self._track_latency else 0
        result = compute_func(key)
        if self._track_latency:
            self._record_latency('compute', compute_start)

        self._l2.set(key, int(result), 'java_hash')

        if self._track_latency:
            self._record_latency('total', total_start)
        return result

    def get_str_hash(self, key: str, seed: int, compute_func: Callable[[str, int], int]) -> int:
        self._stats['str_hash_calls'] += 1
        total_start = time.perf_counter() if self._track_latency else 0
        cache_key = f"{key}:{seed}"

        # L0: Check pre-computed cache
        l0_start = time.perf_counter() if self._track_latency else 0
        result = self._l0.get_str_hash(key, seed)
        if self._track_latency:
            self._record_latency('l0_lookup', l0_start)

        if result is not None:
            self._stats['l0_hits'] += 1
            if self._track_latency:
                self._record_latency('total', total_start)
            return result

        # L2: Check local cache
        l2_start = time.perf_counter() if self._track_latency else 0
        result = self._l2.get(cache_key, 'str_hash')
        if self._track_latency:
            self._record_latency('l2_lookup', l2_start)

        if result is not None:
            self._stats['l2_hits'] += 1
            if self._track_latency:
                self._record_latency('total', total_start)
            return result

        # MISS: Compute and store
        self._stats['misses'] += 1
        self._stats['computes'] += 1

        compute_start = time.perf_counter() if self._track_latency else 0
        result = compute_func(key, seed)
        if self._track_latency:
            self._record_latency('compute', compute_start)

        self._l2.set(cache_key, int(result), 'str_hash')

        if self._track_latency:
            self._record_latency('total', total_start)
        return result

    def _record_latency(self, category: str, start_time: float):
        """Record latency in microseconds."""
        if len(self._latencies[category]) < self._max_latency_samples:
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000
            self._latencies[category].append(elapsed_us)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_calls = self._stats['java_hash_calls'] + self._stats['str_hash_calls']
        total_hits = self._stats['l0_hits'] + self._stats['l2_hits']

        result = {
            'l0_hits': self._stats['l0_hits'],
            'l2_hits': self._stats['l2_hits'],
            'misses': self._stats['misses'],
            'computes': self._stats['computes'],
            'total_calls': total_calls,
            'hit_rate': f"{(total_hits / total_calls * 100):.1f}%" if total_calls > 0 else "0%",
            'l0_hit_rate': f"{(self._stats['l0_hits'] / total_calls * 100):.1f}%" if total_calls > 0 else "0%",
            'l2_hit_rate': f"{(self._stats['l2_hits'] / total_calls * 100):.1f}%" if total_calls > 0 else "0%",
            'l0_size': len(self._l0),
            'l2_size': len(self._l2),
        }

        return result

    def latency_stats(self) -> Dict[str, Any]:
        """Return latency statistics (p50, p95, p99)."""
        result = {}
        for category, values in self._latencies.items():
            if len(values) >= 4:
                sorted_vals = sorted(values)
                q = quantiles(sorted_vals, n=100)
                result[category] = {
                    'count': len(values),
                    'avg_us': mean(values),
                    'p50_us': q[49],
                    'p95_us': q[94],
                    'p99_us': q[98],
                    'min_us': min(values),
                    'max_us': max(values),
                }
            elif values:
                result[category] = {
                    'count': len(values),
                    'avg_us': mean(values),
                }
        return result

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            'l0_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'computes': 0,
            'java_hash_calls': 0,
            'str_hash_calls': 0,
        }
        self._latencies = {k: [] for k in self._latencies}
