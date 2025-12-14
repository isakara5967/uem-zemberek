import numpy as np

from typing import List, BinaryIO, Tuple, Optional, Union
from struct import unpack
from functools import lru_cache

from zemberek.core.hash.mphf import Mphf

np.seterr(over='ignore')

# Try to import Cython optimized functions
try:
    from zemberek.cython.hash_functions import hash_for_str_cy, hash_for_int_tuple_cy
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

# Cache manager instance (lazy initialized)
_cache_manager = None

def _get_cache_manager():
    """Lazy initialization of cache manager."""
    global _cache_manager
    if _cache_manager is None:
        from zemberek.cache import CacheManager
        _cache_manager = CacheManager.get_instance()
    return _cache_manager


class MultiLevelMphf(Mphf):
    """
    Minimum Perfect Hash Function Implementation. Detailed explanation can be found in original zemberek file
    """

    HASH_MULTIPLIER: np.int32 = np.int32(16777619)
    INITIAL_HASH_SEED: np.int32 = np.int32(-2128831035)
    BIT_MASK_21: np.int32 = np.int32(2097151)  # np.int32((1 << 21) - 1)

    def __init__(self, hash_level_data: Tuple['MultiLevelMphf.HashIndexes']):
        self.hash_level_data = hash_level_data

    @staticmethod
    def deserialize(f: BinaryIO) -> 'MultiLevelMphf':
        level_count, = unpack('>i', f.read(4))
        indexes: List['MultiLevelMphf.HashIndexes'] = []

        for i in range(level_count):
            key_count, = unpack('>i', f.read(4))
            bucket_amount, = unpack('>i', f.read(4))
            hash_seed_values: bytes = f.read(bucket_amount)
            failed_indexes_count, = unpack('>i', f.read(4))
            failed_indexes: np.ndarray = np.zeros(failed_indexes_count, dtype=np.int32)
            for j in range(failed_indexes_count):
                failed_indexes[j], = unpack('>i', f.read(4))

            indexes.append(MultiLevelMphf.HashIndexes(key_count, bucket_amount, hash_seed_values, failed_indexes))
        return MultiLevelMphf(tuple(indexes))

    @staticmethod
    @lru_cache(maxsize=50000)
    def _hash_for_str_compute(data: str, seed: int) -> np.int32:
        """L1 cached computation of string hash. Uses Cython if available."""
        if _USE_CYTHON:
            return np.int32(hash_for_str_cy(data, seed))

        # Fallback to NumPy implementation
        d = np.int32(seed) if seed > 0 else MultiLevelMphf.INITIAL_HASH_SEED

        for c in data:
            d = (d ^ np.int32(ord(c))) * MultiLevelMphf.HASH_MULTIPLIER

        return d & np.int32(0x7fffffff)

    @staticmethod
    def hash_for_str(data: str, seed: int) -> np.int32:
        """
        Compute string hash with multi-level caching.
        L0 (pre-computed) -> L2 (shared) -> L1 (lru_cache) -> compute
        """
        manager = _get_cache_manager()
        return np.int32(manager.get_str_hash(data, seed, MultiLevelMphf._hash_for_str_compute))

    @staticmethod
    def hash_for_int_tuple(data: Tuple[int, ...], seed: int) -> np.int32:
        """Hash function for integer tuples. Uses Cython if available."""
        if _USE_CYTHON:
            return np.int32(hash_for_int_tuple_cy(data, seed))

        # Fallback to NumPy implementation
        d = np.int32(seed) if seed > 0 else MultiLevelMphf.INITIAL_HASH_SEED
        for a in np.asarray(data, dtype=np.int32):
            d = (d ^ a) * MultiLevelMphf.HASH_MULTIPLIER

        return d & np.int32(0x7fffffff)

    @staticmethod
    def hash_(
            data: Union[Tuple[int, ...], str],
            seed: int
    ) -> np.int32:

        if isinstance(data, str):
            return MultiLevelMphf.hash_for_str(data, seed)
        elif isinstance(data, tuple):
            return MultiLevelMphf.hash_for_int_tuple(data, seed)
        else:
            raise ValueError(f"(data) parameter type not supported: {type(data)}")


    def get_for_str(self, key: str, initial_hash: Optional[int] = None):

        if initial_hash is None:
            initial_hash = self.hash_for_str(key, seed=-1)

        for i, hd in enumerate(self.hash_level_data):
            seed = hd.get_seed(initial_hash)

            if seed != 0:
                if i == 0:
                    return self.hash_for_str(key, seed) % self.hash_level_data[0].key_amount
                else:
                    return self.hash_level_data[i - 1].failed_indexes[self.hash_for_str(key, seed) %
                                                                      self.hash_level_data[i].key_amount]

        return BaseException("Cannot be here")

    def get_for_tuple(self, key: Tuple[int, ...], initial_hash: int) -> np.int32:
        for i in range(len(self.hash_level_data)):
            seed = self.hash_level_data[i].get_seed(initial_hash)
            if seed != 0:
                if i == 0:
                    return self.hash_(key, seed) % self.hash_level_data[0].key_amount
                else:
                    return self.hash_level_data[i - 1].failed_indexes[self.hash_(key, seed) %
                                                                      self.hash_level_data[i].key_amount]
        raise BaseException("Cannot be here.")

    def get_(
            self,
            key: Union[Tuple[int, ...], str],
            initial_hash: int = None
    ) -> np.int32:

        if isinstance(key, str):
            return self.get_for_str(key, initial_hash)
        elif isinstance(key, tuple):
            return self.get_for_tuple(key, initial_hash)
        else:
            raise ValueError(f"(key) parameter type not supported: {type(key)}")

    class HashIndexes:
        def __init__(self, key_amount: int, bucket_amount: int, bucket_hash_seed_values: bytes,
                     failed_indexes: np.ndarray):
            self.key_amount = key_amount
            self.bucket_amount = bucket_amount
            self.bucket_hash_seed_values = bucket_hash_seed_values
            self.failed_indexes = failed_indexes

        def get_seed(self, finger_print: int) -> int:
            return (self.bucket_hash_seed_values[finger_print % self.bucket_amount]) & 0xFF
