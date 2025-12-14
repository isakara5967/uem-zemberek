from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO
from functools import lru_cache

import struct
import numpy as np

if TYPE_CHECKING:
    from zemberek.core.hash.mphf import Mphf

from zemberek.core.hash.multi_level_mphf import MultiLevelMphf

# Try to import Cython optimized functions
try:
    from zemberek.cython.hash_functions import java_hash_code_cy
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


class LossyIntLookup:

    MAGIC: np.int32 = np.int32(-889274641)

    def __init__(self, mphf: Mphf, data: np.ndarray):
        self.mphf = mphf
        self.data = data

    def get_(self, s: str) -> np.int32:
        index = self.mphf.get_(s) * 2
        fingerprint: int = LossyIntLookup.get_fingerprint(s)

        if fingerprint == self.data[index]:
            return self.data[index + 1]
        else:
            return np.int32(0)

    def size_(self) -> int:
        return self.data.shape[0] // 2

    def get_as_float(self, s: str) -> np.int32:
        return self.java_int_bits_to_float(self.get_(s))

    @staticmethod
    def get_fingerprint(s: str) -> np.int32:
        """
        This method performs a bitwise and operation for the hash of a string. It uses java's string hash method
        (hashCode()) therefore we implemented java's hash code in python. From java doc:
        s[0]*31^(n-1) + s[1]*31^(n-2) + ... + s[n-1]

        using int arithmetic, where s[i] is the ith character of the string, n is the length of the string,
        and ^ indicates exponentiation. (The hash value of the empty string is zero.)
        :param s:
        :return:
        """
        return LossyIntLookup.java_hash_code(s) & 0x7ffffff

    @staticmethod
    def java_int_bits_to_float(b: np.int32) -> np.float32:
        s = struct.pack('>i', b)
        return np.float32(struct.unpack('>f', s)[0])

    @staticmethod
    @lru_cache(maxsize=50000)
    def _java_hash_code_compute(s: str) -> np.int32:
        """L1 cached computation of java hash code. Uses Cython if available."""
        if _USE_CYTHON:
            return np.int32(java_hash_code_cy(s))

        # Fallback to NumPy implementation
        arr = np.asarray([ord(c) for c in s], dtype=np.int32)
        powers = np.arange(arr.shape[0], dtype=np.int32)[::-1]
        bases = np.full((arr.shape[0],), 31, dtype=np.int32)
        result = np.sum(arr * (np.power(bases, powers)), dtype=np.int32)
        return np.int32(result)

    @staticmethod
    def java_hash_code(s: str) -> np.int32:
        """
        Compute java-compatible hash code with multi-level caching.
        L0 (pre-computed) -> L2 (shared) -> L1 (lru_cache) -> compute
        """
        manager = _get_cache_manager()
        return np.int32(manager.get_java_hash(s, LossyIntLookup._java_hash_code_compute))

    @classmethod
    def deserialize(cls, dis: BinaryIO) -> 'LossyIntLookup':
        magic = np.int32(struct.unpack('>i', dis.read(4))[0])

        if magic != cls.MAGIC:
            raise ValueError(f"File does not carry expected value in the beginning. magic != LossyIntLookup.magid")

        length = np.int32(struct.unpack('>i', dis.read(4))[0])
        data = np.empty((length, ), dtype=np.int32)
        for i in range(length):
            data[i] = struct.unpack('>i', dis.read(4))[0]

        mphf: 'Mphf' = MultiLevelMphf.deserialize(dis)
        return cls(mphf, data)
