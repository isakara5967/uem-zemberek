# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized hash functions for Zemberek.

These are drop-in replacements for the NumPy-based hash functions,
providing significant speedup for hash computation.
"""

cimport cython
from libc.stdint cimport int32_t, uint32_t

# Constants matching Java/Zemberek
DEF HASH_MULTIPLIER = 16777619
DEF INITIAL_HASH_SEED = -2128831035  # 0x811c9dc5 as signed int32


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int32_t hash_for_str_cy(str data, int seed):
    """
    Cython version of MultiLevelMphf._hash_for_str_compute

    Computes FNV-1a style hash with Java int32 overflow behavior.
    """
    cdef int32_t d
    cdef int32_t c_ord
    cdef Py_UCS4 c

    # Initialize seed
    if seed > 0:
        d = <int32_t>seed
    else:
        d = INITIAL_HASH_SEED

    # Process each character
    for c in data:
        c_ord = <int32_t>c
        # XOR and multiply with natural int32 overflow
        d = (d ^ c_ord) * HASH_MULTIPLIER

    # Mask to positive int32
    return d & 0x7fffffff


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int32_t java_hash_code_cy(str s):
    """
    Cython version of LossyIntLookup._java_hash_code_compute

    Java String.hashCode() compatible:
    s[0]*31^(n-1) + s[1]*31^(n-2) + ... + s[n-1]
    """
    cdef int32_t result = 0
    cdef int32_t c_ord
    cdef Py_UCS4 c

    for c in s:
        c_ord = <int32_t>c
        # This is equivalent to: result = result * 31 + c_ord
        # with natural int32 overflow (Java behavior)
        result = (result * 31) + c_ord

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int32_t hash_for_int_tuple_cy(tuple data, int seed):
    """
    Cython version of MultiLevelMphf.hash_for_int_tuple

    Hash function for integer tuples (word index tuples).
    """
    cdef int32_t d
    cdef int32_t a
    cdef int i
    cdef int n = len(data)

    # Initialize seed
    if seed > 0:
        d = <int32_t>seed
    else:
        d = INITIAL_HASH_SEED

    # Process each integer
    for i in range(n):
        a = <int32_t>data[i]
        d = (d ^ a) * HASH_MULTIPLIER

    return d & 0x7fffffff


# Batch processing for multiple strings (optional optimization)
@cython.boundscheck(False)
@cython.wraparound(False)
def hash_for_str_batch_cy(list strings, int seed):
    """
    Batch hash computation for multiple strings.
    Returns list of hash values.
    """
    cdef list results = []
    cdef str s

    for s in strings:
        results.append(hash_for_str_cy(s, seed))

    return results
