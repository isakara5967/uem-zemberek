#!/usr/bin/env python3
"""
Generate L0 pre-computed cache from profiled keys.
Uses actual feature strings from disambiguation profiling.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zemberek.cache.precomputed import PrecomputedCache
from zemberek.core.compression.lossy_int_lookup import LossyIntLookup
from zemberek.core.hash.multi_level_mphf import MultiLevelMphf


def main():
    print("=" * 70)
    print("L0 PRECOMPUTED CACHE GENERATOR v2")
    print("=" * 70)

    # Load profiled keys
    profile_path = Path(__file__).parent / "profiled_keys.json"
    if not profile_path.exists():
        print(f"HATA: {profile_path} bulunamadi")
        print("Once profile_cache_keys.py calistirin!")
        return

    print("\n[1/4] Loading profiled keys...")
    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    java_keys = data['java_keys']
    str_keys = [(k, s) for k, s in data['str_keys']]

    print(f"   Java keys: {len(java_keys)}")
    print(f"   Str keys: {len(str_keys)}")

    # Initialize cache (fresh)
    print("\n[2/4] Computing java_hash_code values...")
    cache = PrecomputedCache()
    # Reset internal dicts for fresh start
    cache._java_hashes = {}
    cache._str_hashes = {}

    java_count = 0
    for i, key in enumerate(java_keys):
        hash_value = int(LossyIntLookup._java_hash_code_compute(key))
        cache.set_java_hash(key, hash_value)
        java_count += 1

        if (i + 1) % 10000 == 0:
            print(f"   Progress: {i+1}/{len(java_keys)}")

    print(f"   Computed {java_count} java_hash values")

    # Compute str_hash values
    print("\n[3/4] Computing hash_for_str values...")
    str_count = 0
    for i, (key, seed) in enumerate(str_keys):
        hash_value = int(MultiLevelMphf._hash_for_str_compute(key, seed))
        cache.set_str_hash(key, seed, hash_value)
        str_count += 1

        if (i + 1) % 20000 == 0:
            print(f"   Progress: {i+1}/{len(str_keys)}")

    print(f"   Computed {str_count} hash_for_str values")

    # Save to disk
    print("\n[4/4] Saving cache...")
    cache.save_cache()

    cache_path = cache._get_cache_path()
    if cache_path.exists():
        size_kb = cache_path.stat().st_size / 1024
        print(f"   File: {cache_path}")
        print(f"   Size: {size_kb:.1f} KB")

    print("\n" + "=" * 70)
    print(f"Cache stats: {cache.stats()}")
    print("L0 Precomputed Cache successfully generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
