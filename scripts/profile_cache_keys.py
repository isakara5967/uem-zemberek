#!/usr/bin/env python3
"""
Profile cache keys used during disambiguation.
Collects all java_hash and str_hash keys for L0 precomputation.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch CacheManager to collect keys BEFORE imports
java_keys = Counter()
str_keys = Counter()  # (key, seed) tuples

original_get_java_hash = None
original_get_str_hash = None

def patched_get_java_hash(self, key, compute_func):
    java_keys[key] += 1
    return original_get_java_hash(self, key, compute_func)

def patched_get_str_hash(self, key, seed, compute_func):
    str_keys[(key, seed)] += 1
    return original_get_str_hash(self, key, seed, compute_func)

# Import and patch
from zemberek.cache import CacheManager
original_get_java_hash = CacheManager.get_java_hash
original_get_str_hash = CacheManager.get_str_hash
CacheManager.get_java_hash = patched_get_java_hash
CacheManager.get_str_hash = patched_get_str_hash

# Now import morphology
from zemberek import TurkishMorphology

def main():
    print("=" * 70)
    print("CACHE KEY PROFILER")
    print("=" * 70)

    # Load morphology
    print("\n[1/4] Loading morphology...")
    morph = TurkishMorphology.create_with_defaults()

    # Load test data
    print("\n[2/4] Loading test sentences...")
    data_path = Path(__file__).parent.parent / "benchmark" / "data" / "tr_benchmark_10000.jsonl"

    if not data_path.exists():
        print(f"HATA: {data_path} bulunamadi")
        return

    sentences = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Profile first 1000 sentences
                break
            data = json.loads(line.strip())
            sentences.append(data['sentence'])

    print(f"   Loaded {len(sentences)} sentences")

    # Run disambiguation
    print("\n[3/4] Running disambiguation...")
    for i, sentence in enumerate(sentences):
        try:
            morph.analyze_and_disambiguate(sentence)
        except Exception as e:
            pass  # Skip errors

        if (i + 1) % 200 == 0:
            print(f"   Progress: {i+1}/{len(sentences)}")

    # Results
    print("\n[4/4] Analyzing keys...")
    print("\n" + "=" * 70)
    print("JAVA HASH KEYS")
    print("=" * 70)
    print(f"Total unique keys: {len(java_keys)}")
    print(f"Total calls: {sum(java_keys.values())}")
    print(f"\nTop 20 most used keys:")
    for key, count in java_keys.most_common(20):
        print(f"  {count:>6}x  '{key[:60]}{'...' if len(key) > 60 else ''}'")

    print("\n" + "=" * 70)
    print("STR HASH KEYS")
    print("=" * 70)
    print(f"Total unique (key, seed) pairs: {len(str_keys)}")
    print(f"Total calls: {sum(str_keys.values())}")

    # Analyze seeds
    seeds = Counter()
    for (key, seed), count in str_keys.items():
        seeds[seed] += count

    print(f"\nSeed distribution:")
    for seed, count in seeds.most_common(10):
        print(f"  seed={seed:>4}: {count:>8} calls")

    print(f"\nTop 20 most used (key, seed) pairs:")
    for (key, seed), count in str_keys.most_common(20):
        print(f"  {count:>6}x  ('{key[:40]}{'...' if len(key) > 40 else ''}', {seed})")

    # Save to file
    output = {
        'java_keys': list(java_keys.keys()),
        'str_keys': [(k, s) for k, s in str_keys.keys()],
        'stats': {
            'java_unique': len(java_keys),
            'java_calls': sum(java_keys.values()),
            'str_unique': len(str_keys),
            'str_calls': sum(str_keys.values()),
        }
    }

    output_path = Path(__file__).parent / "profiled_keys.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 70)
    print(f"Saved to: {output_path}")
    print(f"Total keys to precompute: {len(java_keys)} java + {len(str_keys)} str")
    print("=" * 70)


if __name__ == "__main__":
    main()
