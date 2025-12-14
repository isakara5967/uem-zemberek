#!/usr/bin/env python3
"""Cache Layer Analysis - Detailed hit/miss and latency metrics."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sentences(path: str, limit: int = None):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            sentences.append(json.loads(line.strip()))
    return sentences


def run_analysis(sentences, limit: int):
    print("=" * 70)
    print("CACHE LAYER ANALYSIS")
    print("=" * 70)

    # Load modules
    print("\n[1/3] Moduller yukleniyor...")
    load_start = time.perf_counter()

    from zemberek import TurkishMorphology
    from zemberek.normalization import TurkishSentenceNormalizer
    from zemberek.cache import CacheManager

    morph = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morph)
    cache = CacheManager.get_instance()

    load_time = (time.perf_counter() - load_start) * 1000
    print(f"   Yukleme: {load_time:.0f}ms")

    # Enable latency tracking and reset stats
    cache.reset_stats()
    cache.enable_latency_tracking(True)

    # Run test
    test_data = sentences[:limit]
    total = len(test_data)
    total_tokens = 0

    print(f"\n[2/3] Test calistiriliyor ({total} cumle)...")
    test_start = time.perf_counter()

    for i, item in enumerate(test_data):
        sentence = item['sentence']
        total_tokens += len(sentence.split())

        # Run normalization (uses cache internally)
        normalizer.normalize(sentence)

        if (i + 1) % 200 == 0:
            print(f"   Ilerleme: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

    test_time = (time.perf_counter() - test_start) * 1000

    # Get stats
    stats = cache.stats()
    latency = cache.latency_stats()

    # Results
    print(f"\n[3/3] Sonuclar...")

    # Table 1: Cache Layer Analysis
    print("\n" + "=" * 70)
    print("TABLO 1: CACHE KATMAN ANALIZI")
    print("=" * 70)
    print(f"{'Katman':<15} {'Hits':>12} {'Miss':>12} {'Hit Rate':>12} {'Avg Latency':>15}")
    print("-" * 70)

    total_calls = stats['total_calls']
    l0_hits = stats['l0_hits']
    l2_hits = stats['l2_hits']
    misses = stats['misses']

    l0_lat = latency.get('l0_lookup', {}).get('avg_us', 0)
    l2_lat = latency.get('l2_lookup', {}).get('avg_us', 0)
    compute_lat = latency.get('compute', {}).get('avg_us', 0)

    print(f"{'L0 (Precomputed)':<15} {l0_hits:>12,} {'-':>12} {stats['l0_hit_rate']:>12} {l0_lat:>12.2f} us")
    print(f"{'L2 (LocalCache)':<15} {l2_hits:>12,} {'-':>12} {stats['l2_hit_rate']:>12} {l2_lat:>12.2f} us")
    print(f"{'Compute (MISS)':<15} {'-':>12} {misses:>12,} {'-':>12} {compute_lat:>12.2f} us")
    print("-" * 70)
    print(f"{'TOPLAM':<15} {l0_hits + l2_hits:>12,} {misses:>12,} {stats['hit_rate']:>12}")

    # Table 2: Latency Distribution
    print("\n" + "=" * 70)
    print("TABLO 2: LATENCY DAGILIMI")
    print("=" * 70)
    print(f"{'Kategori':<15} {'Count':>10} {'Avg':>12} {'p50':>12} {'p95':>12} {'p99':>12}")
    print("-" * 70)

    for cat in ['l0_lookup', 'l2_lookup', 'compute', 'total']:
        if cat in latency:
            l = latency[cat]
            print(f"{cat:<15} {l.get('count', 0):>10,} {l.get('avg_us', 0):>10.2f}us "
                  f"{l.get('p50_us', 0):>10.2f}us {l.get('p95_us', 0):>10.2f}us {l.get('p99_us', 0):>10.2f}us")

    # Table 3: Summary
    print("\n" + "=" * 70)
    print("TABLO 3: OZET METRIKLER")
    print("=" * 70)
    print(f"{'Metrik':<30} {'Deger':>20}")
    print("-" * 70)
    print(f"{'Toplam cumle':<30} {total:>20,}")
    print(f"{'Toplam token':<30} {total_tokens:>20,}")
    print(f"{'Toplam cache call':<30} {total_calls:>20,}")
    print(f"{'Call/token':<30} {total_calls/total_tokens:>20.1f}")
    print(f"{'Test suresi':<30} {test_time:>18.0f}ms")
    print(f"{'Throughput':<30} {total/(test_time/1000):>17.0f}/s")
    print(f"{'L0 size':<30} {stats['l0_size']:>20,}")
    print(f"{'L2 size':<30} {stats['l2_size']:>20,}")

    # Save results
    output = {
        'meta': {
            'sentences': total,
            'tokens': total_tokens,
            'test_time_ms': test_time,
        },
        'cache_stats': stats,
        'latency_stats': latency,
    }

    output_path = Path(__file__).parent / 'cache_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nKaydedildi: {output_path}")

    print("\n" + "=" * 70)
    print("ANALIZ TAMAMLANDI")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Cache Layer Analysis')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    if args.data:
        data_path = args.data
    else:
        data_path = Path(__file__).parent / 'data' / 'tr_benchmark_10000.jsonl'

    if not Path(data_path).exists():
        print(f"HATA: {data_path} bulunamadi")
        sys.exit(1)

    sentences = load_sentences(str(data_path), args.limit)
    run_analysis(sentences, args.limit)


if __name__ == '__main__':
    main()
