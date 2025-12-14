#!/usr/bin/env python3
"""Zemberek Full Benchmark - 10K Sentence Comprehensive Test."""

import argparse
import json
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ModuleStats:
    name: str
    cold_start_ms: float = 0.0
    times_ms: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def avg_ms(self) -> float:
        return mean(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def throughput(self) -> float:
        return 1000 / self.avg_ms if self.avg_ms > 0 else 0.0

    @property
    def count(self) -> int:
        return len(self.times_ms)


@dataclass
class VariantStats:
    variant: str
    count: int = 0
    total_ms: float = 0.0
    successes: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def throughput(self) -> float:
        return 1000 / self.avg_ms if self.avg_ms > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.successes / self.count * 100) if self.count > 0 else 0.0


def load_sentences(path: str, limit: int = None) -> List[Dict[str, Any]]:
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            sentences.append(json.loads(line.strip()))
    return sentences


def measure(func: Callable) -> tuple:
    start = time.perf_counter()
    try:
        result = func()
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, result, None
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, None, str(e)


def run_benchmark(sentences: List[Dict], limit: int):
    print("=" * 80)
    print("ZEMBEREK FULL BENCHMARK (10K)")
    print("=" * 80)

    tracemalloc.start()
    mem_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

    # Load modules
    print("\n[1/4] Moduller yukleniyor...")
    load_start = time.perf_counter()

    from zemberek import TurkishMorphology
    from zemberek.tokenization import TurkishTokenizer, TurkishSentenceExtractor
    from zemberek.normalization import TurkishSpellChecker, TurkishSentenceNormalizer
    from zemberek.cache import CacheManager

    morph = TurkishMorphology.create_with_defaults()
    tokenizer = TurkishTokenizer.DEFAULT
    extractor = TurkishSentenceExtractor()
    spell_checker = TurkishSpellChecker(morph)
    normalizer = TurkishSentenceNormalizer(morph)
    cache_manager = CacheManager()

    load_time = (time.perf_counter() - load_start) * 1000
    print(f"   Yukleme suresi: {load_time:.0f}ms")

    mem_after_load = tracemalloc.get_traced_memory()[0] / 1024 / 1024

    # Initialize stats
    module_stats = {
        'tokenizer': ModuleStats('TurkishTokenizer.tokenize()'),
        'extractor': ModuleStats('TurkishSentenceExtractor.extract()'),
        'analyze_word': ModuleStats('TurkishMorphology.analyze()'),
        'analyze_sentence': ModuleStats('TurkishMorphology.analyze_sentence()'),
        'disambiguate': ModuleStats('TurkishMorphology.disambiguate()'),
        'spell_check': ModuleStats('TurkishSpellChecker.suggest()'),
        'normalizer': ModuleStats('TurkishSentenceNormalizer.normalize()'),
    }

    variant_stats: Dict[str, VariantStats] = {}

    test_sentences = sentences[:limit]
    total = len(test_sentences)

    print(f"\n[2/4] Benchmark basliyor ({total} cumle)...")
    benchmark_start = time.perf_counter()

    for i, item in enumerate(test_sentences):
        sentence = item['sentence']
        variant = item.get('variant', 'unknown')

        if variant not in variant_stats:
            variant_stats[variant] = VariantStats(variant)
        variant_stats[variant].count += 1

        words = sentence.split()
        first_word = words[0] if words else "test"

        # 1. Tokenizer
        elapsed, _, err = measure(lambda: tokenizer.tokenize(sentence))
        if i == 0:
            module_stats['tokenizer'].cold_start_ms = elapsed
        module_stats['tokenizer'].times_ms.append(elapsed)
        if err:
            module_stats['tokenizer'].errors += 1

        # 2. Sentence Extractor
        elapsed, _, err = measure(lambda: extractor.from_paragraph(sentence))
        if i == 0:
            module_stats['extractor'].cold_start_ms = elapsed
        module_stats['extractor'].times_ms.append(elapsed)
        if err:
            module_stats['extractor'].errors += 1

        # 3. Analyze word
        elapsed, _, err = measure(lambda: morph.analyze(first_word))
        if i == 0:
            module_stats['analyze_word'].cold_start_ms = elapsed
        module_stats['analyze_word'].times_ms.append(elapsed)
        if err:
            module_stats['analyze_word'].errors += 1

        # 4. Analyze sentence
        elapsed, _, err = measure(lambda: morph.analyze_sentence(sentence))
        if i == 0:
            module_stats['analyze_sentence'].cold_start_ms = elapsed
        module_stats['analyze_sentence'].times_ms.append(elapsed)
        if err:
            module_stats['analyze_sentence'].errors += 1

        # 5. Disambiguate
        elapsed, _, err = measure(lambda: morph.analyze_and_disambiguate(sentence))
        if i == 0:
            module_stats['disambiguate'].cold_start_ms = elapsed
        module_stats['disambiguate'].times_ms.append(elapsed)
        if err:
            module_stats['disambiguate'].errors += 1

        # 6. Spell checker (every 10th)
        if i % 10 == 0:
            elapsed, _, err = measure(lambda: spell_checker.suggest_for_word(first_word))
            if i == 0:
                module_stats['spell_check'].cold_start_ms = elapsed
            module_stats['spell_check'].times_ms.append(elapsed)
            if err:
                module_stats['spell_check'].errors += 1

        # 7. Normalizer
        norm_start = time.perf_counter()
        try:
            result = normalizer.normalize(sentence)
            elapsed = (time.perf_counter() - norm_start) * 1000
            success = result is not None and len(result) > 0
        except Exception:
            elapsed = (time.perf_counter() - norm_start) * 1000
            success = False
            module_stats['normalizer'].errors += 1

        if i == 0:
            module_stats['normalizer'].cold_start_ms = elapsed
        module_stats['normalizer'].times_ms.append(elapsed)

        variant_stats[variant].total_ms += elapsed
        if success:
            variant_stats[variant].successes += 1

        # Progress
        if (i + 1) % 100 == 0:
            pct = (i + 1) / total * 100
            elapsed_total = time.perf_counter() - benchmark_start
            eta = (elapsed_total / (i + 1)) * (total - i - 1)
            print(f"   Ilerleme: {i+1}/{total} ({pct:.0f}%) - ETA: {eta:.0f}s")

    benchmark_time = (time.perf_counter() - benchmark_start) * 1000
    mem_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    tracemalloc.stop()

    cache_stats = cache_manager.stats()

    # Results
    print(f"\n[3/4] Sonuclar hazirlaniyor...")

    # Table 1: Module Performance
    print("\n" + "=" * 80)
    print("TABLO 1: MODUL PERFORMANSI")
    print("=" * 80)
    print(f"{'Modul':<42} {'Cold':>8} {'Ort':>8} {'Min':>8} {'Max':>8} {'Tput':>8} {'N':>6}")
    print("-" * 80)

    bottlenecks = []
    for key, stats in module_stats.items():
        if stats.times_ms:
            print(f"{stats.name:<42} {stats.cold_start_ms:>6.1f}ms {stats.avg_ms:>6.2f}ms "
                  f"{stats.min_ms:>6.2f}ms {stats.max_ms:>6.1f}ms {stats.throughput:>6.0f}/s {stats.count:>6}")
            bottlenecks.append((stats.name, stats.avg_ms))

    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    print(f"\n   Bottleneck: {bottlenecks[0][0]} ({bottlenecks[0][1]:.2f}ms)")

    # Table 2: Variant Analysis
    print("\n" + "=" * 80)
    print("TABLO 2: VARIANT BAZLI (Normalizer)")
    print("=" * 80)
    print(f"{'Variant':<15} {'Adet':>8} {'Ort Sure':>12} {'Throughput':>12} {'Basari':>10}")
    print("-" * 80)

    for variant, stats in sorted(variant_stats.items()):
        print(f"{variant:<15} {stats.count:>8} {stats.avg_ms:>10.2f}ms {stats.throughput:>10.0f}/s "
              f"{stats.success_rate:>9.1f}%")

    # Table 3: Cache Status
    print("\n" + "=" * 80)
    print("TABLO 3: CACHE DURUMU")
    print("=" * 80)
    print(f"{'Cache':<20} {'Hit':>10} {'Miss':>10} {'Hit Rate':>12}")
    print("-" * 80)
    print(f"{'L0 (Precomputed)':<20} {cache_stats.get('l0_hits', 0):>10} {'-':>10} {'-':>12}")
    print(f"{'L1 (@lru_cache)':<20} {'N/A':>10} {'N/A':>10} {'Dahili':>12}")
    print(f"{'L2 (LocalCache)':<20} {cache_stats.get('l2_hits', 0):>10} {cache_stats.get('misses', 0):>10} "
          f"{cache_stats.get('hit_rate', '0%'):>12}")

    # Table 4: General
    print("\n" + "=" * 80)
    print("TABLO 4: GENEL METRIKLER")
    print("=" * 80)
    print(f"{'Metrik':<30} {'Deger':>20}")
    print("-" * 80)
    print(f"{'Toplam cumle':<30} {total:>20}")
    print(f"{'Yukleme suresi':<30} {load_time:>18.0f}ms")
    print(f"{'Benchmark suresi':<30} {benchmark_time/1000:>18.1f}s")
    print(f"{'Toplam sure':<30} {(load_time + benchmark_time)/1000:>18.1f}s")
    print(f"{'Bellek (baslangic)':<30} {mem_start:>18.1f}MB")
    print(f"{'Bellek (yukleme sonrasi)':<30} {mem_after_load:>18.1f}MB")
    print(f"{'Bellek (son)':<30} {mem_end:>18.1f}MB")
    print(f"{'Ortalama throughput':<30} {total / (benchmark_time / 1000):>17.0f}/s")

    # Save JSON
    print(f"\n[4/4] Sonuclar kaydediliyor...")

    results = {
        'meta': {
            'total_sentences': total,
            'load_time_ms': load_time,
            'benchmark_time_ms': benchmark_time,
            'memory_start_mb': mem_start,
            'memory_end_mb': mem_end,
            'beam_width': 10,
        },
        'modules': {k: {
            'name': v.name,
            'cold_start_ms': v.cold_start_ms,
            'avg_ms': v.avg_ms,
            'min_ms': v.min_ms,
            'max_ms': v.max_ms,
            'std_ms': v.std_ms,
            'throughput': v.throughput,
            'errors': v.errors,
            'sample_count': v.count,
        } for k, v in module_stats.items()},
        'variants': {k: {
            'count': v.count,
            'avg_ms': v.avg_ms,
            'throughput': v.throughput,
            'success_rate': v.success_rate,
        } for k, v in variant_stats.items()},
        'cache': cache_stats,
        'bottleneck': bottlenecks[0][0] if bottlenecks else None,
    }

    output_path = Path(__file__).parent / 'benchmark_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"   Kaydedildi: {output_path}")
    print("\n" + "=" * 80)
    print("BENCHMARK TAMAMLANDI")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Zemberek Full Benchmark')
    parser.add_argument('--limit', type=int, default=10000, help='Number of sentences (default: 10000)')
    parser.add_argument('--data', type=str, default=None, help='Path to JSONL data file')
    args = parser.parse_args()

    if args.data:
        data_path = args.data
    else:
        data_path = Path(__file__).parent / 'data' / 'tr_benchmark_10000.jsonl'

    if not Path(data_path).exists():
        print(f"HATA: Veri dosyasi bulunamadi: {data_path}")
        sys.exit(1)

    print(f"Veri dosyasi: {data_path}")
    sentences = load_sentences(str(data_path), args.limit)
    print(f"Yuklenen cumle: {len(sentences)}")

    run_benchmark(sentences, args.limit)


if __name__ == '__main__':
    main()
