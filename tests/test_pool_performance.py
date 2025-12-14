"""Pool performance test: Sequential vs Parallel."""

import time
from zemberek.cache import PrecomputedCache
from zemberek.pool import WorkerPool

# Test kelimeleri: L0'da olanlar + olmayanlar karışık
WORDS_IN_L0 = ["ev", "gel", "git", "al", "ver", "yap", "et", "ol", "de", "gör"] * 25  # 250
WORDS_NOT_L0 = [f"testkelime{i}" for i in range(250)]  # 250
TEST_WORDS = WORDS_IN_L0 + WORDS_NOT_L0  # 500 kelime

RUNS = 3


def test_sequential():
    """Sequential processing with PrecomputedCache."""
    cache = PrecomputedCache()
    results = {}
    for word in TEST_WORDS:
        h = cache.get_str_hash(word)
        if h is not None:
            results[word] = h
    return results


def test_pool(workers=2):
    """Parallel processing with WorkerPool."""
    with WorkerPool(workers=workers) as pool:
        return pool.process_words(TEST_WORDS)


def benchmark(name, func, runs=RUNS):
    """Run function multiple times, return average time."""
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_time = sum(times) / len(times)
    return avg_time, result


if __name__ == "__main__":
    print(f"Test: {len(TEST_WORDS)} kelime, {RUNS} çalıştırma\n")

    # Sequential test
    seq_time, seq_result = benchmark("Sequential", test_sequential)
    seq_hits = len(seq_result)

    # Pool test
    pool_time, pool_result = benchmark("Pool(2w)", lambda: test_pool(2))
    pool_hits = len(pool_result)

    # Hesaplamalar
    seq_ms = seq_time * 1000
    pool_ms = pool_time * 1000
    speedup = seq_time / pool_time if pool_time > 0 else 0
    seq_throughput = len(TEST_WORDS) / seq_time
    pool_throughput = len(TEST_WORDS) / pool_time

    # Sonuç tablosu
    print("| Metrik      | Sequential | Pool (2w)  | Speedup |")
    print("|-------------|------------|------------|---------|")
    print(f"| Süre        | {seq_ms:.1f}ms     | {pool_ms:.1f}ms    | {speedup:.2f}x    |")
    print(f"| Throughput  | {seq_throughput:.0f}/s      | {pool_throughput:.0f}/s     | -       |")
    print(f"| Cache hits  | {seq_hits}        | {pool_hits}        | -       |")
