#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for uem-zemberek v2
===================================================

Tests all major components with detailed metrics:
- Loading performance
- Module-by-module analysis
- Cache efficiency
- Variant-based analysis
- Cython vs NumPy comparison

Usage:
    python benchmark/full_benchmark_v2.py --limit 10000
    python benchmark/full_benchmark_v2.py --limit 5000 --output my_results.json
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np

# Try to get memory info
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LatencyStats:
    """Latency statistics for a test."""
    count: int = 0
    total_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput: float = 0.0  # items/sec

    @classmethod
    def from_times(cls, times_ms: List[float], unit_name: str = "items") -> 'LatencyStats':
        if not times_ms:
            return cls()

        count = len(times_ms)
        total = sum(times_ms)
        mean = statistics.mean(times_ms)
        min_val = min(times_ms)
        max_val = max(times_ms)
        std = statistics.stdev(times_ms) if count > 1 else 0.0

        sorted_times = sorted(times_ms)
        p50 = sorted_times[int(count * 0.50)] if count > 0 else 0
        p95 = sorted_times[int(count * 0.95)] if count > 0 else 0
        p99 = sorted_times[int(count * 0.99)] if count > 0 else 0

        throughput = (count / (total / 1000)) if total > 0 else 0

        return cls(
            count=count,
            total_ms=total,
            mean_ms=mean,
            min_ms=min_val,
            max_ms=max_val,
            std_ms=std,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            throughput=throughput
        )


@dataclass
class CacheStats:
    """Cache statistics."""
    name: str
    hits: int = 0
    misses: int = 0
    total: int = 0
    hit_rate: float = 0.0
    size: int = 0


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    # General info
    timestamp: str = ""
    python_version: str = ""
    platform_info: str = ""
    cpu_info: str = ""
    ram_gb: float = 0.0
    cython_enabled: bool = False

    # Loading metrics
    morphology_load_ms: float = 0.0
    normalizer_load_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_load_mb: float = 0.0
    memory_final_mb: float = 0.0

    # Module performance
    module_stats: Dict[str, Dict] = field(default_factory=dict)

    # Cache analysis
    cache_stats: Dict[str, Dict] = field(default_factory=dict)

    # Variant analysis
    variant_stats: Dict[str, Dict] = field(default_factory=dict)

    # Cython comparison
    cython_comparison: Dict[str, Dict] = field(default_factory=dict)

    # Totals
    total_sentences: int = 0
    total_time_sec: float = 0.0
    overall_throughput: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def get_cpu_info() -> str:
    """Get CPU info string."""
    if HAS_PSUTIL:
        try:
            return f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"
        except:
            pass
    return platform.processor() or "Unknown"


def get_ram_gb() -> float:
    """Get total RAM in GB."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().total / (1024**3)
    return 0.0


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def print_table(headers: List[str], rows: List[List], col_widths: Optional[List[int]] = None):
    """Print a formatted table."""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]

    # Header
    header_str = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(f"|{header_str}|")
    print("|" + "|".join("-" * w for w in col_widths) + "|")

    # Rows
    for row in rows:
        row_str = "|".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(f"|{row_str}|")


def time_function(func: Callable, *args, **kwargs) -> tuple:
    """Time a function call and return (result, time_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


# =============================================================================
# SENTENCE GENERATORS
# =============================================================================

def generate_sentences(limit: int) -> Dict[str, List[str]]:
    """Generate test sentences by variant type."""

    # Base clean sentences
    clean_sentences = [
        "Bugün hava çok güzel.",
        "Yarın toplantıya gideceğim.",
        "Türkçe doğal dil işleme çalışıyorum.",
        "Bu cümle belirsizlik içeriyor.",
        "Evden işe arabayla gittim.",
        "Kitabı okudum ve çok beğendim.",
        "Kahvaltıda çay içtim.",
        "İstanbul'da yaşıyorum.",
        "Anneme telefon ettim.",
        "Yarın sabah erkenden kalkacağım.",
        "Öğrenciler sınava çalışıyor.",
        "Akşam yemeği hazırladım.",
        "Hafta sonu pikniğe gideceğiz.",
        "Telefon şarjı bitmek üzere.",
        "Kedim çok tembel bir hayvan.",
        "Türkiye'nin başkenti Ankara'dır.",
        "Yeni bir araba almayı düşünüyorum.",
        "Bu restoranın yemekleri lezzetli.",
        "Yazılım geliştirme zor bir iş.",
        "Matematik dersine geç kaldım.",
        "Arkadaşımla sinemaya gittik.",
        "Tatil için Antalya'ya gideceğiz.",
        "Bahçedeki çiçekler açmış.",
        "Kütüphanede ders çalıştım.",
        "Müzik dinlemeyi seviyorum.",
    ]

    # Devrik (inverted) sentences
    devrik_sentences = [
        "Gittim bugün okula.",
        "Seviyorum seni çok.",
        "Yapacağım bunu mutlaka.",
        "Geldi sonunda bahar.",
        "Bitirdi işini erken.",
        "Okudum kitabı dün gece.",
        "Aldım hediyeyi anneme.",
        "Başladı yağmur aniden.",
        "Gördüm onu parkta.",
        "Yaptım kahvaltıyı erken.",
    ]

    # Slang/informal sentences
    slang_sentences = [
        "Naber kanka nasılsın?",
        "Çok fena bi film izledim.",
        "Adam harbiden çılgın.",
        "Bi dakka bekle geliyom.",
        "Efsane bi gün geçirdik.",
        "Aynen öyle ya kesinlikle.",
        "Nası ya inanamıyorum.",
        "Acayip güzel olmuş bu.",
        "Kafayı yicem artık.",
        "Çok sağlam iş çıkardın.",
    ]

    # Vowel drop sentences (konuşma dili)
    vowel_drop_sentences = [
        "Napıyosun bu aralar?",
        "Nereye gidiyon acaba?",
        "Anlıyo musun beni?",
        "Geliyo musun bizimle?",
        "Bakıyom şimdi sana.",
        "Yapıyoz elimizden geleni.",
        "Gidiyo muyuz artık?",
        "Bekliyo musun bizi?",
    ]

    # Typo sentences
    typo_sentences = [
        "Bugn hava çok gzel.",
        "Yarin toplantiya gidecegim.",
        "Kitabi okdum ve begendim.",
        "Istanbulda yasiyorum.",
        "Telefn şarji bitmek üzre.",
    ]

    # Mixed complexity sentences
    mixed_sentences = [
        "Dün akşam yediğimiz yemek gerçekten çok lezzetliydi ve herkes beğendi.",
        "Önümüzdeki hafta yapılacak olan toplantıya katılmam gerekiyor.",
        "Bu konuda farklı görüşler olmasına rağmen sonunda anlaştık.",
        "Teknolojinin hızlı gelişmesi hayatımızı kolaylaştırıyor.",
        "Sanatın toplum üzerindeki etkisi tartışılmaz bir gerçektir.",
    ]

    # Calculate counts based on limit
    total_ratio = 10000  # Base ratio

    variants = {
        "clean": (clean_sentences, int(limit * 5000 / total_ratio)),
        "devrik": (devrik_sentences, int(limit * 1500 / total_ratio)),
        "slang": (slang_sentences, int(limit * 1500 / total_ratio)),
        "vowel_drop": (vowel_drop_sentences, int(limit * 1000 / total_ratio)),
        "typo": (typo_sentences, int(limit * 500 / total_ratio)),
        "mixed": (mixed_sentences, int(limit * 500 / total_ratio)),
    }

    result = {}
    for variant_name, (sentences, count) in variants.items():
        # Extend sentences to reach count
        multiplier = (count // len(sentences)) + 1
        extended = (sentences * multiplier)[:count]
        result[variant_name] = extended

    return result


# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class FullBenchmark:
    """Main benchmark runner."""

    def __init__(self, limit: int = 10000, output_path: str = "benchmark_results_v2.json"):
        self.limit = limit
        self.output_path = output_path
        self.results = BenchmarkResults()

        # Will be initialized during run
        self.morphology = None
        self.normalizer = None
        self.spell_checker = None
        self.tokenizer = None
        self.sentence_extractor = None

    def run(self):
        """Run the complete benchmark."""
        print("\n" + "=" * 60)
        print("  UEM-ZEMBEREK COMPREHENSIVE BENCHMARK v2")
        print("=" * 60)

        try:
            self._collect_system_info()
            self._benchmark_loading()
            self._benchmark_modules()
            self._collect_cache_stats()
            self._benchmark_variants()
            self._benchmark_cython()
            self._calculate_totals()
            self._print_summary()
            self._save_results()
        except Exception as e:
            print(f"\nBenchmark failed with error: {e}")
            traceback.print_exc()
            raise

    def _collect_system_info(self):
        """Collect system information."""
        print_header("SYSTEM INFORMATION")

        self.results.timestamp = datetime.now().isoformat()
        self.results.python_version = sys.version.split()[0]
        self.results.platform_info = f"{platform.system()} {platform.release()}"
        self.results.cpu_info = get_cpu_info()
        self.results.ram_gb = round(get_ram_gb(), 1)
        self.results.memory_before_mb = round(get_memory_mb(), 1)

        # Check Cython status
        try:
            from zemberek.cython.hash_functions import hash_for_str_cy
            self.results.cython_enabled = True
        except ImportError:
            self.results.cython_enabled = False

        print(f"  Timestamp:      {self.results.timestamp}")
        print(f"  Python:         {self.results.python_version}")
        print(f"  Platform:       {self.results.platform_info}")
        print(f"  CPU:            {self.results.cpu_info}")
        print(f"  RAM:            {self.results.ram_gb} GB")
        print(f"  Memory (start): {self.results.memory_before_mb} MB")
        print(f"  Cython:         {'Enabled' if self.results.cython_enabled else 'Disabled'}")

    def _benchmark_loading(self):
        """Benchmark module loading times."""
        print_header("LOADING METRICS")

        # Load TurkishMorphology
        print("  Loading TurkishMorphology...", end=" ", flush=True)
        start = time.perf_counter()
        from zemberek.morphology import TurkishMorphology
        self.morphology = TurkishMorphology.create_with_defaults()
        self.results.morphology_load_ms = (time.perf_counter() - start) * 1000
        print(f"{self.results.morphology_load_ms:.0f}ms")

        # Load Normalizer
        print("  Loading Normalizer...", end=" ", flush=True)
        start = time.perf_counter()
        from zemberek.normalization import TurkishSentenceNormalizer
        self.normalizer = TurkishSentenceNormalizer(self.morphology)
        self.results.normalizer_load_ms = (time.perf_counter() - start) * 1000
        print(f"{self.results.normalizer_load_ms:.0f}ms")

        # Load other components
        print("  Loading other components...", end=" ", flush=True)
        from zemberek.tokenization import TurkishTokenizer
        from zemberek.tokenization import TurkishSentenceExtractor
        self.tokenizer = TurkishTokenizer.DEFAULT
        self.sentence_extractor = TurkishSentenceExtractor()

        try:
            from zemberek.normalization import TurkishSpellChecker
            self.spell_checker = TurkishSpellChecker(self.morphology)
        except Exception as e:
            print(f"(SpellChecker unavailable: {e})")
            self.spell_checker = None

        print("done")

        self.results.memory_after_load_mb = round(get_memory_mb(), 1)
        print(f"\n  Memory after load: {self.results.memory_after_load_mb} MB")
        print(f"  Memory increase:   {self.results.memory_after_load_mb - self.results.memory_before_mb:.1f} MB")

    def _benchmark_modules(self):
        """Benchmark individual modules."""
        print_header("MODULE PERFORMANCE")

        # Generate test sentences (use clean for module tests)
        test_sentences = generate_sentences(self.limit)["clean"]
        test_words = []
        for s in test_sentences[:100]:
            test_words.extend(s.replace(".", "").replace(",", "").split())
        test_words = test_words[:500]  # Limit words

        # 1. Tokenizer
        print("\n  [1/6] Tokenizer...")
        times = []
        for sentence in test_sentences[:1000]:
            _, t = time_function(self.tokenizer.tokenize, sentence)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["tokenizer"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

        # 2. SentenceExtractor
        print("  [2/6] SentenceExtractor...")
        long_text = " ".join(test_sentences[:100])
        times = []
        for _ in range(100):
            _, t = time_function(self.sentence_extractor.from_paragraph, long_text)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["sentence_extractor"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

        # 3. analyze() - word level
        print("  [3/6] analyze() (word-level)...")
        times = []
        for word in test_words:
            _, t = time_function(self.morphology.analyze, word)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["analyze_word"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

        # 4. analyze_sentence()
        print("  [4/6] analyze_sentence()...")
        times = []
        for sentence in test_sentences:
            _, t = time_function(self.morphology.analyze_sentence, sentence)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["analyze_sentence"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

        # 5. disambiguate()
        print("  [5/6] disambiguate()...")
        times = []
        for sentence in test_sentences:
            analysis = self.morphology.analyze_sentence(sentence)
            _, t = time_function(self.morphology.disambiguate, sentence, analysis)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["disambiguate"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

        # 6. Normalizer
        print("  [6/6] Normalizer...")
        times = []
        for sentence in test_sentences[:500]:
            _, t = time_function(self.normalizer.normalize, sentence)
            times.append(t)
        stats = LatencyStats.from_times(times)
        self.results.module_stats["normalizer"] = asdict(stats)
        print(f"        Mean: {stats.mean_ms:.3f}ms | p99: {stats.p99_ms:.3f}ms | {stats.throughput:.0f}/s")

    def _collect_cache_stats(self):
        """Collect cache statistics."""
        print_header("CACHE ANALYSIS")

        # Get CacheManager stats (L0 + L2)
        from zemberek.cache import CacheManager
        manager = CacheManager.get_instance()
        cm_stats = manager.stats()

        # Store raw stats for JSON
        self.results.cache_stats["cache_manager"] = cm_stats

        # L1 LRU cache stats
        from zemberek.core.hash.multi_level_mphf import MultiLevelMphf
        from zemberek.core.compression.lossy_int_lookup import LossyIntLookup

        info1 = MultiLevelMphf._hash_for_str_compute.cache_info()
        info2 = LossyIntLookup._java_hash_code_compute.cache_info()

        l1_hash_for_str = CacheStats(
            name="L1_hash_for_str",
            hits=info1.hits,
            misses=info1.misses,
            total=info1.hits + info1.misses,
            hit_rate=info1.hits / (info1.hits + info1.misses) if (info1.hits + info1.misses) > 0 else 0,
            size=info1.currsize
        )

        l1_java_hash = CacheStats(
            name="L1_java_hash_code",
            hits=info2.hits,
            misses=info2.misses,
            total=info2.hits + info2.misses,
            hit_rate=info2.hits / (info2.hits + info2.misses) if (info2.hits + info2.misses) > 0 else 0,
            size=info2.currsize
        )

        self.results.cache_stats["L1_hash_for_str"] = asdict(l1_hash_for_str)
        self.results.cache_stats["L1_java_hash_code"] = asdict(l1_java_hash)

        # Print cache stats
        print("\n  === CacheManager Stats (L0 + L2) ===")
        print(f"  Total Calls:    {cm_stats['total_calls']:,}")
        print(f"  L0 Hits:        {cm_stats['l0_hits']:,} ({cm_stats['l0_hit_rate']})")
        print(f"  L2 Hits:        {cm_stats['l2_hits']:,} ({cm_stats['l2_hit_rate']})")
        print(f"  Misses:         {cm_stats['misses']:,}")
        print(f"  Computes:       {cm_stats['computes']:,}")
        print(f"  Overall Hit:    {cm_stats['hit_rate']}")
        print(f"  L0 Size:        {cm_stats['l0_size']:,} entries")
        print(f"  L2 Size:        {cm_stats['l2_size']:,} entries")

        print("\n  === L1 LRU Cache Stats ===")
        print(f"  hash_for_str:   hits={info1.hits:,}, misses={info1.misses:,}, size={info1.currsize:,}")
        print(f"  java_hash_code: hits={info2.hits:,}, misses={info2.misses:,}, size={info2.currsize:,}")

    def _benchmark_variants(self):
        """Benchmark different sentence variants."""
        print_header("VARIANT-BASED ANALYSIS")

        variants = generate_sentences(self.limit)

        print(f"\n  {'Variant':<12} | {'Count':>6} | {'Mean ms':>8} | {'p99 ms':>8} | {'Throughput':>10}")
        print("  " + "-" * 56)

        for variant_name, sentences in variants.items():
            if not sentences:
                continue

            times = []
            for sentence in sentences:
                analysis = self.morphology.analyze_sentence(sentence)
                _, t = time_function(self.morphology.disambiguate, sentence, analysis)
                times.append(t)

            stats = LatencyStats.from_times(times)
            self.results.variant_stats[variant_name] = asdict(stats)

            print(f"  {variant_name:<12} | {stats.count:>6} | {stats.mean_ms:>8.3f} | {stats.p99_ms:>8.3f} | {stats.throughput:>10.1f}/s")

    def _benchmark_cython(self):
        """Compare Cython vs NumPy hash implementations."""
        print_header("CYTHON VS NUMPY COMPARISON")

        if not self.results.cython_enabled:
            print("\n  Cython not available, skipping comparison.")
            return

        from zemberek.cython.hash_functions import hash_for_str_cy, java_hash_code_cy

        # NumPy implementations
        HASH_MULTIPLIER = np.int32(16777619)
        INITIAL_HASH_SEED = np.int32(-2128831035)

        def hash_for_str_numpy(data: str, seed: int) -> np.int32:
            d = np.int32(seed) if seed > 0 else INITIAL_HASH_SEED
            for c in data:
                d = (d ^ np.int32(ord(c))) * HASH_MULTIPLIER
            return d & np.int32(0x7fffffff)

        def java_hash_code_numpy(s: str) -> np.int32:
            arr = np.asarray([ord(c) for c in s], dtype=np.int32)
            powers = np.arange(arr.shape[0], dtype=np.int32)[::-1]
            bases = np.full((arr.shape[0],), 31, dtype=np.int32)
            return np.sum(arr * (np.power(bases, powers)), dtype=np.int32)

        test_words = ["merhaba", "dünya", "test", "zemberek", "morfoloji",
                      "disambiguate", "türkçe", "kelime", "cümle", "analiz"] * 100
        iterations = 10

        # hash_for_str
        start = time.perf_counter()
        for _ in range(iterations):
            for word in test_words:
                hash_for_str_numpy(word, -1)
        numpy_time_str = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        for _ in range(iterations):
            for word in test_words:
                hash_for_str_cy(word, -1)
        cython_time_str = (time.perf_counter() - start) * 1000

        speedup_str = numpy_time_str / cython_time_str if cython_time_str > 0 else 0

        # java_hash_code
        start = time.perf_counter()
        for _ in range(iterations):
            for word in test_words:
                java_hash_code_numpy(word)
        numpy_time_java = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        for _ in range(iterations):
            for word in test_words:
                java_hash_code_cy(word)
        cython_time_java = (time.perf_counter() - start) * 1000

        speedup_java = numpy_time_java / cython_time_java if cython_time_java > 0 else 0

        self.results.cython_comparison = {
            "hash_for_str": {
                "numpy_ms": numpy_time_str,
                "cython_ms": cython_time_str,
                "speedup": speedup_str
            },
            "java_hash_code": {
                "numpy_ms": numpy_time_java,
                "cython_ms": cython_time_java,
                "speedup": speedup_java
            }
        }

        print(f"\n  {'Function':<18} | {'NumPy ms':>10} | {'Cython ms':>10} | {'Speedup':>8}")
        print("  " + "-" * 54)
        print(f"  {'hash_for_str':<18} | {numpy_time_str:>10.1f} | {cython_time_str:>10.1f} | {speedup_str:>7.1f}x")
        print(f"  {'java_hash_code':<18} | {numpy_time_java:>10.1f} | {cython_time_java:>10.1f} | {speedup_java:>7.1f}x")

    def _calculate_totals(self):
        """Calculate total benchmark statistics."""
        self.results.memory_final_mb = round(get_memory_mb(), 1)

        # Sum all variant sentences
        total_sentences = sum(
            stats.get('count', 0)
            for stats in self.results.variant_stats.values()
        )
        self.results.total_sentences = total_sentences

        # Sum all variant times
        total_time_ms = sum(
            stats.get('total_ms', 0)
            for stats in self.results.variant_stats.values()
        )
        self.results.total_time_sec = total_time_ms / 1000

        # Calculate overall throughput
        if self.results.total_time_sec > 0:
            self.results.overall_throughput = total_sentences / self.results.total_time_sec

    def _print_summary(self):
        """Print final summary."""
        print_header("BENCHMARK SUMMARY", "=")

        print(f"""
  Test Configuration:
    Limit:              {self.limit} sentences
    Cython:             {'Enabled' if self.results.cython_enabled else 'Disabled'}

  Loading Performance:
    TurkishMorphology:  {self.results.morphology_load_ms:.0f}ms
    Normalizer:         {self.results.normalizer_load_ms:.0f}ms

  Memory Usage:
    Before:             {self.results.memory_before_mb:.1f} MB
    After load:         {self.results.memory_after_load_mb:.1f} MB
    Final:              {self.results.memory_final_mb:.1f} MB

  Overall Performance:
    Total sentences:    {self.results.total_sentences}
    Total time:         {self.results.total_time_sec:.2f}s
    Throughput:         {self.results.overall_throughput:.1f} sentences/sec
    Per sentence:       {(self.results.total_time_sec / self.results.total_sentences * 1000) if self.results.total_sentences > 0 else 0:.2f}ms
""")

        # Comparison table (if previous results exist)
        print("  Performance Metrics by Module:")
        print("  " + "-" * 56)
        print(f"  {'Module':<20} | {'Mean ms':>10} | {'p99 ms':>10} | {'Throughput':>10}")
        print("  " + "-" * 56)
        for module, stats in self.results.module_stats.items():
            mean = stats.get('mean_ms', 0)
            p99 = stats.get('p99_ms', 0)
            throughput = stats.get('throughput', 0)
            print(f"  {module:<20} | {mean:>10.3f} | {p99:>10.3f} | {throughput:>10.1f}/s")

    def _save_results(self):
        """Save results to JSON file."""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.results), f, indent=2, ensure_ascii=False)

        print(f"\n  Results saved to: {output_path.absolute()}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for uem-zemberek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10000,
        help="Number of sentences to test"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark/benchmark_results_v2.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    benchmark = FullBenchmark(limit=args.limit, output_path=args.output)
    benchmark.run()

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
