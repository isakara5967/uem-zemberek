#!/usr/bin/env python3
"""
Zemberek-Python Performance Benchmark Suite

This script measures the performance of various zemberek-python components:
- Morphological Analysis
- Sentence Normalization
- Spell Checking
- Tokenization
- Memory Usage

Usage:
    python benchmarks/benchmark.py
"""

import time
import sys
import tracemalloc
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass

# Sample data for benchmarks
SAMPLE_WORDS = [
    "kitap", "kalem", "masa", "sandalye", "bilgisayar", "telefon", "araba", "ev", "okul", "hastane",
    "doktor", "mühendis", "öğretmen", "avukat", "polis", "asker", "pilot", "şoför", "aşçı", "garson",
    "güzel", "çirkin", "büyük", "küçük", "uzun", "kısa", "geniş", "dar", "yüksek", "alçak",
    "koşmak", "yürümek", "oturmak", "kalkmak", "yemek", "içmek", "uyumak", "uyanmak", "gülmek", "ağlamak",
    "hızlı", "yavaş", "sessiz", "gürültülü", "temiz", "kirli", "sıcak", "soğuk", "tatlı", "acı",
    "evler", "arabalar", "kitaplar", "kalemler", "masalar", "sandalyeler", "bilgisayarlar", "telefonlar",
    "okullar", "hastaneler", "doktorlar", "mühendisler", "öğretmenler", "avukatlar", "polisler",
    "gidiyorum", "geliyorsun", "yapıyor", "ediyoruz", "biliyorsunuz", "seviyorlar", "istiyorum",
    "aldım", "verdim", "gördüm", "duydum", "anladım", "öğrendim", "unuttum", "hatırladım",
    "gideceğim", "geleceğiz", "yapacaksın", "edecekler", "olacak", "bitecek", "başlayacak",
    "güzellik", "çirkinlik", "büyüklük", "küçüklük", "uzunluk", "kısalık", "genişlik", "darlık",
    "Türkiye", "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep",
    "cumhuriyet", "demokrasi", "özgürlük", "eşitlik", "adalet", "barış", "sevgi", "saygı", "hoşgörü",
]

SAMPLE_SENTENCES_INFORMAL = [
    "slm nslsn",
    "naber nasılsın",
    "bugn hava cok guzel",
    "yarin okula gidcem",
    "bu kitabi okumalisin",
    "cok yorgunum bugun",
    "aksam yemegi ne yiyek",
    "hafta sonu ne yapiyosun",
    "film izlemek istiyom",
    "kahve icmek ister misin",
    "bu gün çok sıcak",
    "yarın hava nasıl olcak",
    "biraz geç kalcam",
    "seni arıycam sonra",
    "bu işi bitirmem lazm",
    "çok acıktım yemek yiyelm",
    "eve gidiyom ben",
    "pazara gidecez mi",
    "bu fiyat çok pahali",
    "indirim var mı acba",
]

SAMPLE_SENTENCES_TOKENIZE = [
    "Merhaba, bugün hava çok güzel!",
    "Türkiye'nin başkenti Ankara'dır.",
    "Dr. Ahmet Bey yarın gelecek.",
    "Bu kitabın fiyatı 50 TL'dir.",
    "İstanbul, Türkiye'nin en büyük şehridir.",
    "Saat 15:30'da toplantımız var.",
    "www.example.com adresini ziyaret edin.",
    "E-posta adresim: ornek@email.com",
    "Telefon numaram: 0532-123-4567",
    "Bugün 25.12.2024 tarihinde buluşalım.",
    "Atatürk, Türkiye Cumhuriyeti'nin kurucusudur.",
    "Prof. Dr. Mehmet Öz konferans verecek.",
    "Bu ürünün %20 indirimi var.",
    "Toplam tutar: 1.250,50 TL",
    "A.B.D. ve İngiltere'den haberler.",
    "Sn. Ali Veli, mektubunuz geldi.",
    "T.C. Kimlik No: 12345678901",
    "Ankara-İstanbul arası 450 km.",
    "Hız limiti 120 km/saat.",
    "Bugünkü döviz kuru: 1$ = 32,50 TL",
]

SAMPLE_WORDS_MISSPELLED = [
    "kitab", "kaelm", "msaa", "sandlaye", "bilgsayar", "telfon", "arba", "okl", "hastaen", "doktr",
    "mühends", "öğretmn", "avkat", "plis", "askr", "pilt", "şofr", "aşcı", "garsn", "güzl",
    "çrikn", "büyk", "küçk", "uzn", "ksa", "gnis", "dar", "yüksk", "alck", "koşmk",
    "yürümk", "otrmak", "kalkmk", "yemk", "içmk", "uymak", "uyanmk", "gülmk", "ağlamk", "hızl",
    "yavaşş", "seesiz", "gürültlü", "tenmiz", "kirlli", "sıcakk", "soğukk", "tatlii", "acıı",
    "evlr", "arablr", "kitplr", "kalemlr", "masalr", "sandlyeler", "bilgisyarlar", "teleflnlar",
    "okullr", "hastanlr", "doktorlr", "mühendslr", "öğretmnler", "avukatlr", "polislr",
    "gidyorum", "gelyorsun", "yapyor", "edyoruz", "bilyorsunuz", "sevyorlar", "istyorum",
    "aldm", "verdm", "gördm", "duydm", "anladm", "öğrendm", "unuttm", "hatırladm",
    "gidecğim", "gelecğiz", "yapacksın", "edecklr", "olack", "biteck", "başlayack",
    "güzllik", "çirknlik", "büyüklk", "küçüklk", "uzunlk", "kısalk", "genişlk", "darlk",
    "Türkye", "İstanbl", "Ankra", "İzmr", "Brsa", "Antalay", "Adna", "Koyna", "Gaziantb",
    "cumhuryet", "demokrsi", "özgürlk", "eşitlk", "adalt", "barş", "sevg", "sayg", "hoşgör",
]


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single test."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    items_per_second: float
    memory_peak_mb: float


def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def measure_memory(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024  # Convert to MB


def run_benchmark(name: str, func: Callable, items: List, iterations: int = 1) -> BenchmarkResult:
    """Run a benchmark and return results."""
    print(f"  Running {name}...", end=" ", flush=True)

    # Warm up
    for item in items[:min(10, len(items))]:
        func(item)

    # Measure memory
    _, memory_peak = measure_memory(lambda: [func(item) for item in items])

    # Measure time
    total_time = 0
    for _ in range(iterations):
        for item in items:
            _, elapsed = measure_time(func, item)
            total_time += elapsed

    total_items = len(items) * iterations
    avg_time = total_time / total_items
    items_per_second = total_items / total_time if total_time > 0 else 0

    print(f"Done ({total_time:.2f}s)")

    return BenchmarkResult(
        name=name,
        iterations=total_items,
        total_time=total_time,
        avg_time=avg_time,
        items_per_second=items_per_second,
        memory_peak_mb=memory_peak
    )


def print_results(results: List[BenchmarkResult], title: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")
    print(f"{'Component':<25} {'Items':<10} {'Total (s)':<12} {'Avg (ms)':<12} {'Items/sec':<12} {'Memory (MB)':<12}")
    print(f"{'-' * 80}")

    for r in results:
        print(f"{r.name:<25} {r.iterations:<10} {r.total_time:<12.3f} {r.avg_time*1000:<12.3f} {r.items_per_second:<12.1f} {r.memory_peak_mb:<12.2f}")

    print(f"{'=' * 80}\n")


def get_system_info() -> str:
    """Get system information."""
    import platform

    info = []
    info.append(f"Python Version: {platform.python_version()}")
    info.append(f"Platform: {platform.system()} {platform.release()}")
    info.append(f"Architecture: {platform.machine()}")

    return "\n".join(info)


def main():
    print("\n" + "=" * 80)
    print(" Zemberek-Python Performance Benchmark")
    print("=" * 80)
    print(f"\n{get_system_info()}\n")

    results = []

    # Initialize components
    print("\nInitializing components...")

    print("  Loading TurkishMorphology...", end=" ", flush=True)
    init_start = time.perf_counter()
    tracemalloc.start()
    from zemberek import TurkishMorphology
    morphology = TurkishMorphology.create_with_defaults()
    _, morphology_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    init_morphology_time = time.perf_counter() - init_start
    print(f"Done ({init_morphology_time:.2f}s, {morphology_memory/1024/1024:.1f} MB)")

    print("  Loading TurkishSentenceNormalizer...", end=" ", flush=True)
    init_start = time.perf_counter()
    tracemalloc.start()
    from zemberek import TurkishSentenceNormalizer
    normalizer = TurkishSentenceNormalizer(morphology)
    _, normalizer_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    init_normalizer_time = time.perf_counter() - init_start
    print(f"Done ({init_normalizer_time:.2f}s, {normalizer_memory/1024/1024:.1f} MB)")

    print("  Loading TurkishSpellChecker...", end=" ", flush=True)
    init_start = time.perf_counter()
    tracemalloc.start()
    from zemberek import TurkishSpellChecker
    spell_checker = TurkishSpellChecker(morphology)
    _, spellchecker_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    init_spellchecker_time = time.perf_counter() - init_start
    print(f"Done ({init_spellchecker_time:.2f}s, {spellchecker_memory/1024/1024:.1f} MB)")

    print("  Loading TurkishTokenizer...", end=" ", flush=True)
    init_start = time.perf_counter()
    from zemberek import TurkishTokenizer
    tokenizer = TurkishTokenizer.DEFAULT
    init_tokenizer_time = time.perf_counter() - init_start
    print(f"Done ({init_tokenizer_time:.2f}s)")

    # Print initialization summary
    print(f"\n{'=' * 80}")
    print(" Initialization Summary")
    print(f"{'=' * 80}")
    print(f"{'Component':<30} {'Time (s)':<15} {'Memory (MB)':<15}")
    print(f"{'-' * 80}")
    print(f"{'TurkishMorphology':<30} {init_morphology_time:<15.2f} {morphology_memory/1024/1024:<15.1f}")
    print(f"{'TurkishSentenceNormalizer':<30} {init_normalizer_time:<15.2f} {normalizer_memory/1024/1024:<15.1f}")
    print(f"{'TurkishSpellChecker':<30} {init_spellchecker_time:<15.2f} {spellchecker_memory/1024/1024:<15.1f}")
    print(f"{'TurkishTokenizer':<30} {init_tokenizer_time:<15.2f} {'N/A':<15}")
    print(f"{'=' * 80}\n")

    # Run benchmarks
    print("\nRunning benchmarks...")

    # Expand word list to 1000
    words_1000 = (SAMPLE_WORDS * 10)[:1000]

    # Morphology Analysis (1000 words)
    results.append(run_benchmark(
        "Morphology Analysis",
        lambda w: morphology.analyze(w),
        words_1000,
        iterations=1
    ))

    # Sentence Normalization (100 sentences)
    sentences_100 = (SAMPLE_SENTENCES_INFORMAL * 5)[:100]
    results.append(run_benchmark(
        "Sentence Normalization",
        lambda s: normalizer.normalize(s),
        sentences_100,
        iterations=1
    ))

    # Spell Checking (100 words)
    misspelled_100 = (SAMPLE_WORDS_MISSPELLED)[:100]
    results.append(run_benchmark(
        "Spell Checker",
        lambda w: spell_checker.suggest_for_word(w),
        misspelled_100,
        iterations=1
    ))

    # Tokenization (100 sentences)
    sentences_tokenize_100 = (SAMPLE_SENTENCES_TOKENIZE * 5)[:100]
    results.append(run_benchmark(
        "Tokenization",
        lambda s: tokenizer.tokenize(s),
        sentences_tokenize_100,
        iterations=1
    ))

    # Print results
    print_results(results, "Benchmark Results")

    # Print summary for README
    print("\n" + "=" * 80)
    print(" Markdown Table for README.md")
    print("=" * 80)
    print("\n```markdown")
    print("| Component | Items | Total Time | Avg Time | Throughput | Memory |")
    print("|-----------|-------|------------|----------|------------|--------|")
    for r in results:
        print(f"| {r.name} | {r.iterations} | {r.total_time:.2f}s | {r.avg_time*1000:.2f}ms | {r.items_per_second:.0f}/sec | {r.memory_peak_mb:.1f} MB |")
    print("```\n")

    print("Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
