# zemberek-python

**High-Performance Turkish NLP Library for Python 3.12+**

A maintained, optimized fork of [Loodos/zemberek-python](https://github.com/Loodos/zemberek-python) — the Python port of [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp).

---

## Highlights

- **Python 3.12+ Compatible** — All overflow and deprecation issues fixed
- **85-190x Faster Hash Functions** — Cython-optimized core operations
- **3-Layer Cache System** — 99.9% cache hit rate for repeated analyses
- **Memory Optimized** — Replaced `deepcopy` with efficient `set.copy()` operations
- **Full Turkish Morphology** — Vowel harmony, consonant mutations, all suffix rules

---

## Performance

Benchmarked on AMD Ryzen 7 5700X (8 cores), Ubuntu WSL2, Python 3.12.3

### Throughput

| Operation | With Cache | Without Cache |
|-----------|------------|---------------|
| Word Analysis | **5,767 words/sec** | 2,044 words/sec |
| Sentence Disambiguation | **428 sent/sec** | 294 sent/sec |
| Tokenization | 10,418 tokens/sec | — |
| Sentence Extraction | 1,045 sent/sec | — |

### Latency (Word Analysis)

| Percentile | Time |
|------------|------|
| Mean | 0.17ms |
| p50 | 0.45ms |
| p99 | 1.95ms |

### Cache Efficiency

```
Total Calls:    5,808,000
L0 Hits (precomputed):  54.6%
L2 Hits (runtime):      45.3%
Overall Hit Rate:       99.9%
```

### Cython Speedup

| Function | Pure Python | Cython | Speedup |
|----------|-------------|--------|---------|
| `hash_for_str` | 28.0ms | 0.3ms | **85x** |
| `java_hash_code` | 55.4ms | 0.3ms | **190x** |

---

## Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/isakara5967/zemberek-python.git

# From source
git clone https://github.com/isakara5967/zemberek-python.git
cd zemberek-python
pip install -e .
```

### Requirements

- Python 3.12+
- `numpy>=1.19.0`
- `antlr4-python3-runtime==4.8`

---

## Quick Start

### Morphological Analysis

```python
from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()

# Analyze a word
for analysis in morphology.analyze("kitaplarımızdan"):
    print(analysis)
# [kitap:Noun] kitap:Noun+lar:A3pl+ımız:P1pl+dan:Abl
```

### Sentence Disambiguation

```python
# Get the most likely analysis for each word in context
result = morphology.analyze_and_disambiguate("Hava çok güzel.")
for word in result.best_analysis():
    print(f"{word.surface}: {word.analysis}")
```

### Text Normalization

```python
from zemberek import TurkishMorphology, TurkishSentenceNormalizer

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

text = "slm nslsn bugun hava cok guzel"
print(normalizer.normalize(text))
# "selam nasılsın bugün hava çok güzel"
```

### Spell Checking

```python
from zemberek import TurkishMorphology, TurkishSpellChecker

morphology = TurkishMorphology.create_with_defaults()
spell_checker = TurkishSpellChecker(morphology)

suggestions = spell_checker.suggest_for_word("kitab")
print(suggestions)  # ("kitap", "kitabı", ...)
```

### Tokenization

```python
from zemberek import TurkishTokenizer, TurkishSentenceExtractor

# Tokenize
tokenizer = TurkishTokenizer.DEFAULT
tokens = tokenizer.tokenize("Dr. Ahmet Bey yarın gelecek.")
for token in tokens:
    print(f"{token.content} ({token.type_})")

# Extract sentences
extractor = TurkishSentenceExtractor()
sentences = extractor.from_paragraph("Merhaba! Nasılsın? İyi misin?")
```

---

## Architecture

### Cache Hierarchy

```
┌─────────────────────────────────────────┐
│            User Request                 │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  L0: PrecomputedCache (from disk)       │ ◄─ 54.6% hit
│  - 132K+ precomputed hash values        │
└─────────────────┬───────────────────────┘
                  ▼ miss
┌─────────────────────────────────────────┐
│  L1: LRU Cache (@lru_cache)             │ ◄─ per-function
│  - Per-process, automatic eviction      │
└─────────────────┬───────────────────────┘
                  ▼ miss
┌─────────────────────────────────────────┐
│  L2: LocalCache (runtime dict)          │ ◄─ 45.3% hit
│  - Session-local, no IPC overhead       │
└─────────────────┬───────────────────────┘
                  ▼ miss
┌─────────────────────────────────────────┐
│  Compute (Cython-optimized)             │ ◄─ 0.1% only
│  - 85-190x faster than pure Python      │
└─────────────────────────────────────────┘
```

### Optimizations Applied

| Optimization | Impact | Location |
|--------------|--------|----------|
| Cython hash functions | 85-190x speedup | `zemberek/core/hash/` |
| 3-layer cache | 99.9% hit rate | `zemberek/cache/` |
| `deepcopy` → `set.copy()` | 109x faster | morphology analysis |
| Precomputed cache | Zero-cost lookups | `resources/cache/` |

---

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `TurkishMorphology` | Word analysis, generation, disambiguation | Full |
| `TurkishTokenizer` | Tokenization with Turkish rules | Full |
| `TurkishSentenceExtractor` | Sentence boundary detection | Full |
| `TurkishSentenceNormalizer` | Informal text normalization | Full |
| `TurkishSpellChecker` | Spelling suggestions | Full |

---

## Why This Fork?

The original [Loodos/zemberek-python](https://github.com/Loodos/zemberek-python) is no longer maintained and has critical issues in Python 3.12+:

| Issue | Impact | Status |
|-------|--------|--------|
| `OverflowError: int 255 out of bounds for int8` | Crashes on load | Fixed |
| `OverflowError: int 4294967296 out of bounds for int32` | Hash failures | Fixed |
| `pkg_resources` deprecation | Breaks in setuptools 81+ | Migrated |
| Slow hash computations | Poor performance | Optimized |

---

## Running Benchmarks

```bash
# Full benchmark with cache
python benchmark/full_benchmark_v2.py --limit 10000

# Cold start benchmark (no cache)
python benchmark/cold_start_benchmark.py --limit 10000
```

---

## Technical Notes

### Java Compatibility

This library replicates Java's integer overflow behavior for hash functions:

```python
# Java: int wraps at 32-bit boundary
# Python: integers have arbitrary precision
# Solution: numpy.int32 with overflow handling
```

### Turkish Language Handling

Full support for Turkish morphology:

- **Vowel harmony** (büyük/küçük ünlü uyumu)
- **Consonant softening** (p→b, ç→c, t→d, k→ğ)
- **Vowel dropping** (burun→burnu, şehir→şehri)
- **Buffer letters** (y, n, s, ş insertion)
- **All suffix combinations** (noun, verb, derivational)

---

## Credits

### Original Projects

- [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp) by Ahmet A. Akın — Original Java library
- [zemberek-python](https://github.com/Loodos/zemberek-python) by Loodos — Python port

### Original Contributors

- [Harun Uz](https://github.com/harun-loodos)
- [Furkan Unlturk](https://github.com/futurk)

### This Fork

Maintained for the [UEM Project](https://github.com/isakara5967/UEM). We provide bug fixes, compatibility updates, and performance optimizations. Core algorithms belong to original authors.

---

## License

Apache License 2.0

---

## Links

- [Original zemberek-nlp (Java)](https://github.com/ahmetaa/zemberek-nlp)
- [Original zemberek-python](https://github.com/Loodos/zemberek-python)
- [This Fork](https://github.com/isakara5967/zemberek-python)
