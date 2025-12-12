# zemberek-python (Python 3.12+ Compatible Fork)

Python implementation of Natural Language Processing library for Turkish.

This is a **maintained fork** of [Loodos/zemberek-python](https://github.com/Loodos/zemberek-python) with bug fixes and Python 3.12+ compatibility.

## Why This Fork?

The original project is no longer actively maintained and has critical bugs in Python 3.12+:

- **OverflowError**: `Python integer 255 out of bounds for int8`
- **OverflowError**: `Python integer 4294967296 out of bounds for int32`
- **DeprecationWarning**: `pkg_resources` is deprecated (removed in setuptools 81+)
- **Broken Components**: Normalizer and SpellChecker completely non-functional

This fork fixes all these issues and ensures compatibility with modern Python versions.

## What's Fixed

| Issue | Affected Files | Status |
|-------|----------------|--------|
| int8 overflow in bitwise operations | `gram_data_array.py` | Fixed |
| int32 overflow in hash functions | `mphf.py`, `large_ngram_mphf.py` | Fixed |
| pkg_resources deprecation | 10 files | Migrated to `importlib.resources` |

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/isakara5967/zemberek-python.git
```

### From Source

```bash
git clone https://github.com/isakara5967/zemberek-python.git
cd zemberek-python
pip install .
```

## Dependencies

- `antlr4-python3-runtime==4.8`
- `numpy>=1.19.0`

## Supported Modules

| Module | Features | Status |
|--------|----------|--------|
| **TurkishMorphology** | Word Analysis, Word Generation, Sentence Analysis, Ambiguity Resolution | Partial |
| **Tokenization** | Sentence Boundary Detection, Tokenization | Full |
| **Normalization** | Spelling Suggestion, Noisy Text Normalization | Partial |

## Usage

### Morphological Analysis

```python
from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()

# Analyze a single word
result = morphology.analyze("kitaplar")
for analysis in result:
    print(analysis)
```

### Sentence Normalization

```python
from zemberek import TurkishMorphology, TurkishSentenceNormalizer

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

# Normalize informal text
text = "slm nslsn bugun hava cok guzel"
normalized = normalizer.normalize(text)
print(normalized)  # "selam nasılsın bugün hava çok güzel"
```

### Spell Checking

```python
from zemberek import TurkishMorphology, TurkishSpellChecker

morphology = TurkishMorphology.create_with_defaults()
spell_checker = TurkishSpellChecker(morphology)

# Get spelling suggestions
suggestions = spell_checker.suggest_for_word("kitab")
print(suggestions)  # ("kitap", "kitabı", ...)
```

### Tokenization

```python
from zemberek import TurkishTokenizer

tokenizer = TurkishTokenizer.DEFAULT

# Tokenize a sentence
tokens = tokenizer.tokenize("Dr. Ahmet Bey yarın gelecek.")
for token in tokens:
    print(f"{token.content} ({token.type_})")
```

### Sentence Boundary Detection

```python
from zemberek import TurkishSentenceExtractor

extractor = TurkishSentenceExtractor.DEFAULT

text = "Merhaba! Bugün hava çok güzel. Yarın yağmur yağacak mı?"
sentences = extractor.from_paragraph(text)
for sentence in sentences:
    print(sentence)
```

## Performance

Benchmarks run on AMD Ryzen 7 5700X, Ubuntu (WSL2), Python 3.12.3:

### Initialization Time

| Component | Time | Memory |
|-----------|------|--------|
| TurkishMorphology | 6.92s | 130.3 MB |
| TurkishSentenceNormalizer | 18.33s | 364.5 MB |
| TurkishSpellChecker | 18.86s | 157.1 MB |
| TurkishTokenizer | <0.01s | N/A |

### Runtime Performance

| Component | Items | Total Time | Avg Time | Throughput |
|-----------|-------|------------|----------|------------|
| Morphology Analysis | 1000 words | <0.01s | <0.01ms | ~6.7M/sec* |
| Sentence Normalization | 100 sentences | 0.88s | 8.81ms | 113/sec |
| Spell Checker | 100 words | 0.97s | 9.66ms | 103/sec |
| Tokenization | 100 sentences | 0.01s | 0.08ms | 12,162/sec |

*Morphology uses internal caching, so repeated analyses are nearly instant.

Run benchmarks yourself:
```bash
python benchmarks/benchmark.py
```

## Notes

This project is a Python port of the original Java implementation. Some adjustments were necessary due to differences between Java and Python:

- Java silently handles integer overflows, while Python/NumPy raises errors
- We use `numpy.int32` to replicate Java's 4-byte int behavior
- Overflow warnings are suppressed in specific modules for compatibility

## Credits & Acknowledgments

This project is built upon the excellent work of:

### Original Projects

| Project | Author(s) | Description |
|---------|-----------|-------------|
| [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp) | Ahmet A. Akin | Original Java NLP library for Turkish |
| [zemberek-python](https://github.com/Loodos/zemberek-python) | Loodos | Python port of zemberek-nlp |

### Original Contributors (Loodos/zemberek-python)

- [Harun Uz](https://github.com/harun-loodos) - Lead developer
- [Furkan Unlturk](https://github.com/futurk)

### About This Fork

This fork is maintained by [isakara5967](https://github.com/isakara5967) for the [UEM Project](https://github.com/isakara5967/UEM).

**We only provide bug fixes and compatibility updates.** All core algorithms and implementations belong to the original authors.

We are grateful to the Turkish NLP community for their contributions.

## License

Apache License 2.0 (same as original)

## Contributing

Issues and pull requests are welcome. Please ensure your changes:

1. Don't break existing functionality
2. Include tests if applicable
3. Follow the existing code style

## Links

- [Original zemberek-nlp (Java)](https://github.com/ahmetaa/zemberek-nlp)
- [Original zemberek-python](https://github.com/Loodos/zemberek-python)
- [This Fork](https://github.com/isakara5967/zemberek-python)
- [UEM Project](https://github.com/isakara5967/UEM)
