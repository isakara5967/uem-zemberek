# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3-fork.1] - 2024-12-12

### Fixed

#### Critical Bug Fixes

- **int8 Overflow Error** (`gram_data_array.py`)
  - Fixed `OverflowError: Python integer 255 out of bounds for int8`
  - Affected methods: `get_probability_rank()`, `get_back_off_rank()`, `check_finger_print()`
  - Solution: Convert numpy int8 values to Python int before bitwise operations
  - Lines affected: 54, 56, 58, 68, 70, 72, 82-94

- **int32 Overflow Error** (`mphf.py`, `large_ngram_mphf.py`)
  - Fixed `OverflowError: Python integer 4294967296 out of bounds for int32`
  - Affected: `rshift()` function used in hash calculations
  - Solution: Convert numpy int32 values to Python int before modulo operations
  - Files:
    - `zemberek/core/hash/mphf.py:22`
    - `zemberek/lm/compression/gram_data_array.py:106`
    - `zemberek/core/hash/large_ngram_mphf.py:39`

#### Deprecation Fixes

- **pkg_resources Deprecation**
  - Migrated from deprecated `pkg_resources.resource_filename` to `importlib.resources.files`
  - This prepares for setuptools 81+ where pkg_resources is removed
  - Files updated:
    - `zemberek/morphology/turkish_morphology.py`
    - `zemberek/morphology/lexicon/root_lexicon.py`
    - `zemberek/morphology/analysis/tr/pronunciation_guesser.py`
    - `zemberek/morphology/analysis/tr/turkish_numbers.py`
    - `zemberek/normalization/stem_ending_graph.py`
    - `zemberek/normalization/turkish_spell_checker.py`
    - `zemberek/normalization/turkish_sentence_normalizer.py`
    - `zemberek/normalization/deasciifier/deasciifier.py`
    - `zemberek/tokenization/perceptron_segmenter.py`
    - `zemberek/tokenization/antlr/turkish_lexer.py`

### Added

- **Benchmark Suite** (`benchmarks/benchmark.py`)
  - Performance measurement for all major components
  - Memory usage tracking
  - Markdown table output for documentation

- **Documentation**
  - Comprehensive README.md with:
    - Installation instructions
    - Usage examples for all components
    - Performance benchmarks
    - Credits and acknowledgments
  - This CHANGELOG.md file

### Changed

- Updated `.gitignore` to include Python cache files and build artifacts

## Technical Details

### Root Cause Analysis

#### int8/int32 Overflow

The original Java implementation uses signed 32-bit integers with silent overflow behavior. When ported to Python with NumPy, the behavior differs:

```java
// Java - silent overflow
int result = value & 0xFF;  // Works even if value overflows
```

```python
# Python/NumPy - raises OverflowError
result = numpy_int8_value & 255  # Raises error if value is outside int8 range
```

**Solution**: Convert NumPy integers to Python native integers before bitwise operations:

```python
# Fixed version
result = int(numpy_int8_value) & 255  # Always works
```

#### pkg_resources Deprecation

`pkg_resources` from setuptools is deprecated and will be removed in setuptools 81+. The `importlib.resources` module (standard library since Python 3.9) is the recommended replacement.

```python
# Old (deprecated)
from pkg_resources import resource_filename
path = resource_filename("package", "path/to/resource")

# New (Python 3.9+)
from importlib.resources import files
path = str(files("package").joinpath("path", "to", "resource"))
```

## Compatibility

| Python Version | Status |
|----------------|--------|
| 3.9 | Supported |
| 3.10 | Supported |
| 3.11 | Supported |
| 3.12 | Supported (primary target) |
| 3.13 | Should work (untested) |

## Migration from Original

If you're migrating from the original `zemberek-python`:

1. Uninstall the original package:
   ```bash
   pip uninstall zemberek-python
   ```

2. Install this fork:
   ```bash
   pip install git+https://github.com/isakara5967/zemberek-python.git
   ```

3. No code changes required - the API is identical.

## [0.2.3] - Original Release

This is the last release from the original [Loodos/zemberek-python](https://github.com/Loodos/zemberek-python) repository.

---

## Links

- [This Fork](https://github.com/isakara5967/zemberek-python)
- [Original Repository](https://github.com/Loodos/zemberek-python)
- [Original zemberek-nlp (Java)](https://github.com/ahmetaa/zemberek-nlp)
