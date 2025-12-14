#!/usr/bin/env python3
"""
Cython build script for Zemberek hash functions.

Usage:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Get the directory containing this script
here = os.path.dirname(os.path.abspath(__file__))

# Define extensions
extensions = [
    Extension(
        "zemberek.cython.hash_functions",
        sources=[os.path.join(here, "zemberek/cython/hash_functions.pyx")],
        extra_compile_args=["-O3"],  # Maximum optimization
    )
]

# Cython compiler directives for performance
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'initializedcheck': False,
}

setup(
    name="zemberek-cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generate HTML annotation file
    ),
)
