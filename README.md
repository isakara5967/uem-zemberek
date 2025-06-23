# ZEMBEREK-PYTHON

Python implementation of Natural Language Processing library 
for Turkish, [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp). It is based on
zemberek 0.17.1 and is completely written in Python meaning there is no need to setup
a Java development environment to run it.

*Source Code*

https://github.com/Loodos/zemberek-python

**Dependencies**
* antlr4-python3-runtime==4.8
* numpy>=1.19.0

## Supported Modules
Currently, following modules are supported.

* Core (Partially)
    
* TurkishMorphology (Partially)
    * Single Word Analysis
    * Diacritics Ignored Analysis
    * Word Generation
    * Sentence Analysis
    * Ambiguity Resolution
* Tokenization
    * Sentence Boundary Detection
    * Tokenization
* Normalization (Partially)
    * Spelling Suggestion
    * Noisy Text Normalization

## Installation
You can install the package with pip

    pip install zemberek-python

## Examples
Example usages can be found in [examples.py](zemberek/examples.py)

## Notes
This project is a Python port of the original Java implementation, and some adjustments were necessary due to differences between the two languages. While we aimed to use Python equivalents for Java-specific features and data structures wherever possible, a few changes were required that may slightly affect the overall performance and accuracy.

In particular, within the MultiLevelMphf class, the original Java code includes several integer multiplication operations. Initially, we reimplemented these using Python’s built-in int type, but the results did not match the original behavior. To more closely replicate Java’s default 4-byte int and float behavior, we switched to using numpy.int32 and numpy.float32. This yielded results consistent with the Java version, but introduced a new issue: frequent RuntimeWarnings due to overflow in multiplication operations.

Java silently handles overflows without warning, while NumPy alerts the user—this discrepancy is likely due to differences in how the two environments manage numeric overflows. Despite efforts, we couldn’t find a more accurate or safer alternative, so overflow warnings have been suppressed specifically for MultiLevelMphf.

Please note: While this approach reproduces the original behavior, suppressing warnings is not ideal. Use this part of the code with caution, and be aware that it may not handle all edge cases reliably.




## Credits
This project is Python port of [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp). 

