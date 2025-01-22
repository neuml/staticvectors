# staticvectors: Work with static vector models

<p align="center">
    <a href="https://github.com/neuml/staticvectors/releases">
        <img src="https://img.shields.io/github/release/neuml/staticvectors.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/staticvectors/releases">
        <img src="https://img.shields.io/github/release-date/neuml/staticvectors.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/stativectors/issues">
        <img src="https://img.shields.io/github/issues/neuml/staticvectors.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/staticvectors">
        <img src="https://img.shields.io/github/last-commit/neuml/staticvectors.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
</p>

`staticvectors` makes it easy to work with static vector models. This includes legacy models such as [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/). While [Transformers-based models](https://github.com/huggingface/transformers) are now the primary way to embed content for vector search, these older models still have a purpose.

For example, this [FastText Language identification model](https://fasttext.cc/docs/en/language-identification.html) is still one of the fastest and most efficient ways to detect languages. N-grams work well for this task and it's lightning fast.

Additionally, there are historical, low resource and other languages where there just isn't enough training data to build a solid language model. In these cases, a simpler model using one of these older techniques might be the best option. 

## What's wrong with the existing libraries

Unfortunately, the tooling to use these models is aging and in some cases unmaintained. The world is moving forward and these libraries are getting harder to install.

As a concrete example, the build script for [txtai](https://github.com/neuml/txtai/blob/master/.github/workflows/build.yml#L42) often has to be modified to get FastText to work on all supported platforms. There are pre-compiled versions but they're often slow to support the latest version of Python or fix issues.

This project breathes life into these models and integrates them with modern tooling such as the [Hugging Face Hub](https://huggingface.co/models) and [Safetensors](https://github.com/huggingface/safetensors). While it's pure Python, it's still fast due to it's heavy usage of [NumPy](https://github.com/numpy/numpy) and [vectorization techniques](https://numpy.org/doc/stable/user/whatisnumpy.html#why-is-numpy-fast). This makes it easier to maintain as it's only a single install package to maintain.

## Installation
The easiest way to install is via pip and PyPI

```
pip install staticvectors
```

Python 3.9+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

`staticvectors` can also be installed directly from GitHub to access the latest, unreleased features.

```
pip install git+https://github.com/neuml/staticvectors
```
