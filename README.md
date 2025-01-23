<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/staticvectors/master/logo.png"/>
</p>

<p align="center">
    <b>Work with static vector models</b>
</p>

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
    <a href="https://github.com/neuml/staticvectors/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/staticvectors/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/staticvectors?branch=master">
        <img src="https://img.shields.io/coverallsCoverage/github/neuml/staticvectors" alt="Coverage Status">
    </a>
</p>

`staticvectors` makes it easy to work with static vector models. This includes word vector models such as [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/). While [Transformers-based models](https://github.com/huggingface/transformers) are now the primary way to embed content for vector search, these older models still have a purpose.

For example, this [FastText Language identification model](https://fasttext.cc/docs/en/language-identification.html) is still one of the fastest and most efficient ways to detect languages. N-grams work well for this task and it's lightning fast.

Additionally, there are historical, low resource and other languages where there just isn't enough training data to build a solid language model. In these cases, a simpler model using one of these older techniques might be the best option. 

## What's wrong with the existing libraries

Unfortunately, the tooling to use word vector models is aging and in some cases unmaintained. The world is moving forward and these libraries are getting harder to install.

As a concrete example, the build script for [txtai](https://github.com/neuml/txtai/blob/master/.github/workflows/build.yml#L42) often has to be modified to get FastText to work on all supported platforms. There are pre-compiled versions but they're often slow to support the latest version of Python or fix issues.

This project breathes life into word vector models and integrates them with modern tooling such as the [Hugging Face Hub](https://huggingface.co/models) and [Safetensors](https://github.com/huggingface/safetensors). While it's pure Python, it's still fast due to it's heavy usage of [NumPy](https://github.com/numpy/numpy) and [vectorization techniques](https://numpy.org/doc/stable/user/whatisnumpy.html#why-is-numpy-fast).

This makes it easier to maintain as it's only a single install package to maintain.

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

## Quickstart

See the following examples on how to use this library.

### Convert an existing FastText model

```python
from staticvectors import FastTextConverter, StaticVectors

# Download https://huggingface.co/julien-c/fasttext-language-id/blob/main/lid.176.bin
converter = FastTextConverter()
converter("lid.176.bin", "langid")

# Load the converted model - runs in pure Python, FastText library install not required for inference
model = StaticVectors("langid")
model.predict(["Hello, what language is this?"])
```

### Load an existing Magnitude SQLite database

This library replaces [Magnitude Lite](https://github.com/neuml/magnitude) which is now deprecated. Magnitude libraries are supported by this library.

```python
from staticvectors import StaticVectors

model = StaticVectors("/path/to/vectors.magnitude")

# Get word vectors
model.embeddings(["hello"])
```

### Convert and quantize

```python
from staticvectors import FastTextConverter, StaticVectors

# Download https://huggingface.co/julien-c/fasttext-language-id/blob/main/lid.176.bin
converter = FastTextConverter()

# Quantize with PQ and two subspaces - model goes from 125MB to 4MB with minimal accuracy impacts!
converter("lid.176.bin", "langid-pq2x256", quantize=2)

# Load the converted model - runs in pure Python, FastText library install not required for inference
model = StaticVectors("langid")
model.predict(["Hello, what language is this?"])
```

See the unit tests in this project here for more examples.

## Libraries for Static Embeddings with Transformers models

This library is primarily focused on word vector models. There is a recent push to distill Transformers models into static embeddings models. The difference between `staticvectors` and these libraries is that the base models are Transformers models. Additionally, they use Transformers tokenizers where as word vector models tokenize on whitespace and use n-grams.

Check out these links for more on static embeddings with Transformers models.

- [Model2Vec](https://github.com/MinishLab/model2vec) - Turn any sentence transformer into a really small static model
- [Static Sentence Transformers](https://huggingface.co/blog/static-embeddings) - Run 100x to 400x faster on CPU than state-of-the-art embedding models
