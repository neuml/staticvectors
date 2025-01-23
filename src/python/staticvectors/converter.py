"""
Converter module
"""

# Conditional import
try:
    import fasttext
    import nanopq

    TRAIN = True
except ImportError:
    TRAIN = False

import re
import os

import numpy as np

from tqdm.auto import tqdm

from .database import Database
from .filesystem import FileSystem
from .model import StaticVectors


class TextConverter:
    """
    Converts pre-trained vectors stored as text to a StaticVectors model.
    """

    def __init__(self):
        """
        Creates a new converter.
        """

        if not TRAIN:
            raise ImportError('Training libraries are not available - install "train" extra to enable')

    def __call__(self, model, path, quantize=None, storage="filesystem"):
        """
        Exports pre-trained vectors stored as text to a StaticVectors model.

        Args:
            model: model path or instance
            path: output path to store exported model
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
            storage: sets the staticvectors storage format (filesystem or database)
        """

        with open(model, encoding="utf-8") as f:
            total, dimensions = [int(x) for x in f.readline().strip().split()]
            tokens, vectors = [], []

            # Read vectors
            for line in tqdm(f, total=total):
                # Read token and vector
                fields = line.strip().split()
                tokens.append(fields[0])
                vectors.append(np.loadtxt(fields[1:], dtype=np.float32))

            # Join into single vectors array
            vectors = np.array(vectors)

        # Generate configuration and tokens
        config = {"format": "vectors", "source": os.path.basename(model), "total": total, "dim": dimensions}
        tokens = {token: x for x, token in enumerate(tokens)}

        # Save model
        self.save(path, storage, config, vectors, quantize, tokens=tokens)

    def save(self, path, storage, config, vectors, quantize, weights=None, tokens=None, labels=None, counts=None):
        """
        Saves a StaticVectors model to the underlying storage layer.

        Args:
            path: output path to store model
            storage: storage format
            config: model configuration
            vectors: model vectors
            quantize: number of subspaces for quantization
            weights: model weights (for classification models)
            tokens: tokens used in model
            labels: classification labels
            counts: label frequency counts
        """

        if storage == "sqlite":
            # Create and save database
            writer = Database(path, create=True)
            writer.save(vectors, tokens)
        else:
            # Initialize writer
            writer = FileSystem(path, create=True)

            # Apply quantization, if necessary
            vectors, pq = self.quantize(vectors, quantize) if quantize else (vectors, None)

            # Save model to output path
            writer.saveconfig(config)
            writer.savetensors(vectors, pq, weights)
            writer.savevocab(tokens, labels, counts)

    def quantize(self, vectors, quantize):
        """
        Quantizes vectors using Product Quantization (PQ).

        Read more on this method at the link below.

        https://fasttext.cc/blog/2017/10/02/blog-post.html#model-compression

        Args:
            vectors: model vectors
            quantize: number of subspaces for quantization

        Returns:
            (quantized vectors, product quantizer)
        """

        # Quantizes vectors using Product Quantization (PQ)
        pq = nanopq.PQ(M=quantize)
        pq.fit(vectors)
        vectors = pq.encode(vectors)

        return vectors, pq


class FastTextConverter(TextConverter):
    """
    Converts a FastText model to a StaticVectors model.
    """

    def __call__(self, model, path, quantize=None, storage="filesystem"):
        """
        Exports a FastText model to output path.

        Args:
            model: model path or instance
            path: output path to store exported model
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
        """

        # Load the model
        source = model if isinstance(model, str) else "memory"
        model = fasttext.load_model(model) if isinstance(model, str) else model
        args = model.f.getArgs()
        supervised = args.model.name == "supervised"

        # Generate configuration
        config = self.config(source, args)

        # Extract model data
        vectors = model.get_input_matrix()
        weights = model.get_output_matrix() if supervised else None

        # Vocabulary parameters
        tokens = {token: x for x, token in enumerate(model.get_words())}
        labels, counts = model.get_labels(include_freq=True) if supervised else (None, None)
        counts = {i: int(x) for i, x in enumerate(counts)} if supervised else None

        # Save model
        self.save(path, storage, config, vectors, quantize, weights, tokens, labels, counts)

    def config(self, source, args):
        """
        Builds model configuration from a FastText args instance.

        Args:
            source: path to input model, if available
            args: FastText args instance

        Returns:
            dict of training parametersarguments
        """

        # Options for FastText
        options = [
            "lr",
            "dim",
            "ws",
            "epoch",
            "minCount",
            "minCountLabel",
            "neg",
            "wordNgrams",
            "loss",
            "model",
            "bucket",
            "minn",
            "maxn",
            "thread",
            "lrUpdateRate",
            "t",
            "label",
            "verbose",
            "pretrainedVectors",
            "saveOutput",
            "seed",
            "qout",
            "retrain",
            "qnorm",
            "cutoff",
            "dsub",
        ]

        # Convert args to a config dictionary
        config = {**{"format": "fasttext", "source": os.path.basename(source)}, **{option: getattr(args, option) for option in options}}
        config["loss"] = config["loss"].name
        config["model"] = config["model"].name

        # Change camel case to underscores to standardize config.json
        config = {re.sub(r"([a-z])([A-Z])", r"\1_\2", k).lower(): v for k, v in config.items()}

        return config


class StaticVectorsConverter(TextConverter):
    """
    Converts a StaticVectors model to another format. This enables rebuilding an existing model with different
    parameters.
    """

    def __call__(self, model, path, quantize=None, storage="filesystem"):
        model = StaticVectors(model)

        # Get model vectors and tokens
        vectors = np.array([model.vectors[x] for x in tqdm(range(len(model.vectors)), total=len(model.vectors))])
        tokens = dict(model.tokens.items())

        # Save model
        self.save(path, storage, model.config, vectors, quantize, model.weights, tokens, model.labels, model.counts)
