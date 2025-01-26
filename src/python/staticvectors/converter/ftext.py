"""
FastText module
"""

import re
import os

# Conditional import
try:
    import fasttext

except ImportError:
    pass

from .base import Converter


class FastTextConverter(Converter):
    """
    Converts a FastText model to a StaticVectors model.
    """

    def __call__(self, model, path, quantize=None, storage="safetensors", storefile=False):
        """
        Converts a FastText model.

        Args:
            model: model path or instance
            path: output path to store exported model
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
            storage: storage layer type, defaults to safetensors
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
        self.save(path, storage, storefile, config, vectors, quantize, weights, tokens, labels, counts)

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
