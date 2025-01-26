"""
StaticVectors module
"""

import numpy as np

from tqdm.auto import tqdm

from ..model import StaticVectors
from .base import Converter


class StaticVectorsConverter(Converter):
    """
    Converts a StaticVectors model. This enables rebuilding an existing model with different
    parameters.
    """

    def __call__(self, model, path, quantize=None, storage="safetensors", storefile=False):
        """
        Converts a StaticVectors model.

        Args:
            model: model path or instance
            path: output path to store exported model
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
            storage: storage layer type, defaults to safetensors
        """

        model = StaticVectors(model)

        # Get model vectors and tokens
        vectors = np.array([model.vectors[x] for x in tqdm(range(len(model.vectors)), total=len(model.vectors))])
        tokens = dict(model.tokens.items())

        # Save model
        self.save(path, storage, storefile, model.config, vectors, quantize, model.weights, tokens, model.labels, model.counts)
