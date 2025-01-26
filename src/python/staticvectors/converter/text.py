"""
Text module
"""

import os

import numpy as np

from tqdm.auto import tqdm

from .base import Converter


class TextConverter(Converter):
    """
    Converts pre-trained text vectors to a StaticVectors model.
    """

    def __call__(self, model, path, quantize=None, storage="safetensors", storefile=False):
        """
        Converts pre-trained text vectors.

        Args:
            model: model path or instance
            path: output path to store exported model
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
            storage: sets the staticvectors storage format (tensors or database)
        """

        with open(model, encoding="utf-8") as f:
            total, dimensions = [int(x) for x in f.readline().strip().split()]
            tokens, vectors = [], []

            # Read vectors
            for line in tqdm(f, total=total):
                # Read token and vector
                fields = line.split(" ")
                tokens.append(fields[0])
                vectors.append(np.loadtxt(fields[1:], dtype=np.float32))

            # Join into single vectors array
            vectors = np.array(vectors)

        # Generate configuration and tokens
        config = {"format": "text", "source": os.path.basename(model), "total": total, "dim": dimensions}
        tokens = {token: x for x, token in enumerate(tokens)}

        # Save model
        self.save(path, storage, storefile, config, vectors, quantize, tokens=tokens)
