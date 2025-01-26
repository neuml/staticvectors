"""
Tensors module
"""

import numpy as np

from safetensors import safe_open
from safetensors.numpy import save_file

from .base import Storage


class Tensors(Storage):
    """
    Safetensors storage format. Tensors are stored in a safetensors file. Configuration and vocabulary are stored as JSON.
    """

    def loadtensors(self):
        with safe_open(self.retrieve(f"{self.path}/model.safetensors"), framework="numpy") as f:
            return (
                f.get_tensor("vectors"),
                (f.get_tensor("pq"), f.get_tensor("codewords")) if "pq" in f.keys() else None,
                f.get_tensor("weights") if "weights" in f.keys() else None,
            )

    def savetensors(self, vectors, pq, weights):
        # Base exports
        tensors = {"vectors": vectors}

        # Vector quantization enabled
        if pq is not None:
            tensors["pq"] = np.array([pq.Ds, pq.M])
            tensors["codewords"] = pq.codewords

        # Classification model weights
        if weights is not None:
            tensors["weights"] = weights

        # Save model.safetensors
        save_file(tensors, f"{self.path}/model.safetensors")

    def storage(self):
        return "safetensors"
