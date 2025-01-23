"""
ModelIO module
"""

import json
import os

import numpy as np

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import save_file


class FileSystem:
    """
    FileSystem tensor storage Stores config, vectors and vocabulary in a file system directory.
    Supports integration with the Hugging Face Hub.
    """

    def __init__(self, path, create=False):
        """
        Creates a new FileSystem instance.

        Args:
            path: model path
            create: create model path locally if True, this is for writing models
        """

        # Model path
        self.path = path

        # Create output directory
        if create:
            os.makedirs(path, exist_ok=True)

    def loadconfig(self):
        """
        Loads model configuration.
        """

        with open(self.retrieve(f"{self.path}/config.json"), encoding="utf-8") as f:
            return json.load(f)

    def loadtensors(self):
        """
        Loads tensor data from Safetensors file.
        """

        with safe_open(self.retrieve(f"{self.path}/model.safetensors"), framework="numpy") as f:
            return (
                f.get_tensor("vectors"),
                (f.get_tensor("pq"), f.get_tensor("codewords")) if "pq" in f.keys() else None,
                f.get_tensor("weights") if "weights" in f.keys() else None,
            )

    def loadvocab(self):
        """
        Loads model vocabulary.

        Args:
            path: model path
        """

        with open(self.retrieve(f"{self.path}/vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
            return (vocab["tokens"], vocab.get("labels"), {int(k): int(v) for k, v in vocab["counts"].items()} if "counts" in vocab else None)

    def saveconfig(self, config):
        """
        Saves model configuration.

        Args:
            path: output path
            config: model configuration
        """

        with open(f"{self.path}/config.json", "w", encoding="utf-8") as f:
            # Add model_type
            config = {**{"model_type": "staticvectors"}, **config}

            # Save config.json
            json.dump(config, f, indent=2)

    def savetensors(self, vectors, pq=None, weights=None):
        """
        Saves tensors data to a Safetensors file.

        Args:
            vectors: model vectors
            pq: product quantization parameters
            weights: model weights (for classification models)
        """

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

    def savevocab(self, tokens, labels=None, counts=None):
        """
        Saves model vocabulary.

        Args:
            tokens: tokens used in model
            labels: classification labels
            counts: label frequency counts
        """

        with open(f"{self.path}/vocab.json", "w", encoding="utf-8") as f:
            data = {"tokens": tokens}
            if labels:
                data["labels"] = labels
            if counts:
                data["counts"] = counts

            # Save to vocab.json
            json.dump(data, f)

    def retrieve(self, path):
        """
        Retrieves file at path locally. Skips downloading if the file already exists.

        Args:
            path: requested file path

        Returns:
            local path to file
        """

        return path if os.path.exists(path) else self.download(path)

    def download(self, path):
        """
        Downloads path from the Hugging Face Hub.

        Args:
            path: full model path

        Returns:
            local cached model path
        """

        # Split into parts
        parts = path.split("/")

        # Calculate repo id split
        repo = 2 if len(parts) > 2 else 1

        # Download and cache file
        return hf_hub_download(repo_id="/".join(parts[:repo]), filename="/".join(parts[repo:]))
