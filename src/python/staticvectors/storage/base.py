"""
Storage module
"""

import json
import os

from .retriever import Retriever


class Storage:
    """
    Base class for Storage instances.
    """

    def __init__(self, path, create=False, storefile=False):
        """
        Creates a new Storage instance.

        Args:
            path: model path
            create: create model path locally if True, this is for writing models
            storefile: if true, this storage instance stores all it's content a single file
        """

        # Model path
        self.path = path

        # File retriever
        self.retrieve = Retriever()

        # Flag for single file model storage
        self.storefile = storefile

        # Create output directory
        if create and not storefile:
            os.makedirs(path, exist_ok=True)

    def load(self):
        """
        Loads model data from storage.

        Returns:
            model data
        """

        # Load model files
        config = self.loadconfig()
        vectors, quantization, weights = self.loadtensors()
        tokens, labels, counts = self.loadvocab()

        # Return model data
        return config, vectors, quantization, weights, tokens, labels, counts

    def save(self, config, vectors, pq=None, weights=None, tokens=None, labels=None, counts=None):
        """
        Saves model data to storage.

        Args:
            config: model configuration
            vectors: model vectors
            pq: product quantization parameters
            weights: model weights (for classification models)
            tokens: tokens used in model
            labels: classification labels
            counts: label frequency counts
        """

        # Save model to output path
        self.saveconfig(config)
        self.savetensors(vectors, pq, weights)
        self.savevocab(tokens, labels, counts)

    def loadconfig(self):
        """
        Loads model configuration.
        """

        with open(self.retrieve(f"{self.path}/config.json"), encoding="utf-8") as f:
            return json.load(f)

    def loadtensors(self):
        """
        Loads model tensor data.
        """

        raise NotImplementedError

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
            config = {**{"model_type": "staticvectors", "storage": self.storage()}, **config}

            # Save config.json
            json.dump(config, f, indent=2)

    def savetensors(self, vectors, pq, weights):
        """
        Saves model tensor data.

        Args:
            vectors: model vectors
            pq: product quantization parameters
            weights: model weights (for classification models)
        """

        raise NotImplementedError

    def savevocab(self, tokens, labels, counts):
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

    def storage(self):
        """
        Gets the name of this storage backend.

        Returns:
            storage name
        """

        raise NotImplementedError
