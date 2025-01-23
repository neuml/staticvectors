"""
Train module
"""

# Conditional import
try:
    import fasttext

    TRAIN = True
except ImportError:
    TRAIN = False

from .converter import FastTextConverter
from .model import StaticVectors


class StaticVectorsTrainer:
    """
    Trains a new StaticVector model using FastText.
    """

    def __init__(self):
        """
        Creates a new trainer.
        """

        if not TRAIN:
            raise ImportError('Training libraries are not available - install "train" extra to enable')

    def __call__(self, data, size, mincount, path, storage="filesystem", classification=False, quantize=None, load=False, **kwargs):
        """
        Trains a new StaticVector model. Additional keyword args are passed to the
        FastText train method.

        See this link: https://fasttext.cc/docs/en/python-module.html#api

        Args:
            data: path to input data file
            size: number of vector dimensions
            mincount: minimum number of occurrences required to register a token
            path: path to store StaticVectors model
            storage: sets the staticvectors storage format (filesystem or database)
            quantize: enables quantization and sets the number of Product Quantization (PQ)
                      subspaces
            load: returns both the StaticVectors and FastText trained model if True, defaults to False

        Returns:
            (StaticVectors model, trained FastText model)
        """

        if classification:
            model = fasttext.train_supervised(data, dim=size, minCount=mincount, **kwargs)
        else:
            model = fasttext.train_unsupervised(data, dim=size, minCount=mincount, **kwargs)

        # Convert the model to a StaticVector model
        converter = FastTextConverter()
        converter(model, path, quantize, storage)

        # Return the StaticVectors model and original trained FastText model
        return StaticVectors(path), model if load else None
