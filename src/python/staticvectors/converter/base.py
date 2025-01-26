"""
Converter module
"""

# Conditional import
try:
    import nanopq

    TRAIN = True
except ImportError:
    TRAIN = False

from ..storage import StorageFactory


class Converter:
    """
    Base converter.
    """

    def __init__(self):
        """
        Creates a new converter.
        """

        if not TRAIN:
            raise ImportError('Training libraries are not available - install "train" extra to enable')

    # pylint: disable=R0913
    def save(self, path, storage, storefile, config, vectors, quantize, weights=None, tokens=None, labels=None, counts=None):
        """
        Saves a StaticVectors model to the underlying storage layer.

        Args:
            path: output path to store model
            storage: storage format
            storefile: store entire model in as a single file for storage formats that support it
            config: model configuration
            vectors: model vectors
            quantize: number of subspaces for quantization
            weights: model weights (for classification models)
            tokens: tokens used in model
            labels: classification labels
            counts: label frequency counts
        """

        # Apply quantization, if necessary
        vectors, pq = self.quantize(vectors, quantize) if quantize else (vectors, None)

        # Store the model
        writer = StorageFactory.create(path, storage, create=True, storefile=storefile)
        writer.save(config, vectors, pq, weights, tokens, labels, counts)

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
