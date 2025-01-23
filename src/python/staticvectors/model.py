"""
Model module
"""

import numpy as np

from .loss import LossFactory
from .tokenizer import NgramTokenizer
from .xhash import FNV

# Storage formats
from .database import Database
from .filesystem import FileSystem


class StaticVectors:
    """
    Generates static embeddings and also supports multi-label classification.
    """

    def __init__(self, path=None):
        # Model configuration
        self.config = None

        # Model parameters
        self.vectors, self.quantization = None, None
        self.weights = None
        self.tokens, self.labels, self.counts = None, None, None

        # Loss instance, used for classification models
        self.loss = None

        # Token cache
        self.cache = {}

        # Tokenizer and Hasher
        self.tokenizer, self.hasher = NgramTokenizer(), FNV()

        # Load model, if provided
        if path:
            self.load(path)

    def load(self, path):
        """
        Loads model at path.

        Args:
            path: model path
        """

        if Database.isdatabase(path):
            # SQLite vectors storage format
            self.vectors = Database(path)
            self.config = self.vectors.config()
            self.tokens = self.vectors.tokens()

        else:
            # File system vectors storage format
            reader = FileSystem(path)

            # Load model files
            self.config = reader.loadconfig()
            self.vectors, self.quantization, self.weights = reader.loadtensors()
            self.tokens, self.labels, self.counts = reader.loadvocab()

            # Cache tokens from vocabulary
            if self.isclassification():
                for token in self.tokens:
                    self.cache[token] = self.tokenize(token)

            # Create model loss when label weights are available
            self.loss = LossFactory.create(self.config["loss"], self.counts, self.weights) if self.weights is not None else None

    def embeddings(self, tokens, normalize=True):
        """
        Gets embeddings vectors for tokens.

        Args:
            tokens: list of tokens to get
            normalize: if True (default), vectors will be normalized

        Returns:
            array of embeddings vectors
        """

        embeddings = []
        for token in tokens:
            if self.isclassification():
                # Vectors from a FastText model
                embeddings.append(self.query(token))
            else:
                # Vectors from a vectors dump
                embeddings.append(self.lookup(token))

        # Get embeddings as np.array
        embeddings = np.array(embeddings)

        # Normalize vectors
        if normalize:
            self.normalize(embeddings)

        return embeddings

    def predict(self, text, limit=1):
        """
        Predicts a label for text. This only works for supervised classification models.

        Args:
            text: input text
            limit: maximum labels to return
        """

        if not self.loss:
            raise ValueError("Predictions only supported with classification models")

        # Create query vector from input text
        vector = self.query(text)
        return [(self.labels[uid].replace(self.config["label"], ""), score) for uid, score in self.loss(vector, limit)]

    def isclassification(self):
        """
        Checks if this model is trained for classification.

        Returns:
            True if this model is trained for classification, False otherwise
        """

        # Bucket is only set for classification models
        return self.config.get("bucket")

    def getvectors(self, tokenids):
        """
        Gets vectors for a set of tokenids. This method handles reconstructing quantized vectors, if necessary.

        Args:
            tokenids: vector ids to get

        Returns:
            array of vectors
        """

        # Decode quantized vectors, if necessary
        return self.decode(self.vectors[tokenids]) if self.quantization else self.vectors[tokenids]

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)

    def lookup(self, token):
        """
        Looks up an embeddings vector for a token. If it's not found, this method
        will break down the query into subtokens and create a vector.

        Args:
            token: input token

        Returns:
            embeddings vector
        """

        # Token is in the vocabulary
        if token in self.tokens:
            return self.getvectors(self.tokens[token])

        # Subtoken parameters
        minn = self.config.get("minn", 3)
        maxn = self.config.get("maxn", 6)

        # Generate vector for out of vocabulary term
        tokenids = [self.tokens[subtoken] for subtoken in self.tokenizer(token, minn, maxn) if subtoken in self.tokens]
        return self.getvectors(tokenids).mean(axis=0)

    def query(self, text):
        """
        Builds a query vector.

        Args:
            text: input text

        Returns:
            query vector
        """

        # Token ids
        tokenids = None

        # Split on whitespace
        for token in text.split():
            # Tokenize each token into subtokens and get as ids
            ids = self.cache[token] if token in self.cache else self.tokenize(token)

            # Add new token ids
            if ids is not None:
                tokenids = np.concatenate((tokenids, ids)) if tokenids is not None else ids

        # Decode quantized vectors, if necessary
        vecs = self.getvectors(tokenids)

        # Default query if no tokens found
        vecs = vecs if vecs.size else self.getvectors(self.tokenize("</s>"))

        # Return mean vector
        return vecs.mean(axis=0)

    def decode(self, codes):
        """
        Decodes a list of Product Quantization (PQ) codes and reconstructs the original vector.

        See https://github.com/matsui528/nanopq/blob/main/nanopq/pq.py#L151

        Args:
            codes: PQ codes

        Returns:
            reconstructed vector
        """

        n, _ = codes.shape
        (d, m), codewords = self.quantization

        vecs = np.empty((n, d * m), dtype=np.float32)
        for x in range(m):
            vecs[:, x * d : (x + 1) * d] = codewords[x][codes[:, x], :]

        return vecs

    def tokenize(self, token, minn=None, maxn=None):
        """
        Tokenizes token into token ids.

        Args:
            token: token text
            minn: min ngram size
            maxn: max ngram size

        Returns:
            [token ids]
        """

        # Subtoken parameters
        minn = minn if minn else self.config.get("minn", 3)
        maxn = maxn if maxn else self.config.get("maxn", 6)

        # Tokenize into subtokens
        tokens = self.tokenizer(f"<{token}>", minn, maxn) if minn else None

        # Create token ids list and return
        return np.concatenate(
            (
                np.array([self.tokens[token]] if token in self.tokens else [], dtype=np.uint32),
                self.ids(tokens) if tokens else np.array([], dtype=np.uint32),
            )
        )

    def ids(self, tokens):
        """
        Gets token ids for a list of tokens. This method applies a hash function then
        uses that to route it to an available bucket id within the input matrix.

        Args:
            tokens: list of tokens

        Returns:
            list of token ids
        """

        return len(self.tokens) + (self.hasher(tokens) % self.config["bucket"])
