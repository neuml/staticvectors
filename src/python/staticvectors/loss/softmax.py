"""
Softmax module
"""

import numpy as np

from .base import Loss


class SoftmaxLoss(Loss):
    """
    Standard softmax loss.
    """

    def __call__(self, vector, limit):
        """
        Predicts the label for vector using softmax.

        Args:
            vector: input vector
            limit: max labels to return

        Returns:
            [(label.id, score)]
        """

        # Softmax calculation
        scores = np.dot(vector, self.weights.T)
        scores = np.exp(scores) / (np.exp(scores)).sum()

        # Get up to limit scores and return
        return [(x, scores[x]) for x in np.argsort(-scores)[:limit]]
