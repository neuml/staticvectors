"""
Factory module
"""

from .hsoftmax import HierarchicalSoftmaxLoss
from .softmax import SoftmaxLoss


class LossFactory:
    """
    Methods to create loss functions.
    """

    @staticmethod
    def create(loss, counts, weights):
        """
        Creates a loss function.

        Args:
            loss: loss name
            counts: frequency count data
            weights: output weight matrix

        Returns:
            Loss
        """

        if loss == "softmax":
            return SoftmaxLoss(counts, weights)
        if loss == "hs":
            return HierarchicalSoftmaxLoss(counts, weights)

        raise ValueError(f"{loss} not currently implemented")
