"""
Loss module
"""


class Loss:
    """
    Base loss.
    """

    def __init__(self, counts, weights):
        """
        Creates a new Loss instance.

        Args:
            counts: frequency count data
            weights: output weight matrix
        """

        self.counts = counts
        self.weights = weights
