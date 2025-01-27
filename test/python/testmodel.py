"""
Model module
"""

import unittest

from staticvectors import StaticVectors


class TestModel(unittest.TestCase):
    """
    Model tests.
    """

    def testGenerate(self):
        """
        Test generating a vector for an out of vocabulary token
        """

        # Create model for testing
        model1, model2 = StaticVectors(), StaticVectors()

        # Set the dimensions for testing
        model1.config = {"dim": 100}
        model2.config = {"dim": 100}

        # Generate vectors from two different models for same token and test they are the same
        self.assertTrue((model1.generate("abc") == model2.generate("abc")).all())

        # Repeat and confirm it's still the same
        self.assertTrue((model1.generate("abc") == model2.generate("abc")).all())
