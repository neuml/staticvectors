"""
Pre-trained vector text model module
"""

import os
import tempfile
import unittest

import numpy as np

from staticvectors import StaticVectors, TextConverter


class TestText(unittest.TestCase):
    """
    Pre-trained vector text model tests.
    """

    def testConvert(self):
        """
        Test converting a vectors text file
        """

        # Generate training data
        train, data = self.build()
        path = os.path.join(tempfile.gettempdir(), "vectors-text")

        # Convert the model
        converter = TextConverter()
        converter(train, path)

        # Load the model
        model = StaticVectors(path)
        self.assertTrue(np.allclose(model.vectors, data, atol=1e-4))

        # Check out of vocab term
        self.assertEqual(model.embeddings(["theand"]).shape[1], 10)

        # Validate normalize method
        data = np.random.rand(5)
        model.normalize(data)
        self.assertTrue(data.shape, (5,))

    def build(self):
        """
        Builds text vectors training file for testing.

        Returns:
            path to training data
        """

        # Generate a sample vectors text file
        train = os.path.join(tempfile.gettempdir(), "vectors-text.txt")

        tokens = ["the", "hello", "and", "you", "I"]
        data = np.random.uniform(-1, 1, (5, 10))
        with open(train, "w", encoding="utf-8") as f:
            f.write("5 10\n")

            for x, row in enumerate(data):
                f.write(f"{tokens[x]} ")
                np.savetxt(f, [row], fmt="%.4f")

        return train, data
