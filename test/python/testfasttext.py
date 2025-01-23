"""
FastText model module
"""

import os
import tempfile
import unittest

import numpy as np

from staticvectors import StaticVectorsTrainer


class TestFastText(unittest.TestCase):
    """
    FastText model tests.
    """

    def testEmbeddings(self):
        """
        Test a FastText model for embeddings generation
        """

        # Train a new model
        path = os.path.join(tempfile.gettempdir(), "vectors-fasttext")
        trainer = StaticVectorsTrainer()
        sv, model = trainer("README.md", size=100, mincount=1, path=path, load=True, bucket=500)

        # Compare StaticVectors model with trained FastText model
        self.assertTrue(np.allclose(sv.vectors, model.get_input_matrix()))

        # Compare generated embeddings
        self.assertTrue(np.allclose(sv.embeddings(["hello"], normalize=False)[0], model.get_word_vector("hello")))

        # Ensure predict throws an error
        with self.assertRaises(ValueError):
            sv.predict("hello")

    def testLabels(self):
        """
        Test a FastText classification model
        """

        # Generate labeled training data
        train = os.path.join(tempfile.gettempdir(), "vectors-text.txt")

        with open(train, "w", encoding="utf-8") as f:
            f.write("__label__positive This is a great!\n")
            f.write("__label__positive This is a great!\n")
            f.write("__label__positive This is a great!\n")
            f.write("__label__negative Bad news.\n")

        for loss in ["hs", "softmax"]:
            # Train the model
            path = os.path.join(tempfile.gettempdir(), f"vectors-fasttext-{loss}")
            trainer = StaticVectorsTrainer()
            sv, model = trainer(train, size=100, mincount=1, path=path, classification=True, load=True, loss=loss)

            # Test the classifier
            self.assertEqual(sv.predict("great")[0][0], "positive")
            self.assertEqual(sv.predict("great")[0][0], model.predict(["great"])[0][0][0].replace("__label__", ""))

        # Test unsupported loss
        with self.assertRaises(ValueError):
            path = os.path.join(tempfile.gettempdir(), "vectors-fasttext-ns")
            trainer = StaticVectorsTrainer()
            sv, model = trainer(train, size=100, mincount=1, path=path, classification=True, load=True, loss="ns")

    def testQuantization(self):
        """
        Test converting and quantizing a model
        """

        # Train a new model
        path = os.path.join(tempfile.gettempdir(), "vectors-fasttext-quant")
        trainer = StaticVectorsTrainer()
        sv, model = trainer("README.md", size=100, mincount=1, path=path, quantize=1, load=True, bucket=500)

        # Compare StaticVectors model with trained FastText model
        self.assertEqual(sv.vectors.shape[0], model.get_input_matrix().shape[0])

        # Test an embeddings vector
        self.assertEqual(sv.embeddings(["hello"]).shape, (1, 100))
