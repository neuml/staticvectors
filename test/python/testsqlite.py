"""
SQLite vector model module
"""

import os
import sqlite3
import tempfile
import time
import unittest

import numpy as np

from staticvectors import StaticVectorsConverter, StaticVectors


class TestSQLite(unittest.TestCase):
    """
    SQLite vector model tests.
    """

    def testEmbeddings(self):
        """
        Test a SQLite model for embeddings generation
        """

        # Generate test model
        model, data = self.build()

        # Test both file and directory storage
        for storefile, path in [(True, "vectors.sqlite"), (False, "vectors-sqlite")]:
            path = os.path.join(tempfile.gettempdir(), path)
            converter = StaticVectorsConverter()
            converter(model, path, storage="sqlite", storefile=storefile)

            # Load the model
            sv = StaticVectors(path)
            self.assertTrue(np.allclose(np.array(sv.vectors), data, atol=1e-5))

            # Compare generated embeddings
            self.assertTrue(np.allclose(sv.embeddings(["hello"])[0], data[1], atol=1e-5))

            # pylint: disable=W0104
            with self.assertRaises(IndexError):
                sv.tokens["abc1234"]

    def testLegacy(self):
        """
        Test directly loading a legacy magnitude database.
        """

        # Generate test model
        model, data = self.build()

        # Load the model
        sv = StaticVectors(model)

        # Compare generated embeddings
        self.assertTrue(np.allclose(sv.embeddings(["hello"])[0], data[1], atol=1e-5))

    def build(self):
        """
        Builds a SQLite database for testing.

        Returns:
            path to SQLite vectors model
        """

        # Generate a test magnitude file
        model = os.path.join(tempfile.gettempdir(), f"vectors-{time.time()}.magnitude")
        connection = sqlite3.connect(model, check_same_thread=False)

        connection.execute(
            """
            CREATE TABLE magnitude_format (
                key TEXT,
                value INTEGER
            )
        """
        )

        connection.execute(
            """
            CREATE TABLE magnitude (
                key TEXT,
                dim_0 INTEGER,
                dim_1 INTEGER,
                dim_2 INTEGER
            )
        """
        )

        # Insert config
        connection.execute("INSERT INTO magnitude_format VALUES('size', 5)")
        connection.execute("INSERT INTO magnitude_format VALUES('dim', 3)")
        connection.execute("INSERT INTO magnitude_format VALUES('precision', 5)")

        # Generate data
        data = np.random.rand(5, 3)

        # Normalize data (to match Magnitude logic)
        data /= np.linalg.norm(data, axis=1)[:, np.newaxis]

        # Convert to integers (to match Magnitude logic)
        data = (data * 10**5).astype(np.int32)

        tokens = ["the", "hello", "and", "you", "I"]

        # Insert data
        for x in range(5):
            connection.execute(
                f"""
                INSERT INTO magnitude 
                VALUES({",".join([f"'{tokens[x]}'"] + [str(y) for y in data[x]])})
            """
            )

        connection.commit()
        connection.close()

        # Return database and data
        return model, data / 10**5
