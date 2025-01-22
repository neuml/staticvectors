"""
Database module
"""

import os
import sqlite3

import numpy as np


class Database:
    """
    Loads tensors from a SQLite file created with the legacy magnitude-light library (https://github.com/neuml/magnitude).
    """

    @staticmethod
    def isdatabase(path):
        """
        Checks if this is a legacy SQLite vectors database.

        Args:
            path: path to check

        Returns:
            True if this is a SQLite database
        """

        if isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) >= 100:
            # Read 100 byte SQLite header
            with open(path, "rb") as f:
                header = f.read(100)

            # Check for SQLite header
            return header.startswith(b"SQLite format 3\000")

        return False

    def __init__(self, path):
        """
        Loads a tensors database file.

        Args:
            path: path to file
        """

        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.cursor = self.connection.cursor()

        # Load parameters
        self.parameters()

    def __getitem__(self, indices):
        """
        Gets vectors for the input indices. This supports a single index or a list of indexes. The return
        value will match the input type.

        Args:
            indices: index or list of indices

        Returns:
            vector(s) for indices
        """

        embeddings = []
        indices = indices if isinstance(indices, (tuple, list)) else (indices,)

        for index in indices:
            # Lookup vector. Convert integer to float.
            self.cursor.execute(f"SELECT {self.columns} FROM magnitude WHERE rowid = ?", [index])
            vector = np.array(self.cursor.fetchone(), dtype=np.float32) / self.divisor

            # Replace 0's with a small number. This is due to storing integers in the database
            vector[vector == 0] = 1e-15

            # Save vector
            embeddings.append(vector)

        # Return type should match indices type (list vs single index)
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]

    def config(self):
        """
        Builds model configuration.

        Returns:
            model configuration
        """

        # Model configuration
        return {"format": "magnitude", "source": os.path.basename(self.path), "total": self.total, "dim": self.dimensions}

    def tokens(self):
        """
        Gets all tokens as a dictionary of {token: token id}.

        Returns:
            {token: token id}
        """

        # Iterate over all tokens. Magnitude ids are 1-based, adjust to 0-based.
        self.cursor.execute("SELECT key, rowid FROM magnitude ORDER BY rowid")
        return {row[0]: row[1] - 1 for row in self.cursor}

    def parameters(self):
        """
        Sets parameters stored in the SQLite database on this instance.
        """

        # Configuration parameters
        self.total = self.cursor.execute("SELECT value FROM magnitude_format WHERE key='size'").fetchone()[0]
        self.dimensions = self.cursor.execute("SELECT value FROM magnitude_format WHERE key='dim'").fetchone()[0]
        precision = self.cursor.execute("SELECT value FROM magnitude_format WHERE key='precision'").fetchone()[0]

        # Vector columns
        self.columns = ",".join(f"dim_{x}" for x in range(self.dimensions))

        # Precision divisor
        self.divisor = 10**precision
