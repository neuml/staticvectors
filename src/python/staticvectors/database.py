"""
Database module
"""

import os
import sqlite3

import numpy as np


class Database:
    """
    SQLite tensor storage format. Also supports the legacy magnitude-light databases (https://github.com/neuml/magnitude).
    """

    @staticmethod
    def isdatabase(path):
        """
        Checks if this is a SQLite vectors database.

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

    def __init__(self, path, create=False):
        """
        Loads a tensors database file.

        Args:
            path: path to file
            create: remove existing database if True, this is for writing models
        """

        # Delete existing file, if building a new database
        if create and os.path.exists(path):
            os.remove(path)

        # Database connection
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.cursor = self.connection.cursor()

        # Model parameters
        self.vectortable, self.configtable = None, None
        self.total, self.dimensions = None, None
        self.columns = None
        self.divisor = None

        # Load parameters
        if not create:
            self.parameters()

    def __getitem__(self, indices):
        """
        Enables in-place access to an existing SQLite vector database. This method adjusts the input 0-based
        indices to 1-based. This supports a single index or a list of indexes. The return value will match the input
        type.

        Args:
            indices: index or list of indices

        Returns:
            vector(s) for indices
        """

        embeddings = []
        indices = indices if isinstance(indices, (tuple, list)) else (indices,)

        for index in indices:
            # Index out of range error
            if index >= self.total:
                raise IndexError

            # SQLite ids are 1-based, adjust to 1-based
            index += 1

            # Lookup vector. Convert integer to float.
            self.cursor.execute(f"SELECT {self.columns} FROM {self.vectortable} WHERE rowid = ?", [index])
            vector = np.array(self.cursor.fetchone(), dtype=np.float32) / self.divisor

            # Replace 0's with a small number. This is due to storing integers in the database
            vector = np.where(vector == 0, 1e-15, vector)

            # Save vector
            embeddings.append(vector)

        # Return type should match indices type (list vs single index)
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]

    def __len__(self):
        """
        Gets the total size of the database.

        Returns:
            total database size
        """

        return self.total

    def save(self, vectors, tokens):
        """
        Saves vectors and tokens to the SQLite vector database.

        Args:
            vectors: model vectors
            tokens: model tokens
        """

        # Default precision
        precision = 7

        # Create dimension columns
        columns = ", ".join(f"dim_{x} INTEGER" for x in range(vectors.shape[1]))

        # Create tables and indexes
        self.cursor.execute("CREATE TABLE config (key TEXT, value INTEGER)")
        self.cursor.execute(f"CREATE TABLE vectors (key TEXT, {columns})")
        self.cursor.execute("CREATE INDEX vectors_key ON vectors(key)")

        # Insert config
        for name, value in {"size": vectors.shape[0], "dim": vectors.shape[1], "precision": precision}.items():
            self.cursor.execute("INSERT INTO config VALUES(?, ?)", [name, value])

        # Normalize data
        vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Convert to integers
        vectors = (vectors * 10**precision).astype(np.int32)

        # Insert vectors
        values = ",".join("?" for _ in range(vectors.shape[1] + 1))
        for token, tokenid in tokens.items():
            # SQL bind parameters
            row = [token] + [int(y) for y in vectors[tokenid]]

            # Insert row, adjust token index to be 1-based
            self.cursor.execute(f"INSERT INTO vectors VALUES ({values})", row)

        # Save results
        self.connection.commit()
        self.connection.close()

    def config(self):
        """
        Builds model configuration.

        Returns:
            model configuration
        """

        # Model configuration
        return {"format": "sqlite", "source": os.path.basename(self.path), "total": self.total, "dim": self.dimensions}

    def tokens(self):
        """
        Gets all tokens as a dictionary of {token: token id}.

        Returns:
            {token: token id}
        """

        return Tokens(self.cursor, self.vectortable)

    def parameters(self):
        """
        Sets parameters stored in the SQLite database on this instance.
        """

        # Check if this database was generated by magnitude or this library
        if self.cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='magnitude'").fetchone():
            self.vectortable, self.configtable = "magnitude", "magnitude_format"
        else:
            self.vectortable, self.configtable = "vectors", "config"

        # Configuration parameters
        self.total = self.cursor.execute(f"SELECT value FROM {self.configtable} WHERE key='size'").fetchone()[0]
        self.dimensions = self.cursor.execute(f"SELECT value FROM {self.configtable} WHERE key='dim'").fetchone()[0]
        precision = self.cursor.execute(f"SELECT value FROM {self.configtable} WHERE key='precision'").fetchone()[0]

        # Vector columns
        self.columns = ",".join(f"dim_{x}" for x in range(self.dimensions))

        # Precision divisor
        self.divisor = 10**precision


class Tokens:
    """
    Dictionary like iterface to model keys. Lookups are run against the backing database.
    """

    def __init__(self, cursor, vectortable):
        """
        Creates a new Tokens instance.

        Args:
            cursor: database cursor
            vectortable: vector table name
        """

        self.cursor = cursor
        self.vectortable = vectortable

    def __getitem__(self, key):
        """
        Gets an item by key.

        Args:
            key: token

        Returns:
            token id
        """

        row = self.cursor.execute(f"SELECT rowid - 1 FROM {self.vectortable} WHERE key = ? COLLATE BINARY", [key]).fetchone()
        if not row:
            raise IndexError

        return row[0]

    def __contains__(self, key):
        """
        Checks if key is present.

        Args:
            key: token

        Returns:
            True if key is present, False otherwise
        """

        return self.cursor.execute(f"SELECT 1 FROM {self.vectortable} WHERE key = ? COLLATE BINARY", [key]).fetchone()

    def items(self):
        """
        Item iterator.

        Returns:
            iterator of (key, value) pairs
        """

        # Iterate over all tokens. SQLite rowids are 1-based, adjust to 0-based.
        self.cursor.execute(f"SELECT key, rowid - 1 FROM {self.vectortable} ORDER BY rowid")
        return self.cursor
