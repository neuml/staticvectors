"""
Factory module
"""

import json

from .database import Database
from .retriever import Retriever
from .tensors import Tensors


class StorageFactory:
    """
    Methods to create Storage instances.
    """

    @staticmethod
    def create(path, storage=None, create=False, storefile=False):
        """
        Creates a Storage instance.

        Args:
            path: model path
            storage: storage backend
            create: create model path locally if True, this is for writing models
            storefile: if true, this storage instance stores all it's content a single file

        Returns:
            Storage
        """

        # Get storage method
        storefile = storefile or Database.isdatabase(path)
        storage = StorageFactory.storage(path, storage, storefile)

        # SQLite model storage
        if storage == "sqlite":
            return Database(path, create, storefile)

        # Default model is safetensors
        return Tensors(path, create, storefile)

    @staticmethod
    def storage(path, storage, storefile):
        """
        Derives the storage method, if it's not provided.

        Args:
            path: model path
            storage: storage backend
            storefile: if true, this storage instance stores all it's content a single file
        """

        # Infer storage with path
        if not storage:
            # The database storage format supports storing the entire model as a database
            if storefile:
                return "sqlite"

            # Default method
            storage = "safetensors"

            # Load config.json
            config = StorageFactory.config(f"{path}/config.json")
            if config and config.get("storage") == "sqlite":
                storage = "sqlite"

        return storage

    @staticmethod
    def config(path):
        """
        Loads a JSON config file from a local or remote model on the Hugging Face Hub.

        Args:
            path: model path

        Returns:
            config
        """

        # Download file and parse JSON
        config = None
        try:
            retriever = Retriever()
            path = retriever(path)
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)

        # Ignore this error - invalid repo or directory
        except OSError:
            pass

        return config
