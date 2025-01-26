"""
Retriever module
"""

import os

from huggingface_hub import hf_hub_download


class Retriever:
    """
    Retrieves files from the Hugging Face Hub. Files are only downloaded if they don't already exist.
    """

    def __call__(self, path):
        """
        Retrieves file at path locally. Skips downloading if the file already exists.

        Args:
            path: requested file path

        Returns:
            local path to file
        """

        # Check if this is a local path, otherwise download from the HF Hub
        return path if os.path.exists(path) or path.startswith("/") else self.download(path)

    def download(self, path):
        """
        Downloads path from the Hugging Face Hub.

        Args:
            path: full model path

        Returns:
            local cached model path
        """

        # Split into parts
        parts = path.split("/")

        # Calculate repo id split
        repo = 2 if len(parts) > 2 else 1

        # Download and cache file
        return hf_hub_download(repo_id="/".join(parts[:repo]), filename="/".join(parts[repo:]))
