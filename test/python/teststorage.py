"""
Storage test module
"""

import unittest

from staticvectors import Storage, StorageFactory


class TestStorage(unittest.TestCase):
    """
    Storage tests.
    """

    def testNotImplemented(self):
        """
        Test storage not implemented errors
        """

        storage = Storage(None)
        self.assertRaises(NotImplementedError, storage.loadtensors)
        self.assertRaises(NotImplementedError, storage.savetensors, None, None, None)
        self.assertRaises(NotImplementedError, storage.storage)

    def testInvalidPath(self):
        """
        Test an invalid storage path
        """

        self.assertIsNone(StorageFactory.config("invalid"))
