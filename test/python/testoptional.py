"""
Optional module tests
"""

import sys
import unittest


# pylint: disable=C0415
class TestOptional(unittest.TestCase):
    """
    Optional tests. Simulates optional dependencies not being installed.
    """

    @classmethod
    def setUpClass(cls):
        """
        Simulate optional packages not being installed
        """

        modules = ["fasttext", "nanopq"]

        # Get handle to all currently loaded staticvectors modules
        modules = modules + [key for key in sys.modules if key.startswith("staticvectors")]
        cls.modules = {module: None for module in modules}

        # Replace loaded modules with stubs. Save modules for later reloading
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]

            # Remove staticvectors modules. Set optional dependencies to None to prevent reloading.
            if "staticvectors" in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    def testTrain(self):
        """
        Test missing training dependencies
        """

        from staticvectors import StaticVectorsTrainer, TextConverter

        with self.assertRaises(ImportError):
            StaticVectorsTrainer()

        with self.assertRaises(ImportError):
            TextConverter()
