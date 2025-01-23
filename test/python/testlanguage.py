"""
Language module tests
"""

import os
import unittest
import tempfile


from staticvectors import FastTextConverter, FileSystem, StaticVectors


class TestLanguage(unittest.TestCase):
    """
    Language model tests
    """

    def testConvert(self):
        """
        Test converting an existing FastText model for language detection to a StaticVectors model
        """

        # Get FastText language detection model
        langmodel = FileSystem(None).download("julien-c/fasttext-language-id/lid.176.bin")
        path = os.path.join(tempfile.gettempdir(), "langid")

        # Test both the standard and a quantized model
        for quantize in [None, 2]:
            converter = FastTextConverter()
            converter(langmodel, path, quantize=quantize)

            model = StaticVectors(path)

            for text, language in self.content():
                self.assertEqual(model.predict(text)[0][0], language)

    def content(self):
        """
        Returns a list of text-language pairs for testing.

        Returns:
            list of text language pairs
        """

        return [
            ("Hello", "en"),
            ("txtai is an all-in-one database for semantic search", "en"),
            ("txtai è un database tutto in uno per la ricerca semantica", "it"),
            ("txtai เป็นฐานข้อมูล all-in-one สำหรับการค้นหาความหมาย", "th"),
            ("txtai はセマンティック検索のためのオールインワン データベースです", "ja"),
            ("txtai הוא מסד נתונים הכל-באחד לחיפוש סמנטי", "he"),
            ("txtai هي قاعدة بيانات شاملة للبحث الدلالي", "ar"),
        ]
