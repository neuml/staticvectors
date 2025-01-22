"""
Tokenizer module
"""


class NgramTokenizer:
    """
    Tokenizes text into n-grams.
    """

    def __call__(self, text, minn, maxn):
        """
        Tokenizes text into n-grams.

        Args:
            text: input text
            minn: min ngram size
            maxn: max ngram size

        Returns:
            list of n-grams for text
        """

        ngrams = []
        for x in range(len(text)):
            for n in range(minn, maxn + 1):
                if x + n <= len(text):
                    ngrams.append(text[x : x + n])

        return ngrams
