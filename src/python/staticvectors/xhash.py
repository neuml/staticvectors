"""
XHash module
"""

import numpy as np


class FNV:
    """
    Calculates FNV hashes. See link below for more.

    https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
    """

    def __call__(self, tokens):
        """
        Calculates FNV hashes for each input token.

        Args:
            tokens list of tokens

        Returns:
            np.array([hash ids])
        """

        # Don't log overflow error, it's intended behavior
        with np.errstate(over="ignore"):
            # Join into single buffer to improve buffer load time
            data = ("\0".join(tokens) + "\0").encode("utf-8")

            # Review comments found at this link for rationale on this conversion
            # https://github.com/facebookresearch/fastText/blob/main/src/dictionary.cc#L157
            data = np.frombuffer(data, dtype=np.int8).astype(np.uint32)

            # Get buffer separator indices
            indices = np.where(data == 0x00)[0]

            # Calculate fnv hashes
            results, index = [], 0
            for separator in indices:
                fnv = np.uint32(2166136261)
                for x in data[index:separator]:
                    fnv = (fnv ^ x) * 16777619

                results.append(fnv)
                index = separator + 1

            return np.array(results)
