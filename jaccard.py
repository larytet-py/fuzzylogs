# jaccard.py

from __future__ import annotations

from typing import Callable, Iterable, Set, List
import re


def default_tokenizer(text: str) -> List[str]:
    """
    Default tokenizer: split on whitespace into non-empty tokens.
    """
    if not text:
        return []
    # Split on any whitespace, filter out empty tokens
    return [t for t in text.split() if t]


class Jaccard:
    """
    Utility class to compute token-based Jaccard similarity between texts.

    J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|

    Tokens are produced by the provided tokenizer, which must return
    an iterable of string tokens.
    """

    def __init__(self, tokenizer: Callable[[str], Iterable[str]] = default_tokenizer):
        self.tokenizer = tokenizer

    def tokens(self, text: str) -> Set[str]:
        """
        Tokenize a string and return a set of tokens.
        """
        return set(self.tokenizer(text))

    def similarity(self, a: str, b: str) -> float:
        """
        Compute the Jaccard similarity between two strings based on tokens.

        - If both token sets are empty, returns 0.0 (undefined Jaccard, but
          0.0 is a practical default).
        - Similarity is in [0.0, 1.0].
        """
        set_a = self.tokens(a)
        set_b = self.tokens(b)

        if not set_a and not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def is_match(self, a: str, b: str, threshold: float) -> bool:
        """
        Return True if Jaccard similarity between a and b is >= threshold.
        """
        return self.similarity(a, b) >= threshold
