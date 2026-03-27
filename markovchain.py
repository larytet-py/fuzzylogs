from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence, Dict, DefaultDict, List, Set
import math


class MarkovChain:
    """
    Character-level Markov model trained on a list of words.

    Typical use: given a word, estimate how "English-like" it is
    based on the training dictionary plus domain-specific words.
    """

    def __init__(
        self,
        dictionary_words: Iterable[str],
        *,
        domain_words: Sequence[str] | None = None,
        order: int = 3,
        smoothing: float = 1.0,
        lowercase: bool = True,
        default_score_threshold: float = -3.5,
    ) -> None:
        """
        Parameters
        ----------
        dictionary_words
            Iterable of base words (e.g., English words) to train on.
        domain_words
            Optional list of domain-specific words to include initially.
        order
            Markov order (n-gram length) on characters. Typical: 2–4.
        smoothing
            Additive smoothing constant for transition probabilities
            (Laplace smoothing).
        lowercase
            If True, all words are lowercased before modeling.
        default_score_threshold
            Per-character log-probability below which tokens are fuzzed.
        """
        self.order = order
        self.smoothing = smoothing
        self.lowercase = lowercase
        self.default_score_threshold = default_score_threshold

        self._start_token = "^"
        self._end_token = "$"

        self._words: List[str] = []
        self._alphabet: Set[str] = set()
        self._transition_counts: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._context_totals: Dict[str, float] = {}
        self._unknown_probability: float = 1e-12

        for word in dictionary_words:
            self._add_word_internal(word)

        if domain_words:
            for word in domain_words:
                self._add_word_internal(word)

        self._fit_markov_model()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def probability(self, word: str) -> float:
        """
        Return P(word) under the learned character-level Markov model.
        """
        if not word:
            return 0.0
        return math.exp(self.log_probability(word))

    def log_probability(self, word: str) -> float:
        """
        Return log P(word) under the model using character transitions.
        """
        if not word:
            return float("-inf")

        normalized = self._normalize(word)
        padded = self._start_token * (self.order - 1) + normalized + self._end_token
        chars = list(padded)

        log_prob = 0.0
        for idx in range(len(chars) - self.order + 1):
            context = "".join(chars[idx : idx + self.order - 1])
            next_char = chars[idx + self.order - 1]
            log_prob += self._log_prob_next_char(context, next_char)

        return log_prob

    def score(self, word: str) -> float:
        """
        Return a per-character log-probability score to compare token length.
        """
        if not word:
            return float("-inf")
        return self.log_probability(word) / len(word)

    def is_english_like(self, word: str, threshold: float | None = None) -> bool:
        """
        Decide whether the token should be kept instead of fuzzed away.
        """
        if threshold is None:
            threshold = self.default_score_threshold
        return self.score(word) >= threshold

    def add_domain_words(self, words: Iterable[str]) -> None:
        """
        Add domain-specific words and refit the transition counts.
        """
        changed = False
        for word in words:
            if self._add_word_internal(word):
                changed = True
        if changed:
            self._fit_markov_model()

    def known_words(self) -> List[str]:
        """Return all words currently known to the model."""
        return list(self._words)

    def order_n(self) -> int:
        """Return the Markov order used by the model."""
        return self.order

    def known_alphabet(self) -> List[str]:
        """Return the characters observed in the training data."""
        return sorted(self._alphabet)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _normalize(self, word: str) -> str:
        normalized = word.strip()
        if self.lowercase:
            normalized = normalized.lower()
        return normalized

    def _add_word_internal(self, word: str) -> bool:
        normalized = self._normalize(word)
        if not normalized:
            return False
        self._words.append(normalized)
        for char in normalized:
            self._alphabet.add(char)
        self._alphabet.add(self._end_token)
        return True

    def _fit_markov_model(self) -> None:
        self._transition_counts.clear()
        self._context_totals.clear()

        if not self._words:
            return

        for word in self._words:
            padded = self._start_token * (self.order - 1) + word + self._end_token
            chars = list(padded)
            for idx in range(len(chars) - self.order + 1):
                context = "".join(chars[idx : idx + self.order - 1])
                next_char = chars[idx + self.order - 1]
                self._transition_counts[context][next_char] += 1.0

        for context, next_counts in self._transition_counts.items():
            self._context_totals[context] = sum(next_counts.values())

    def _log_prob_next_char(self, context: str, next_char: str) -> float:
        counts = self._transition_counts.get(context)
        if not counts:
            return math.log(self._unknown_probability)

        total = self._context_totals[context]
        vocab_size = max(len(self._alphabet), 1)

        if next_char not in self._alphabet:
            return math.log(self._unknown_probability)

        count_next = counts.get(next_char, 0.0)
        probability = (count_next + self.smoothing) / (total + self.smoothing * vocab_size)
        if probability <= 0.0:
            probability = self._unknown_probability

        return math.log(probability)
