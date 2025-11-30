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
        """
        self.order = order
        self.smoothing = smoothing
        self.lowercase = lowercase

        # Special tokens
        self._start_token = "^"
        self._end_token = "$"

        # All words known to the model (base + domain)
        self._words: List[str] = []

        # Character vocabulary (excluding start token, including end token)
        self._alphabet: Set[str] = set()

        # Context -> next-char counts
        self._transition_counts: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Pre-summed totals for each context
        self._context_totals: Dict[str, float] = {}

        # Fallback probability for unknown contexts/chars
        self._unknown_prob: float = 1e-12

        # Initialize words list
        for w in dictionary_words:
            self._add_word_internal(w)

        if domain_words:
            for w in domain_words:
                self._add_word_internal(w)

        # Fit the Markov model
        self._fit_markov_model()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def probability(self, word: str) -> float:
        """
        Return P(word) under the learned character-level Markov model.

        This is usually very small for long words; for comparisons
        between words, prefer `score` or `log_probability`.
        """
        if not word:
            return 0.0
        log_p = self.log_probability(word)
        return math.exp(log_p)

    def log_probability(self, word: str) -> float:
        """
        Return log P(word) under the model.

        Uses character transitions with start/end tokens:
          P(word) = Π P(c_i | previous (order-1) characters)
        """
        if not word:
            return float("-inf")

        w = self._normalize(word)
        padded = self._start_token * (self.order - 1) + w + self._end_token
        chars = list(padded)

        log_p = 0.0
        for i in range(len(chars) - self.order + 1):
            context = "".join(chars[i : i + self.order - 1])
            next_char = chars[i + self.order - 1]
            log_p += self._log_prob_next_char(context, next_char)

        return log_p

    def score(self, word: str) -> float:
        """
        Return a per-character log-probability score:

            score(word) = log P(word) / len(word)

        This makes scores more comparable across different word lengths.
        Higher scores -> more "English-like" for this model.
        """
        if not word:
            return float("-inf")

        lp = self.log_probability(word)
        return lp / len(word)

    def is_english_like(self, word: str, threshold: float = -3.5) -> bool:
        """
        Heuristic yes/no decision: is this word plausibly part of English?

        Parameters
        ----------
        word
            String to test.
        threshold
            Per-character log-probability threshold. You should tune
            this empirically for your use case by checking some known
            English / non-English examples.

        Returns
        -------
        bool
            True if score(word) >= threshold.
        """
        return self.score(word) >= threshold

    def add_domain_words(self, words: Iterable[str]) -> None:
        """
        Add domain-specific words to the model and update statistics.

        After this call, probabilities and scores will reflect the
        new words as part of the "dictionary".
        """
        changed = False
        for w in words:
            if self._add_word_internal(w):
                changed = True

        if changed:
            self._fit_markov_model()

    def known_words(self) -> List[str]:
        """
        Return all words currently known to the model.
        """
        return list(self._words)

    def order_n(self) -> int:
        """Return the Markov order used by the model."""
        return self.order

    def known_alphabet(self) -> List[str]:
        """
        Return the set of characters observed in the training data
        (dictionary + domain words), including the end-of-word token '$'.
        """
        return sorted(self._alphabet)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _normalize(self, word: str) -> str:
        w = word.strip()
        if self.lowercase:
            w = w.lower()
        return w

    def _add_word_internal(self, word: str) -> bool:
        """
        Add a single word to self._words and update alphabet.
        Returns True if a non-empty word was added.
        """
        w = self._normalize(word)
        if not w:
            return False

        self._words.append(w)

        for ch in w:
            self._alphabet.add(ch)
        # Ensure end token is part of alphabet
        self._alphabet.add(self._end_token)

        return True

    def _fit_markov_model(self) -> None:
        """
        Build transition counts and context totals from the current words.
        """
        self._transition_counts.clear()
        self._context_totals.clear()

        if not self._words:
            return

        for w in self._words:
            padded = self._start_token * (self.order - 1) + w + self._end_token
            chars = list(padded)

            for i in range(len(chars) - self.order + 1):
                context = "".join(chars[i : i + self.order - 1])
                next_char = chars[i + self.order - 1]
                self._transition_counts[context][next_char] += 1.0

        for context, next_counts in self._transition_counts.items():
            self._context_totals[context] = sum(next_counts.values())

    def _log_prob_next_char(self, context: str, next_char: str) -> float:
        """
        Compute log P(next_char | context) with additive smoothing.
        """
        counts = self._transition_counts.get(context)
        if not counts:
            # Unknown context: assign a very small probability.
            return math.log(self._unknown_prob)

        total = self._context_totals[context]
        vocab_size = max(len(self._alphabet), 1)

        if next_char not in self._alphabet:
            # Character never seen in training: extremely unlikely.
            return math.log(self._unknown_prob)

        count_next = counts.get(next_char, 0.0)
        p = (count_next + self.smoothing) / (total + self.smoothing * vocab_size)

        # Safety clamp in case of numerical issues
        if p <= 0.0:
            p = self._unknown_prob

        return math.log(p)


