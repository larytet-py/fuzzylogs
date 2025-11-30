import unittest
from markovchain import MarkovChain


class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        # Base English-like words
        self.base_dict = ["cat", "dog", "mouse", "house", "car", "apple", "banana"]

        # Domain-specific words (e.g. tech / medical jargon)
        self.domain = ["microservice", "kubernetes", "hyperparameter"]

        self.mc = MarkovChain(self.base_dict, domain_words=self.domain, order=3)

    # ------------------------------------------------------------------ #
    # Basic construction & introspection
    # ------------------------------------------------------------------ #

    def test_order_and_known_words(self):
        self.assertEqual(self.mc.order_n(), 3)
        # All base words should be in known_words
        for w in self.base_dict:
            self.assertIn(w.lower(), self.mc.known_words())

    def test_known_alphabet_not_empty(self):
        alphabet = self.mc.known_alphabet()
        self.assertTrue(len(alphabet) > 0)
        # End-of-word token should be known
        self.assertIn("$", alphabet)

    # ------------------------------------------------------------------ #
    # Probability / scoring behavior
    # ------------------------------------------------------------------ #

    def test_probability_positive_for_known_word(self):
        p = self.mc.probability("banana")
        self.assertGreater(p, 0.0)

    def test_log_probability_for_empty_word_is_minus_inf(self):
        self.assertEqual(self.mc.log_probability(""), float("-inf"))

    def test_score_compares_english_like_vs_gibberish(self):
        english_like = "banana"
        gibberish = "xqzpt"

        score_en = self.mc.score(english_like)
        score_gib = self.mc.score(gibberish)

        # English-like word should have higher per-character log-probability
        self.assertGreater(score_en, score_gib)

    def test_is_english_like_flag(self):
        # We don't rely on a specific numeric threshold,
        # just that an obviously English word passes and gibberish fails
        self.assertTrue(self.mc.is_english_like("banana"))
        self.assertFalse(self.mc.is_english_like("xqzpt"))

    # ------------------------------------------------------------------ #
    # Domain words behavior
    # ------------------------------------------------------------------ #

    def test_domain_words_are_in_known_words(self):
        for w in self.domain:
            self.assertIn(w.lower(), self.mc.known_words())

    def test_add_domain_words_increases_score(self):
        word = "OpenAI"

        # Model without 'OpenAI' as domain word
        base_mc = MarkovChain(self.base_dict, domain_words=self.domain, order=3)
        score_before = base_mc.score(word)

        # Now add 'OpenAI' and refit
        base_mc.add_domain_words([word])
        score_after = base_mc.score(word)

        # Score should increase once the word is part of the training data
        self.assertGreater(score_after, score_before)

    # ------------------------------------------------------------------ #
    # Example-usage equivalence sanity check
    # ------------------------------------------------------------------ #

    def test_example_usage_runs_without_errors(self):
        words_to_check = ["banana", "xqzpt", "kubernetes", "asdfghjklq"]

        # Just make sure the calls don't raise and return sensible types
        for word in words_to_check:
            lp = self.mc.log_probability(word)
            score = self.mc.score(word)
            flag = self.mc.is_english_like(word)

            self.assertIsInstance(lp, float)
            self.assertIsInstance(score, float)
            self.assertIsInstance(flag, bool)

    def test_add_domain_words_updates_known_words(self):
        original_len = len(self.mc.known_words())
        self.mc.add_domain_words(["ChatGPT"])
        new_len = len(self.mc.known_words())

        self.assertGreater(new_len, original_len)
        self.assertIn("chatgpt", self.mc.known_words())


if __name__ == "__main__":
    unittest.main()
