import unittest
from jaccard import Jaccard, default_tokenizer


class TestJaccard(unittest.TestCase):
    def setUp(self):
        self.j = Jaccard()

    def test_default_tokenizer_basic(self):
        tokens = default_tokenizer("foo bar   baz")
        self.assertEqual(tokens, ["foo", "bar", "baz"])

    def test_similarity_identical_strings(self):
        s = "foo bar baz"
        self.assertAlmostEqual(self.j.similarity(s, s), 1.0)

    def test_similarity_disjoint(self):
        a = "foo bar"
        b = "baz qux"
        self.assertAlmostEqual(self.j.similarity(a, b), 0.0)

    def test_similarity_partial_overlap(self):
        a = "foo bar baz"
        b = "bar baz qux"

        sim = self.j.similarity(a, b)
        # tokens: a={foo,bar,baz}, b={bar,baz,qux}
        # intersection={bar,baz} -> 2, union={foo,bar,baz,qux} -> 4
        # J=2/4=0.5
        self.assertAlmostEqual(sim, 0.5)

    def test_similarity_is_symmetric(self):
        a = "foo bar baz"
        b = "bar baz qux"
        self.assertAlmostEqual(self.j.similarity(a, b), self.j.similarity(b, a))

    def test_empty_strings(self):
        self.assertEqual(self.j.similarity("", ""), 0.0)

    def test_one_empty_one_nonempty(self):
        self.assertEqual(self.j.similarity("", "foo bar"), 0.0)

    def test_is_match_true(self):
        a = "foo bar baz"
        b = "foo baz qux"
        sim = self.j.similarity(a, b)
        self.assertTrue(self.j.is_match(a, b, threshold=sim - 1e-9))

    def test_is_match_false(self):
        a = "foo bar"
        b = "baz qux"
        self.assertFalse(self.j.is_match(a, b, threshold=0.1))

