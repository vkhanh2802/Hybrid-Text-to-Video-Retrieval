import unittest

from tools.benchmark_school_retrieval import GROUPS, QUERIES, summary


class SchoolRetrievalBenchmarkTest(unittest.TestCase):
    def test_scope_is_explicit_and_excludes_missing_l18(self):
        self.assertEqual(len(GROUPS), 19)
        self.assertNotIn("L18", GROUPS)
        self.assertEqual(len(set(QUERIES)), 10)

    def test_percentiles_use_linear_interpolation(self):
        result = summary([4., 1., 2., 3.])
        self.assertEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertEqual(result["count"], 4)


if __name__ == "__main__":
    unittest.main()
