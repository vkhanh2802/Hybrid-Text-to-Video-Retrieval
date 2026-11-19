import unittest

from tools.vlm_result_report import timing_stats, transitions


class VLMResultReportTest(unittest.TestCase):
    def test_paired_wins_and_losses_not_hidden_by_equal_mean(self):
        rows = [{"case_id": name, "baseline_metrics": {"hit": a}, "vlm_metrics": {"hit": b}}
                for name, a, b in (("wins", 0, 1), ("losses", 1, 0), ("same", 1, 1))]
        result = transitions(rows, "hit")
        self.assertEqual(result["counts"], {"improved": 1, "regressed": 1, "unchanged": 1})
        self.assertEqual(result["case_ids"]["regressed"], ["losses"])

    def test_timing_empty_and_median_do_not_invent_latency(self):
        self.assertIsNone(timing_stats([])["median_seconds"])
        self.assertEqual(timing_stats([3, 1, 2])["sum_seconds"], 6)
        self.assertEqual(timing_stats([3, 1, 2])["median_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
