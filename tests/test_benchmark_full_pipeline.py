import unittest

from tools.benchmark_full_pipeline import result_checks


class FullPipelineTimingTest(unittest.TestCase):
    def response(self):
        return {"results": [{"video_id": "L01_V001", "frame_id": i} for i in range(100)]}

    def test_requires_real_qwen_score_and_distinct_100_frames(self):
        result = result_checks(self.response(), [{"valid_score": True}] * 20)
        self.assertTrue(result["complete_100"])
        self.assertTrue(result["qwen_executed"])
        self.assertTrue(result["qwen_budget_ok"])

    def test_skipped_invalid_and_over_budget_cannot_pass(self):
        self.assertFalse(result_checks(self.response(), [])["qwen_executed"])
        self.assertFalse(result_checks(self.response(), [{"valid_score": False}])["qwen_executed"])
        self.assertFalse(result_checks(self.response(), [{"valid_score": True}] * 21)["qwen_budget_ok"])
        response = self.response()
        response["results"][1] = response["results"][0]
        self.assertFalse(result_checks(response, [{"valid_score": True}])["complete_100"])


if __name__ == "__main__":
    unittest.main()
