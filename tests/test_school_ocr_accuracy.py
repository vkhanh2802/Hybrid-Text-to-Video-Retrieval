import unittest

from tools.school_ocr_accuracy import metrics


class SchoolOCRAccuracyTest(unittest.TestCase):
    def test_known_positive_metrics_never_call_unjudged_negative(self):
        query = {"positives": [{"video_id": "L01_V001", "keyframe_n": 2}]}
        rows = [{"video_id": "L02_V001", "keyframe_n": 2},
                {"video_id": "L01_V001", "keyframe_n": 1},
                {"video_id": "L01_V001", "keyframe_n": 2}]
        m = metrics(rows, query)
        self.assertEqual(m["first_known_positive_rank"], 3)
        self.assertEqual(m["known_hit_at_1"], 0)
        self.assertEqual(m["known_hit_at_5"], 1)
        self.assertAlmostEqual(m["known_mrr_at_100"], 1 / 3)
        self.assertNotIn("precision", m)
        self.assertNotIn("accuracy", m)

    def test_missing_known_target_does_not_invent_a_rank(self):
        m = metrics([], {"positives": [{"video_id": "L01_V001", "keyframe_n": 1}]})
        self.assertIsNone(m["first_known_positive_rank"])
        self.assertEqual(m["known_hit_at_100"], 0)


if __name__ == "__main__":
    unittest.main()
