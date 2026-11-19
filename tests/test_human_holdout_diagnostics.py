import copy
import unittest

from kis.temporal import TemporalConfig
from tools.human_holdout_diagnostics import classify


class HumanHoldoutDiagnosticsTest(unittest.TestCase):
    def fixture(self):
        row = {"task": "sequence", "target_video_in_pool": True, "failures": {},
               "query": {"video_id": "L01_V001", "event_relevant_keyframes": [[1], [2]]},
               "modes": {"baseline": {"metrics": {"known_path_hit_at_20": 0}}}}
        evidence = {"actions": ["first", "second"], "frames": [
            {"keyframe_n": 1, "pts_time": 1}, {"keyframe_n": 2, "pts_time": 2}],
            "visual": [[0.9, 0], [0, 0.9]], "semantic": None}
        return row, evidence

    def test_missing_pool_support_and_selection_are_different_stages(self):
        row, evidence = self.fixture()
        self.assertEqual(classify(row, evidence, TemporalConfig()), "alignment_chose_other_frames_than_known_positives")
        low = copy.deepcopy(evidence)
        low["visual"][1][1] = 0.1
        self.assertEqual(classify(row, low, TemporalConfig()), "known_positive_support_failure")
        row["target_video_in_pool"] = False
        self.assertEqual(classify(row, None, TemporalConfig()), "target_video_missing_from_pool")

    def test_sparse_known_hit_not_reclassified_as_failure(self):
        row, evidence = self.fixture()
        row["modes"]["baseline"]["metrics"]["known_path_hit_at_20"] = 1
        self.assertEqual(classify(row, evidence, TemporalConfig()), "known_positive_hit_at_20")

    def test_temporal_gap_and_missing_local_frame_are_not_caption_failures(self):
        row, evidence = self.fixture()
        evidence["frames"][1]["pts_time"] = 100
        self.assertEqual(classify(row, evidence, TemporalConfig()), "known_positive_temporal_gap_failure")
        evidence["frames"][1]["keyframe_n"] = 3
        self.assertEqual(classify(row, evidence, TemporalConfig()), "known_positive_missing_from_local_frames")


if __name__ == "__main__":
    unittest.main()
