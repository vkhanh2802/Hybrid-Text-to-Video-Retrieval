import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kis.temporal import TemporalConfig, align_events, evidence_scores
from tools.temporal_multipath import compact_path
from tools.temporal_multipath_experiment import path_feasibility, replay_case, verify_hashes
from tools.human_holdout import sha


class TemporalMultipathExperimentTest(unittest.TestCase):
    def fixture(self, root, semantic):
        table = [{"video_id": "v", "keyframe_n": i+1, "frame_id": i, "pts_time": i+1}
                 for i in range(4)]
        config = TemporalConfig()
        scores, support = evidence_scores(None, semantic, (2, 4), config)
        baseline = align_events(table, scores, support, config)
        raw, saved = root / "raw.json", root / "saved.json"
        raw.write_text(json.dumps({"split": "dev", "actions": ["a", "b"], "frames": table,
                                   "visual": None, "semantic": semantic}), encoding="utf-8")
        saved.write_text(json.dumps({"modes": {"baseline": {"results":
            [compact_path("v", table, baseline, 2)] if baseline else []}}}), encoding="utf-8")
        query = {"query_id": "q", "video_id": "v", "event_relevant_keyframes": [[2], [4]]}
        return {"query": query, "actions": ["a", "b"], "candidate_videos": ["v"],
                "baseline_record": str(saved), "evidence": {"v": str(raw)}}

    def test_replay_recovers_alternative_without_using_qrels_for_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(Path(directory), [[.9, .8, .1, .1], [.1, .1, .9, .8]])
            row = replay_case(case, TemporalConfig(), "development")
            self.assertEqual(row["attribution"], "known_supported_path_lost_by_single_alignment")
            self.assertFalse(row["arms"]["single_12"]["metrics"]["known_path_candidate_coverage"])
            self.assertTrue(row["arms"]["top4_expanded48"]["metrics"]["known_path_candidate_coverage"])
            other = copy.deepcopy(case)
            other["query"]["event_relevant_keyframes"] = [[1000], [2000]]
            poisoned = replay_case(other, TemporalConfig(), "development")
            for arm in row["arms"]:
                self.assertEqual(row["arms"][arm]["results"], poisoned["arms"][arm]["results"])

    def test_gate_failure_is_not_called_single_path_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(Path(directory), [[.9, .1, .1, .1], [.1, .1, .9, .1]])
            row = replay_case(case, TemporalConfig(), "development")
            self.assertEqual(row["attribution"], "known_path_blocked_by_gate")
            self.assertFalse(row["target_feasibility"]["known_ordered_supported_path"])

    def test_source_hash_and_saved_baseline_mismatch_stop_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(Path(directory), [[.9, .8, .1, .1], [.1, .1, .9, .8]])
            saved = Path(case["baseline_record"])
            hashes = {str(saved): sha(saved)}
            verify_hashes(hashes)
            saved.write_text(json.dumps({"modes": {"baseline": {"results": []}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_hashes(hashes)
            with self.assertRaises(ValueError):
                replay_case(case, TemporalConfig(), "development")

    def test_order_or_frame_absence_is_distinct_from_gate(self):
        table = [{"keyframe_n": 1, "pts_time": 1}, {"keyframe_n": 2, "pts_time": 1}]
        scores, mask = np.ones((2, 2)), np.ones((2, 2), bool)
        query = {"event_relevant_keyframes": [[1], [2]]}
        diag = path_feasibility(table, scores, mask, query, TemporalConfig(), None)
        self.assertEqual(diag["known_frame_supported_by_event"], [True, True])
        self.assertFalse(diag["known_ordered_supported_path"])


if __name__ == "__main__":
    unittest.main()
