import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kis.temporal import TemporalConfig
from tools.candidate_pool_ablation import compact_alignment, load_dev_records


class CandidatePoolAblationTest(unittest.TestCase):
    def test_opens_only_development_record_files(self):
        dev = {"query_id": "dev_sequence", "split": "dev", "task": "sequence"}
        holdout = {"query_id": "holdout_sequence", "split": "holdout", "task": "sequence"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.json"
            labels_path.write_text("frozen labels", encoding="utf-8")
            (root / "context.json").write_text(json.dumps({"label_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest()}))
            (root / "dev_sequence.json").write_text(json.dumps({"split": "dev", "query": dev}))
            # Deliberately no holdout record file: attempting to open it fails.
            with patch("tools.candidate_pool_ablation.load_labels", return_value={"queries": [dev, holdout]}):
                records, _ = load_dev_records(labels_path, root)
            self.assertEqual(len(records), 1)

    def test_changed_label_hash_stops_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.json"
            labels_path.write_text("changed", encoding="utf-8")
            (root / "context.json").write_text(json.dumps({"label_sha256": "different"}))
            with patch("tools.candidate_pool_ablation.load_labels", return_value={"queries": []}):
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    load_dev_records(labels_path, root)

    def test_compact_result_retains_unknown_event_not_fake_full_support(self):
        frames = [{"video_id": "L01_V001", "frame_id": 10, "keyframe_n": 1, "pts_time": 1}]
        result = compact_alignment("L01_V001", frames, np.array([[0.8], [0.1]]),
                                   np.array([[True], [False]]), TemporalConfig())
        self.assertEqual(result["temporal"]["coverage"], 0.5)
        self.assertIsNone(result["temporal"]["events"][1]["frame_id"])
        self.assertFalse(result["temporal"]["verified"])


if __name__ == "__main__":
    unittest.main()
