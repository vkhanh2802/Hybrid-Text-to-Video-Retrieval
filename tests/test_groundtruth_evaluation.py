import json
from pathlib import Path
import tempfile
import unittest

from tools.groundtruth_evaluation import known_path_metrics, load_labels, structured, summarize


class GroundTruthEvaluationTest(unittest.TestCase):
    def query(self):
        return {"query_id": "image_01", "split": "dev", "video_id": "L01_V001", "frame_id": 30,
                "annotation_source": "ai_visual_observation", "image_sha256": "abc", "raw_query": "test",
                "english": "test image", "keywords": "image", "anchor_keyframe_n": 1, "relevant_keyframes": [1]}

    def test_static_image_does_not_invent_sequence(self):
        plan = structured(self.query())
        self.assertEqual(plan["events"], [])
        self.assertEqual(plan["temporal_constraints"], [])

    def test_rejects_video_leakage_or_missing_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "labels.json"
            a = self.query()
            b = dict(a, query_id="image_02", split="holdout")
            for queries in ([a, b], [dict(a, image_sha256=None)]):
                path.write_text(json.dumps({"schema_version": 1, "queries": queries}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_labels(path)

    def test_empty_mode_metrics_are_not_zero_accuracy(self):
        result = summarize([])
        self.assertIsNone(result["holdout"]["temporal"]["hit_at_1"])

    def test_video_match_or_full_support_does_not_imply_known_path_match(self):
        query = {"video_id": "L01_V001", "event_relevant_keyframes": [[1], [2]]}
        result = {"video_id": "L01_V001", "temporal": {"events": [
            {"status": "retrieved", "keyframe_n": 1}, {"status": "retrieved", "keyframe_n": 99}]}}
        self.assertEqual(known_path_metrics([result], query)["known_path_hit_at_20"], 0)
        result["temporal"]["events"][1]["keyframe_n"] = 2
        self.assertEqual(known_path_metrics([result], query)["known_path_hit_at_1"], 1)


if __name__ == "__main__":
    unittest.main()
