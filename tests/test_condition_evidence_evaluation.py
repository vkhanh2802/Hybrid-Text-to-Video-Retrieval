import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kis.temporal import TemporalConfig
from tools.condition_evidence_evaluation import ArtifactCaptions, evaluate_pairs, load_dev_scores, replay


class ConditionEvaluationTest(unittest.TestCase):
    def test_saved_raw_sequence_replay_does_not_load_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            actions = ["A man in a blue shirt.", "A man in a red tie."]
            video_id = "L01_V001"
            frames = [{"video_id": video_id, "frame_id": i * 10, "keyframe_n": i, "pts_time": i}
                      for i in (1, 2)]
            evidence = {"split": "dev", "actions": actions, "frames": frames, "stats": {},
                        "visual": None, "semantic": [[0.9, 0.9], [0.9, 0.9]]}
            (root / f"q_{video_id}_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            record = {"query": {"query_id": "q", "video_id": video_id, "event_relevant_keyframes": [[1], [2]]},
                      "modes": {"temporal": {"results": [{"video_id": video_id}], "diagnostics": {"temporal": {
                          "candidate_videos": [video_id], "events": [{"action": action, "streams": {"visual": [frames[i]]}}
                                                                    for i, action in enumerate(actions)]}}}}}
            captions = SimpleNamespace(get_caption=lambda *args: {"caption": "A man in a white shirt and green tie."})
            with patch("tools.temporal_pilot.require_model_cache", side_effect=AssertionError("No models allowed")):
                report = replay([record], {"temporal_config": TemporalConfig().public()}, captions, output, root)
            self.assertEqual(report["failures"], 0)
            self.assertEqual(report["baseline_matches_source_top"], 1)
            self.assertEqual(report["support_removed"], 4)
            self.assertEqual(report["modes"]["baseline"]["top_target_video"], 1)
            self.assertEqual(report["modes"]["semantic_guard"]["top_target_video"], 0)

    def test_pair_metrics_keep_unknown_judgments_and_missing_captions(self):
        rows = [{"query_id": "q", "split": "dev", "event_text": "A man in a blue shirt.",
                 "video_id": "L01_V001", "frame_id": i, "judgment": judgment,
                 "visual": 0.1, "semantic": 0.8}
                for i, judgment in enumerate(("positive", "negative", "unknown"))]
        captions = SimpleNamespace(get_caption=lambda video_id, frame_id:
                                   {"caption": "A man in a white shirt."} if frame_id else None)
        evaluated, report = evaluate_pairs(rows, captions, TemporalConfig())
        self.assertTrue(evaluated[0]["guarded_support"])
        self.assertFalse(evaluated[1]["guarded_support"])
        self.assertEqual(report["semantic_guard"]["unknown_rows_skipped"], 1)
        self.assertEqual(report["baseline"]["macro_fpr"], 1)
        self.assertEqual(report["semantic_guard"]["macro_fpr"], 0)

    def test_holdout_rows_rejected_before_labels_or_caption_io(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps([{"split": "holdout"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "development-only"):
                load_dev_scores(path, Path(directory) / "missing.json")

    def test_frozen_labels_must_match_every_cached_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"query_id": "q", "split": "dev", "event_text": "Blue shirt", "video_id": "L01_V001",
                   "frame_id": 10, "keyframe_n": 1, "judgment": "positive", "image_sha256": "abc"}
            label = {"queries": [{"query_id": "q", "split": "dev", "event_text": "Blue shirt", "judgments": [row]}]}
            scores, labels = root / "scores.json", root / "labels.json"
            labels.write_text(json.dumps(label), encoding="utf-8")
            scores.write_text(json.dumps([row]), encoding="utf-8")
            self.assertEqual(load_dev_scores(scores, labels), [row])
            for data in ([row, row], [{**row, "judgment": "negative"}], [{**row, "keyframe_n": 2}]):
                scores.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "frozen labels"):
                    load_dev_scores(scores, labels)

    def test_caption_adapter_missing_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captions = ArtifactCaptions(root)
            self.assertIsNone(captions.get_caption("L01_V001", 10))
            folder = root / "caption_generator"
            folder.mkdir()
            item = {"video_id": "L01_V002", "frame_id": 10, "caption": "A man."}
            (folder / "L01_V002.json").write_text(json.dumps([item, item]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                captions.get_caption("L01_V002", 10)
            captions._video.cache_clear()


if __name__ == "__main__":
    unittest.main()
