import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.temporal import TemporalConfig
from tests.test_human_holdout import synthetic_annotations, synthetic_packet
from tools.human_holdout import freeze
from tools.human_holdout_evaluation import (adoption_gate, compare_unit, frame_metrics,
    load_frozen_labels, ranking_summary, run, score_labeled_pairs, sequence_queries)


class HumanHoldoutEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = synthetic_packet(Path(self.temp.name) / "packet")
        self.source = self.root / "approved.json"
        atomic_write_json(self.source, synthetic_annotations(self.root))
        self.labels = self.root / "frozen.json"
        freeze(self.root, self.source, self.labels)

    def test_human_holdout_loads_without_relabeling_as_ai_or_dev(self):
        labels, _ = load_frozen_labels(self.root, self.labels)
        self.assertEqual(labels["annotation_source"], "human_visual_observation")
        self.assertTrue(all(q["split"] == "holdout" for q in labels["queries"]))

    def test_changed_qrels_or_unapproved_english_rejected_before_models(self):
        original = json.loads(self.labels.read_text(encoding="utf-8"))
        for mutate in ("qrels", "approval", "provenance"):
            data = copy.deepcopy(original)
            if mutate == "qrels":
                data["queries"][0]["judgments"][0]["judgment"] = "negative"
            elif mutate == "approval":
                data["human_annotations"]["samples"][0]["events"][0]["english_approved"] = False
            else:
                data["annotation_source"] = "ai_visual_observation"
            path = self.root / f"invalid_{mutate}.json"
            atomic_write_json(path, data)
            with patch("tools.temporal_pilot.require_model_cache", side_effect=AssertionError("Must not load")):
                with self.assertRaises(ValueError):
                    run(self.root, path, self.root / "out")
            self.assertFalse((self.root / "out").exists())

    def test_known_frame_metrics_only_count_explicit_positives(self):
        query = {"video_id": "L01_V001", "judgments": [
            {"video_id": "L01_V001", "keyframe_n": 1, "judgment": "positive"},
            {"video_id": "L01_V001", "keyframe_n": 2, "judgment": "unknown"}]}
        ranked = [{"video_id": "L01_V001", "keyframe_n": 2}, {"video_id": "L01_V001", "keyframe_n": 1}]
        metrics = frame_metrics(ranked, query)
        self.assertEqual(metrics["known_frame_hit_at_1"], 0)
        self.assertEqual(metrics["known_frame_hit_at_20"], 1)
        self.assertEqual(metrics["mrr_at_20"], 0.5)
        self.assertTrue(metrics["top_target_video"])
        self.assertEqual(frame_metrics([], query)["known_frame_hit_at_20"], 0)

    def test_sequence_queries_use_only_approved_order_and_own_positives(self):
        labels, _ = load_frozen_labels(self.root, self.labels)
        self.assertEqual(sequence_queries(labels), [])
        sample = labels["human_annotations"]["samples"][0]
        sample["events"].append(copy.deepcopy(sample["events"][0]))
        second = copy.deepcopy(labels["queries"][0])
        second["query_id"] = "drive_01_event_2"
        second["event_text"] = "Second approved event."
        second["judgments"] = [{**j, "judgment": "positive" if j["keyframe_n"] == 2 else "negative"}
                               for j in second["judgments"]]
        labels["queries"].append(second)
        sequences = sequence_queries(labels)
        self.assertEqual(sequences[0]["event_relevant_keyframes"], [[1], [2]])
        self.assertEqual(sequences[0]["actions"][-1], "Second approved event.")

    def fake_dependencies(self):
        frames = [{"video_id": "L01_V001", "frame_id": n*10, "keyframe_n": n, "pts_time": n} for n in (1, 2)]
        scorer = SimpleNamespace(score=Mock(return_value=(frames, np.array([[0.8, 0.9]]),
            np.array([[True, True]]), np.array([[0.1, 0.3]]), np.array([[0.8, 0.8]]), {})))
        visual = SimpleNamespace(search=Mock(return_value=[{"video_id": "L01_V001", "frame_id": 10, "score": 1}]))
        bm25 = SimpleNamespace(search_bm25=Mock(return_value=[]))
        captions = SimpleNamespace(search=Mock(return_value=[]),
                                   get_caption=lambda *args: {"caption": "A man in a white shirt."})
        return frames, scorer, visual, bm25, captions

    def test_both_modes_share_scores_and_searches_preserve_visual_support(self):
        frames, scorer, visual, bm25, captions = self.fake_dependencies()
        query = {"query_id": "q", "video_id": "L01_V001", "event_text": "A man in a blue shirt.",
                 "judgments": [{**frames[0], "judgment": "positive"}]}
        vectors, hits = {query["event_text"]: [1, 0]}, {}
        row = compare_unit(query, [query["event_text"]], vectors, hits, scorer, visual, bm25, captions,
                           TemporalConfig(), self.root / "scores")
        self.assertEqual(row["modes"]["baseline"]["candidate_count"], 2)
        self.assertEqual(row["modes"]["semantic_guard"]["candidate_count"], 1)
        self.assertEqual(row["modes"]["semantic_guard"]["results"][0]["frame_id"], 20)
        self.assertEqual(row["modes"]["baseline"]["metrics"]["known_frame_hit_at_20"], 1)
        self.assertEqual(row["modes"]["semantic_guard"]["metrics"]["known_frame_hit_at_20"], 0)
        compare_unit(query, [query["event_text"]], vectors, hits, scorer, visual, bm25, captions,
                     TemporalConfig(), self.root / "scores2")
        visual.search.assert_called_once()
        self.assertEqual(ranking_summary([row])["support_removed"], 1)
        json.dumps(row, allow_nan=False)

    def test_known_target_not_injected_into_candidate_pool(self):
        frames, scorer, visual, bm25, captions = self.fake_dependencies()
        query = {"query_id": "q", "video_id": "L01_V999", "event_text": "A man in a blue shirt.",
                 "judgments": [{"video_id": "L01_V999", "keyframe_n": 1, "judgment": "positive"}]}
        row = compare_unit(query, [query["event_text"]], {query["event_text"]: [1, 0]}, {},
                           scorer, visual, bm25, captions, TemporalConfig(), self.root / "scores")
        self.assertFalse(row["target_video_in_pool"])
        self.assertEqual(row["candidate_videos"], ["L01_V001"])
        self.assertEqual(row["modes"]["baseline"]["metrics"]["known_frame_hit_at_20"], 0)

    def test_pair_scoring_preserves_missing_caption_scores_and_exact_identity(self):
        frames, _, _, _, _ = self.fake_dependencies()
        query = {"query_id": "q", "event_text": "A blue shirt.",
                 "judgments": [{**frame, "judgment": "positive", "image_sha256": "human-image"} for frame in frames]}
        encoder = SimpleNamespace(encode_text=Mock(return_value=[1, 0]))
        captions = SimpleNamespace(score_frames=lambda *args: np.array([np.nan, 0.8]),
                                   get_caption=lambda *args: {"image_sha256": "human-image"})
        with patch("tools.human_holdout_evaluation.LocalEventScorer._video_data", return_value=(frames, np.eye(2))):
            rows, _ = score_labeled_pairs([query], encoder, captions)
        self.assertEqual(rows[0]["visual"], 1)
        self.assertIsNone(rows[0]["semantic"])
        self.assertTrue(all(row["caption_image_hash_matches"] for row in rows))
        self.assertTrue(all(row["split"] == "holdout" for row in rows))
        bad = copy.deepcopy(query)
        bad["judgments"][0]["keyframe_n"] = 999
        with patch("tools.human_holdout_evaluation.LocalEventScorer._video_data", return_value=(frames, np.eye(2))):
            with self.assertRaisesRegex(ValueError, "mismatched"):
                score_labeled_pairs([bad], encoder, captions)

    def test_sequence_known_path_requires_all_events_not_just_correct_video(self):
        frames, scorer, visual, bm25, captions = self.fake_dependencies()
        actions = ["A man in a blue shirt.", "A person stirs soup."]
        scorer.score.return_value = (frames, np.array([[0.9, 0], [0, 0.8]]),
                                     np.array([[True, False], [False, True]]), None,
                                     np.array([[0.9, 0], [0, 0.8]]), {})
        query = {"query_id": "q_sequence", "video_id": "L01_V001", "event_relevant_keyframes": [[1], [2]]}
        row = compare_unit(query, actions, {action: [1, 0] for action in actions}, {},
                           scorer, visual, bm25, captions, TemporalConfig(), self.root / "seq", sequence=True)
        self.assertTrue(row["modes"]["semantic_guard"]["metrics"]["top_target_video"])
        self.assertEqual(row["modes"]["baseline"]["metrics"]["known_path_hit_at_20"], 1)
        self.assertEqual(row["modes"]["semantic_guard"]["metrics"]["known_path_hit_at_20"], 0)
        self.assertFalse(row["modes"]["baseline"]["results"][0]["temporal"]["verified"])
        self.assertEqual(ranking_summary([row], sequence=True)["queries"], 1)

    def test_adoption_requires_actual_fpr_improvement_and_no_path_regression(self):
        pairs = {"baseline": {"eligible_queries": 26, "macro_fpr": 0.2, "macro_tpr": 0.8},
                 "semantic_guard": {"eligible_queries": 26, "macro_fpr": 0.2, "macro_tpr": 0.8}}
        events = {"queries": 26, "video_scoring_failures": 0, "modes": {
            name: {"known_frame_hit_at_20": 0.5} for name in ("baseline", "semantic_guard")}}
        sequences = {"queries": 10, "video_scoring_failures": 0, "modes": {
            name: {"known_path_hit_at_20": 0.3} for name in ("baseline", "semantic_guard")}}
        self.assertFalse(adoption_gate(pairs, events, sequences)["all_checks_passed"])
        pairs["semantic_guard"]["macro_fpr"] = 0.1
        decision = adoption_gate(pairs, events, sequences)
        self.assertTrue(decision["all_checks_passed"])
        self.assertFalse(decision["automatically_promoted"])
        sequences["modes"]["semantic_guard"]["known_path_hit_at_20"] = 0.2
        self.assertFalse(adoption_gate(pairs, events, sequences)["all_checks_passed"])


if __name__ == "__main__":
    unittest.main()
