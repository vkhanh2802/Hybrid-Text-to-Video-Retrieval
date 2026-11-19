from dataclasses import replace
from contextlib import ExitStack
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from kis.event_retrieval import LocalEventScorer, retrieve_event_segments
from kis.query_plan import build_event_plan
from kis.temporal import TemporalConfig, align_events, evidence_scores
from kis.solve_kis import solve_kis


def sequence(actions=("pour eggs", "stir soup")):
    return {"raw_query": "then".join(actions), "query_variants": ["soup"], "entities": [],
            "events": [{"action": action} for action in actions],
            "temporal_constraints": [{"type": "sequence", "order": list(actions)}]}


def frames(times=(1.0, 2.0, 3.0), video_id="L01_V001"):
    return [{"video_id": video_id, "frame_id": 10 * (i + 1), "keyframe_n": i + 1, "pts_time": t}
            for i, t in enumerate(times)]


class EventPlanTest(unittest.TestCase):
    def test_requires_explicit_order(self):
        query = sequence()
        query["temporal_constraints"] = []
        self.assertFalse(build_event_plan(query).active)

    def test_uses_constraint_order_not_event_array(self):
        query = sequence()
        query["temporal_constraints"][0]["order"].reverse()
        self.assertEqual(build_event_plan(query).actions, ("stir soup", "pour eggs"))

    def test_rejects_partial_duplicate_unknown_or_overbudget(self):
        for order in (["pour eggs"], ["pour eggs", "pour eggs"], ["pour eggs", "invented action"]):
            query = sequence()
            query["temporal_constraints"][0]["order"] = order
            self.assertFalse(build_event_plan(query).active)
        self.assertEqual(build_event_plan(sequence(), max_events=1).reason, "event_budget_exceeded")

    def test_malformed_data_falls_back(self):
        for data in ({"events": [None], "temporal_constraints": [None]},
                     {"events": [{"action": 3}], "temporal_constraints": [{"type": "sequence", "order": [1, 2]}]}):
            self.assertFalse(build_event_plan(data).active)


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.config = TemporalConfig()

    def test_order_is_sensitive_and_frame_cannot_satisfy_two_events(self):
        scores = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.9]])
        good = align_events(frames(), scores, scores > 0, self.config)
        reverse = align_events(frames(), scores[::-1], scores[::-1] > 0, self.config)
        self.assertEqual(good["status"], "retrieved_complete")
        self.assertEqual(good["path"], ((0, 0), (1, 2)))
        self.assertEqual(reverse["coverage"], 0.5)
        one = align_events(frames((1.0,)), np.ones((2, 1)), np.ones((2, 1), bool), self.config)
        self.assertEqual(one["coverage"], 0.5)

    def test_unknown_event_is_skipped_not_compensated_by_strong_events(self):
        scores = np.array([[0.9, 0, 0], [0, 0.1, 0], [0, 0, 0.9]])
        result = align_events(frames(), scores, scores > 0.2, self.config)
        self.assertEqual(result["path"], ((0, 0), (2, 2)))
        self.assertEqual(result["status"], "partial")
        self.assertAlmostEqual(result["coverage"], 2 / 3)

    def test_equal_timestamps_gaps_and_invalid_timestamps(self):
        scores = np.eye(2)
        for times in ((1, 1), (1, 100)):
            result = align_events(frames(times), scores, scores > 0, self.config)
            self.assertEqual(result["coverage"], 0.5)
        for times in ((2, 1), (0, float("nan")), (-1, 0)):
            with self.assertRaises(ValueError):
                align_events(frames(times), scores, scores > 0, self.config)

    def test_absent_event_support_abstains(self):
        self.assertIsNone(align_events(frames(), np.zeros((2, 3)), np.zeros((2, 3), bool), self.config))
        score, support = evidence_scores(np.full((2, 3), 0.1), None, (2, 3), self.config)
        self.assertFalse(support.any())  # No relative min-max promotion of weak evidence.

    def test_missing_text_is_not_negative_evidence(self):
        score, support = evidence_scores(np.array([[0.8]]), np.array([[float("nan")]]), (1, 1), self.config)
        self.assertAlmostEqual(score[0, 0], 0.8)
        self.assertTrue(support[0, 0])

    def test_compactness_penalty_and_zero_penalty_control(self):
        scores = np.array([[0.9, 0.85, 0], [0, 0, 0.9]])
        times = frames((0, 58, 59))
        no_penalty = align_events(times, scores, scores > 0, replace(self.config, gap_penalty=0))
        penalized = align_events(times, scores, scores > 0, self.config)
        self.assertEqual(no_penalty["path"][0][1], 0)
        self.assertEqual(penalized["path"][0][1], 1)

    def test_complete_support_always_outranks_partial(self):
        low = np.array([[0.01, 0], [0, 0.01]])
        complete = align_events(frames((1, 2)), low, low > 0, self.config)
        partial = align_events(frames((1, 2)), np.ones((2, 2)), np.array([[True, True], [False, False]]), self.config)
        self.assertGreater(complete["score"], partial["score"])

    def test_optimized_dp_matches_exhaustive_search_including_skips_and_ties(self):
        rng = np.random.default_rng(123)
        config = replace(self.config, max_gap_seconds=2.0, gap_penalty=0.03)
        for _ in range(60):
            times = np.sort(rng.integers(0, 7, 5)).astype(float)
            scores = rng.random((3, 5))
            support = rng.random((3, 5)) > 0.5
            expected = (0, 0.0)
            for assignment in product(range(-1, 5), repeat=3):
                path = [(event, frame) for event, frame in enumerate(assignment) if frame >= 0]
                if any(not support[event, frame] for event, frame in path):
                    continue
                if any(not 0 < times[b] - times[a] <= config.max_gap_seconds
                       for (_, a), (_, b) in zip(path, path[1:])):
                    continue
                quality = sum(scores[event, frame] for event, frame in path)
                if path:
                    quality -= config.gap_penalty * (times[path[-1][1]] - times[path[0][1]])
                expected = max(expected, (len(path), quality))
            actual = align_events(frames(times), scores, support, config)
            if expected[0] == 0:
                self.assertIsNone(actual)
            else:
                self.assertEqual(len(actual["path"]), expected[0])
                self.assertAlmostEqual(actual["quality"], max(0, min(1, expected[1] / expected[0])))


class LocalScoringTest(unittest.TestCase):
    def test_condition_guard_keeps_scores_and_visual_evidence(self):
        self.csv.write_text("n,frame_idx,pts_time\n1,10,1\n2,20,2\n", encoding="utf-8")
        np.save(self.npy, np.array([[1, 0], [0, 1]], np.float32))
        captions = SimpleNamespace(encode_query=lambda action: [[1, 0]],
                                  score_frames=lambda *args, **kwargs: [0.8, 0.8],
                                  get_caption=lambda *args: {"caption": "A man in a white shirt."})
        with patch("kis.event_retrieval.video_csv_path", return_value=self.csv), \
             patch("kis.event_retrieval.video_clip_path", return_value=self.npy):
            baseline = LocalEventScorer(None, captions, TemporalConfig()).score(
                "L01_V001", ["A man in a blue shirt."], [[1, 0]], set())
            guarded = LocalEventScorer(None, captions, TemporalConfig(condition_evidence="semantic_guard")).score(
                "L01_V001", ["A man in a blue shirt."], [[1, 0]], set())
        np.testing.assert_array_equal(baseline[1], guarded[1])
        self.assertEqual(baseline[2].tolist(), [[True, True]])
        self.assertEqual(guarded[2].tolist(), [[True, False]])
        self.assertEqual(guarded[-1]["condition_evidence"]["support_removed"], 1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "video.csv"
        self.npy = self.root / "video.npy"
        LocalEventScorer._video_data.cache_clear()
        self.addCleanup(LocalEventScorer._video_data.cache_clear)

    def scorer(self, config=None):
        return LocalEventScorer(None, object(), config or TemporalConfig())

    def test_duplicate_legacy_frame_ids_excluded_without_changing_vector_row_join(self):
        self.csv.write_text("n,frame_idx,pts_time\n1,0,0\n2,0,0.0333333\n3,30,1\n4,60,2\n", encoding="utf-8")
        np.save(self.npy, np.array([[1, 0], [1, 0], [0, 1], [1, 0]], np.float32))
        with patch("kis.event_retrieval.video_csv_path", return_value=self.csv), \
             patch("kis.event_retrieval.video_clip_path", return_value=self.npy):
            rows, scores, _, _, _, stats = self.scorer().score("L21_V006", ("A",), [[0, 1]], {0, 30})
        self.assertEqual([row["frame_id"] for row in rows], [30, 60])
        self.assertEqual(scores.tolist(), [[1, 0]])
        self.assertEqual(stats["ambiguous_frame_rows_excluded"], 2)
        self.assertEqual(stats["seed_frames_scored"], 1)

    def test_reads_real_timestamp_sorts_and_preserves_seed_under_budget(self):
        self.csv.write_text("n,frame_idx,pts_time\n1,10,100\n2,20,1\n3,30,3\n4,40,7\n", encoding="utf-8")
        np.save(self.npy, np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32))
        with patch("kis.event_retrieval.video_csv_path", return_value=self.csv), \
             patch("kis.event_retrieval.video_clip_path", return_value=self.npy):
            rows, scores, support, *_rest, stats = self.scorer(replace(TemporalConfig(), max_frames_per_video=2)).score(
                "L01_V001", ("A", "B"), [[1, 0], [0, 1]], {20})
        self.assertEqual(rows[0]["frame_id"], 20)
        self.assertEqual(rows[0]["pts_time"], 1)
        self.assertEqual(scores.shape, (2, 2))
        self.assertTrue(stats["sampled"])
        self.assertTrue(support.any())

    def test_rejects_mismatched_features_or_missing_timestamp(self):
        self.csv.write_text("n,frame_idx,pts_time\n1,10,1\n", encoding="utf-8")
        np.save(self.npy, np.ones((2, 2), np.float32))
        with patch("kis.event_retrieval.video_csv_path", return_value=self.csv), \
             patch("kis.event_retrieval.video_clip_path", return_value=self.npy):
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                self.scorer().score("L01_V001", ("A",), [[1, 0]], set())
            self.csv.write_text("n,frame_idx\n1,10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pts_time"):
                self.scorer().score("L01_V001", ("A",), [[1, 0]], set())

    def test_caption_exact_scores_missing_identity_and_no_model_loading(self):
        import faiss
        from kis.text import CaptionRetriever

        caption = CaptionRetriever.__new__(CaptionRetriever)
        caption._caption_position = {("L01_V001", 10): 0}
        caption.caption_data = [{"keyframe_n": 1}]
        caption.index = faiss.IndexFlatIP(2)
        caption.index.add(np.array([[1, 0]], np.float32))
        caption.encode_query = Mock(return_value=np.array([[1, 0]], np.float32))
        result = caption.score_frames("pour", frames())
        self.assertAlmostEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1:]).all())
        bad = [dict(frames()[0], keyframe_n=999)]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            caption.score_frames("pour", bad)


class EventRetrievalTest(unittest.TestCase):
    def test_end_to_end_local_visual_rescore_distinguishes_same_events_wrong_order(self):
        LocalEventScorer._video_data.cache_clear()
        self.addCleanup(LocalEventScorer._video_data.cache_clear)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            # Windows forbids deleting a memory-mapped NPY until cache release.
            stack.callback(LocalEventScorer._video_data.cache_clear)
            root = Path(directory)
            good, bad = "L01_V001", "L01_V002"
            for video_id in (good, bad):
                (root / f"{video_id}.csv").write_text("n,frame_idx,pts_time\n1,10,1\n2,20,2\n3,30,3\n", encoding="utf-8")
            good_vectors = np.array([[1, 0], [0.7, 0.7], [0, 1]], np.float32)
            np.save(root / f"{good}.npy", good_vectors)
            np.save(root / f"{bad}.npy", good_vectors[::-1])
            encoder = SimpleNamespace(encode_text=lambda action: [1, 0] if action == "pour eggs" else [0, 1])
            visual = SimpleNamespace(search=lambda vector, top_k: [
                {"video_id": bad, "frame_id": 30 if vector[0] else 10, "score": 1},
                {"video_id": good, "frame_id": 10 if vector[0] else 30, "score": 0.9}])
            bm25 = SimpleNamespace(search_bm25=lambda action, top_k: [])
            captions = SimpleNamespace(search=lambda action, top_k: [])
            diagnostics = {}
            with patch("kis.event_retrieval.video_csv_path", side_effect=lambda vid: root / f"{vid}.csv"), \
                 patch("kis.event_retrieval.video_clip_path", side_effect=lambda vid: root / f"{vid}.npy"):
                results = solve_kis(sequence(), visual, encoder, bm25, captions, None, diagnostics=diagnostics,
                                    temporal_config=TemporalConfig(min_visual_similarity=0.9))
            # Unordered max-over-frames gives both videos the same perfect score.
            self.assertEqual(np.max(good_vectors, axis=0).sum(), np.max(good_vectors[::-1], axis=0).sum())
            self.assertEqual([item["video_id"] for item in results], [good, bad])
            self.assertEqual(results[0]["temporal"]["coverage"], 1)
            self.assertEqual(results[1]["temporal"]["coverage"], 0.5)
            json.dumps({"results": results, "diagnostics": diagnostics}, allow_nan=False)

    def dependencies(self):
        encoder = Mock(encode_text=Mock(side_effect=lambda action: [1, 0] if action == "pour eggs" else [0, 1]))
        visual = Mock(search=Mock(side_effect=lambda vector, top_k: [
            {"video_id": "L01_V001", "frame_id": 10 if vector[0] else 30, "score": 0.8}]))
        bm25 = Mock(search_bm25=Mock(return_value=[]))
        captions = Mock(search=Mock(return_value=[]), get_caption=Mock(return_value=None))
        return visual, encoder, bm25, captions

    def test_sequence_skips_pointwise_vlm_and_keeps_unknown_evidence(self):
        visual, encoder, bm25, captions = self.dependencies()
        config = TemporalConfig()
        scorer = Mock()
        scorer.score.return_value = (frames(), np.array([[0.8, 0, 0], [0, 0, 0.8]]),
                                     np.array([[True, False, False], [False, False, True]]),
                                     None, None, {"sampled": False})
        reranker, diagnostics = Mock(), {}
        result = solve_kis(sequence(), visual, encoder, bm25, captions, object(),
                           vlm_reranker=reranker, temporal_config=config, event_scorer=scorer, diagnostics=diagnostics)
        self.assertEqual(result[0]["temporal"]["status"], "retrieved_complete")
        self.assertFalse(result[0]["temporal"]["verified"])
        self.assertEqual(result[0]["temporal"]["end_seconds"], 3)
        self.assertEqual(diagnostics["temporal"]["status"], "applied")
        self.assertEqual(diagnostics["stage_seconds"]["temporal"], diagnostics["temporal"]["seconds"])
        reranker.score.assert_not_called()
        self.assertEqual(scorer.score.call_args.args[-1], {10, 30})

    def test_text_rerank_temporal_path_preserves_events_support_and_skips_vlm(self):
        from copy import deepcopy
        from kis.text_reranker import TextRerankerConfig
        from tests.test_text_reranker import segment

        visual, encoder, bm25, captions = self.dependencies()
        segments = [segment(1, .9, ["pour eggs", "stir"]), segment(10, .8, ["pour eggs", "stir"])]
        original = deepcopy(segments)
        def retrieve(*args, **kwargs):
            args[6].update(seconds=.1, status="applied")
            return deepcopy(segments)
        text = Mock(score_pairs=Mock(return_value=[-3, -2, 2, 3]))
        vlm, diagnostics = Mock(), {}
        with patch("kis.event_retrieval.retrieve_event_segments", side_effect=retrieve):
            results = solve_kis(sequence(), visual, encoder, bm25, captions, None, vlm_reranker=vlm,
                                diagnostics=diagnostics, text_reranker_config=TextRerankerConfig(mode="rerank"),
                                text_reranker=text)
        self.assertEqual([r["frame_id"] for r in results], [10, 1])
        self.assertEqual(results[0]["temporal"], original[1]["temporal"])
        self.assertFalse(results[0]["temporal"]["verified"])
        self.assertEqual(results[0]["score"], .8)
        self.assertEqual(results[0]["final_score"], .9)
        self.assertEqual(diagnostics["stage_seconds"]["temporal"], .1)
        vlm.score.assert_not_called()

    def test_condition_diagnostics_with_legacy_custom_scorer(self):
        visual, encoder, bm25, captions = self.dependencies()
        captions.get_caption.return_value = {"caption": "A cook pours eggs."}
        scorer = Mock()
        scorer.score.return_value = (frames(), np.array([[0.8, 0, 0], [0, 0, 0.8]]),
                                     np.array([[True, False, False], [False, False, True]]),
                                     None, None, {"sampled": False})
        result = solve_kis(sequence(), visual, encoder, bm25, captions, object(), event_scorer=scorer,
                           temporal_config=TemporalConfig(condition_evidence="diagnostics"))
        event = result[0]["temporal"]["events"][0]
        self.assertEqual(event["condition_evidence"]["status"], "unknown")
        self.assertFalse(event["condition_evidence"]["pixel_verified"])
        self.assertFalse(result[0]["temporal"]["verified"])
        json.dumps(result, allow_nan=False)

    def test_local_missing_artifacts_fall_back_to_frame_pipeline(self):
        visual, encoder, bm25, captions = self.dependencies()
        # Baseline query is distinct from the event queries.
        encoder.encode_text.side_effect = lambda action: [1, 0]
        captions.get_caption.return_value = None
        scorer, diagnostics = Mock(), {}
        scorer.score.side_effect = FileNotFoundError("missing local mapping")
        result = solve_kis(sequence(), visual, encoder, bm25, captions, object(),
                           event_scorer=scorer, diagnostics=diagnostics, rerank_top_k=0)
        self.assertTrue(result)
        self.assertNotIn("temporal", result[0])
        self.assertEqual(diagnostics["temporal"]["status"], "fallback")
        self.assertIn("L01_V001", diagnostics["temporal"]["failures"])
        self.assertEqual(diagnostics["stage_seconds"]["temporal"], diagnostics["temporal"]["seconds"])
        self.assertGreaterEqual(diagnostics["stage_seconds"]["temporal"], 0.0)

    def test_fallback_keeps_temporal_timing_without_changing_baseline_results(self):
        visual, encoder, bm25, captions = self.dependencies()
        encoder.encode_text.side_effect = lambda action: [1, 0]
        diagnostics = {}

        def no_supported_path(*args, **kwargs):
            args[6].update(status="fallback", seconds=.125)
            return []

        retrieve = Mock(side_effect=no_supported_path)
        with patch("kis.event_retrieval.retrieve_event_segments", retrieve):
            fallback = solve_kis(sequence(), visual, encoder, bm25, captions, object(),
                                 diagnostics=diagnostics, rerank_top_k=0)
        control_visual, control_encoder, control_bm25, control_captions = self.dependencies()
        control_encoder.encode_text.side_effect = lambda action: [1, 0]
        baseline = solve_kis(sequence(), control_visual, control_encoder, control_bm25,
                             control_captions, object(), rerank_top_k=0,
                             temporal_config=TemporalConfig(enabled=False))
        fields = ("video_id", "frame_id", "score", "final_score")
        self.assertEqual([tuple(item[field] for field in fields) for item in fallback],
                         [tuple(item[field] for field in fields) for item in baseline])
        self.assertEqual([tuple(item.get(field) for field in fields) for item in fallback],
                         [("L01_V001", 10, .5, .5)])
        self.assertEqual(diagnostics["temporal"]["status"], "fallback")
        self.assertEqual(diagnostics["temporal"]["reason"], "no_supported_temporal_path")
        self.assertEqual(diagnostics["stage_seconds"]["temporal"], .125)
        retrieve.assert_called_once()
        self.assertEqual(visual.search.call_count, control_visual.search.call_count)
        self.assertEqual(bm25.search_bm25.call_count, control_bm25.search_bm25.call_count)
        self.assertEqual(captions.search.call_count, control_captions.search.call_count)
        self.assertEqual(encoder.encode_text.call_count, control_encoder.encode_text.call_count)

    def test_single_query_has_no_temporal_stage_timing(self):
        visual, encoder, bm25, captions = self.dependencies()
        encoder.encode_text.side_effect = lambda action: [1, 0]
        diagnostics = {}
        results = solve_kis({"raw_query": "pour eggs", "query_variants": ["pour eggs"], "entities": []},
                            visual, encoder, bm25, captions, object(), diagnostics=diagnostics,
                            rerank_top_k=0)
        self.assertTrue(results)
        self.assertEqual(diagnostics["temporal"]["status"], "inactive")
        self.assertNotIn("temporal", diagnostics["stage_seconds"])
        self.assertEqual(visual.search.call_count, 1)

    def test_disabled_temporal_does_not_search_events(self):
        visual, encoder, bm25, captions = self.dependencies()
        encoder.encode_text.side_effect = lambda action: [1, 0]
        diagnostics = {}
        solve_kis(sequence(), visual, encoder, bm25, captions, object(), rerank_top_k=0,
                  temporal_config=TemporalConfig(enabled=False), diagnostics=diagnostics)
        self.assertEqual(encoder.encode_text.call_count, 1)
        self.assertEqual(diagnostics["temporal"]["status"], "disabled")
        self.assertNotIn("temporal", diagnostics["stage_seconds"])

    def test_video_budget_round_robin_includes_later_event(self):
        visual, encoder, bm25, captions = self.dependencies()
        visual.search.side_effect = lambda vector, top_k: [
            {"video_id": f"L01_V{offset + i:03d}", "frame_id": 10, "score": 0.9}
            for i in range(5) for offset in ([1] if vector[0] else [10])]
        scorer = Mock(score=Mock(side_effect=FileNotFoundError("missing")))
        diagnostics = {}
        retrieve_event_segments(build_event_plan(sequence()), visual, encoder, bm25, captions,
                                TemporalConfig(max_videos=2), diagnostics, scorer)
        self.assertEqual(diagnostics["candidate_videos"], ["L01_V001", "L01_V010"])


if __name__ == "__main__":
    unittest.main()
