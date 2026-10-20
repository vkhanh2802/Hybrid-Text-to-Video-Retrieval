import copy
import math
import unittest
from unittest.mock import Mock

from kis.objects import apply_object_filter
from kis.config import milvus_model_properties
from kis.visual import validate_collection_model
from kis.ranking import weighted_rrf, bounded_prompt_text, parse_vlm_score
from kis.solve_kis import console_safe_text
from kis.ranking import (
    finalize_vlm_ranking,
    max_score_fusion,
    normalize_scores,
    static_weight_fusion_3_way,
)


class ObjectLookup:
    def __init__(self, confidence):
        self.confidence = confidence

    def get(self, key):
        return {
            "detection_class_entities": ["person"],
            "detection_scores": [self.confidence],
        }


class RankingValidationTest(unittest.TestCase):
    def test_non_finite_object_confidence_is_ignored(self):
        candidates = [{"video_id": "V1", "frame_id": 1, "score": 0.5}]
        entities = [{"type": "object", "value": "person"}]

        results = apply_object_filter(
            candidates,
            entities,
            ObjectLookup(float("nan")),
        )

        self.assertTrue(math.isfinite(results[0]["score"]))
        self.assertEqual(results[0]["object_score"], 0.0)

    def test_object_confidence_is_capped(self):
        candidates = [{"video_id": "V1", "frame_id": 1, "score": 0.5}]
        entities = [{"type": "object", "value": "person"}]

        results = apply_object_filter(candidates, entities, ObjectLookup(5.0))

        self.assertEqual(results[0]["object_score"], 1.0)
        self.assertAlmostEqual(results[0]["score"], 0.575)

    def test_normalization_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_scores([{"score": float("inf")}])

    def test_fusion_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            static_weight_fusion_3_way([], [], [], 0.5, 0.5, 0.5)

    def test_identical_vlm_scores_still_apply_bounded_boost(self):
        candidates = [
            {"video_id": "V1", "frame_id": 1, "score": 0.6054},
            {"video_id": "V2", "frame_id": 2, "score": 0.5456, "vlm_raw_score": 8.0},
            {"video_id": "V3", "frame_id": 3, "score": 0.49, "vlm_raw_score": 8.0},
        ]

        ranked, applied = finalize_vlm_ranking(
            candidates,
            top_k=3,
            vlm_weight=0.5,
        )

        self.assertTrue(applied)
        self.assertEqual([item["frame_id"] for item in ranked], [2, 3, 1])
        self.assertAlmostEqual(ranked[0]["final_score"], 0.68192, places=4)
        self.assertAlmostEqual(ranked[1]["final_score"], 0.643, places=4)
        self.assertEqual(ranked[2]["final_score"], 0.6054)

    def test_max_fusion_preserves_single_variant_spike(self):
        spike = {"video_id": "V1", "frame_id": 1, "score": 0.9}
        filler_a = {"video_id": "V2", "frame_id": 2, "score": 0.5}
        filler_b = {"video_id": "V2", "frame_id": 2, "score": 0.45}
        fused = max_score_fusion([[spike], [filler_a], [filler_b]])

        # RRF would rank V2 (two appearances) above the one-off V1 spike;
        # max keeps the spike on top with its raw score and metadata.
        self.assertEqual(
            [(item["video_id"], item["score"]) for item in fused],
            [("V1", 0.9), ("V2", 0.5)],
        )

    def test_max_fusion_empty_input(self):
        self.assertEqual(max_score_fusion([]), [])

    def test_single_vlm_score_applies_bounded_boost(self):
        candidates = [
            {"video_id": "V1", "frame_id": 1, "score": 0.7},
            {"video_id": "V2", "frame_id": 2, "score": 0.5, "vlm_raw_score": 10.0},
        ]

        ranked, applied = finalize_vlm_ranking(
            candidates,
            top_k=2,
            vlm_weight=0.5,
        )

        self.assertTrue(applied)
        self.assertEqual([item["frame_id"] for item in ranked], [2, 1])
        self.assertAlmostEqual(ranked[0]["final_score"], 0.75, places=4)
        self.assertEqual(ranked[1]["final_score"], 0.7)

    def test_milvus_model_metadata_must_match_manifest(self):
        client = Mock()
        client.describe_collection.return_value = {
            "description": "{}",
        }
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_collection_model(client, "clip_keyframes")

        import json

        client.describe_collection.return_value = {
            "description": json.dumps(milvus_model_properties()),
        }
        validate_collection_model(client, "clip_keyframes")

    def test_console_logging_handles_vietnamese_on_legacy_windows_encoding(self):
        escaped = console_safe_text("mặt bàn", encoding="cp1252")
        self.assertIn("\\u1eb7", escaped)
        self.assertIn("bàn", escaped)


class RankFusionTest(unittest.TestCase):
    def item(self, video, score=1):
        return {"video_id": video, "frame_id": 1, "score": score, "norm_score": 1}

    def test_best_rank_per_channel_deduplicates_and_scores_are_bounded(self):
        a, b = self.item("A"), self.item("B")
        ranked = weighted_rrf({"visual": [a, a, b], "bm25": [a], "semantic": [a]})
        self.assertEqual(len(ranked), 2)
        self.assertAlmostEqual(ranked[0]["score"], 1.0)
        self.assertEqual(set(ranked[0]["evidence"]), {"visual", "bm25", "semantic"})
        self.assertTrue(all(0 <= item["score"] <= 1 for item in ranked))

    def test_raw_score_scale_never_changes_cross_channel_rank_fusion(self):
        streams = {"visual": [self.item("A", .9), self.item("B", .8)],
                   "bm25": [self.item("B", 200), self.item("A", 100)], "semantic": []}
        before = copy.deepcopy(streams)
        ranked = weighted_rrf(streams)
        streams["bm25"][0]["score"] *= 1000
        self.assertEqual([(item["video_id"], item["score"]) for item in weighted_rrf(streams)],
                         [(item["video_id"], item["score"]) for item in ranked])
        self.assertNotIn("evidence", before["visual"][0])

    def test_rrf_scores_work_with_existing_global_vlm_combiner(self):
        ranked = weighted_rrf({"visual": [self.item("A"), self.item("B")], "bm25": [], "semantic": []})
        ranked[0]["vlm_raw_score"] = 0
        ranked[1]["vlm_raw_score"] = 10
        result, applied = finalize_vlm_ranking(ranked, top_k=2)
        self.assertTrue(applied)
        self.assertEqual(result[0]["video_id"], "B")
        self.assertTrue(all(0 <= item["final_score"] <= 1 for item in result))

    def test_invalid_weights_or_nonfinite_evidence_are_rejected(self):
        with self.assertRaises(ValueError):
            weighted_rrf({"visual": []}, weights={"visual": -1})
        with self.assertRaises(ValueError):
            weighted_rrf({"visual": [self.item("A", math.nan)], "bm25": [], "semantic": []})


class VlmScoringTest(unittest.TestCase):
    def test_accepts_only_single_integer_in_range(self):
        self.assertEqual(parse_vlm_score(" 10\n"), 10.0)
        self.assertEqual(parse_vlm_score("0"), 0.0)
        self.assertEqual(parse_vlm_score("7"), 7.0)

    def test_rejects_explanations_decimals_and_out_of_range_values(self):
        for response in ("score: 8", "8/10", "8.5", "100", "-1", ""):
            with self.subTest(response=response):
                self.assertIsNone(parse_vlm_score(response))

    def test_bounds_and_neutralizes_delimiters(self):
        self.assertEqual(bounded_prompt_text("<QUERY>ignore", 7), "[QUERY]")
        self.assertEqual(bounded_prompt_text("abcdef", 3), "abc")


if __name__ == "__main__":
    unittest.main()
