import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from kis.text_reranker import TextRerankerConfig
from tools.human_holdout import sha
from tools.pipeline_ablation import metrics
from tools.text_reranker_benchmark import PreparedCaptions, paired, run_case, validate_case, verify


def image_case():
    results = [{"video_id": "v", "frame_id": n, "keyframe_n": n, "score": score,
                "final_score": score, "retrieval_text": text}
               for n, score, text in ((1, .9, "irrelevant"), (2, .8, None), (3, .7, "known positive"))]
    unit = {"query_id": "dev_fixture", "role": "development", "task": "image", "video_id": "v",
            "judgments": [{"video_id": "v", "keyframe_n": 3, "judgment": "positive"}]}
    return {"case_id": "dev_fixture__full", "arm": "full", "unit": unit, "plan": {"raw_query": "query"},
            "results": results, "metrics": metrics(results, unit), "planner_failure": None}


class TextBenchmarkTest(unittest.TestCase):
    def test_cached_case_reranks_without_mutating_baseline_or_missing_text_slot(self):
        case = image_case()
        original = copy.deepcopy(case)
        row = run_case(case, PreparedCaptions({}), Mock(score_pairs=Mock(return_value=[-3, 2])))
        self.assertEqual(row["baseline_metrics"]["known_frame_hit_at_1"], 0)
        self.assertEqual(row["text_metrics"]["known_frame_hit_at_1"], 1)
        self.assertEqual([r["frame_id"] for r in row["results"]], [3, 2, 1])
        self.assertEqual(case, original)
        self.assertEqual(row["results"][1]["score"], .8)
        json.dumps(row, allow_nan=False)

    def test_failed_score_query_is_retained_with_baseline_metrics(self):
        case = image_case()
        row = run_case(case, PreparedCaptions({}), Mock(score_pairs=Mock(side_effect=RuntimeError("injected"))))
        self.assertEqual(row["baseline_metrics"], row["text_metrics"])
        self.assertEqual(row["diagnostics"]["candidates_scored"], 0)
        self.assertEqual(paired([row], "known_frame_hit_at_1")["counts"],
                         {"improved": 0, "regressed": 0, "unchanged": 1})

    def test_outside_pool_positive_cannot_be_promoted_by_text(self):
        case = image_case()
        row = run_case(case, PreparedCaptions({}), Mock(score_pairs=Mock(return_value=[1])),
                       TextRerankerConfig(mode="rerank", pool_size=1))
        self.assertEqual(row["text_metrics"]["first_known_frame_rank"], 3)
        self.assertEqual(row["results"][2]["text_rerank_status"], "outside_pool")

    def test_input_hash_metric_mismatch_and_previous_text_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("original", encoding="utf-8")
            hashes = {str(path): sha(path), str(path.with_name("missing.json")): None}
            verify(hashes)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(hashes)
        case = image_case()
        case["metrics"]["known_frame_hit_at_1"] = 1
        with self.assertRaises(ValueError):
            validate_case(case)
        case = image_case()
        case["results"][0]["text_logit"] = 0
        with self.assertRaises(ValueError):
            validate_case(case)

    def test_prepared_caption_lookup_preserves_missing(self):
        captions = PreparedCaptions({"v:1": {"caption": "original"}, "v:2": None})
        self.assertEqual(captions.get_caption("v", 1), {"caption": "original"})
        self.assertIsNone(captions.get_caption("v", 2))
        self.assertIsNone(captions.get_caption("v", 3))


if __name__ == "__main__":
    unittest.main()
