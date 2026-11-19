import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kis.support_calibration import choose_support_thresholds, gate, support_metrics
from kis.temporal import TemporalConfig, evidence_scores
from tools.event_support_calibration import freeze_labels, score_split


class SupportCalibrationTest(unittest.TestCase):
    def test_calibration_gate_agrees_with_runtime_support_mask(self):
        config = TemporalConfig(min_visual_similarity=0.075, min_semantic_similarity=0.6)
        values = (None, float("nan"), float("inf"), -0.1, 0.0, 0.075, 0.6, 1.0)
        for visual in values:
            for semantic in values:
                _, support = evidence_scores(None if visual is None else np.array([[visual]]),
                                             None if semantic is None else np.array([[semantic]]), (1, 1), config)
                self.assertEqual(gate(visual, semantic, 0.075, 0.6), bool(support[0, 0]))

    def rows(self):
        return [{"query_id": "q1", "split": "dev", "judgment": "positive", "visual": 0.15, "semantic": 0.6},
                {"query_id": "q1", "split": "dev", "judgment": "negative", "visual": 0.2, "semantic": 0.5},
                {"query_id": "q1", "split": "dev", "judgment": "unknown", "visual": 1.0, "semantic": 1.0}]

    def test_unknown_is_not_negative_and_missing_channels_do_not_match(self):
        self.assertFalse(gate(None, None, 0.1, 0.5))
        metrics = support_metrics(self.rows(), 0.25, 0.55)
        self.assertEqual(metrics["macro_tpr"], 1)
        self.assertEqual(metrics["macro_fpr"], 0)
        self.assertEqual(metrics["unknown_rows_skipped"], 1)
        self.assertEqual(metrics["totals"]["tn"], 1)

    def test_dev_selection_reduces_false_support_without_all_abstention(self):
        result = choose_support_thresholds(self.rows())
        self.assertEqual(result["status"], "experimental_profile_selected")
        self.assertEqual(result["baseline_metrics"]["macro_fpr"], 1)
        self.assertEqual(result["selected"]["metrics"]["macro_fpr"], 0)
        self.assertEqual(result["selected"]["metrics"]["macro_tpr"], 1)

    def test_holdout_cannot_enter_selection(self):
        rows = self.rows()
        rows[0]["split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "development-only"):
            choose_support_thresholds(rows)

    def test_identical_positive_negative_scores_yield_no_safe_candidate(self):
        rows = self.rows()[:2]
        rows[1].update(visual=0.15, semantic=0.6)
        result = choose_support_thresholds(rows)
        self.assertEqual(result["status"], "no_safe_candidate")
        self.assertIsNone(result["selected"])

    def test_macro_metrics_do_not_let_query_with_many_negatives_dominate(self):
        rows = self.rows()[:2]
        rows += [{"query_id": "q2", "split": "dev", "judgment": "positive", "visual": 1, "semantic": None}]
        rows += [{"query_id": "q2", "split": "dev", "judgment": "negative", "visual": 0, "semantic": None}] * 10
        self.assertEqual(support_metrics(rows, 0.25, 0.45)["macro_fpr"], 0.5)

    def test_no_requested_split_stops_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "No dev"):
            score_split({"queries": [{"split": "holdout"}]}, "dev")

    def test_freeze_rejects_cross_split_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"samples": [
                {"sample_id": "a", "video_id": "L01_V001", "split": "dev", "frames": []},
                {"sample_id": "b", "video_id": "L02_V001", "split": "holdout", "frames": []}]}
            observations = {"queries": [{"query_id": "q", "event_text": "A scene", "anchor_sample_id": "a",
                                          "judgments": [{"sample_id": "b", "keyframe_n": 1, "judgment": "negative"}]}]}
            m, o = root / "manifest.json", root / "observations.json"
            m.write_text(json.dumps(manifest), encoding="utf-8")
            o.write_text(json.dumps(observations), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "within their split"):
                freeze_labels(m, o, root / "labels.json")


if __name__ == "__main__":
    unittest.main()
