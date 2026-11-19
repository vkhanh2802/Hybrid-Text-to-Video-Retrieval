from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kis.paths import KIS_ROOT
from kis.temporal import TemporalConfig
from tools.temporal_pilot import alignment_controls, load_queries, run


class TemporalPilotTest(unittest.TestCase):
    def test_missing_model_cache_stops_before_retrieval_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            with patch("tools.temporal_pilot.require_model_cache", side_effect=RuntimeError("STOP: missing cached model")), \
                 patch.dict("os.environ", {}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "STOP"):
                    run(KIS_ROOT / "benchmarks/temporal_pilot.json", output)
            self.assertEqual(json.loads((output / "failure.json").read_text(encoding="utf-8"))["type"], "RuntimeError")
            self.assertFalse((output / "retrieval.log").exists())
            with self.assertRaises(FileExistsError):
                run(KIS_ROOT / "benchmarks/temporal_pilot.json", output)

    def test_default_queries_are_explicit_fixed_sequences(self):
        queries = load_queries(KIS_ROOT / "benchmarks/temporal_pilot.json")
        self.assertEqual(len(queries), 3)
        for query in queries:
            self.assertEqual(query["temporal_constraints"][0]["order"], [event["action"] for event in query["events"]])
            self.assertEqual(query["entities"], [])

    def test_bad_fixture_rejected_before_models_load(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "query.json"
            for queries in ([], [{"query_id": "../escape"}], [{"query_id": "good", "raw_query": "x"}]):
                path.write_text(json.dumps({"schema_version": 1, "queries": queries}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_queries(path)

    def test_same_pool_controls_distinguish_order_from_unordered_coverage(self):
        frames = [{"frame_id": 10, "pts_time": 1}, {"frame_id": 20, "pts_time": 2}]
        scores = np.eye(2)
        controls = alignment_controls(frames, scores, scores > 0, TemporalConfig())
        self.assertEqual(controls["ordered"]["coverage"], 1)
        self.assertEqual(controls["unordered"]["coverage"], 1)
        self.assertEqual(controls["reversed_order"]["coverage"], 0.5)
        json.dumps(controls, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
