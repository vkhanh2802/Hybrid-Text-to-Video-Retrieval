import ast
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import l04_l05_milvus_staging as staging


class FakeSchema:
    def __init__(self, **kwargs):
        self.description = {**kwargs, "fields": []}

    def add_field(self, field_name, datatype, **kwargs):
        self.description["fields"].append({
            "name": field_name, "type": datatype,
            "is_primary": kwargs.get("is_primary", False),
            "params": {key: value for key, value in kwargs.items() if key in {"dim", "max_length"}},
        })


class FakeIndex:
    def add_index(self, **kwargs):
        self.description = kwargs


class FakeClient:
    """Only the permitted API exists; any alias/drop/rename/discovery call fails."""

    def __init__(self):
        self.exists = False
        self.rows = []
        self.calls = []
        self.insert_calls = 0
        self.fail_batch = None
        self.short_batch = None
        self.bad_count = False
        self.bad_identity = False
        self.bad_schema = False
        self.skip_top_hit = False
        self.bad_scores = False
        self.reverse_ties = False

    def has_collection(self, **kwargs):
        self.calls.append("has_collection")
        return self.exists

    def create_schema(self, **kwargs):
        return FakeSchema(**kwargs)

    def prepare_index_params(self):
        return FakeIndex()

    def create_collection(self, collection_name, schema, index_params, **kwargs):
        assert not self.exists
        assert kwargs["consistency_level"] == "Strong"
        self.calls.append("create_collection")
        self.exists = True
        self.description = schema.description
        self.index = index_params.description

    def insert(self, collection_name, data, **kwargs):
        self.calls.append("insert")
        self.insert_calls += 1
        if self.insert_calls == self.fail_batch:
            raise RuntimeError("fake transport failure with secret=do-not-record")
        inserted = len(data) - 1 if self.insert_calls == self.short_batch else len(data)
        for row in data[:inserted]:
            assert set(row) == {"video_id", "frame_id", "embedding"}
            assert row["video_id"] in staging.VIDEO_IDS
            np.testing.assert_allclose(np.linalg.norm(row["embedding"]), 1, atol=1e-6)
            self.rows.append({**row, "id": len(self.rows) + 1})
        return {"insert_count": inserted}

    def flush(self, **kwargs):
        self.calls.append("flush")

    def load_collection(self, **kwargs):
        self.calls.append("load_collection")

    def describe_collection(self, **kwargs):
        self.calls.append("describe_collection")
        if self.bad_schema:
            return {**self.description, "enable_dynamic_field": True}
        return self.description

    def describe_index(self, **kwargs):
        return {**self.index, "total_rows": len(self.rows), "indexed_rows": len(self.rows),
                "pending_index_rows": 0, "state": "Finished"}

    def get_load_state(self, **kwargs):
        return {"state": "Loaded"}

    def get_collection_stats(self, **kwargs):
        return {"row_count": len(self.rows) + int(self.bad_count)}

    def query(self, collection_name, output_fields, consistency_level, **kwargs):
        assert consistency_level == "Strong"
        self.calls.append("query")
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(self.rows)}]
        match = re.fullmatch(r'video_id == "(L0[45]_V\d{3})"', kwargs["filter"])
        assert match and match[1] in staging.VIDEO_IDS
        rows = [{key: row[key] for key in output_fields}
                for row in self.rows if row["video_id"] == match[1]]
        if self.bad_identity and rows:
            rows[-1] = dict(rows[0])
        return rows

    def search(self, collection_name, data, consistency_level, **kwargs):
        assert consistency_level == "Strong"
        assert kwargs["search_params"] == {"metric_type": "IP", "params": {"nprobe": 128}}
        self.calls.append("search")
        scores = np.asarray([row["embedding"] for row in self.rows]) @ np.asarray(data[0])
        tie_order = -np.arange(len(scores)) if self.reverse_ties else np.arange(len(scores))
        order = np.lexsort((tie_order, -scores))
        offset = int(self.skip_top_hit)
        order = order[offset:offset + kwargs["limit"]]
        return [[{"id": self.rows[i]["id"], "distance": float(scores[i]) + (0.001 if self.bad_scores else 0),
                  "entity": {key: self.rows[i][key] for key in ("video_id", "frame_id")}}
                 for i in order]]

    def close(self):
        self.calls.append("close")


class MilvusReadinessTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.client.rows = [{}, {}]
        self.client.index = deepcopy(staging.SCHEMA["index"])
        self.report = {"counts": {}}
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds

    def wait(self):
        with patch.object(staging.time, "monotonic", side_effect=lambda: self.now), \
                patch.object(staging.time, "sleep", side_effect=self.advance), \
                patch.object(staging, "READINESS_TIMEOUT", 2):
            staging.wait_for_readiness(self.client, "staging", 2, self.report)

    def test_full_index_and_loaded_state_are_required_without_inventing_progress(self):
        self.wait()
        gate = self.report["readiness"]
        self.assertEqual(gate["status"], "ok")
        self.assertEqual(gate["terminal_reason"], "full_index_coverage_and_loaded")
        self.assertEqual(gate["final_snapshot"]["indexed_rows"], 2)
        self.assertIsNone(gate["final_snapshot"]["load_progress"])
        self.assertEqual(self.now, 0)

    def test_finished_with_zero_indexed_rows_waits_for_full_coverage(self):
        ready = self.client.describe_index()
        incomplete = {**ready, "total_rows": 0, "indexed_rows": 0}
        with patch.object(self.client, "describe_index", side_effect=[incomplete, ready]):
            self.wait()
        gate = self.report["readiness"]
        self.assertEqual(gate["status"], "ok")
        self.assertEqual(len(gate["snapshots"]), 2)
        self.assertEqual(gate["snapshots"][0]["indexed_rows"], 0)
        self.assertEqual(gate["snapshots"][1]["indexed_rows"], 2)
        self.assertEqual(self.now, 1)

    def test_loading_state_waits_instead_of_assuming_loaded(self):
        with patch.object(self.client, "get_load_state", side_effect=[
                {"state": "Loading", "progress": 50}, {"state": "Loaded"}]):
            self.wait()
        gate = self.report["readiness"]
        self.assertEqual(gate["status"], "ok")
        self.assertEqual(gate["snapshots"][0]["load_progress"], 50)
        self.assertEqual(len(gate["snapshots"]), 2)

    def test_missing_index_counts_fail_closed(self):
        for field in ("total_rows", "indexed_rows", "pending_index_rows"):
            with self.subTest(field=field):
                index = self.client.describe_index()
                del index[field]
                with patch.object(self.client, "describe_index", return_value=index):
                    with self.assertRaises(staging.ReadinessError):
                        self.wait()
                self.assertEqual(self.report["readiness"]["status"], "failed")
                self.assertEqual(self.report["readiness"]["terminal_reason"], "missing_or_invalid_count_field")
                self.assertNotIn("search", self.client.calls)

    def test_failed_index_stops_without_search(self):
        index = {**self.client.describe_index(), "state": "Failed"}
        with patch.object(self.client, "describe_index", return_value=index):
            with self.assertRaises(staging.ReadinessError):
                self.wait()
        self.assertEqual(self.report["readiness"]["terminal_reason"], "index_failed")
        self.assertNotIn("search", self.client.calls)

    def test_unknown_load_state_fails_closed(self):
        with patch.object(self.client, "get_load_state", return_value={}):
            with self.assertRaises(staging.ReadinessError):
                self.wait()
        self.assertEqual(self.report["readiness"]["terminal_reason"], "missing_or_invalid_load_state")

    def test_incomplete_index_times_out_without_real_sleep_or_search(self):
        index = {**self.client.describe_index(), "indexed_rows": 1, "pending_index_rows": 1}
        with patch.object(self.client, "describe_index", return_value=index):
            with self.assertRaises(staging.ReadinessError):
                self.wait()
        gate = self.report["readiness"]
        self.assertEqual(gate["status"], "failed")
        self.assertEqual(gate["terminal_reason"], "timeout")
        self.assertEqual(gate["elapsed_seconds"], 2)
        self.assertNotIn("search", self.client.calls)


class MilvusStagingTest(unittest.TestCase):
    def setUp(self):
        approved = Path.home() / "AppData/Local/Temp/opencode"
        directory = tempfile.TemporaryDirectory(dir=approved if approved.is_dir() else None)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = self.root / "dataset"
        self.features = self.dataset / "clip-features-32"
        self.output = self.root / "reports"
        self.client = FakeClient()
        self.collection = "kis_l04_l05_stg_20261108_01"
        self.fixture_expected = {"L04": {"videos": 30, "records": 60},
                                 "L05": {"videos": 31, "records": 62}}

    def fixture(self):
        for i, video in enumerate(staging.VIDEO_IDS):
            mapping = staging.offline.artifact_path(self.dataset, "mapping", video)
            mapping.parent.mkdir(parents=True, exist_ok=True)
            mapping.write_text("n,pts_time,frame_idx\n1,0.5,10\n2,1,20\n", encoding="utf-8")
            features = staging.local.safe_path(self.features, video[:3], video + ".npy")
            features.parent.mkdir(parents=True, exist_ok=True)
            vectors = np.zeros((2, 768), dtype=np.float64)
            vectors[0, i * 2] = 2
            vectors[1, i * 2 + 1] = 3
            np.save(features, vectors)

    def run_mode(self, mode="ingest", **kwargs):
        with patch.object(staging, "EXPECTED", self.fixture_expected):
            return staging.run(mode, self.dataset, self.features, self.output,
                               "http://localhost:19530", self.collection,
                               client_factory=lambda **options: self.client, **kwargs)

    def test_constants_and_reject_scope_before_io(self):
        self.assertEqual(len(staging.VIDEO_IDS), 61)
        self.assertEqual(staging.EXPECTED, {"L04": {"videos": 30, "records": 7471},
                                            "L05": {"videos": 31, "records": 7447}})
        with patch.object(Path, "lstat", side_effect=AssertionError("no IO")), \
                patch.object(Path, "open", side_effect=AssertionError("no IO")):
            for video in ("L06", "L06_V001", "L04_V031", "L05_V032", "L03_V001", "../L04_V001"):
                with self.assertRaises(ValueError):
                    staging.read_video(self.dataset, self.features, video)
                with self.assertRaises(ValueError):
                    staging.run("ingest", self.dataset, self.root / video, self.output,
                                "http://localhost:19530", self.collection)

    def test_reject_remote_credentials_and_production_before_io(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("no IO")):
            for uri in ("http://remote:19530", "http://127.0.0.2:19530", "https://localhost:19530",
                        "http://user:secret@localhost:19530", "http://localhost:19530/db",
                        "http://localhost:19530?token=secret", "http://localhost", "file:///tmp/x"):
                with self.assertRaises(ValueError):
                    staging.run("ingest", self.dataset, self.features, self.output, uri, self.collection)
            for collection in ("clip_keyframes", "arbitrary", "kis_l04_l05_stg_20261108_01;drop"):
                with self.assertRaises(ValueError):
                    staging.validate_target("http://localhost:19530", collection)
        for uri in ("http://localhost:19530", "http://127.0.0.1:19530", "http://[::1]:19530"):
            self.assertEqual(staging.validate_target(uri, self.collection), "http://localhost:19530")

    def test_existing_collection_aborts_without_mutation(self):
        self.fixture()
        self.client.exists = True
        report = self.run_mode()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.client.calls, ["has_collection", "close"])
        self.assertFalse(report["create_attempted"])

    def test_idempotent_create_race_cannot_insert_into_other_process_collection(self):
        self.fixture()
        first = self.run_mode()
        self.assertEqual(first["status"], "ok")
        owner = first["ownership"]["owner_token"]
        self.assertRegex(owner, r"^[0-9a-f]{32}$")

        class RacingClient(FakeClient):
            def has_collection(self, **kwargs):
                self.calls.append("has_collection")
                return False  # B's stale check before A's create became visible.

            def create_collection(self, **kwargs):
                self.calls.append("create_collection")
                # Milvus may succeed idempotently without changing A's schema.

        second_client = RacingClient()
        second_client.exists = True
        second_client.description = deepcopy(self.client.description)
        second_client.index = deepcopy(self.client.index)
        second_client.rows = deepcopy(self.client.rows)
        self.client = second_client
        self.output = self.root / "racing_process_b"
        second = self.run_mode()
        self.assertEqual(second["status"], "failed")
        self.assertNotEqual(second["ownership"]["owner_token"], owner)
        self.assertEqual(second["ownership"]["observed_owner_token"], owner)
        self.assertEqual(second["ownership"]["status"], "mismatch")
        self.assertEqual(self.client.insert_calls, 0)
        self.assertEqual(len(self.client.rows), 122)
        self.assertFalse(second["collection_created"])
        self.assertNotIn("insert", self.client.calls)

    def test_missing_owner_marker_aborts_before_insert(self):
        self.fixture()
        with patch.object(self.client, "describe_collection", return_value={"description": staging.TAG}):
            report = self.run_mode()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["ownership"]["status"], "unverified")
        self.assertEqual(self.client.insert_calls, 0)

    def test_full_ingest_and_verify_no_discovery_config_or_excluded_reads(self):
        self.fixture()
        original_open, original_lstat = Path.open, Path.lstat

        def check(path):
            self.assertNotIn("L06", str(path))
            self.assertNotIn("config", Path(path).parts)

        def guarded_open(path, *args, **kwargs):
            check(path)
            return original_open(path, *args, **kwargs)

        def guarded_lstat(path, *args, **kwargs):
            check(path)
            return original_lstat(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", guarded_open))
            stack.enter_context(patch.object(Path, "lstat", guarded_lstat))
            for method in ("glob", "rglob", "iterdir"):
                stack.enter_context(patch.object(Path, method, side_effect=AssertionError("no discovery")))
            first = self.run_mode(batch_size=17)
            second = self.run_mode("verify")
        for report in (first, second):
            self.assertEqual(report["status"], "ok", report.get("error"))
            self.assertEqual(report["counts"]["collection_entity_count"], 122)
            self.assertEqual(report["counts"]["per_video"], dict.fromkeys(staging.VIDEO_IDS, 2))
            self.assertTrue(report["counts"]["identities_verified"])
            self.assertEqual(report["verified_schema"], {"status": "ok", **staging.SCHEMA})
            self.assertEqual(report["query_count"], 61)
            self.assertEqual(report["top_k_comparison"]["status"], "ok")
            self.assertEqual(report["top_k_comparison"]["matched_queries"], 61)
            self.assertEqual(report["top_k_comparison"]["mismatched_queries"], 0)
            self.assertFalse(report["production_published"])
            self.assertFalse(report["l06_included"])
            self.assertTrue(report["normalized"])
            self.assertEqual(report["accuracy"]["Hit@K"], "not_evaluated")
            self.assertEqual(report["normalized_offline_baseline"]["normalization"], "float32_l2_builder_convention")
        self.assertEqual(first["counts"]["actual_inserted_this_run"], 122)
        self.assertEqual(first["ownership"]["status"], "owned_by_this_run")
        self.assertEqual(first["ownership"]["owner_token"], first["ownership"]["observed_owner_token"])
        self.assertEqual(second["ownership"]["status"], "marker_verified_read_only")
        self.assertEqual(second["ownership"]["observed_owner_token"], first["ownership"]["owner_token"])
        self.assertLess(self.client.calls.index("describe_collection"), self.client.calls.index("insert"))
        self.assertEqual(second["counts"]["actual_inserted_this_run"], 0)
        self.assertEqual(self.client.calls.count("create_collection"), 1)
        self.assertEqual(self.client.calls.count("flush"), 1)

    def test_missing_high_score_hit_fails_even_with_valid_identities_and_scores(self):
        self.fixture()
        self.client.skip_top_hit = True
        report = self.run_mode()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["counts"]["identities_verified"])
        self.assertEqual(report["query_count"], 61)
        self.assertEqual(report["top_k_comparison"]["status"], "issues")
        self.assertEqual(report["top_k_comparison"]["matched_queries"], 0)
        self.assertEqual(report["top_k_comparison"]["mismatched_queries"], 61)
        for query in report["queries"]:
            self.assertEqual(len(query["results"]), 5)
            self.assertIn("missing_high_score_ids", [issue["reason"] for issue in query["issues"]])
            for result in query["results"]:
                self.assertEqual(result["score"], result["normalized_offline_ip"])
        self.assertTrue(self.client.exists)

    def test_wrong_scores_fail_and_record_mismatches(self):
        self.fixture()
        self.client.bad_scores = True
        report = self.run_mode()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["top_k_comparison"]["mismatched_queries"], 61)
        self.assertIn("score_mismatch", [issue["reason"] for issue in report["queries"][0]["issues"]])

    def test_boundary_tied_ids_may_differ_from_deterministic_baseline(self):
        self.fixture()
        self.client.reverse_ties = True
        report = self.run_mode()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["top_k_comparison"]["matched_queries"], 61)
        self.assertEqual(report["top_k_comparison"]["tie_adjusted_matches"], 61)
        self.assertEqual(report["top_k_comparison"]["strict_id_rank_matches"], 0)

    def test_normal_non_tied_exact_top_k_matches(self):
        self.fixture()
        rng = np.random.default_rng(42)
        for video in staging.VIDEO_IDS:
            features = self.features / video[:3] / (video + ".npy")
            np.save(features, rng.normal(size=(2, 768)))
        report = self.run_mode()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["top_k_comparison"]["strict_id_rank_matches"], 61)
        self.assertEqual(report["top_k_comparison"]["tie_adjusted_matches"], 0)

    def test_batch_failure_retains_collection_stops_and_reports_uncertainty(self):
        self.fixture()
        self.client.fail_batch = 2
        report = self.run_mode(batch_size=10)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(self.client.exists)
        self.assertTrue(report["collection_retained"])
        self.assertEqual(report["counts"]["acknowledged_inserted_this_run"], 10)
        self.assertIsNone(report["counts"]["actual_inserted_this_run"])
        self.assertEqual(self.client.calls[-3:], ["insert", "insert", "close"])
        self.assertNotIn("do-not-record", (self.output / "report.json").read_text())

    def test_short_insert_acknowledgement_stops_without_cleanup(self):
        self.fixture()
        self.client.short_batch = 2
        report = self.run_mode(batch_size=10)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["counts"]["actual_inserted_this_run"], 19)
        self.assertEqual(len(self.client.rows), 19)
        self.assertNotIn("flush", self.client.calls)

    def test_count_duplicate_and_schema_mismatches_fail(self):
        for flag in ("bad_count", "bad_identity", "bad_schema"):
            with self.subTest(flag=flag):
                self.output = self.root / flag
                self.client = FakeClient()
                setattr(self.client, flag, True)
                self.fixture()
                report = self.run_mode()
                self.assertEqual(report["status"], "failed" if flag == "bad_count" else "partial")
                if flag == "bad_count":
                    self.assertEqual(report["readiness"]["terminal_reason"], "count_mismatch")
                self.assertTrue(self.client.exists)
                self.assertNotIn("search", self.client.calls)

    def test_preflight_only_and_exclusive_output(self):
        self.fixture()
        self.output = self.root / "l04_l05_milvus_staging_20261108_v2"
        report = self.run_mode("preflight")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(self.client.calls, ["has_collection", "close"])
        before = (self.output / "preflight.json").read_bytes()
        with patch.object(staging, "preflight", side_effect=AssertionError("no payload read")):
            with self.assertRaises(FileExistsError):
                self.run_mode("preflight")
        self.assertEqual((self.output / "preflight.json").read_bytes(), before)

    def test_schema_fields_and_index_parameters_are_exact(self):
        staging.create_collection(self.client, self.collection, "a" * 32)
        original_schema = deepcopy(self.client.description)
        original_index = deepcopy(self.client.index)
        for field_name, key, value in (("embedding", "dim", 384), ("video_id", "max_length", 128)):
            self.client.description = deepcopy(original_schema)
            field = next(f for f in self.client.description["fields"] if f["name"] == field_name)
            field["params"][key] = value
            with self.assertRaises(ValueError):
                staging.verify_schema(self.client, self.collection)
        self.client.description = original_schema
        for key, value in (("metric_type", "L2"), ("index_type", "FLAT"), ("params", {"nlist": 64})):
            self.client.index = {**original_index, key: value}
            with self.assertRaises(ValueError):
                staging.verify_schema(self.client, self.collection)
        self.client.index = original_index
        self.assertEqual(staging.verify_schema(self.client, self.collection)["status"], "ok")

    def test_invalid_vectors_fail_before_client(self):
        self.fixture()
        features = self.features / "L04/L04_V001.npy"
        for i, vectors in enumerate((np.ones((1, 768)), np.ones((2, 384)), np.zeros((2, 768)),
                                     np.full((2, 768), np.nan))):
            np.save(features, vectors)
            self.output = self.root / f"invalid_vectors_{i}"
            self.assertEqual(self.run_mode()["status"], "failed")
            self.assertEqual(self.client.calls, [])

    def test_preflight_count_or_duplicate_blocks_create(self):
        self.fixture()
        mapping = staging.offline.artifact_path(self.dataset, "mapping", "L04_V001")
        mapping.write_text("n,pts_time,frame_idx\n1,0,10\n2,1,10\n", encoding="utf-8")
        report = self.run_mode()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.client.calls, [])
        self.fixture()
        self.output = self.root / "count_mismatch"
        with patch.object(staging, "EXPECTED", {"L04": {"videos": 30, "records": 61},
                                                "L05": {"videos": 31, "records": 62}}):
            report = staging.run("ingest", self.dataset, self.features, self.output,
                                 "http://localhost:19530", self.collection,
                                 client_factory=lambda **options: self.client)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.client.calls, [])

    def test_static_no_unsafe_imports_or_mutating_calls(self):
        tree = ast.parse(Path(staging.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("kis", "tools.build_milvus")))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"drop_collection", "rename_collection", "create_alias",
                                                  "alter_alias", "drop_alias", "delete", "upsert",
                                                  "list_collections", "rglob", "glob", "iterdir"})
        self.assertEqual(json.loads(json.dumps(staging.SCHEMA))["index"]["params"], {"nlist": 128})


if __name__ == "__main__":
    unittest.main()
