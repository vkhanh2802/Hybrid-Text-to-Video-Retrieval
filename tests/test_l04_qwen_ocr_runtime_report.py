"""SYNTHETIC, temporary fixtures only. Never inspect a real benchmark run."""

import ast
import copy
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import l04_qwen_ocr_runtime_report as audit


class SyntheticRuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(tempfile.gettempdir()) / "opencode"
        self.temp = tempfile.TemporaryDirectory(
            prefix="l04_runtime_synthetic_", dir=temp_root if temp_root.is_dir() else None)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.run = self.root / "benchmark_results" / "SYNTHETIC"
        self.run.mkdir(parents=True)
        self.output = self.base / "audit"
        self.sources = []
        for name in sorted(audit.SOURCES):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SYNTHETIC source fixture\n", encoding="utf-8")
            self.sources.append({"path": name, "sha256": audit.schema.sha(path)})
        config = copy.deepcopy(audit.schema.CONFIG)
        self.pools = [{"id": qid, "query": "SYNTHETIC robot", "candidates": [
            {"video_id": "L04_V001", "keyframe_n": n, "baseline_position": n,
             "caption_text": "robot", "ocr_status": "missing", "ocr_text": ""}
            for n in range(1, 3 if i == 0 else 2)]} for i, qid in enumerate(audit.schema.IDS)]
        artifacts = {"protocol.json": {"config": config, "query_ids": audit.schema.IDS,
                                       "query_packet_sha256": "a" * 64},
                     "config.json": config, "queries.private.json": {
                         "label": "SYNTHETIC", "queries": [{"id": p["id"], "text": p["query"]} for p in self.pools]},
                     "pools.json": self.pools, "input_manifest.json": [], "coverage.json": {},
                     "preservation.json": [], "source_manifest.json": self.sources}
        for name, value in artifacts.items():
            self.save(name, value)
        self.save("freeze.json", {"files": {name: audit.schema.sha(self.run / name) for name in artifacts},
                                  "planned_pairs": 17, "query_source_sha256": "a" * 64, "frozen_before_model": True})
        cache = {"verification": "SYNTHETIC_no_model"}
        self.save("preflight.json", {"status": "PASS", "cache": cache, "packages_metadata_only": {}})
        self.context = {"freeze_sha256": audit.schema.sha(self.run / "freeze.json"),
                        "preflight_sha256": audit.schema.sha(self.run / "preflight.json"),
                        "source_manifest": self.sources, "config": config, "cache": cache, "packages": {}}
        self.save("run_context.json", self.context)
        self.save("worker_process.json", {"returncode": 0})
        self.save("smoke.json", {"status": "PASS", "included_in_run": True, "warmup_excluded": False})
        self.rows = {key: {"task_key": key, "query_id": p["id"], "baseline_position": d["baseline_position"],
                           "arm": arm, "pair_index": pair, "status": "scored", "score": 0.5,
                           "raw": '{"score":0.5}', "seconds": 0.25 if arm == "A" else 0.5,
                           "input_tokens": 10 if arm == "A" else 12, "generated_tokens": 3}
                     for key, p, d, arm, pair in audit.schema.tasks(self.pools, self.context)}
        self.save("checkpoint.json", self.rows)
        self.report = {"status": "COMPLETE", "metrics": {"SYNTHETIC_UNCHANGED": 1},
                       "preflight_status": "PASS", "blocker": None, "freeze_sha256": self.context["freeze_sha256"],
                       "planned_queries": 16, "planned_pairs": 17, "planned_arm_scores": 34,
                       "completed_arm_scores": 34, "complete_pairs": 17,
                       "per_query": [{"id": p["id"], "pool_size": len(p["candidates"])} for p in self.pools],
                       "runtime": {arm: audit.aggregate([r for r in self.rows.values() if r["arm"] == arm])
                                   for arm in ("A", "B")}}
        self.save("report.json", self.report)

    def save(self, name, value):
        (self.run / name).write_text(json.dumps(value, allow_nan=True), encoding="utf-8")

    def generate(self, **kwargs):
        return audit.write_audit(self.run, self.output, repo_root=self.root, synthetic_test_only=True, **kwargs)

    def refused(self):
        with self.assertRaises((ValueError, OSError)):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_complete_aggregates_and_preserves_every_input(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.generate()
        self.assertEqual(result["status"], "SYNTHETIC_TEST_ONLY")
        self.assertEqual(result["per_query"][0]["A"], {"calls": 2, "seconds_total": 0.5,
                                                     "input_tokens_total": 20, "generated_tokens_total": 6})
        self.assertEqual(result["per_query"][0]["B"]["seconds_total"], 1)
        self.assertEqual(result["runtime"]["A"]["calls"], 17)
        self.assertEqual(result["runtime"]["B"]["input_tokens_total"], 204)
        for arm in ("A", "B"):
            for field in result["runtime"][arm]:
                self.assertEqual(sum(r[arm][field] for r in result["per_query"]), result["runtime"][arm][field])
        self.assertNotIn("metrics", result)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        saved = json.loads((self.output / "runtime_audit.json").read_bytes())
        self.assertEqual(saved, result)
        self.assertEqual(saved["source_sha256"][str(self.run / "checkpoint.json")], audit.schema.sha(self.run / "checkpoint.json"))
        markdown = (self.output / "runtime_audit.md").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC_TEST_ONLY", markdown)
        self.assertIn("| TOTAL | B | 17 |", markdown)

    def test_synthetic_cannot_claim_real_run(self):
        with self.assertRaisesRegex(ValueError, "Synthetic inputs"):
            audit.write_audit(self.run, self.output, repo_root=self.root)
        self.assertFalse(self.output.exists())

    def test_partial_missing_foreign_or_duplicate_pair_rejected(self):
        original = copy.deepcopy(self.rows)
        for change in ("missing", "foreign", "pair", "arm", "task_key", "status", "score", "raw"):
            with self.subTest(change=change):
                self.rows = copy.deepcopy(original)
                key = next(iter(self.rows))
                if change == "missing":
                    del self.rows[key]
                elif change == "foreign":
                    self.rows["f" * 64] = self.rows.pop(key)
                else:
                    field = "pair_index" if change == "pair" else change
                    self.rows[key][field] = {"pair": 999, "arm": "B", "task_key": "bad", "status": "error",
                                             "score": True, "raw": '{"score":0,"score":0.5}'}[change]
                self.save("checkpoint.json", self.rows)
                self.refused()

    def test_invalid_timing_and_tokens_fail_closed(self):
        for field, values in (("seconds", [None, True, "1", 0, -1, float("nan"), float("inf"), -float("inf"), 1e309]),
                              ("input_tokens", [None, True, "10", 0, -1, 1.5]),
                              ("generated_tokens", [None, False, "3", 0, -1, 1.5, 33])):
            for value in values:
                with self.subTest(field=field, value=value):
                    rows = copy.deepcopy(self.rows)
                    rows[next(iter(rows))][field] = value
                    self.save("checkpoint.json", rows)
                    self.refused()

    def test_report_state_counts_and_runtime_must_agree(self):
        for field, value in (("status", "PARTIAL_NO_ACCURACY_CLAIM"), ("metrics", None), ("blocker", "failed"),
                             ("completed_arm_scores", 33), ("complete_pairs", True), ("freeze_sha256", "0" * 64),
                             ("per_query", []), ("runtime", {"A": {"calls": 1}})):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = value
                self.save("report.json", report)
                self.refused()
        for field in self.report["runtime"]["A"]:
            report = copy.deepcopy(self.report)
            report["runtime"]["A"][field] += 1
            self.save("report.json", report)
            self.refused()

    def test_hashes_and_context_must_match(self):
        for filename in ("config.json", "preflight.json", "freeze.json"):
            with self.subTest(filename=filename):
                path = self.run / filename
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.refused()
                path.write_bytes(original)
        path = self.root / self.sources[0]["path"]
        path.write_text("changed", encoding="utf-8")
        self.refused()

    def test_missing_schema_and_unsuccessful_execution(self):
        for filename, value in (("worker_process.json", {"returncode": 1}), ("smoke.json", {}),
                                ("run_context.json", {}), ("report.json", {})):
            with self.subTest(filename=filename):
                path = self.run / filename
                original = path.read_bytes()
                self.save(filename, value)
                self.refused()
                path.write_bytes(original)
        (self.run / "report.json").unlink()
        self.refused()

    def test_duplicate_json_and_overflow_rejected(self):
        (self.run / "checkpoint.json").write_text('{"x":1,"x":2}', encoding="utf-8")
        self.refused()
        for row in self.rows.values():
            row["seconds"] = 1e308
        self.save("checkpoint.json", self.rows)
        self.refused()

    def test_output_scope_existing_and_traversal(self):
        for output in (self.run, self.run / "audit", self.root, self.run / ".." / "audit"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                audit.write_audit(self.run, output, repo_root=self.root, synthetic_test_only=True)
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        for name in ("../escape", "C:/escape", "\\\\server\\share", "/absolute", "a/../../escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.scoped(self.run, name)

    def test_symlink_rejected_when_platform_allows(self):
        link = self.base / "link"
        try:
            link.symlink_to(self.run, target_is_directory=True)
        except OSError as exc:
            self.skipTest("Symlink creation unavailable: " + str(exc))
        with self.assertRaisesRegex(ValueError, "Linked"):
            audit.checked_path(link)

    def test_reparse_and_cloud_flags_rejected_without_symlink_privilege(self):
        for flag in (0x400, 0x1000, 0x40000, 0x400000):
            info = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=flag)
            with self.subTest(flag=flag), patch.object(Path, "lstat", return_value=info):
                with self.assertRaisesRegex(ValueError, "Linked/reparse/cloud"):
                    audit.checked_path(self.run)

    def test_manifest_paths_cannot_escape_even_with_valid_freeze_hash(self):
        for filename, value in (("source_manifest.json", [{"path": "../outside", "sha256": "0" * 64}]),
                                ("preservation.json", [{"path": "benchmarks/../../outside", "sha256": "0" * 64}]),
                                ("preservation.json", [{"path": "dataset/outside", "sha256": "0" * 64}])):
            with self.subTest(filename=filename, value=value):
                path = self.run / filename
                freeze_path = self.run / "freeze.json"
                old, old_freeze = path.read_bytes(), freeze_path.read_bytes()
                self.save(filename, value)
                frozen = json.loads(old_freeze)
                frozen["files"][filename] = audit.schema.sha(path)
                self.save("freeze.json", frozen)
                self.refused()
                path.write_bytes(old)
                freeze_path.write_bytes(old_freeze)

    def test_concurrent_input_mutation_rejected_before_output(self):
        real_aggregate = audit.aggregate

        def mutate(rows):
            path = self.run / "report.json"
            path.write_bytes(path.read_bytes() + b" ")
            return real_aggregate(rows)

        with patch.object(audit, "aggregate", side_effect=mutate):
            self.refused()

    def test_no_model_or_heavy_runtime_calls(self):
        with patch.object(audit.schema, "run", side_effect=AssertionError("model forbidden")), \
                patch.object(audit.schema, "run_context", side_effect=AssertionError("cache forbidden")), \
                patch.object(audit.schema, "verify", side_effect=AssertionError("original verifier forbidden")), \
                patch.object(audit.schema, "lexical", side_effect=AssertionError("OCR forbidden")):
            self.generate()
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        self.assertFalse(imports & {"torch", "numpy", "transformers", "bitsandbytes", "PIL", "socket", "requests"})


if __name__ == "__main__":
    unittest.main()
