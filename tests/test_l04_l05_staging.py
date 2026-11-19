import ast
import builtins
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

import numpy as np

from tools import l04_l05_staging as staging


class StagingTest(unittest.TestCase):
    def setUp(self):
        approved = Path.home() / "AppData/Local/Temp/opencode"
        directory = tempfile.TemporaryDirectory(dir=approved if approved.is_dir() else None)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "data"
        self.images.mkdir(parents=True)
        (self.root / "config").mkdir()
        self.models = {"visual_embedding": {"dimension": 768},
                       "fast_caption_generation": {"model": "fixture", "revision": "pinned"}}
        (self.root / "config/models.json").write_text(json.dumps(self.models), encoding="utf-8")

    def fixture_video(self, video_id="L04_V001"):
        try:
            from PIL import Image
        except ImportError as exc:
            self.skipTest(str(exc))
        path = self.images / video_id[:3] / video_id / "001.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), (20, 30, 40)).save(path)
        caption = [{"video_id": video_id, "frame_id": 10, "keyframe_n": 1,
                    "caption": "fixture", "image_path": f"data/{video_id[:3]}/{video_id}/001.jpg",
                    "image_sha256": staging.local.sha256(path)}]
        for kind in ("mapping", "visual_features", "caption_generator", "caption_mapping"):
            path = staging.artifact_path(self.dataset, kind, video_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "mapping":
                path.write_text("n,pts_time,frame_idx\n1,0.5,10\n", encoding="utf-8")
            elif kind == "visual_features":
                np.save(path, np.ones((1, 768), dtype=np.float32))
            else:
                path.write_text(json.dumps(caption), encoding="utf-8")
        path = self.dataset / "objects" / video_id / "001.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def run_staging(self, name, mode="structural-smoke"):
        with redirect_stderr(io.StringIO()):
            return staging.run(mode, self.dataset, self.images, self.root / name, repo_root=self.root)

    def test_scope_exact_and_reject_before_any_io(self):
        self.assertEqual(staging.VIDEO_IDS, staging.local.VIDEO_IDS)
        self.assertEqual(len(staging.VIDEO_IDS), 61)
        self.assertEqual(staging.EXPECTED, {"L04": {"videos": 30, "records": 7471},
                                          "L05": {"videos": 31, "records": 7447}})
        with patch.object(Path, "lstat", side_effect=AssertionError("no IO")), \
             patch.object(Path, "open", side_effect=AssertionError("no IO")):
            for video in ("L06", "L06_V001", "L04_V031", "L05_V032", "L03_V001", "../L04_V001"):
                with self.subTest(video=video):
                    with self.assertRaises(staging.local.UnsafePath):
                        staging.artifact_path(self.dataset, "mapping", video)
                    with self.assertRaises(staging.local.UnsafePath):
                        staging.check_video(self.dataset, self.images, video, {}, None)
                    with self.assertRaises(staging.local.UnsafePath):
                        staging.validate_identities([(video, 1)])
            with self.assertRaises(staging.local.UnsafePath):
                staging.run("preflight", self.root / "L06", self.root / "L06/data", self.root / "out")

    def test_vector_dimension_count_dtype_finite_zero(self):
        for value, count in ((np.ones((1, 384)), 1), (np.ones((2, 768)), 1),
                             (np.ones(768), 1), (np.ones((1, 768), dtype=int), 1),
                             (np.zeros((1, 768)), 1), (np.full((1, 768), np.nan), 1),
                             (np.full((1, 768), np.inf), 1), (np.empty((0, 768)), 0)):
            with self.subTest(shape=value.shape, count=count):
                with self.assertRaises(ValueError):
                    staging.validate_vectors(value, count)
        self.assertEqual(staging.validate_vectors(np.ones((1, 768)), 1)["dimension"], 768)

    def test_module_imports_and_calls_have_no_discovery_or_service_clients(self):
        tree = ast.parse(Path(staging.__file__).read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"rglob", "glob", "iterdir", "walk", "scandir",
                                                 "connect", "create_connection", "urlopen"})
        self.assertEqual(imports, {"argparse", "collections", "hashlib", "json", "pathlib",
                                   "sys", "numpy", "tools"})

    def test_duplicate_scope_and_identity_order(self):
        for pairs in ([("L04_V001", 1), ("L04_V001", 1)],
                      [("L04_V001", 2), ("L04_V001", 1)],
                      [("L05_V001", 1), ("L04_V001", 1)], [("L04_V001", True)]):
            with self.assertRaises(ValueError):
                staging.validate_identities(pairs)
        staging.validate_identities([("L04_V001", 1), ("L05_V031", 1)])

    def test_exact_visual_ip_ties_scope_mapping_and_not_self_top1(self):
        vectors = np.zeros((4, 768), dtype=np.float32)
        vectors[:, 0] = [1, 1, 1, 2]
        pairs = [("L04_V001", i) for i in range(4)]
        report = staging.exact_ip(vectors, pairs, [0, 2], k=2)
        self.assertEqual(report["dimension"], 768)
        self.assertEqual(report["query_count"], 2)
        self.assertFalse(report["semantic_quality_evaluated"])
        query = report["queries"][0]
        self.assertEqual([result["row_id"] for result in query["results"]], [3, 0])
        self.assertEqual([result["score"] for result in query["results"]], [2.0, 1.0])
        self.assertEqual(query["boundary_score_ties"], 3)
        self.assertEqual(query["boundary_ties_returned"], 1)
        self.assertEqual(query["self_rank"], 2)
        self.assertEqual(report["queries"][1]["self_rank"], 4)
        for result in query["results"]:
            self.assertEqual((result["video_id"], result["frame_id"]), pairs[result["row_id"]])
        self.assertEqual(report, staging.exact_ip(vectors, pairs, [0, 2], k=2))
        all_tied = staging.exact_ip(vectors[:3], pairs[:3], [2], k=2)["queries"][0]
        self.assertEqual(all_tied["top_score_ties"], 3)
        self.assertEqual([result["row_id"] for result in all_tied["results"]], [0, 1])

    def test_bad_query_k_and_mapping_count(self):
        vectors = np.ones((1, 768))
        pairs = [("L04_V001", 1)]
        for queries, k in (([-1], 1), ([1], 1), ([True], 1), ([0], 0), ([0], True)):
            with self.assertRaises(ValueError):
                staging.exact_ip(vectors, pairs, queries, k)
        with self.assertRaises(ValueError):
            staging.exact_ip(vectors, pairs + [("L04_V001", 2)], [0])

    def test_direct_artifact_readiness_and_count_mismatch(self):
        self.fixture_video()
        from PIL import Image
        detail, vectors, pairs = staging.check_video(self.dataset, self.images, "L04_V001", self.models, Image)
        self.assertEqual(detail["status"], "ok")
        self.assertEqual(detail["objects"]["records"], 1)
        self.assertEqual(detail["images"]["decoded"], 1)
        self.assertEqual(pairs, [("L04_V001", 10)])
        self.assertEqual(vectors.shape, (1, 768))
        path = staging.artifact_path(self.dataset, "visual_features", "L04_V001")
        np.save(path, np.ones((2, 768)))
        with self.assertRaisesRegex(ValueError, "count"):
            staging.check_video(self.dataset, self.images, "L04_V001", self.models, Image)

    def test_mapping_frame_and_keyframe_order_rejected(self):
        self.fixture_video()
        path = staging.artifact_path(self.dataset, "mapping", "L04_V001")
        for rows in ("1,0.5,10\n2,0.6,10\n", "1,0.5,20\n2,0.6,10\n", "2,0.5,10\n"):
            path.write_text("n,pts_time,frame_idx\n" + rows, encoding="utf-8")
            with self.assertRaises(ValueError):
                staging.check_video(self.dataset, self.images, "L04_V001", self.models, None)

    def test_missing_hash_or_decoder_stays_unknown(self):
        self.fixture_video()
        detail, _, _ = staging.check_video(self.dataset, self.images, "L04_V001", self.models, None)
        self.assertEqual(detail["status"], "unknown")
        from PIL import Image
        path = staging.artifact_path(self.dataset, "caption_generator", "L04_V001")
        captions = json.loads(path.read_text(encoding="utf-8"))
        del captions[0]["image_sha256"]
        path.write_text(json.dumps(captions), encoding="utf-8")
        detail, _, _ = staging.check_video(self.dataset, self.images, "L04_V001", self.models, Image)
        self.assertEqual(detail["status"], "unknown")

    def test_full_run_no_enumeration_network_milvus_or_excluded_io_and_determinism(self):
        for video in staging.VIDEO_IDS:
            self.fixture_video(video)
        original_open, original_lstat, original_import = Path.open, Path.lstat, builtins.__import__
        touched = []

        def check_path(path):
            self.assertNotIn("L06", str(path))
            self.assertNotIn("pipeline.json", str(path))
            self.assertNotIn("images.local.json", str(path))

        def guarded_open(path, *args, **kwargs):
            check_path(path)
            touched.append(str(path))
            return original_open(path, *args, **kwargs)

        def guarded_lstat(path, *args, **kwargs):
            check_path(path)
            return original_lstat(path, *args, **kwargs)

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("pymilvus", "milvus", "faiss", "kis", "google", "requests", "torch", "transformers")):
                raise AssertionError(f"Forbidden import: {name}")
            return original_import(name, *args, **kwargs)

        # Full allowlist with reduced temp-fixture record totals only.
        fixture_expected = {"L04": {"videos": 30, "records": 30}, "L05": {"videos": 31, "records": 31}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(staging, "EXPECTED", fixture_expected))
            stack.enter_context(patch.object(Path, "open", guarded_open))
            stack.enter_context(patch.object(Path, "lstat", guarded_lstat))
            stack.enter_context(patch.object(builtins, "__import__", guarded_import))
            blocked = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
                       for owner, name in ((Path, "rglob"), (Path, "glob"), (Path, "iterdir"),
                                           (socket.socket, "connect"), (socket, "create_connection"),
                                           (urllib.request, "urlopen"))]
            first = self.run_staging("one")
            self.run_staging("two")
            for mock in blocked:
                mock.assert_not_called()
        self.assertTrue(touched)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["index"]["query_count"], 61)
        self.assertEqual(first["milvus"]["status"], "not_started")
        self.assertEqual(first["accuracy"]["status"], "not_evaluated")
        self.assertEqual(first["accuracy"]["Hit@K"], "not_evaluated")
        self.assertEqual(first["accuracy"]["MRR"], "not_evaluated")
        self.assertEqual((self.root / "one/report.json").read_bytes(), (self.root / "two/report.json").read_bytes())

    def test_output_collision_and_input_protection(self):
        output = self.root / "existing"
        output.mkdir()
        path = output / "report.json"
        path.write_bytes(b"preserve")
        with patch.object(staging, "check_video", side_effect=AssertionError("no input reads")):
            with self.assertRaises(FileExistsError):
                self.run_staging("existing")
        self.assertEqual(path.read_bytes(), b"preserve")
        for output in (self.dataset / "output", self.root):
            with self.assertRaises(staging.local.UnsafePath):
                staging.run("preflight", self.dataset, self.images, output)
        with self.assertRaises(staging.local.UnsafePath):
            staging.run("preflight", self.dataset, self.root / "drive", self.root / "other")

    def test_missing_scope_counts_blocks_index_and_accuracy_not_evaluated(self):
        self.fixture_video()
        with patch.object(staging, "exact_ip", side_effect=AssertionError("not ready")):
            report = self.run_staging("incomplete")
        self.assertEqual(report["status"], "issues")
        self.assertEqual(report["summary"]["exact_counts_status"], "issues")
        self.assertEqual(report["index"]["status"], "not_started")
        self.assertEqual(report["index"]["query_count"], 0)
        self.assertEqual(report["accuracy"]["status"], "not_evaluated")

    def test_preflight_never_queries_and_cli_only_safe_modes(self):
        with patch.object(staging, "exact_ip", side_effect=AssertionError("preflight has no queries")):
            report = self.run_staging("preflight", mode="preflight")
        self.assertEqual(report["index"]["reason"], "preflight_only")
        with patch.object(Path, "lstat", side_effect=AssertionError("no IO")):
            with self.assertRaises(ValueError):
                staging.run("ingest", self.dataset, self.images, self.root / "ingest")
        for status, code in (("ok", 0), ("unknown", 1), ("issues", 1)):
            with patch.object(staging, "run", return_value={"status": status, "summary": {},
                                                          "index": {"status": "not_started", "query_count": 0}}):
                with patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(staging.main(["preflight", "--dataset-root", str(self.dataset),
                                                   "--image-root", str(self.images),
                                                   "--output-dir", str(self.root / "cli")]), code)


if __name__ == "__main__":
    unittest.main()
