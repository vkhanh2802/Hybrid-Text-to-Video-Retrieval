import builtins
import hashlib
import json
from pathlib import Path
import socket
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request

import numpy as np

from tools import l04_l05_local_smoke as smoke


class LocalSmokeTest(unittest.TestCase):
    def setUp(self):
        approved_temp = Path.home() / "AppData/Local/Temp/opencode"
        self.directory = tempfile.TemporaryDirectory(dir=approved_temp if approved_temp.is_dir() else None)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "data"
        self.images.mkdir(parents=True)
        self.config = {
            "visual_embedding": {"dimension": 768},
            "caption_embedding": {"model": "fixture", "revision": "pinned", "dimension": 384,
                                  "normalized": True, "metric": "IP"},
            "fast_caption_generation": {"model": "fixture", "revision": "pinned"},
        }
        (self.root / "config").mkdir()
        self.write_json(self.root / "config/models.json", self.config)
        (self.root / "config/pipeline.json").write_text('{"video_groups": ["L01"]}', encoding="utf-8")
        self.video_id = "L04_V001"
        self.frames = [{"frame_id": 10, "keyframe_n": 1, "pts_time": 0.5}]

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self):
        try:
            import faiss
            from PIL import Image
        except ImportError as exc:
            self.skipTest(f"Optional fixture dependency unavailable: {exc}")
        image_path = self.images / "L04" / self.video_id / "001.jpg"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 3), (20, 30, 40)).save(image_path)
        captions = [{"video_id": self.video_id, "frame_id": 10, "keyframe_n": 1,
                     "image_path": "data/L04/L04_V001/001.jpg", "caption": "fixture",
                     "image_sha256": smoke.sha256(image_path)}]
        for kind in smoke.ARTIFACTS:
            path = smoke.canonical_path(self.dataset, kind, self.video_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "mapping":
                path.write_text("n,pts_time,fps,frame_idx\n1,0.5,25,10\n", encoding="utf-8")
            elif kind in {"caption_generator", "caption_mapping"}:
                self.write_json(path, captions)
            elif kind == "visual_features":
                np.save(path, np.ones((1, 768), dtype=np.float32))
            elif kind == "index":
                index = faiss.IndexFlatIP(384)
                index.add(np.ones((1, 384), dtype=np.float32) / np.sqrt(384))
                faiss.write_index(index, str(path))
            else:
                names = {"caption_sha256": "caption_generator", "mapping_sha256": "caption_mapping", "index_sha256": "index"}
                self.write_json(path, {"schema_version": 1, "video_id": self.video_id, "count": 1,
                                       "embedding": self.config["caption_embedding"],
                                       "files": {name: smoke.sha256(smoke.canonical_path(self.dataset, kind, self.video_id))
                                                 for name, kind in names.items()}})
        self.write_json(self.dataset / "objects/L04/L04_V001/001.json", {})
        return faiss, Image, image_path

    def check_video(self, faiss, decoder):
        return smoke.check_video(self.dataset, self.images, self.video_id, self.config, faiss, decoder)

    def test_fixed_scope_and_canonical_paths_never_contain_excluded_group(self):
        self.assertEqual(len(smoke.VIDEO_IDS), 61)
        self.assertEqual(sum(value["records"] for value in smoke.EXPECTED.values()), 14918)
        self.assertEqual(smoke.VIDEO_IDS[:30], tuple(f"L04_V{i:03d}" for i in range(1, 31)))
        self.assertEqual(smoke.VIDEO_IDS[30:], tuple(f"L05_V{i:03d}" for i in range(1, 32)))
        for video in smoke.VIDEO_IDS:
            for kind, (directory, suffix) in smoke.ARTIFACTS.items():
                path = smoke.canonical_path(self.dataset, kind, video)
                self.assertEqual(path, self.dataset / directory / video.split("_")[0] / (video + suffix))
                self.assertNotIn("L06", str(path))
        with patch.object(Path, "lstat", side_effect=AssertionError("no excluded IO")):
            for video in ("L06_V001", "L04_V031", "../L04_V001", "L05/../L06_V001"):
                with self.assertRaises(smoke.UnsafePath):
                    smoke.canonical_path(self.dataset, "mapping", video)

    def test_traversal_and_network_roots_rejected_before_io(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("no unsafe IO")):
            for part in ("..", "../outside", "L06_V001", "a/b", "a\\b", "C:drive", ""):
                with self.assertRaises(smoke.UnsafePath):
                    smoke.safe_path(self.images, part)
            for root in (self.root / "../outside", Path("//server/share")):
                with self.assertRaises(smoke.UnsafePath):
                    smoke.safe_path(root)

    def test_reparse_symlink_and_parent_placeholder_not_opened(self):
        original = Path.lstat
        for mode, attributes, expected in ((stat.S_IFLNK, 0, smoke.UnsafePath),
                                           (stat.S_IFDIR, smoke.REPARSE_FLAG, smoke.UnsafePath),
                                           (stat.S_IFDIR, smoke.PLACEHOLDER_FLAGS, smoke.PlaceholderPath)):
            def fake(path, *args, **kwargs):
                if path == self.images:
                    return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
                return original(path, *args, **kwargs)

            with patch.object(Path, "lstat", fake), patch.object(Path, "open", side_effect=AssertionError("no payload")):
                with self.assertRaises(expected):
                    smoke.safe_path(self.images, "L04", "L04_V001", "001.jpg")

    def test_valid_video_counts_hashes_decode_and_faiss(self):
        faiss, decoder, _ = self.fixture()
        result = self.check_video(faiss, decoder)
        self.assertEqual(result["readiness"], {"artifact": "ok", "image": "ok", "faiss": "ok"})
        self.assertEqual(result["artifacts"]["visual_features"]["shape"], [1, 768])
        index = result["artifacts"]["index"]
        self.assertEqual((index["rows"], index["dimension"], index["metric"]), (1, 384, "IP"))
        self.assertEqual(index["query"], "synthetic_caption_space_unit_vector")
        self.assertEqual(result["images"][0]["dimensions"], [4, 3])
        self.assertEqual(result["images"][0]["caption_hash_status"], "ok")

    def test_missing_corrupt_zero_and_decoder_unknown_semantics(self):
        _, decoder, path = self.fixture()
        result = smoke.check_image(self.images, self.video_id, self.frames[0], None, None)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["decoded"])
        for payload in (b"not an image", b""):
            path.write_bytes(payload)
            self.assertEqual(smoke.check_image(self.images, self.video_id, self.frames[0], None, decoder)["status"], "invalid")
        path.unlink()
        self.assertEqual(smoke.check_image(self.images, self.video_id, self.frames[0], None, decoder)["status"], "missing")

    def test_image_placeholder_is_unknown_without_payload_read(self):
        original = Path.lstat

        def placeholder(path, *args, **kwargs):
            if path == self.images:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x1000)
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", placeholder), patch.object(Path, "open", side_effect=AssertionError("no hydrate")):
            result = smoke.check_image(self.images, self.video_id, self.frames[0], None, None)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["decoded"])

    def test_caption_local_image_hash_mismatch_is_not_pass(self):
        _, decoder, _ = self.fixture()
        result = smoke.check_image(self.images, self.video_id, self.frames[0], {"image_sha256": "0" * 64}, decoder)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["caption_hash_status"], "invalid")

    def test_caption_missing_or_malformed_hash_cannot_pass_after_decode(self):
        _, decoder, _ = self.fixture()
        for caption, expected in ((None, "unknown"), ({}, "unknown"), ({"image_sha256": None}, "unknown"),
                                  ({"image_sha256": ""}, "invalid"), ({"image_sha256": "not-sha256"}, "invalid"),
                                  ({"image_sha256": 123}, "invalid"), ({"image_sha256": "A" * 64}, "invalid")):
            with self.subTest(caption=caption):
                result = smoke.check_image(self.images, self.video_id, self.frames[0], caption, decoder)
                self.assertEqual(result["caption_hash_status"], expected)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["decoded"], expected == "unknown")
                self.assertIn("SHA-256", result["reason"])

    def test_caption_identities_are_canonicalized_before_lookup(self):
        self.fixture()
        for kind in ("caption_generator", "caption_mapping"):
            path = smoke.canonical_path(self.dataset, kind, self.video_id)
            original = json.loads(path.read_text(encoding="utf-8"))
            for number in (1, "1", "001"):
                with self.subTest(kind=kind, number=number):
                    self.write_json(path, [dict(original[0], frame_id="10", keyframe_n=number)])
                    before = path.read_bytes()
                    records = smoke.read_captions(path, self.video_id, self.frames, kind, self.config)
                    self.assertEqual(records[0]["keyframe_n"], 1)
                    self.assertIs(type(records[0]["keyframe_n"]), int)
                    self.assertIs(type(records[0]["frame_id"]), int)
                    self.assertEqual(records[0]["image_sha256"], original[0]["image_sha256"])
                    self.assertEqual(path.read_bytes(), before)

    def test_caption_hash_readiness_summary_and_exit_code(self):
        faiss, decoder, image_path = self.fixture()
        path = smoke.canonical_path(self.dataset, "caption_generator", self.video_id)
        original = json.loads(path.read_text(encoding="utf-8"))[0]
        manifest_path = smoke.canonical_path(self.dataset, "index_manifest", self.video_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid_hash = smoke.sha256(image_path)
        cases = ((1, {}, "unknown"), (1, {"image_sha256": "bad"}, "invalid"),
                 (1, {"image_sha256": "0" * 64}, "invalid"),
                 ("1", {"image_sha256": "0" * 64}, "invalid"),
                 (1, {"image_sha256": valid_hash}, "ok"), ("1", {"image_sha256": valid_hash}, "ok"))
        for number, hashes, expected in cases:
            with self.subTest(number=number, hashes=hashes):
                record = dict(original, keyframe_n=number)
                del record["image_sha256"]
                self.write_json(path, [dict(record, **hashes)])
                # Keep unrelated artifact checks valid so exit failure is caused by image readiness alone.
                manifest["files"]["caption_sha256"] = smoke.sha256(path)
                self.write_json(manifest_path, manifest)
                video = self.check_video(faiss, decoder)
                readiness = "issues" if expected == "invalid" else expected
                self.assertEqual(video["readiness"], {"artifact": "ok", "image": readiness, "faiss": "ok"})
                self.assertEqual(len(video["images"]), 1)
                self.assertEqual(video["images"][0]["status"], expected)
                self.assertEqual(video["images"][0]["caption_hash_status"], expected)
                totals = smoke.summary([video])
                totals.update(exact_counts_status="ok")
                totals["readiness"]["dependency"] = "ok"
                self.assertEqual(totals["images"], {expected: 1})
                self.assertEqual(totals["caption_image_hashes"], {expected: 1})
                self.assertEqual(totals["readiness"]["image"], readiness)
                with patch.object(smoke, "run", return_value={"summary": totals}):
                    self.assertEqual(smoke.main(["--image-root", str(self.images), "--output-dir", str(self.root / "out")]),
                                     0 if expected == "ok" else 1)

    def test_caption_identity_order_and_stored_path_validation(self):
        self.fixture()
        path = smoke.canonical_path(self.dataset, "caption_mapping", self.video_id)
        original = json.loads(path.read_text())
        for field, value in (("image_path", "data/L06/L06_V001/001.jpg"),
                             ("image_path", "data/L04/L04_V001/../001.jpg"),
                             ("video_id", "L06_V001"), ("frame_id", 11), ("keyframe_n", True)):
            records = [dict(original[0], **{field: value})]
            self.write_json(path, records)
            with self.assertRaises(ValueError):
                smoke.read_captions(path, self.video_id, self.frames, "caption_mapping", self.config)
        self.write_json(path, original + original)
        with self.assertRaisesRegex(ValueError, "row order"):
            smoke.read_captions(path, self.video_id, self.frames, "caption_mapping", self.config)

    def test_mapping_duplicate_and_timestamp_order_validation(self):
        path = self.root / "mapping.csv"
        for rows in ("1,10,0.5\n1,11,1.0\n", "1,10,0.5\n2,10,1.0\n",
                     "1,10,1.0\n2,11,0.5\n", "1,10,nan\n", ""):
            path.write_text("n,frame_idx,pts_time\n" + rows, encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke.read_mapping(path)

    def test_caption_same_identities_in_different_order_rejected(self):
        path = self.root / "caption_mapping.json"
        frames = self.frames + [{"frame_id": 20, "keyframe_n": 2, "pts_time": 1.0}]
        records = [{"video_id": self.video_id, "frame_id": frame["frame_id"],
                    "keyframe_n": frame["keyframe_n"],
                    "image_path": smoke.image_relative(self.video_id, frame["keyframe_n"])}
                   for frame in reversed(frames)]
        self.write_json(path, records)
        with self.assertRaisesRegex(ValueError, "row order"):
            smoke.read_captions(path, self.video_id, frames, "caption_mapping", self.config)

    def test_bad_feature_shape_dimension_nan_and_zero(self):
        path = self.root / "features.npy"
        for array in (np.ones((2, 768)), np.ones((1, 384)), np.ones(768),
                      np.full((1, 768), np.nan), np.full((1, 768), np.inf), np.zeros((1, 768))):
            np.save(path, array)
            with self.assertRaises(ValueError):
                smoke.check_features(path, 1, self.config["visual_embedding"])

    def test_manifest_hash_and_embedding_mismatch(self):
        faiss, decoder, _ = self.fixture()
        path = smoke.canonical_path(self.dataset, "index_manifest", self.video_id)
        original = json.loads(path.read_text())
        for bad in (dict(original, count=2), dict(original, video_id="L06_V001"),
                    dict(original, embedding=dict(original["embedding"], dimension=768)),
                    dict(original, files=dict(original["files"], mapping_sha256="0" * 64))):
            self.write_json(path, bad)
            self.assertEqual(self.check_video(faiss, decoder)["artifacts"]["index_manifest"]["status"], "invalid")

    def test_index_corrupt_dimension_metric_and_dependency_unknown(self):
        faiss, decoder, _ = self.fixture()
        path = smoke.canonical_path(self.dataset, "index", self.video_id)
        for index in (faiss.IndexFlatIP(768), faiss.IndexFlatL2(384), faiss.IndexFlatIP(384)):
            index.add(np.ones((1 if index.d != 384 or index.metric_type == faiss.METRIC_L2 else 2, index.d), dtype=np.float32))
            faiss.write_index(index, str(path))
            with self.assertRaisesRegex(ValueError, "FAISS"):
                smoke.check_index(path, 1, self.config["caption_embedding"], faiss)
        path.write_bytes(b"corrupt index")
        self.assertEqual(self.check_video(faiss, decoder)["artifacts"]["index"]["status"], "invalid")
        self.assertEqual(self.check_video(None, decoder)["artifacts"]["index"]["status"], "unknown")

    def test_structural_query_uses_caption_dimension_and_checks_labels(self):
        calls = []

        class Index:
            ntotal, d, metric_type, is_trained = 1, 384, 0, True

            def search(self, query, k):
                calls.append((query.copy(), k))
                return np.zeros((1, 1), dtype=np.float32), np.array([[-1]])

        fake = SimpleNamespace(METRIC_INNER_PRODUCT=0, METRIC_L2=1, read_index=lambda path: Index())
        with self.assertRaisesRegex(ValueError, "labels"):
            smoke.check_index(self.root / "unused", 1, self.config["caption_embedding"], fake)
        self.assertEqual(calls[0][0].shape, (1, 384))
        self.assertEqual(float(np.linalg.norm(calls[0][0])), 1.0)

    def test_explicit_legacy_objects_supported_and_ambiguity_not_pass(self):
        path = self.dataset / "objects/L04_V001/001.json"
        self.write_json(path, {})
        result = smoke.check_objects(self.dataset, self.video_id, self.frames)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["coverage"], {"ok": 1})
        self.assertEqual(result["layout"], "legacy_explicit")
        self.assertEqual(result["path"], "objects/L04_V001")
        self.write_json(self.dataset / "objects/L04/L04_V001/001.json", [])
        with patch.object(Path, "read_text", side_effect=AssertionError("no ambiguous payload read")):
            result = smoke.check_objects(self.dataset, self.video_id, self.frames)
        self.assertEqual(result["coverage"], {"invalid": 1})
        self.assertIn("Ambiguous", result["reason"])
        self.assertEqual(result["records"], 0)

    def test_explicit_l05_objects_no_scan_and_excluded_id_no_io(self):
        video_id = "L05_V031"
        self.write_json(self.dataset / "objects" / video_id / "001.json", {})
        with patch.object(Path, "rglob", side_effect=AssertionError("no recursive discovery")) as rglob, \
             patch.object(Path, "iterdir", side_effect=AssertionError("no enumeration")) as iterdir:
            result = smoke.check_objects(self.dataset, video_id, self.frames)
        rglob.assert_not_called()
        iterdir.assert_not_called()
        self.assertEqual(result["coverage"], {"ok": 1})
        self.assertEqual(result["path"], "objects/L05_V031")
        with patch.object(Path, "lstat", side_effect=AssertionError("no excluded IO")):
            with self.assertRaises(smoke.UnsafePath):
                smoke.check_objects(self.dataset, "L06_V001", self.frames)

    def test_missing_and_corrupt_objects_not_pass(self):
        self.assertEqual(smoke.check_objects(self.dataset, self.video_id, self.frames)["coverage"], {"missing": 1})
        path = self.dataset / "objects/L04/L04_V001/001.json"
        self.write_json(path, [])
        self.assertEqual(smoke.check_objects(self.dataset, self.video_id, self.frames)["coverage"], {"invalid": 1})

    def test_legacy_object_reparse_and_placeholder_no_payload_read(self):
        self.write_json(self.dataset / "objects/L04_V001/001.json", {})
        original = Path.lstat
        for attributes, expected in ((smoke.REPARSE_FLAG, "unsafe"), (smoke.PLACEHOLDER_FLAGS, "unknown")):
            def fake(path, *args, **kwargs):
                if path == self.dataset / "objects/L04_V001":
                    return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=attributes)
                return original(path, *args, **kwargs)

            with patch.object(Path, "lstat", fake), \
                 patch.object(Path, "read_text", side_effect=AssertionError("no unsafe payload")):
                result = smoke.check_objects(self.dataset, self.video_id, self.frames)
            self.assertEqual(result["coverage"], {expected: 1})
            self.assertEqual(result["records"], 0)

    def test_run_no_discovery_network_drive_import_or_excluded_path_and_determinism(self):
        self.fixture()
        original_open, original_lstat, original_import = Path.open, Path.lstat, builtins.__import__
        touched = []

        def checked_open(path, *args, **kwargs):
            self.assertNotIn("L06", str(path))
            self.assertNotIn("images.local.json", str(path))
            touched.append(str(path))
            return original_open(path, *args, **kwargs)

        def checked_lstat(path, *args, **kwargs):
            self.assertNotIn("L06", str(path))
            return original_lstat(path, *args, **kwargs)

        def checked_import(name, *args, **kwargs):
            if name.startswith(("kis", "google", "pymilvus", "requests")):
                raise AssertionError(f"No runtime/Drive import: {name}")
            return original_import(name, *args, **kwargs)

        pipeline_before = smoke.sha256(self.root / "config/pipeline.json")
        with patch.object(Path, "open", checked_open), patch.object(Path, "lstat", checked_lstat), \
             patch.object(Path, "rglob", side_effect=AssertionError("no recursive discovery")) as rglob, \
             patch.object(Path, "glob", side_effect=AssertionError("no discovery")) as glob, \
             patch.object(Path, "iterdir", side_effect=AssertionError("no directory enumeration")) as iterdir, \
             patch.object(socket.socket, "connect", side_effect=AssertionError("no network")) as connect, \
             patch.object(socket, "create_connection", side_effect=AssertionError("no network")) as create_connection, \
             patch.object(urllib.request, "urlopen", side_effect=AssertionError("no network")) as urlopen, \
             patch.object(builtins, "__import__", checked_import):
            first = smoke.run(self.images, self.root / "out1", dataset_root=self.dataset, repo_root=self.root)
            smoke.run(self.images, self.root / "out2", dataset_root=self.dataset, repo_root=self.root)
            for forbidden in (rglob, glob, iterdir, connect, create_connection, urlopen):
                forbidden.assert_not_called()
        self.assertTrue(touched)
        self.assertEqual(first["summary"]["videos"], 61)
        self.assertEqual(first["summary"]["records"], 1)
        self.assertEqual(first["videos"][0]["readiness"], {"artifact": "ok", "image": "ok", "faiss": "ok"})
        self.assertEqual(first["summary"]["exact_counts_status"], "issues")
        self.assertEqual(first["scope"]["video_ids"], list(smoke.VIDEO_IDS))
        one, two = self.root / "out1/report.json", self.root / "out2/report.json"
        self.assertEqual(one.read_bytes(), two.read_bytes())
        self.assertEqual(hashlib.sha256(one.read_bytes()).hexdigest(), smoke.sha256(two))
        self.assertEqual(smoke.sha256(self.root / "config/pipeline.json"), pipeline_before)
        with patch.object(smoke, "check_video", side_effect=AssertionError("stop before input checks")):
            with self.assertRaises(FileExistsError):
                smoke.run(self.images, self.root / "out1", dataset_root=self.dataset, repo_root=self.root)
        self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_output_under_input_and_nonlocal_image_root_rejected(self):
        for image_root, output in ((self.images, self.dataset / "output"),
                                   (self.images, self.root), (self.root / "drive", self.root / "out")):
            with self.assertRaises(smoke.UnsafePath):
                smoke.run(image_root, output, dataset_root=self.dataset, repo_root=self.root)

    def test_optional_import_failure_is_unknown_without_install(self):
        with patch.object(smoke.importlib, "import_module", side_effect=ImportError("missing fixture")):
            module, detail = smoke.optional_dependency("PIL.Image")
        self.assertIsNone(module)
        self.assertEqual(detail["status"], "unknown")

    def test_missing_optional_dependencies_stay_unknown_in_report(self):
        self.fixture()
        with patch.object(smoke, "optional_dependency", return_value=(None, {"status": "unknown"})):
            report = smoke.run(self.images, self.root / "unknown", dataset_root=self.dataset, repo_root=self.root)
        self.assertEqual(report["summary"]["readiness"]["dependency"], "unknown")
        self.assertEqual(report["videos"][0]["readiness"],
                         {"artifact": "ok", "image": "unknown", "faiss": "unknown"})


if __name__ == "__main__":
    unittest.main()
