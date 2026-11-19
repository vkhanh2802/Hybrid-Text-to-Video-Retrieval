import json
import hashlib
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
from zipfile import ZipFile, ZipInfo

import numpy as np

from tools.temporal_preflight import check_video, run
from tools.temporal_inventory import archive_inventory, local_images, parse_groups


class TemporalPreflightTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.frames = [{"frame_id": 10, "keyframe_n": 1, "pts_time": 0.5}]

    def check(self, vectors):
        with patch("tools.temporal_preflight.LocalEventScorer._video_data", return_value=(self.frames, vectors)), \
             patch("tools.temporal_preflight.resolve_video_artifact", return_value=self.root / "caption.json"):
            return check_video("L01_V001", self.root)

    def test_visual_only_is_ready_without_loading_models(self):
        result = self.check(np.ones((1, 768), dtype=np.float32))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["searchable_captions"], 0)

    def test_bad_embeddings_or_both_channels_missing_are_not_ready(self):
        for vectors in (None, np.zeros((1, 768)), np.ones((1, 4)), np.full((1, 768), np.nan)):
            self.assertEqual(self.check(vectors)["status"], "error")

    def test_identity_mismatch_is_reported(self):
        (self.root / "caption.json").write_text(json.dumps([
            {"video_id": "L01_V001", "frame_id": 10, "keyframe_n": 2, "caption": "test"}]), encoding="utf-8")
        self.assertIn("identity", self.check(np.ones((1, 768)))["error"])

    def test_legacy_duplicates_are_visible_warnings_not_whole_video_errors(self):
        self.frames = [{"frame_id": fid, "keyframe_n": i + 1, "pts_time": i / 30}
                       for i, fid in enumerate((0, 0, 30))]
        result = self.check(np.ones((3, 768)))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["ambiguous_frame_rows_excluded"], 2)
        self.assertEqual(result["ambiguous_frame_ids"], [0])
        self.assertTrue(result["warnings"])

    def test_inventory_respects_active_scope_and_reports_missing_groups(self):
        for group in ("L01", "L22"):
            (self.root / f"{group}_V001.csv").touch()
        config = SimpleNamespace(video_groups=("L01", "L02"), caption_root=self.root)
        with patch("tools.temporal_preflight.check_video", return_value={"video_id": "L01_V001", "status": "ready", "frames": 1}) as check:
            report = run(self.root / "report.json", config, self.root)
        check.assert_called_once_with("L01_V001", self.root)
        self.assertEqual(report["summary"]["missing_mapping_groups"], ["L02"])

    def inventory_fixture(self, video_id):
        dataset = self.root / "dataset"
        mapping = dataset / "mapping" / f"{video_id}.csv"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_text("n,frame_idx,pts_time\n1,10,0.5\n", encoding="utf-8")
        captions = [{"video_id": video_id, "keyframe_n": 1, "frame_id": 10, "caption": "test"}]
        hashes = {}
        for kind, hash_name in (("caption_generator", "caption_sha256"), ("caption_mapping", "mapping_sha256")):
            directory = dataset / kind
            directory.mkdir(exist_ok=True)
            path = directory / f"{video_id}.json"
            path.write_text(json.dumps(captions), encoding="utf-8")
            hashes[hash_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (dataset / "clip-features-32").mkdir(exist_ok=True)
        np.save(dataset / "clip-features-32" / f"{video_id}.npy", np.ones((1, 768), dtype=np.float32))
        (dataset / "index").mkdir(exist_ok=True)
        index = dataset / "index" / f"{video_id}.index"
        index.write_bytes(b"not loaded by offline inventory")
        hashes["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
        from kis.config import CAPTION_CONFIG

        (dataset / "index" / f"{video_id}.meta.json").write_text(json.dumps({
            "schema_version": 1, "video_id": video_id, "count": 1, "files": hashes,
            "embedding": CAPTION_CONFIG}), encoding="utf-8")
        objects = dataset / "objects" / video_id
        objects.mkdir(parents=True)
        (objects / "001.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(video_groups=("L01",), caption_root=dataset)

    def test_groups_override_missing_l06_images_and_deterministic_reports(self):
        config = self.inventory_fixture("L06_V001")
        self.inventory_fixture("L04_V001")
        archive = self.root / "L04.zip"
        with ZipFile(archive, "w") as handle:
            handle.writestr("data/L04/L04_V001/001.jpg", b"listed only")
        kwargs = dict(config=config, mapping_root=config.caption_root / "mapping",
                      groups=("L04", "L06"), image_roots=(self.root / "images",), archives=(archive,))
        with patch("tools.temporal_inventory.audit._faiss_status", side_effect=AssertionError("no FAISS import")), \
             patch("tools.temporal_preflight.LocalEventScorer._video_data", side_effect=AssertionError("no runtime scorer")):
            report = run(self.root / "first.json", **kwargs)
            run(self.root / "second.json", **kwargs)
        self.assertEqual((self.root / "first.json").read_bytes(), (self.root / "second.json").read_bytes())
        self.assertEqual(config.video_groups, ("L01",))
        self.assertEqual(report["video_groups"], ["L04", "L06"])
        self.assertEqual(report["summary"]["videos"], 2)
        l04, l06 = report["videos"]
        self.assertEqual(l04["images"]["archives"]["status"], "listed")
        self.assertEqual(l04["images"]["local"]["status"], "missing")
        self.assertEqual(l06["images"]["local"]["status"], "missing")
        self.assertEqual(l06["images"]["archives"]["status"], "missing")
        self.assertEqual(l06["runtime_status"], "blocked")
        self.assertEqual(l06["artifacts"]["index"]["status"], "unknown")
        self.assertEqual(l06["artifacts"]["index_manifest"]["status"], "ok")
        self.assertEqual(l06["artifacts"]["objects"]["status"], "present")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            run(self.root / "first.json", **kwargs)

    def test_group_validation_and_missing_mapping_group(self):
        self.assertEqual(parse_groups("L06,L04,L06"), ("L04", "L06"))
        for value in ("", "../L04", "L04,", "L04_V001"):
            with self.assertRaises(ValueError):
                parse_groups(value)
        config = self.inventory_fixture("L04_V001")
        report = run(self.root / "missing.json", config, config.caption_root / "mapping", groups=("L06",))
        self.assertEqual(report["missing_mapping_groups"], ["L06"])
        self.assertEqual(report["summary"]["runtime_ready"], 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            run(config.caption_root / "new.json", config, config.caption_root / "mapping", groups=("L06",))
        self.assertFalse((config.caption_root / "new.json").exists())

    def test_zip_inventory_rejects_traversal_absolute_paths_and_symlinks(self):
        archive = self.root / "unsafe.zip"
        unsafe = ("../data/L06/L06_V001/001.jpg", "data/../data/L06/L06_V001/002.jpg",
                  "/data/L06/L06_V001/003.jpg", "C:/data/L06/L06_V001/004.jpg",
                  "..\\data\\L06\\L06_V001\\005.jpg", "data\\L06\\L06_V001\\007.jpg",
                  "data//L06/L06_V001/008.jpg", "data/L06/L06_V001/009.jpg.",
                  "data/L06/L06_V001/010?.jpg", "data/L06/L06_V001/NUL.jpg",
                  "data/L06/L06_V001/011.jpg ", "data/L06/L06_V001/012\x00.jpg", ".")
        with ZipFile(archive, "w") as handle:
            for name in unsafe:
                member = ZipInfo(name)
                # Preserve malicious separators in the fixture even on Windows.
                member.filename = name
                handle.writestr(member, b"rejected")
            link = ZipInfo("data/L06/L06_V001/006.jpg")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            handle.writestr(link, b"/outside")
            handle.writestr("data/L04/L04_V001/001.jpg", b"safe listing")
        with patch.object(ZipFile, "open", side_effect=AssertionError("no payload access")), \
             patch.object(ZipFile, "extract", side_effect=AssertionError("no extraction")):
            detail, images = archive_inventory(archive, ("L04", "L06"))
        self.assertEqual(detail["status"], "unsafe")
        self.assertEqual(detail["rejected_members"], sorted((*unsafe, link.filename)))
        self.assertEqual(images, {})
        self.assertEqual(detail["image_entries"], 0)
        self.assertFalse((self.root / "data").exists())

    def test_zip_inventory_rejects_duplicate_canonical_destinations(self):
        canonical = "data/L04/L04_V001/001.jpg"
        config = self.inventory_fixture("L04_V001")
        for index, alias in enumerate((canonical, "data/L04/L04_V001/./001.jpg",
                                       "data/L04//L04_V001/001.jpg", "DATA/l04/l04_v001/001.JPG",
                                       canonical + "/")):
            with self.subTest(alias=alias):
                archive = self.root / "duplicate.zip"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with ZipFile(archive, "w") as handle:
                        handle.writestr(canonical, b"listed only")
                        handle.writestr(alias, b"collision")
                with patch.object(ZipFile, "open", side_effect=AssertionError("no payload access")), \
                     patch.object(ZipFile, "extractall", side_effect=AssertionError("no extraction")):
                    detail, images = archive_inventory(archive, ("L04",))
                self.assertEqual(detail["status"], "unsafe")
                self.assertEqual(detail["duplicate_members"], [canonical.casefold()])
                self.assertEqual(images, {})
                self.assertEqual(detail["image_entries"], 0)
                report = run(self.root / f"duplicate_{index}.json",
                             config, config.caption_root / "mapping", groups=("L04",), archives=(archive,))
                self.assertEqual(report["summary"]["archive_image_frames"], 0)
                self.assertEqual(report["videos"][0]["images"]["archives"]["status"], "unknown")
        self.assertFalse((self.root / "data").exists())

    def test_zip_inventory_normalizes_dot_components_and_allows_directory_entries(self):
        archive = self.root / "safe.zip"
        with ZipFile(archive, "w") as handle:
            handle.writestr("data/L04/L04_V001/", b"")
            handle.writestr("data/L04/L04_V001/./001.jpg", b"listed only")
        with patch.object(ZipFile, "open", side_effect=AssertionError("no payload access")):
            detail, images = archive_inventory(archive, ("L04",))
        self.assertEqual(detail["status"], "listed")
        self.assertEqual(detail["duplicate_members"], [])
        self.assertEqual(images, {"L04_V001": {1}})

    def test_unreadable_archive_and_local_placeholder_are_unknown(self):
        detail, _ = archive_inventory(self.root / "absent.zip", ("L06",))
        self.assertEqual(detail["status"], "unknown")
        directory = self.root / "images" / "L06" / "L06_V001"
        directory.mkdir(parents=True)
        (directory / "001.jpg").write_bytes(b"metadata only")
        real_stat = Path.stat

        def placeholder(path, *args, **kwargs):
            if path.name == "001.jpg":
                return SimpleNamespace(st_file_attributes=0x1000)
            return real_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", placeholder):
            result = local_images(self.root / "images", "L06_V001", {1})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["present"], 0)


if __name__ == "__main__":
    unittest.main()
