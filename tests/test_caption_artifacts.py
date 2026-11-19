import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kis import artifact_io
from kis.artifact_io import atomic_write_json, publish_staged_files
from kis.caption_schema import manifest_data, validate_caption_manifest
from kis.caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    rebuild_retrieval_text,
    validate_unique_caption_records,
)
from kis.paths import (
    KIS_ROOT,
    DATASET_ROOT,
    canonical_video_artifact_path,
    discover_video_image_dirs,
    resolve_video_artifact,
    resolve_video_image_dir,
    resolve_video_object_dir,
    resolve_stored_image_path,
    is_keyframe_image,
    configured_data_dir,
    safe_filename_component,
    video_group,
    video_object_dir,
)


class CaptionArtifactTest(unittest.TestCase):
    def test_explicit_group_scope_filters_before_artifact_and_image_traversal(self):
        from kis.text import CaptionRetriever

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group in ("L04", "L05", "L06"):
                (root / group / f"{group}_V001").mkdir(parents=True)
                (root / group / f"{group}_V001.json").write_text("[]", encoding="utf-8")
                (root / group / f"{group}_V001/001.jpg").write_bytes(b"fixture")
            original_iterdir, original_rglob = Path.iterdir, Path.rglob

            def guarded_iterdir(path):
                self.assertFalse({"L05", "L06"}.intersection(path.parts))
                return original_iterdir(path)

            def guarded_rglob(path, pattern):
                self.assertNotEqual(path, root)
                self.assertFalse({"L05", "L06"}.intersection(path.parts))
                return original_rglob(path, pattern)

            with patch.object(Path, "iterdir", guarded_iterdir), patch.object(Path, "rglob", guarded_rglob):
                artifacts = CaptionRetriever._collect_artifact_map(root, "L*_V*.json", {"L04"})
                self.assertEqual(set(artifacts), {"L04_V001"})
                self.assertEqual(set(discover_video_image_dirs(root, video_groups=("L04",))), {"L04_V001"})
                self.assertEqual(resolve_video_artifact(root, "L04_V001", ".json", video_groups=("L04",)),
                                 root / "L04/L04_V001.json")
                with patch.dict("os.environ", {"KIS_VIDEO_GROUPS": "L04"}):
                    self.assertEqual(resolve_video_artifact(root, "L04_V001", ".json"), root / "L04/L04_V001.json")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_path_and_legacy_resolution(self):
        artifact_root = self.root / "captions"
        legacy_path = artifact_root / "L26a" / "L26_V001.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text("[]", encoding="utf-8")

        self.assertEqual(video_group("L26_V001"), "L26")
        self.assertEqual(
            canonical_video_artifact_path(
                artifact_root,
                "L26_V001",
                ".json",
            ),
            artifact_root / "L26" / "L26_V001" / "L26_V001.json",
        )
        self.assertEqual(
            resolve_video_artifact(artifact_root, "L26_V001", ".json"),
            legacy_path.resolve(),
        )

    def test_duplicate_artifacts_are_rejected(self):
        artifact_root = self.root / "captions"
        for group in ("L26", "L26a"):
            path = artifact_root / group / "L26_V001.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Duplicate artifacts"):
            resolve_video_artifact(artifact_root, "L26_V001", ".json")

    def test_atomic_json_write_keeps_previous_file_on_failure(self):
        destination = self.root / "artifact.json"
        atomic_write_json(destination, {"state": "valid"})

        with self.assertRaises(TypeError):
            atomic_write_json(destination, {"invalid": object()})

        self.assertEqual(
            json.loads(destination.read_text(encoding="utf-8")),
            {"state": "valid"},
        )

    def test_staged_publish_restores_previous_file_on_failure(self):
        destination = self.root / "artifact.json"
        staged = self.root / "staged.json"
        destination.write_text("old", encoding="utf-8")
        staged.write_text("new", encoding="utf-8")
        real_replace = artifact_io.os.replace

        def fail_publish(source, target):
            if Path(source) == staged:
                raise OSError("simulated publish failure")
            return real_replace(source, target)

        with patch("kis.artifact_io.os.replace", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "simulated"):
                publish_staged_files([(destination, staged)])

        self.assertEqual(destination.read_text(encoding="utf-8"), "old")

    def test_normalizes_mapping_path_and_repairs_ocr_text(self):
        mapping_dir = self.root / "mapping"
        mapping_dir.mkdir()
        (mapping_dir / "L21_V001.csv").write_text(
            "n,frame_idx\n1,123\n",
            encoding="utf-8",
        )
        frame_mapping = load_frame_mapping("L21_V001", mapping_dir)
        record = {
            "video_id": "L21_V001",
            "frame_id": None,
            "keyframe_n": "1",
            "image_path": "data\\L21_V001\\001.jpg",
            "caption": "A sign",
            "details": "outside",
            "ocr_text": "OPEN",
            "retrieval_text": "stale",
        }

        normalized = normalize_caption_record(
            record,
            "L21_V001",
            frame_mapping,
        )

        self.assertEqual(normalized["frame_id"], 123)
        self.assertEqual(normalized["image_path"], "data/L21_V001/001.jpg")
        self.assertEqual(
            normalized["retrieval_text"],
            "A sign. outside. [OCR]: OPEN",
        )

    def test_existing_frame_id_must_match_mapping(self):
        record = {
            "video_id": "L21_V001",
            "frame_id": 999,
            "keyframe_n": 1,
            "image_path": "data/L21_V001/001.jpg",
        }
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            normalize_caption_record(record, "L21_V001", {1: 123})

    def test_duplicate_caption_identities_are_rejected(self):
        records = [
            {"keyframe_n": 1, "frame_id": 10},
            {"keyframe_n": 1, "frame_id": 11},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate keyframe_n"):
            validate_unique_caption_records(records, "L21_V001")

    def test_empty_ocr_is_a_completed_result(self):
        record = {"caption": "A room", "details": "indoor", "ocr_text": ""}
        self.assertEqual(rebuild_retrieval_text(record), "A room. indoor")

    def test_manifest_detects_tampered_caption(self):
        caption_path = self.root / "caption.json"
        mapping_path = self.root / "mapping.json"
        index_path = self.root / "caption.index"
        manifest_path = self.root / "caption.meta.json"
        caption_path.write_text("[]", encoding="utf-8")
        mapping_path.write_text("[]", encoding="utf-8")
        index_path.write_bytes(b"index")
        manifest = manifest_data(
            "L21_V001",
            caption_path,
            mapping_path,
            index_path,
            0,
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        index = SimpleNamespace(d=384)

        validate_caption_manifest(
            "L21_V001",
            caption_path,
            mapping_path,
            index_path,
            manifest_path,
            index,
            0,
        )
        caption_path.write_text("[{}]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_caption_manifest(
                "L21_V001",
                caption_path,
                mapping_path,
                index_path,
                manifest_path,
                index,
                0,
            )

    def test_filename_component_is_sanitized(self):
        self.assertEqual(
            safe_filename_component("../q:01?"),
            "q_01",
        )

    def test_grouped_and_flat_image_layouts_are_resolved(self):
        data_root = self.root / "data"
        flat_video = data_root / "L21_V001"
        flat_video.mkdir(parents=True)
        (flat_video / "001.jpg").write_bytes(b"flat")
        grouped_video = data_root / "L27" / "L27_V001"
        grouped_video.mkdir(parents=True)
        (grouped_video / "001.jpg").write_bytes(b"grouped")

        self.assertEqual(
            resolve_video_image_dir("L21_V001", image_root=data_root),
            flat_video.resolve(),
        )
        self.assertEqual(
            resolve_video_image_dir("L27_V001", image_root=data_root),
            grouped_video.resolve(),
        )
        discovered = discover_video_image_dirs(data_root)
        self.assertEqual(
            discovered,
            {
                "L21_V001": flat_video,
                "L27_V001": grouped_video,
            },
        )

    def test_duplicate_image_layouts_are_rejected(self):
        data_root = self.root / "data"
        flat_video = data_root / "L21_V001"
        flat_video.mkdir(parents=True)
        grouped_video = data_root / "L21" / "L21_V001"
        grouped_video.mkdir(parents=True)
        (flat_video / "001.jpg").write_bytes(b"image")
        (grouped_video / "001.jpg").write_bytes(b"image")

        with self.assertRaisesRegex(RuntimeError, "Duplicate image"):
            resolve_video_image_dir("L21_V001", image_root=data_root)
        with self.assertRaisesRegex(RuntimeError, "Duplicate image"):
            discover_video_image_dirs(data_root)

    def test_external_image_root_resolves_existing_caption_paths(self):
        external = self.root / "drive" / "data"
        self.assertEqual(configured_data_dir({"KIS_DATA_ROOT": str(external)}), external.resolve())
        self.assertEqual(
            resolve_stored_image_path("data/L15/L15_V001/001.jpg", data_root=external),
            external / "L15" / "L15_V001" / "001.jpg",
        )

    def test_machine_local_image_root_is_used_without_changing_shared_config(self):
        settings = self.root / "images.local.json"
        external = self.root / "external-keyframes"
        atomic_write_json(settings, {"data_root": str(external)})
        self.assertEqual(configured_data_dir({}, settings), external.resolve())
        override = self.root / "explicit-override"
        self.assertEqual(configured_data_dir({"KIS_DATA_ROOT": str(override)}, settings), override.resolve())
        self.assertEqual(configured_data_dir({}, self.root / "missing.json"), DATASET_ROOT / "data")

    def test_discovery_ignores_original_videos_archives_and_non_keyframe_folders(self):
        root = self.root / "drive"
        files = {
            "L15/L15_V001/001.JPG": b"keyframe",
            "video_L15/L15_V001/001.jpg": b"excluded source folder",
            "video_L15/L15_V002.mp4": b"original video",
            "L16/L16_V001/001.mp4": b"video inside a canonical-looking folder",
            "L19/L19_V001/cover.jpg": b"not a numbered keyframe",
            "L15/L16_V002/001.jpg": b"wrong group",
            "L21_V001.mp4": b"video",
            "L13.zip": b"archive",
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        discovered = discover_video_image_dirs(root)
        self.assertEqual(set(discovered), {"L15_V001"})
        self.assertTrue(is_keyframe_image("001.JPG", 1))
        for name in ("001.mp4", "001.zip", "cover.jpg", "desktop.ini", "002.jpg"):
            self.assertFalse(is_keyframe_image(name, 1), name)

    def test_object_layout_uses_group_folder(self):
        self.assertEqual(
            video_object_dir("L21_V001"),
            DATASET_ROOT / "objects" / "L21" / "L21_V001",
        )
        objects_root = self.root / "objects"
        grouped = objects_root / "L21" / "L21_V001"
        grouped.mkdir(parents=True)
        self.assertEqual(
            resolve_video_object_dir("L21_V001", objects_root),
            grouped.resolve(),
        )

    def test_duplicate_object_layouts_are_rejected(self):
        objects_root = self.root / "objects"
        flat = objects_root / "L21_V001"
        flat.mkdir(parents=True)
        grouped = objects_root / "L21" / "L21_V001"
        grouped.mkdir(parents=True)

        with self.assertRaisesRegex(RuntimeError, "Duplicate object"):
            resolve_video_object_dir("L21_V001", objects_root)


if __name__ == "__main__":
    unittest.main()
