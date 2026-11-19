import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.normalize_school_dataset import artifact_destination, normalize_artifacts, extract_group, image_relative
from kis.paths import canonical_video_artifact_path, resolve_video_artifact, resolve_video_image_dir


class CanonicalDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_components_share_group_video_file_layout(self):
        for component, name in [("caption_generator", "L02_V001.json"), ("caption_mapping", "L02_V001.json"),
                                ("mapping", "L02_V001.csv"), ("clip-features-32", "L02_V001.npy"),
                                ("index", "L02_V001.meta.json"), ("preparation", "L02_V001.json")]:
            self.assertEqual(artifact_destination(f"dataset/{component}/L02/{name}").as_posix(),
                             f"dataset/{component}/L02/L02_V001/{name}")
        self.assertEqual(artifact_destination("dataset/objects/L02/L02_V001/001.json").as_posix(),
                         "dataset/objects/L02/L02_V001/001.json")

    def test_normalize_copies_bytes_checks_hash_and_retains_original(self):
        source, dest = self.root / "source", self.root / "new"
        source.mkdir(); dest.mkdir()
        name = "dataset/caption_generator/L02/L02_V001.json"
        p = source / name
        p.parent.mkdir(parents=True)
        data = b'[{"caption":"same original bytes"}]'
        p.write_bytes(data)
        row = {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (source / "TRANSFER_MANIFEST.json").write_text(json.dumps({"files": [row]}))
        report = normalize_artifacts(source, dest)
        self.assertEqual(report["status"], "verified")
        self.assertFalse(report["sources_deleted"])
        self.assertEqual(p.read_bytes(), data)
        self.assertEqual((dest / artifact_destination(name)).read_bytes(), data)
        with self.assertRaises(FileExistsError):
            normalize_artifacts(source, dest)

    def test_unsafe_or_conflicting_identity_is_rejected(self):
        for p in ("../dataset/mapping/L02_V001.csv", "dataset/index/L02/L02_V001/L03_V001.index",
                  "config/dres.local.json", "dataset/data/L02/L02_V001/001.jpg"):
            with self.subTest(path=p), self.assertRaises(ValueError):
                artifact_destination(p)

    def test_canonical_resolver_supports_new_and_old_without_silently_choosing_duplicates(self):
        canonical = canonical_video_artifact_path(self.root, "L02_V001", ".csv")
        self.assertEqual(canonical, self.root / "L02/L02_V001/L02_V001.csv")
        legacy = self.root / "L02/L02_V001.csv"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("n,frame_idx\n1,0\n")
        self.assertEqual(resolve_video_artifact(self.root, "L02_V001", ".csv"), legacy.resolve())
        canonical.parent.mkdir()
        canonical.write_bytes(legacy.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            resolve_video_artifact(self.root, "L02_V001", ".csv")
        self.assertEqual(resolve_video_image_dir("L03_V001", self.root), self.root / "L03/L03_V001")

    def test_caption_discovery_reads_per_video_directories(self):
        from kis.text import CaptionRetriever
        p = self.root / "L02/L02_V001/L02_V001.json"
        p.parent.mkdir(parents=True)
        p.write_text("[]")
        self.assertEqual(CaptionRetriever._collect_artifact_map(self.root, "L*_V*.json", {"L02"}),
                         {"L02_V001": p})

    def make_zip(self, entries):
        p = self.root / "images.zip"
        with zipfile.ZipFile(p, "w") as z:
            for name, data in entries:
                z.writestr(name, data)
        return p

    def test_nested_group_paths_and_identical_browser_duplicates_are_safe(self):
        archive = self.make_zip([("L03/L03_V024/020(1).jpg", b"same"),
                                 ("L03/L03_V024/020.jpg", b"same")])
        report = extract_group(archive, self.root / "data", "L03")
        self.assertEqual(report["images"], 1)
        self.assertEqual(len(report["identical_zip_duplicates"]), 1)
        self.assertEqual((self.root / "data/L03/L03_V024/020.jpg").read_bytes(), b"same")
        self.assertTrue(archive.is_file())
        self.assertEqual(image_relative("L06-KF/L06/L06_V001/001.jpg", "L06").as_posix(),
                         "L06/L06_V001/001.jpg")

    def test_conflicting_duplicate_is_never_overwritten_or_extracted(self):
        archive = self.make_zip([("L07/L07_V031/226.jpg", b"first"),
                                 ("L07/L07_V031/226 (1).jpg", b"wrong")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            extract_group(archive, self.root / "data", "L07")
        self.assertFalse((self.root / "data").exists())

    def test_duplicate_suffix_without_primary_is_not_silently_renumbered(self):
        archive = self.make_zip([("L07/L07_V031/226 (1).jpg", b"image")])
        with self.assertRaisesRegex(ValueError, "no primary"):
            extract_group(archive, self.root / "data", "L07")


if __name__ == "__main__":
    unittest.main()
