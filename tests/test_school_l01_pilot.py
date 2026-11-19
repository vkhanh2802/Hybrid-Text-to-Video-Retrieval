from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from tools.school_l01_pilot import image_destination, extract_images


class SchoolL01ZipTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_zip(self, names):
        path = self.root / "L01.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"image fixture bytes")
        return path

    def test_nested_drive_zip_paths_become_canonical_without_renumbering(self):
        relative = image_destination("L01/L01-20261108T062703Z-1-001/L01/L01_V001/001.jpg")
        self.assertEqual(relative.as_posix(), "L01/L01_V001/001.jpg")

    def test_rejects_traversal_absolute_non_l01_and_non_numeric_images(self):
        for name in ("../L01_V001/001.jpg", "/L01_V001/001.jpg", "C:/L01_V001/001.jpg",
                     "L01\\L01_V001\\001.jpg", "L02/L02_V001/001.jpg", "L01_V001/photo.jpg",
                     "L01_V001/000.jpg", "L01_V001/secret.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                image_destination(name)

    def test_extract_keeps_zip_and_creates_checksums_and_crc_receipt(self):
        path = self.make_zip(["L01/nested/L01/L01_V001/001.jpg", "L01/L01_V002/002.jpg"])
        report = extract_images(path, self.root / "images")
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["videos"], 2)
        self.assertTrue(report["crc_checked"])
        self.assertFalse(report["sources_deleted"])
        self.assertEqual(len(report["files"][0]["sha256"]), 64)
        self.assertTrue(path.is_file())
        self.assertEqual((self.root / "images/L01/L01_V001/001.jpg").read_bytes(), b"image fixture bytes")

    def test_duplicate_identity_after_normalization_is_rejected_before_extract(self):
        path = self.make_zip(["old/L01_V001/001.jpg", "new/L01_V001/001.jpg"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            extract_images(path, self.root / "images")
        self.assertFalse((self.root / "images").exists())

    def test_existing_destination_is_not_overwritten(self):
        path = self.make_zip(["L01/L01_V001/001.jpg"])
        destination = self.root / "images"
        destination.mkdir()
        original = destination / "keep.txt"
        original.write_text("user data")
        with self.assertRaises(FileExistsError):
            extract_images(path, destination)
        self.assertEqual(original.read_text(), "user data")

    def test_zip_symlink_is_rejected(self):
        path = self.root / "L01.zip"
        with zipfile.ZipFile(path, "w") as archive:
            member = zipfile.ZipInfo("L01/L01_V001/001.jpg")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "../../private")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            extract_images(path, self.root / "images")


if __name__ == "__main__":
    unittest.main()
