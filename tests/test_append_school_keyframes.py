import tempfile
from pathlib import Path
import unittest
import zipfile

from tools.append_school_keyframes import preflight


class AppendSchoolKeyframesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        mapping = self.root / "mapping/L16/L16_V001"
        mapping.mkdir(parents=True)
        (mapping / "L16_V001.csv").write_text("n,frame_idx\n1,100\n", encoding="utf-8")

    def archive(self, entries):
        path = self.root / "L16.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name, data in entries:
                z.writestr(name, data)
        return path

    def check(self, entries):
        return preflight(self.archive(entries), self.root / "mapping", "L16")

    def test_complete_zip_preserved_and_duplicates_checked(self):
        report = self.check([("L16/L16_V001/001(1).jpg", b"image"),
                             ("L16/L16_V001/001.jpg", b"image")])
        self.assertEqual(report["images"], 1)
        self.assertEqual(len(report["duplicates"]), 1)
        self.assertEqual(len(report["sha256"]), 64)
        self.assertTrue((self.root / "L16.zip").is_file())

    def test_missing_extra_and_conflicting_duplicates_are_rejected(self):
        for entries in [[], [("L16/L16_V001/002.jpg", b"image")],
                        [("L16/L16_V001/001.jpg", b"one"), ("L16/L16_V001/001(1).jpg", b"two")],
                        [("L16/L16_V001/001(1).jpg", b"image")]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.check(entries)

    def test_other_group_and_unsafe_members_are_rejected(self):
        for name in ["L17/L17_V001/001.jpg", "../L16/L16_V001/001.jpg"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check([(name, b"image")])


if __name__ == "__main__":
    unittest.main()
