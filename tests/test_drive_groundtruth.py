from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.drive_groundtruth import fetch_window, sample_windows


class DriveGroundTruthTest(unittest.TestCase):
    def test_holdout_only_sampling_does_not_create_development_rows(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(4):
                (Path(root) / f"L01_V{index:03d}.csv").write_text("n,frame_idx,pts_time\n1,30,1\n", encoding="utf-8")
            samples = sample_windows(root, {"L01"}, count=4, split_mode="holdout")
            self.assertEqual({s["split"] for s in samples}, {"holdout"})
            with self.assertRaisesRegex(ValueError, "split mode"):
                sample_windows(root, {"L01"}, count=4, split_mode="training")

    def test_excluded_old_video_never_enters_new_split(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(4):
                (Path(root) / f"L01_V{index:03d}.csv").write_text("n,frame_idx,pts_time\n1,30,1\n", encoding="utf-8")
            samples = sample_windows(root, {"L01"}, count=3, exclude_video_ids={"L01_V001"})
            self.assertNotIn("L01_V001", {sample["video_id"] for sample in samples})

    def test_reproducible_video_disjoint_sample_and_excludes_ambiguous_frames(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(5):
                (Path(root) / f"L01_V{index:03d}.csv").write_text("n,frame_idx,pts_time\n1,0,0\n2,0,0.03\n3,30,1\n4,60,2\n", encoding="utf-8")
            a = sample_windows(root, {"L01"}, count=4)
            self.assertEqual(a, sample_windows(root, {"L01"}, count=4))
            self.assertEqual(len({window["video_id"] for window in a}), 4)
            self.assertEqual([window["split"] for window in a], ["dev", "dev", "holdout", "holdout"])
            self.assertTrue(all(frame["frame_id"] != 0 for window in a for frame in window["frames"]))

    def test_drive_reader_bytes_become_contact_sheet_not_original_file(self):
        buffer = BytesIO()
        Image.new("RGB", (16, 10), "blue").save(buffer, format="PNG")
        class Source:
            def read(self, video, number):
                return buffer.getvalue(), "image/png"
        window = {"sample_id": "drive_01", "split": "dev", "video_id": "L01_V001",
                  "frames": [{"video_id": "L01_V001", "frame_id": 30, "keyframe_n": 1, "pts_time": 1}]}
        with tempfile.TemporaryDirectory() as root:
            result = fetch_window(window, Source(), Path(root))
            self.assertEqual([path.name for path in Path(root).iterdir()], ["drive_01.jpg"])
        self.assertFalse(result["caption_accessed"])
        self.assertEqual(result["frames"][0]["status"], "downloaded")


if __name__ == "__main__":
    unittest.main()
