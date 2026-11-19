import json
from pathlib import Path
import tempfile
import unittest

from tools.build_visual_labels import build


class BuildVisualLabelsTest(unittest.TestCase):
    def test_freezes_checksums_and_split_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            frames = [{"keyframe_n": n, "frame_id": n * 30, "pts_time": n,
                       "image_sha256": str(n) * 64, "status": "downloaded"} for n in (1, 2)]
            manifest = {"samples": [{"sample_id": "sample_01", "video_id": "L01_V001", "split": "holdout",
                                     "anchor": {"keyframe_n": 1}, "frames": frames}]}
            observations = {"excluded_samples": [], "samples": [{"sample_id": "sample_01", "video_id": "L01_V001",
                "image": {"raw_query": "a", "english": "a", "keywords": "a", "relevant_keyframes": [1]},
                "sequence": {"raw_query": "a then b", "english": "a then b", "keywords": "a b",
                             "events": ["a", "b"], "event_relevant_keyframes": [[1], [2]]}}]}
            m, o, output = (root / name for name in ("manifest.json", "observations.json", "labels.json"))
            m.write_text(json.dumps(manifest), encoding="utf-8")
            o.write_text(json.dumps(observations), encoding="utf-8")
            build(m, o, output)
            labels = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(labels["queries"]), 2)
            self.assertTrue(all(query["split"] == "holdout" for query in labels["queries"]))
            self.assertEqual(labels["queries"][0]["image_sha256"], "1" * 64)
            self.assertFalse(labels["human_verified"])
            with self.assertRaises(FileExistsError):
                build(m, o, output)
            manifest["samples"][0]["frames"][1]["status"] = "fetch_failed"
            m.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not fetched"):
                build(m, o, root / "invalid_labels.json")


if __name__ == "__main__":
    unittest.main()
