from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from tools.review_school_ocr_accuracy import prepare


class BlindReviewTest(unittest.TestCase):
    def test_review_deduplicates_and_images_hide_arm_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            query = {"id": "q01", "text": "A public scene."}
            row = {"video_id": "L01_V001", "keyframe_n": 1}
            report = {"records": [{"query": query, "arms": {
                arm: {"final_ranking": [row]} for arm in ("A", "B", "C")}}]}
            (root / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
            raw = BytesIO()
            Image.new("RGB", (16, 16), "red").save(raw, format="PNG")
            worker = Mock(read=Mock(return_value=(raw.getvalue(), "image/png")))
            with patch("tools.review_school_ocr_accuracy.client", return_value=worker):
                prepare(root)
            worker.read.assert_called_once_with("L01_V001", 1)
            key = json.loads((root / "BLIND_REVIEW_KEY.json").read_text(encoding="utf-8"))
            self.assertEqual(key["q01"]["item1"]["arms"], ["A", "B", "C"])
            self.assertEqual(key["q01"]["item1"]["path"], "blind_top1/q01_item1.png")
            self.assertTrue((root / "blind_top1/q01_sheet.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
