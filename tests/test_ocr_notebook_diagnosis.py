import unittest

from tools.audit_ocr_alphabet import missing_characters
from tools.diagnose_school_ocr_images import ROOT, detector_size, notebook_worker


class OCRNotebookDiagnosisTest(unittest.TestCase):
    def test_notebook_pins_latin_recognizer_and_max960_detection(self):
        _, source = notebook_worker(ROOT / "notebooks/kaggle_ocr_simple_gpu.ipynb")
        namespace = {"__name__": "audit_test", "__file__": "audit_test.py"}
        exec(compile(source, "<notebook-stdlib-worker>", "exec"), namespace)
        self.assertEqual(namespace["MODELS"]["recognition"], "latin_PP-OCRv5_mobile_rec")
        self.assertEqual(namespace["OPTIONS"]["text_det_limit_side_len"], 960)
        self.assertEqual(namespace["OPTIONS"]["text_det_limit_type"], "max")
        text = "Quảng Ngãi\n3,2 tấn cá khoai chứa formol"
        result = namespace["public_record"]({"status": "ok", "video_id": "L01_V001", "keyframe_n": 1,
            "relative_path": "L01_V001/001.jpg", "lines": [{"text": s} for s in text.splitlines()]})
        self.assertEqual(result["ocr_text"], text)
        self.assertEqual(result["retrieval_text"], text)

    def test_detector_geometry_is_padded_then_stride_rounded(self):
        self.assertEqual(detector_size((1280, 720)), (960, 576))
        self.assertEqual(detector_size((1280, 720), limit=1600), (1344, 768))

    def test_missing_alphabet_handles_nfc_not_silent_accent_folding(self):
        self.assertEqual(missing_characters("Quảng Ngãi", set("Qung Ngãi")), ["ả"])
        self.assertEqual(missing_characters("ta\u0302\u0301n", set("tan")), ["ấ"])
        self.assertEqual(missing_characters("tấn", set("tấn")), [])


if __name__ == "__main__":
    unittest.main()
