import math
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch


try:
    from kis.web_app import serialize_search_results
except ModuleNotFoundError:
    serialize_search_results = None


@unittest.skipIf(serialize_search_results is None, "web dependencies are not installed")
class WebResultSerializationTest(unittest.TestCase):
    def test_serializes_card_metadata_and_scores(self):
        candidate = {
            "video_id": "L27_V013",
            "frame_id": 8876,
            "final_score": 0.7178,
            "fusion_score": 0.701,
            "retrieval_text": "Two people on a boat.",
        }

        def image_resolver(item, _lookup):
            item["keyframe_n"] = 186
            return Path("186.jpg")

        results = serialize_search_results(
            [candidate],
            object_lookup=None,
            image_resolver=image_resolver,
        )

        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["video_id"], "L27_V013")
        self.assertEqual(results[0]["image_number"], 186)
        self.assertEqual(results[0]["frame_id"], 8876)
        self.assertEqual(results[0]["image_url"], "/api/images/L27_V013/186")
        self.assertEqual(results[0]["caption"], "Two people on a boat.")

    def test_non_finite_scores_are_json_safe(self):
        candidate = {
            "video_id": "L27_V013",
            "frame_id": 1,
            "final_score": math.nan,
        }
        results = serialize_search_results(
            [candidate],
            object_lookup=None,
            image_resolver=lambda *_: None,
        )

        self.assertIsNone(results[0]["score"])
        self.assertIsNone(results[0]["image_url"])

    def test_remote_images_are_returned_from_memory_without_http_disk_cache(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        source = Mock(read=Mock(return_value=(b"image bytes", "image/jpeg")))
        service = SimpleNamespace(image_source=source, video_groups=("L15",))
        with patch.object(web_app, "service", service), patch.object(web_app, "mapped_keyframes", return_value={1}):
            result = TestClient(web_app.app).get("/api/images/L15_V001/1")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, b"image bytes")
        self.assertIn("no-store", result.headers["cache-control"])
        source.read.assert_called_once_with("L15_V001", 1)

    def test_unmapped_keyframe_is_rejected_before_any_drive_request(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        source = Mock()
        with patch.object(web_app, "service", SimpleNamespace(image_source=source, video_groups=("L15",))), \
             patch.object(web_app, "mapped_keyframes", return_value={1}):
            result = TestClient(web_app.app).get("/api/images/L15_V001/99999")
        self.assertEqual(result.status_code, 404)
        source.read.assert_not_called()

    def test_local_source_images_are_not_deleted_by_image_responses(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "001.jpg"
            image.write_bytes(b"original image")
            with patch.object(web_app, "service", SimpleNamespace(image_source=None, video_groups=("L15",))), \
                 patch.object(web_app, "find_keyframe_image", return_value=image), \
                 patch.object(web_app, "mapped_keyframes", return_value={1}):
                result = TestClient(web_app.app).get("/api/images/L15_V001/1")
            self.assertEqual(result.content, b"original image")
            self.assertIn("no-store", result.headers["cache-control"])
            self.assertEqual(image.read_bytes(), b"original image")


if __name__ == "__main__":
    unittest.main()
