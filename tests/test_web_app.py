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


def frame_candidates():
    return [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1,
             "score": .8, "final_score": .8} for i in range(200)]


@unittest.skipIf(serialize_search_results is None, "web dependencies are not installed")
class WebResultSerializationTest(unittest.TestCase):
    def test_text_ranking_fields_do_not_replace_original_retrieval_or_claim_verification(self):
        candidate = {"video_id": "L01_V001", "frame_id": 1, "score": .7, "final_score": .9,
                     "text_ranking_score": .9, "text_logit": -2.5, "text_rerank_status": "scored"}
        result = serialize_search_results([candidate], None, lambda *_: None)[0]
        self.assertEqual(result["score"], .9)
        self.assertEqual(result["retrieval_score"], .7)
        self.assertEqual(result["text_logit"], -2.5)
        self.assertEqual(result["text_rerank_status"], "scored")

    def test_serializes_temporal_events_and_preserves_unknown(self):
        candidate = {"video_id": "L01_V001", "frame_id": 10, "keyframe_n": 1,
                     "final_score": 0.8, "vlm_status": "skipped_temporal",
                     "temporal": {"status": "partial", "verified": False, "coverage": 0.5,
                                  "events": [{"action": "pour", "video_id": "L01_V001", "frame_id": 10,
                                              "keyframe_n": 1, "pts_time": 2, "status": "retrieved"},
                                             {"action": "stir", "frame_id": None, "keyframe_n": None,
                                              "pts_time": None, "status": "unknown"}]}}
        resolver = Mock(return_value=Path("001.jpg"))
        result = serialize_search_results([candidate], None, resolver)[0]
        self.assertEqual(result["temporal"]["events"][0]["image_url"], "/api/images/L01_V001/1")
        self.assertIsNone(result["temporal"]["events"][1]["image_url"])
        self.assertFalse(result["temporal"]["verified"])
        self.assertNotIn("image_url", candidate["temporal"]["events"][0])

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


@unittest.skipIf(serialize_search_results is None, "web dependencies are not installed")
class WebQueryAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Windows/OpenSSL while its host environment is intact.
        # clear=True credential-isolation fixtures below can otherwise fail on
        # the FIRST TLS context, depending on test order. No network request.
        from kis.llm import _api_ssl_context
        _api_ssl_context()

    def test_status_exposes_text_config_without_loading_model(self):
        from kis.web_app import SearchService

        with patch.dict("os.environ", {"KIS_TEXT_RERANKER_MODE": "rerank"}, clear=True), \
             patch("kis.text_reranker.LazyCrossEncoder._load") as loader:
            service = SearchService()
            status = service.status()
        self.assertEqual(status["text_reranker"]["mode"], "rerank")
        self.assertFalse(status["text_reranker"]["loaded"])
        self.assertFalse(status["text_reranker"]["verified"])
        loader.assert_not_called()

    def test_public_ocr_startup_selects_adapter_without_loading_models_or_source(self):
        from kis.web_app import SearchService
        from kis.public_ocr import PublicOCRRetriever

        env = {"KIS_OCR_FORMAT": "public_json", "KIS_OCR_ROOT": "dataset/ocr/L04",
               "KIS_OCR_MODE": "retrieve", "KIS_VIDEO_GROUPS": "L04"}
        with patch.dict("os.environ", env, clear=True), \
             patch.object(PublicOCRRetriever, "_ensure") as loader:
            service = SearchService()
            status = service.status()["ocr"]
            self.assertIsInstance(service.ocr_retriever, PublicOCRRetriever)
            self.assertFalse(status["loaded"])
            self.assertFalse(status["verified"])
            self.assertEqual(status["source_format"], "public_json")
            with self.assertRaisesRegex(ValueError, "no confidence/region"):
                service.search("classroom", ocr_region="content")
        loader.assert_not_called()

    def test_lazy_vlm_does_not_load_until_scoring_and_reuses_model(self):
        from kis.web_app import LazyVLMReranker

        factory = Mock(return_value=SimpleNamespace(score=Mock(return_value=8)))
        with patch.dict("sys.modules", {"kis.vlm_reranker": SimpleNamespace(VLMReranker=factory)}):
            reranker = LazyVLMReranker()
            factory.assert_not_called()
            self.assertEqual(reranker.score("001.jpg", "query"), 8)
            self.assertEqual(reranker.score("002.jpg", "query"), 8)
            factory.assert_called_once()

    def test_temporal_request_override_reaches_service(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        service = SimpleNamespace(search=Mock(return_value={"results": [], "temporal": {"status": "disabled"}}))
        with patch.object(web_app, "service", service):
            response = TestClient(web_app.app).post("/api/search", json={"query": "query", "temporal": False})
        self.assertEqual(response.status_code, 200)
        service.search.assert_called_once_with("query", None, False)

    def test_exact_bird_query_reaches_retrieval_after_schema_error_and_entity_normalization(self):
        from kis import web_app
        from kis.structure_query import create_query_structurer
        from tests.test_llm import FakeOpener, completion, groq_generation_error
        from tests.test_structure_query import BIRD_QUERY, bird_generated_query

        opener = FakeOpener(groq_generation_error(), completion(bird_generated_query()))
        with patch.dict("os.environ", {"GROQ_API_KEY": "environment-test-key"}, clear=True):
            service = web_app.SearchService()
            service._load = Mock()
            generator = create_query_structurer()
            generator.opener = opener
            with patch("kis.structure_query.create_query_structurer", return_value=generator), \
                 patch.object(web_app, "solve_kis", return_value=frame_candidates()) as retrieve:
                response = service.search(BIRD_QUERY)
        structured = retrieve.call_args.kwargs["structured_query"]
        self.assertEqual(structured["raw_query"], BIRD_QUERY)
        self.assertEqual(structured["entities"][4], {"type": "concept", "value": "vivid red eyes", "attributes": {}})
        self.assertTrue(all(isinstance(entity, dict) for entity in structured["entities"]))
        self.assertEqual(response["raw_query"], BIRD_QUERY)
        self.assertEqual(len(opener.calls), 2)

    def test_query_only_flow_uses_groq_api_key_from_environment(self):
        from kis import web_app
        from tests.test_structure_query import valid_generated_query

        with patch.dict("os.environ", {"GROQ_API_KEY": "environment-test-key"}, clear=True):
            service = web_app.SearchService()
            service._load = Mock()
            generated = valid_generated_query()
            generated["stream_queries"] = {stream: ["an artisan draws a portrait"] for stream in ("visual", "bm25", "semantic")}
            with patch("kis.llm.HTTPQueryStructurer.generate", return_value=generated), \
                 patch.object(web_app, "solve_kis", return_value=frame_candidates()):
                response = service.search("query")
        self.assertEqual(service.query_structurer.config.provider, "groq")
        self.assertEqual(service.query_structurer.config.model, "openai/gpt-oss-20b")
        self.assertEqual(service.query_structurer.config.api_key, "environment-test-key")
        self.assertEqual(response["query_api"]["provider"], "groq")
        self.assertNotIn("environment-test-key", str(response))

    def test_web_returns_two_hundred_and_caps_verified40_profile_at_twenty(self):
        from kis import web_app
        from kis.ranking import get_search_profile
        from tests.test_structure_query import valid_generated_query

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            service = web_app.SearchService()
            service.profile = get_search_profile("verified40")
            service._load = Mock()
            candidates = [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1,
                           "score": .8, "final_score": .8, "vlm_raw_score": 8 if i < 20 else None,
                           "vlm_status": "reranked" if i < 20 else "unscored"}
                          for i in range(200)]
            structured = {"query_id": "test", "raw_query": "query", "query_variants": ["query"]}
            with patch.object(web_app, "create_structured_query", return_value=structured), \
                    patch.object(web_app, "solve_kis", return_value=candidates) as solve, \
                    patch.object(web_app, "_resolve_candidate_image_path", return_value=None):
                response = service.search("query")
        self.assertEqual(solve.call_args.kwargs["rerank_top_k"], 20)
        self.assertEqual(solve.call_args.kwargs["result_top_k"], 200)
        self.assertEqual(solve.call_args.kwargs["retrieval_top_k"], 200)
        self.assertEqual(len(response["results"]), 200)
        self.assertEqual(len({(r['video_id'], r['frame_id']) for r in response['results']}), 200)
        self.assertEqual(response["result_limit"], 200)
        self.assertEqual(response["llm_scoring_limit"], 20)
        self.assertEqual(response["llm_scored_count"], 20)

    def test_request_provider_settings_reach_service_without_secret_repr(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        settings = {"provider": "custom", "model": "test-model", "base_url": "https://example.com/v1", "api_key": "private-key"}
        self.assertNotIn("private-key", repr(web_app.QueryAPISettings(**settings)))
        service = SimpleNamespace(search=Mock(return_value={"results": []}))
        with patch.object(web_app, "service", service):
            response = TestClient(web_app.app).post("/api/search", json={"query": "query", "llm": settings})
        self.assertEqual(response.status_code, 200)
        service.search.assert_called_once_with("query", settings)
        self.assertNotIn("private-key", response.text)

    def test_query_only_request_remains_compatible_and_unknown_settings_are_rejected(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        service = SimpleNamespace(search=Mock(return_value={"results": []}))
        with patch.object(web_app, "service", service):
            client = TestClient(web_app.app)
            self.assertEqual(client.post("/api/search", json={"query": "query"}).status_code, 200)
            self.assertEqual(client.post("/api/search", json={"query": "query", "llm": {"api_key_env": "SERVER_SECRET"}}).status_code, 422)
        service.search.assert_called_once_with("query")

    def test_status_reports_selected_provider_without_credentials(self):
        from kis.web_app import SearchService

        env = {"KIS_LLM_PROVIDER": "groq", "KIS_LLM_MODEL": "test-model", "KIS_LLM_API_KEY": "private-key"}
        with patch.dict("os.environ", env, clear=True):
            status = SearchService().status()
        self.assertTrue(status["query_planner_configured"])
        self.assertEqual(status["query_api"]["provider"], "groq")
        self.assertNotIn("private-key", str(status))

    def test_per_request_provider_does_not_replace_server_generator(self):
        from kis import web_app
        from tests.test_structure_query import valid_generated_query

        generator = Mock(generate=Mock(return_value=valid_generated_query()))
        with patch.dict("os.environ", {}, clear=True):
            service = web_app.SearchService()
            service._load = Mock()
            settings = {"provider": "groq", "model": "test-model", "api_key": "request-key"}
            with patch("kis.structure_query.create_query_structurer", return_value=generator) as factory, \
                 patch.object(web_app, "solve_kis", return_value=frame_candidates()):
                response = service.search("query", llm=settings)
        self.assertIsNone(service.query_structurer)
        self.assertEqual(factory.call_args.args[0].api_key, "request-key")
        generator.close.assert_called_once()
        self.assertEqual(response["query_api"]["provider"], "groq")
        self.assertNotIn("request-key", str(response))


if __name__ == "__main__":
    unittest.main()


class SimilarFrameVectorTest(unittest.TestCase):
    def test_vector_row_follows_mapping_row_and_rejects_unknown_frame(self):
        import numpy as np
        from kis import web_app
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mapping" / "L01").mkdir(parents=True)
            (root / "mapping" / "L01" / "L01_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0,25,0\n2,4,25,100\n", encoding="utf-8")
            features = root / "L01_V001.npy"
            np.save(features, np.array([[1, 0], [0, 1]], dtype=np.float16))
            with patch.object(web_app, "MAPPING_DIR", root / "mapping"), patch.object(web_app, "video_clip_path", return_value=features):
                self.assertEqual(web_app.frame_vector("L01_V001", 100).tolist(), [0.0, 1.0])
                with self.assertRaises(ValueError):
                    web_app.frame_vector("L01_V001", 50)
