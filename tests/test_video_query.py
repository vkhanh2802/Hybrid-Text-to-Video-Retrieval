from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from kis import video_query as video


def probe(duration=2, **overrides):
    stream = {"width": 320, "height": 180, "duration": str(duration), "avg_frame_rate": "10/1", **overrides}
    return json.dumps({"streams": [stream], "format": {"duration": str(duration)}}).encode()


class VideoDecodeTest(unittest.TestCase):
    def test_samples_are_closed_and_upload_deleted_before_inference(self):
        paths = []

        def run(command):
            if "ffprobe" in command[0]:
                path = Path(command[-1])
                self.assertEqual(path.read_bytes(), b"video bytes")
                paths.append(path)
                return probe()
            self.assertIn("file,pipe", command)
            self.assertIn(video.INPUT_FORMATS, command)
            return bytes([50]) * (video.SAMPLE_SIZE ** 2 * 3 * 4)

        with patch.object(video.shutil, "which", side_effect=lambda name: name), \
                patch.object(video, "_run_media_tool", side_effect=run):
            with video.decode_video_query(b"video bytes") as clip:
                self.assertFalse(paths[0].exists())
                self.assertEqual(len(clip.images), 4)
                self.assertEqual(clip.metadata["sampled_frames"], 4)
                self.assertFalse(clip.metadata["upload_stored"])
                self.assertFalse(clip.metadata["temporal_verified"])
                image = clip.images[0]
                self.assertEqual(image.getpixel((0, 0)), (50, 50, 50))
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_input_and_missing_tools_rejected_without_subprocess(self):
        with patch.object(video, "_run_media_tool") as run:
            for data in (b"", "not bytes", b"x" * (video.MAX_VIDEO_BYTES + 1)):
                with self.subTest(size=len(data)):
                    with self.assertRaises(ValueError):
                        with video.decode_video_query(data):
                            self.fail("invalid input")
            with patch.object(video.shutil, "which", return_value=None):
                with self.assertRaises(RuntimeError):
                    with video.decode_video_query(b"video"):
                        self.fail("tools unavailable")
            run.assert_not_called()

    def test_duration_resolution_and_fps_are_bounded(self):
        for data in (probe(.5), probe(3.1), probe("nan"), probe(2, width=0),
                     probe(2, width=10000, height=10000), probe(2, avg_frame_rate="121/1"),
                     probe(2, avg_frame_rate="0/0"), b'{"streams":[]}'):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    video._video_metadata(data)
        for duration in (1, 2, 3):
            self.assertEqual(video._video_metadata(probe(duration))["duration_seconds"], duration)

    def test_incomplete_or_excess_samples_rejected_and_temp_cleaned(self):
        for raw in (b"", b"incomplete", b"x" * (video.SAMPLE_SIZE ** 2 * 3 * 7)):
            with self.subTest(size=len(raw)), patch.object(video.shutil, "which", side_effect=lambda name: name), \
                    patch.object(video, "_run_media_tool", side_effect=[probe(), raw]):
                with self.assertRaises(ValueError):
                    with video.decode_video_query(b"video"):
                        self.fail("bad output")

    def test_tool_errors_are_sanitized_and_have_timeout(self):
        with patch.object(video.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=b"private-path-and-secret")) as run:
            with self.assertRaises(ValueError) as error:
                video._run_media_tool(["ffmpeg", "args"])
            self.assertNotIn("private", str(error.exception))
            self.assertEqual(run.call_args.kwargs["timeout"], video.DECODE_TIMEOUT_SECONDS)
        with patch.object(video.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 20)):
            with self.assertRaisesRegex(ValueError, "timed out"):
                video._run_media_tool(["ffmpeg"])


class VideoRetrievalTest(unittest.TestCase):
    def test_required_hundred_expands_search_without_reencoding_clip(self):
        encoder = Mock(encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)))
        rows = [{"video_id": "L01_V001", "frame_id": i, "score": .9 - i / 1000}
                for i in range(100)]
        retriever = Mock(search=Mock(side_effect=[rows[:10] * 10, rows]))
        diagnostics = {}
        results = video.retrieve_video_query([object()], encoder, retriever, None,
            top_k=100, require_full_results=True, pool_diagnostics=diagnostics)
        self.assertEqual(len(results), 100)
        self.assertEqual(len({row["frame_id"] for row in results}), 100)
        encoder.encode_images.assert_called_once()
        self.assertEqual([call.kwargs["top_k"] for call in retriever.search.call_args_list], [100, 200])
        np.testing.assert_array_equal(retriever.search.call_args_list[0].args[0],
                                      retriever.search.call_args_list[1].args[0])
        self.assertEqual(diagnostics["expansions"], 1)

    def test_status_describes_full_text_pipeline_rank_fusion(self):
        fusion = video.video_query_status()["text_fusion"]
        self.assertEqual(fusion["mode"], "weighted_rrf")
        self.assertTrue(fusion["full_text_pipeline"])
        self.assertEqual(fusion["text_weight"], .7)
        self.assertEqual(fusion["k"], 60)

    def test_full_text_rank_fusion_weights_union_metadata_and_no_input_mutation(self):
        text = [{"video_id": "L01_V001", "frame_id": 1, "score": 20,
                 "final_score": .9, "keyframe_n": 2, "retrieval_text": "red shirt",
                 "text_ranking_score": .9, "vlm_raw_score": 10,
                 "evidence": {"bm25": {"raw_score": 20}}},
                {"video_id": "L01_V001", "frame_id": 2, "score": .5}]
        visual = [{"video_id": "L01_V001", "frame_id": 3, "score": .99},
                  {"video_id": "L01_V001", "frame_id": 2, "score": .8}]
        before = json.dumps([text, visual], sort_keys=True)
        result = video.fuse_video_text_results(text, visual)
        by_frame = {item["frame_id"]: item for item in result}
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(by_frame[1]["score"], .7)
        self.assertAlmostEqual(by_frame[3]["score"], .3)
        self.assertAlmostEqual(by_frame[2]["score"], 61 / 62)
        self.assertEqual(result[0]["frame_id"], 2)
        self.assertEqual(by_frame[1]["keyframe_n"], 2)
        self.assertEqual(by_frame[1]["evidence"]["text_pipeline"]["evidence"], text[0]["evidence"])
        self.assertNotIn("vlm_raw_score", by_frame[1])
        self.assertNotIn("text_ranking_score", by_frame[1])
        self.assertEqual(json.dumps([text, visual], sort_keys=True), before)

    def test_full_text_rank_fusion_deduplicates_caps_and_rejects_nonfinite_scores(self):
        rows = [{"video_id": "L01_V001", "frame_id": i, "score": .8} for i in range(120)]
        result = video.fuse_video_text_results([rows[0], *rows], [])
        self.assertEqual(len(result), 100)
        self.assertEqual(len({item["frame_id"] for item in result}), 100)
        self.assertAlmostEqual(result[1]["score"], .7 * 61 / 62)
        self.assertEqual(video.fuse_video_text_results([], []), [])
        for field in ("score", "final_score"):
            with self.assertRaises(ValueError):
                video.fuse_video_text_results([{**rows[0], field: float("nan")}], [])

    def test_normalized_pool_uses_image_encoder_and_hydrates_without_text_search(self):
        vectors = np.zeros((2, 768), dtype=np.float32)
        vectors[0, 0], vectors[1, 1] = 2, 4
        encoder = Mock(encode_images=Mock(return_value=vectors))
        retriever = Mock(search=Mock(return_value=[
            {"video_id": "L01_V001", "frame_id": 10, "score": .8},
            {"video_id": "L01_V001", "frame_id": 10, "score": .8},
            {"video_id": "L02_V001", "frame_id": 20, "score": .9},
        ]))
        captions = Mock(get_caption=Mock(return_value={"caption": "Scene", "retrieval_text": "Scene", "keyframe_n": 1}))
        images = [object(), object()]
        results = video.retrieve_video_query(images, encoder, retriever, captions)
        actual = retriever.search.call_args.args[0]
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1, places=6)
        np.testing.assert_allclose(actual[:2], np.ones(2) / np.sqrt(2), atol=1e-6)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["video_id"], "L02_V001")
        self.assertEqual(results[0]["caption"], "Scene")
        self.assertEqual(results[0]["vlm_status"], "skipped_video_query")
        captions.search.assert_not_called()
        encoder.encode_text.assert_not_called()

    def test_text_embedding_is_weighted_seventy_percent_and_reported(self):
        video_vector = np.zeros((1, 768), dtype=np.float32)
        video_vector[0, 0] = 1
        text_vector = np.zeros(768, dtype=np.float32)
        text_vector[1] = 1
        encoder = Mock(encode_images=Mock(return_value=video_vector),
                       encode_text=Mock(return_value=text_vector))
        retriever = Mock(search=Mock(return_value=[
            {"video_id": "L01_V001", "frame_id": 10, "score": .8}]))
        results = video.retrieve_video_query([object()], encoder, retriever, None,
                                             top_k=100, text_query="red shirt")
        actual = retriever.search.call_args.args[0]
        self.assertAlmostEqual(float(actual[0]), .3 / np.sqrt(.58), places=6)
        self.assertAlmostEqual(float(actual[1]), .7 / np.sqrt(.58), places=6)
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1, places=6)
        encoder.encode_text.assert_called_once_with("red shirt")
        self.assertEqual(results[0]["evidence"]["text_fusion"]["text_weight"], .7)
        self.assertEqual(results[0]["evidence"]["visual"]["query_source"],
                         "video_image_mean_pool+text_embedding")

    def test_invalid_text_embedding_never_reaches_milvus(self):
        encoder = Mock(encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)),
                       encode_text=Mock(return_value=np.zeros(768, dtype=np.float32)))
        retriever = Mock()
        with self.assertRaises(RuntimeError):
            video.retrieve_video_query([object()], encoder, retriever, None, text_query="red shirt")
        retriever.search.assert_not_called()

    def test_invalid_embeddings_do_not_reach_milvus(self):
        for vectors in (np.zeros((1, 768)), np.ones((1, 384)), np.full((1, 768), np.nan)):
            retriever = Mock()
            with self.assertRaises(RuntimeError):
                video.retrieve_video_query([object()], Mock(encode_images=Mock(return_value=vectors)), retriever, None)
            retriever.search.assert_not_called()


class VideoAPITest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from kis import web_app
        self.web = web_app
        self.service = Mock(search_video=Mock(return_value={"input_type": "video", "results": []}))
        self.patcher = patch.object(web_app, "service", self.service)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = TestClient(web_app.app)

    def test_raw_upload_works_without_multipart_and_without_llm(self):
        response = self.client.post("/api/search/video", content=b"video", headers={"Content-Type": "video/mp4"})
        self.assertEqual(response.status_code, 200)
        self.service.search_video.assert_called_once_with(b"video")
        self.service.search.assert_not_called()

    def test_bad_mime_empty_and_declared_oversize_are_rejected(self):
        cases = [(b"video", {"Content-Type": "application/json"}, 415),
                 (b"", {"Content-Type": "video/mp4"}, 422),
                 (b"x", {"Content-Type": "video/mp4", "Content-Length": str(video.MAX_VIDEO_BYTES + 1)}, 413)]
        for data, headers, status in cases:
            self.assertEqual(self.client.post("/api/search/video", content=data, headers=headers).status_code, status)
        self.service.search_video.assert_not_called()

    def test_streamed_upload_limit_cannot_be_bypassed_by_content_length(self):
        with patch.object(self.web, "MAX_VIDEO_BYTES", 3):
            response = self.client.post("/api/search/video", content=b"1234",
                headers={"Content-Type": "video/mp4", "Content-Length": "1"})
        self.assertEqual(response.status_code, 413)
        self.service.search_video.assert_not_called()

    def test_decode_errors_are_422_and_missing_runtime_503(self):
        for error, status in ((ValueError("invalid clip"), 422), (RuntimeError("missing tools"), 503)):
            self.service.search_video.side_effect = error
            response = self.client.post("/api/search/video", content=b"video", headers={"Content-Type": "video/mp4"})
            self.assertEqual(response.status_code, status)

    def test_video_service_does_not_need_query_planner_or_qwen(self):
        from kis.web_app import SearchService

        @contextmanager
        def decode(data):
            yield video.DecodedVideoQuery([object()], {"duration_seconds": 2, "sampled_frames": 1})

        service = SearchService()
        service._load = Mock()
        service.visual_encoder = Mock(encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)))
        service.visual_retriever = Mock(search=Mock(return_value=[
            {"video_id": "L01_V001", "frame_id": i, "score": .9} for i in range(200)]))
        service.caption_retriever = Mock(get_caption=Mock(return_value={"keyframe_n": 1}))
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"))
        with patch.object(self.web, "decode_video_query", decode), \
                patch("kis.structure_query.create_query_structurer", side_effect=AssertionError("No LLM for video")):
            result = service.search_video(b"video")
        self.assertEqual(result["input_type"], "video")
        self.assertIsNone(result["query_api"])
        self.assertFalse(result["vlm_enabled"])
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(service.visual_retriever.search.call_args.kwargs["top_k"], 200)
        self.assertEqual(result["result_limit"], 200)
        self.assertEqual(result["llm_scored_count"], 0)

    def test_runtime_connection_failure_is_actionable_and_closes_samples(self):
        from kis.web_app import SearchService

        image = Image.new("RGB", (4, 4))

        @contextmanager
        def decode(data):
            try:
                yield video.DecodedVideoQuery([image], {})
            finally:
                image.close()

        service = SearchService()
        service._load = Mock(side_effect=Exception("Synthetic Milvus transport error"))
        with patch.object(self.web, "decode_video_query", decode), self.assertLogs(self.web.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "Docker/Milvus"):
                service.search_video(b"video")
        with self.assertRaises(ValueError):
            image.getpixel((0, 0))

    def test_official_video_description_scores_only_first_twenty_of_two_hundred_without_backfill(self):
        from kis.web_app import SearchService

        @contextmanager
        def decode(data, *, official=False):
            self.assertTrue(official)
            yield video.DecodedVideoQuery([object()], {"duration_seconds": 5, "sampled_frames": 1})

        @contextmanager
        def open_candidate(candidate, lookup):
            yield None if candidate["frame_id"] == 0 else Path(f"{candidate['frame_id']}.jpg")

        service = SearchService()
        service._load = Mock()
        service.vlm_enabled = True
        service.vlm_reranker = Mock(score=Mock(return_value=8))
        service.visual_encoder = Mock(encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)),
                                      encode_text=Mock(return_value=np.ones(768, dtype=np.float32)))
        service.visual_retriever = Mock(search=Mock(return_value=[
            {"video_id": "L01_V001", "frame_id": i, "score": 1 - i / 201} for i in range(200)]))
        service.caption_retriever = Mock(get_caption=Mock(side_effect=lambda video_id, frame: {"keyframe_n": frame + 1}))
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"), open_candidate=open_candidate)
        service.query_structurer = SimpleNamespace(config=SimpleNamespace(provider="groq", protocol="openai", model="test"))
        text_results = [{"video_id": "L01_V001", "frame_id": i, "score": .8,
                         "final_score": .8, "keyframe_n": i + 1} for i in range(200)]
        structured = {"query_id": "text", "raw_query": "red shirt", "query_variants": ["person wearing red"]}
        with patch.object(self.web, "decode_video_query", decode), \
                patch.object(self.web, "create_structured_query", return_value=structured) as structure, \
                patch.object(self.web, "solve_kis", return_value=text_results) as solve:
            result = service.search_video(b"official-video", official=True, description="  red shirt  ")
        structure.assert_called_once_with("red shirt", output_path=None, generator=service.query_structurer)
        self.assertEqual(solve.call_args.kwargs["rerank_top_k"], 0)
        self.assertIsNone(solve.call_args.kwargs["vlm_reranker"])
        self.assertIs(solve.call_args.kwargs["text_bm25_retriever"], service.bm25_retriever)
        self.assertIs(solve.call_args.kwargs["text_semantic_retriever"], service.caption_retriever)
        self.assertIs(solve.call_args.kwargs["text_reranker_config"], service.runtime_config.text_reranker)
        self.assertIs(solve.call_args.kwargs["ocr_config"], service.runtime_config.ocr)
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(service.visual_retriever.search.call_args.kwargs["top_k"], 200)
        self.assertEqual(service.vlm_reranker.score.call_count, 19)
        self.assertEqual({int(call.args[0].stem) for call in service.vlm_reranker.score.call_args_list}, set(range(1, 20)))
        self.assertTrue(all(call.args[1] == "red shirt" for call in service.vlm_reranker.score.call_args_list))
        by_frame = {item["frame_id"]: item for item in result["results"]}
        self.assertEqual(by_frame[0]["vlm_status"], "missing_image")
        self.assertTrue(all(by_frame[i]["vlm_status"] == "unscored" for i in range(20, 200)))
        self.assertEqual(result["llm_scored_count"], 19)
        self.assertEqual(result["llm_scoring_limit"], 20)
        self.assertEqual(result["supplemental_description"], "red shirt")
        self.assertEqual(result["video_text_guidance"]["status"], "reranked")
        self.assertEqual(result["query_api"]["provider"], "groq")
        self.assertEqual(result["query_variants"], ["person wearing red"])
        self.assertEqual(result["video_text_guidance"]["text_fusion"]["mode"], "weighted_rrf")
        service.visual_encoder.encode_text.assert_not_called()  # text is handled by solve_kis, not embedding fusion

    def test_disabled_vlm_still_runs_full_text_pipeline_and_changes_retrieval(self):
        from kis.web_app import SearchService

        @contextmanager
        def decode(data, **kwargs):
            yield video.DecodedVideoQuery([object()], {"duration_seconds": 5})

        service = SearchService()
        service._load = Mock()
        service.vlm_enabled = False
        service.vlm_reranker = Mock()
        service.visual_encoder = Mock(encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)),
                                      encode_text=Mock(return_value=np.ones(768, dtype=np.float32)))
        service.visual_retriever = Mock(search=Mock(return_value=[
            {"video_id": "L01_V001", "frame_id": i, "score": .8} for i in range(200)]))
        service.caption_retriever = Mock(get_caption=Mock(return_value={"keyframe_n": 1}))
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"))
        service.query_structurer = SimpleNamespace(config=SimpleNamespace(provider="groq", protocol="openai", model="test"))
        structured = {"query_id": "text", "raw_query": "DRES sentence\nred shirt", "query_variants": ["red shirt"]}
        text_results = [{"video_id": "L01_V001", "frame_id": 200, "score": .8, "keyframe_n": 201}]
        with patch.object(self.web, "decode_video_query", decode), \
                patch.object(self.web, "create_structured_query", return_value=structured) as structure, \
                patch.object(self.web, "solve_kis", return_value=text_results):
            result = service.search_video(b"video", official=True, hint_text="DRES sentence",
                                          description="red shirt")
        self.assertEqual(result["results"][0]["frame_id"], 200)  # text-only candidate beats video-only candidate
        self.assertAlmostEqual(result["results"][0]["score"], .7)
        self.assertEqual(result["results"][0]["vlm_status"], "disabled")
        self.assertEqual(result["combined_text_query"], "DRES sentence\nred shirt")
        self.assertEqual(result["video_text_guidance"]["text_fusion"]["text_weight"], .7)
        structure.assert_called_once_with("DRES sentence\nred shirt", output_path=None, generator=service.query_structurer)
        self.assertEqual(result["video_text_guidance"]["status"], "disabled")
        self.assertEqual(result["llm_scored_count"], 0)
        self.assertFalse(result["vlm_enabled"])
        service.vlm_reranker.score.assert_not_called()

    def test_hint_only_video_runs_real_text_solver_with_planner_three_streams_and_text_reranker(self):
        from kis.web_app import SearchService
        from kis.text_reranker import TextRerankerConfig
        from kis.llm import _api_ssl_context
        from tests.test_structure_query import valid_generated_query

        _api_ssl_context()  # Initialize Windows certificates before isolating environment.

        @contextmanager
        def decode(data, **kwargs):
            yield video.DecodedVideoQuery([object()], {"duration_seconds": 5})

        def hit(frame, score=.8):
            return {"video_id": "L01_V001", "frame_id": frame, "score": score}

        generated = valid_generated_query()
        generated["entities"] = []
        generated["stream_queries"] = {stream: list(generated["query_variants"])
                                       for stream in ("visual", "bm25", "semantic")}
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            service = SearchService()
            service._load = Mock()
            service.vlm_enabled = False
            service.vlm_reranker = Mock()
            service.runtime_config = replace(service.runtime_config,
                text_reranker=TextRerankerConfig(mode="rerank"))
            service.text_reranker = Mock(score_pairs=Mock(side_effect=lambda pairs: list(range(len(pairs)))))
            service.visual_encoder = Mock(
                encode_images=Mock(return_value=np.eye(1, 768, dtype=np.float32)),
                encode_text=Mock(return_value=np.eye(1, 768, k=1, dtype=np.float32)[0]))
            service.visual_retriever = Mock(search=Mock(side_effect=lambda vector, **kw:
                [hit(i, .9 - i / 1000) for i in range(200)] if vector[0] else [hit(101)]))
            service.bm25_retriever = Mock(search_bm25=Mock(return_value=[hit(2, 10)]))
            service.caption_retriever = Mock(search=Mock(return_value=[hit(i + 100) for i in range(200)]),
                get_caption=Mock(side_effect=lambda vid, frame: {
                    "retrieval_text": f"caption {frame}", "keyframe_n": frame + 1}))
            service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"))
            with patch.object(self.web, "decode_video_query", decode), \
                    patch("kis.llm.HTTPQueryStructurer.generate", return_value=generated) as planner, \
                    patch("kis.llm.HTTPQueryStructurer._post", side_effect=AssertionError("No live API in tests")):
                result = service.search_video(b"video", official=True, hint_text="First DRES sentence.\nSecond DRES sentence.")
        planner.assert_called_once_with("First DRES sentence.\nSecond DRES sentence.")
        self.assertGreater(service.visual_encoder.encode_text.call_count, 0)
        self.assertGreater(service.bm25_retriever.search_bm25.call_count, 0)
        self.assertGreater(service.caption_retriever.search.call_count, 0)
        self.assertGreater(service.text_reranker.score_pairs.call_count, 0)
        self.assertEqual(result["text_reranker"]["status"], "applied")
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(len({row["frame_id"] for row in result["results"]}), 200)
        self.assertIn(101, {row["frame_id"] for row in result["results"]})
        self.assertNotEqual(result["results"][0]["frame_id"], 0)
        self.assertEqual(result["query_api"]["provider"], "groq")
        self.assertEqual(result["query_variants"], [result["combined_text_query"], *generated["query_variants"]])
        self.assertNotIn("test-key", json.dumps(result))
        service.vlm_reranker.score.assert_not_called()

    def test_temporal_auxiliary_evidence_does_not_reduce_frames_or_receive_vlm_scores(self):
        from kis.web_app import SearchService

        @contextmanager
        def decode(data, **kwargs):
            yield video.DecodedVideoQuery([object()], {})

        temporal = {"status": "partial", "verified": False, "events": []}
        service = SearchService()
        service._load = Mock()
        service.vlm_enabled = True
        service.vlm_reranker = Mock()
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"))
        config = SimpleNamespace(provider="groq", protocol="openai", model="test")
        service._retrieve_text = Mock(return_value=(
            [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1, "score": .8}
             for i in range(200)],
            {"query_variants": ["pour then stir"]},
            {"temporal": {"status": "applied", "presentation": "auxiliary"}, "temporal_results": [
                {"video_id": "L01_V001", "frame_id": 200, "keyframe_n": 201, "score": .8,
                 "temporal": temporal, "vlm_status": "skipped_temporal"}]}, config))
        with patch.object(self.web, "decode_video_query", decode), \
                patch.object(self.web, "retrieve_video_query", return_value=[]):
            result = service.search_video(b"video", official=True, description="pour then stir")
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(result["llm_scoring_limit"], 20)
        self.assertEqual(result["llm_scored_count"], 0)
        self.assertFalse(result["temporal_results"][0]["temporal"]["verified"])
        self.assertTrue(all(row["temporal"] is None for row in result["results"]))
        self.assertIsNone(result["temporal_results"][0]["vlm_score"])
        service.vlm_reranker.score.assert_not_called()

    def test_vlm_budget_applies_after_union_and_selects_text_only_candidates(self):
        from kis.web_app import SearchService

        @contextmanager
        def decode(data, **kwargs):
            yield video.DecodedVideoQuery([object()], {})

        @contextmanager
        def open_candidate(candidate, lookup):
            yield Path(f"{candidate['frame_id']}.jpg")

        def rows(start):
            return [{"video_id": "L01_V001", "frame_id": start + i, "score": .8,
                     "keyframe_n": start + i + 1} for i in range(200)]

        service = SearchService()
        service._load = Mock()
        service.vlm_enabled = True
        service.vlm_reranker = Mock(score=Mock(return_value=8))
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"), open_candidate=open_candidate)
        service._retrieve_text = Mock(return_value=(rows(200), {"query_variants": ["red shirt"]}, {},
            SimpleNamespace(provider="groq", protocol="openai", model="test")))
        with patch.object(self.web, "decode_video_query", decode), \
                patch.object(self.web, "retrieve_video_query", return_value=rows(0)):
            result = service.search_video(b"video", official=True, description="red shirt")
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(service.vlm_reranker.score.call_count, 20)
        self.assertEqual({int(call.args[0].stem) for call in service.vlm_reranker.score.call_args_list},
                          set(range(200, 220)))
        self.assertEqual(result["llm_scored_count"], 20)
        service._retrieve_text.assert_called_once_with("red shirt", rerank_top_k=0)

    def test_text_planner_failure_does_not_silently_fall_back_to_video_and_closes_images(self):
        from kis.web_app import SearchService
        image = Image.new("RGB", (4, 4))

        @contextmanager
        def decode(data, **kwargs):
            try:
                yield video.DecodedVideoQuery([image], {})
            finally:
                image.close()

        service = SearchService()
        service._load = Mock()
        service._retrieve_text = Mock(side_effect=RuntimeError("planner unavailable"))
        with patch.object(self.web, "decode_video_query", decode), \
                patch.object(self.web, "retrieve_video_query", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "planner unavailable"):
                service.search_video(b"video", official=True, description="red shirt")
        with self.assertRaises(ValueError):
            image.getpixel((0, 0))

    def test_invalid_or_non_official_description_fails_before_decoding_or_loading(self):
        from kis.web_app import SearchService
        service = SearchService()
        service._load = Mock()
        with patch.object(self.web, "decode_video_query") as decode:
            for value in (None, "x" * 2001):
                with self.assertRaises(ValueError):
                    service.search_video(b"video", official=True, description=value)
            with self.assertRaises(ValueError):
                service.search_video(b"manual", description="red shirt")
            with self.assertRaises(ValueError):
                service.search_video(b"manual", official=True, hint_text="x" * 16001)
            with self.assertRaises(ValueError):
                service.search_video(b"manual", hint_text="official text")
            with self.assertRaisesRegex(ValueError, "Tổng text"):
                service.search_video(b"video", official=True, hint_text="x" * 8000, description="extra")
        decode.assert_not_called()
        service._load.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires installed FFmpeg/ffprobe")
class RealVideoDecodeTest(unittest.TestCase):
    def test_official_five_second_video_samples_full_clip_with_six_frame_budget(self):
        with tempfile.TemporaryDirectory(prefix="kis-dres-video-test-") as directory:
            path = Path(directory) / "official.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=10", "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                capture_output=True, check=True, timeout=30)
            with video.decode_video_query(path.read_bytes(), official=True) as clip:
                self.assertEqual(clip.metadata["duration_seconds"], 5)
                self.assertEqual(len(clip.images), 6)
                self.assertAlmostEqual(clip.metadata["sample_fps"], 1.2)
                self.assertEqual(clip.metadata["query_source"], "dres_official")
            with self.assertRaises(ValueError):
                with video.decode_video_query(path.read_bytes()):
                    self.fail("Manual input must keep its 1-3 second limit")

    def test_real_two_second_clip_and_playlist_rejection(self):
        with tempfile.TemporaryDirectory(prefix="kis-video-test-") as directory:
            path = Path(directory) / "test.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=10", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                capture_output=True, check=True, timeout=30)
            with video.decode_video_query(path.read_bytes()) as clip:
                self.assertEqual(clip.metadata["duration_seconds"], 2)
                self.assertEqual(len(clip.images), 4)
            with self.assertRaises(ValueError):
                with video.decode_video_query(b"#EXTM3U\nhttp://127.0.0.1:1/private.ts\n"):
                    self.fail("Playlist was accepted")


if __name__ == "__main__":
    unittest.main()
