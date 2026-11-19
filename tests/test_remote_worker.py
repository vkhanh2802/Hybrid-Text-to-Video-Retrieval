from dataclasses import replace
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlopen

from kis.remote_worker import RemoteWorker, runtime_signature, worker_settings
from kis.school_worker import handler, WorkerRuntime


TOKEN = "test-worker-token-" * 4


def rows(count=240):
    return [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1,
             "score": .9, "retrieval_text": "original caption"} for i in range(count)]


class RemoteWorkerTest(unittest.TestCase):
    def test_video_fused_candidates_get_ocr_before_single_qwen_pass(self):
        from kis import web_app
        from kis.ocr import OCRConfig
        from kis.video_query import DecodedVideoQuery
        service = web_app.SearchService()
        service._load = Mock()
        service.runtime_config = replace(service.runtime_config, ocr_format="school_snapshot", ocr=OCRConfig(mode="retrieve"))
        service.remote_worker = Mock(transport_failed=False)
        service.image_source = SimpleNamespace(reference=lambda *_: Path("001.jpg"))
        service.vlm_enabled = True
        service.vlm_reranker = Mock()
        order = []
        video = rows(200)
        text = [dict(rows(1)[0], frame_id=200, keyframe_n=201)]
        def retrieve(*a, **kw):
            order.append("text")
            self.assertEqual(kw["rerank_top_k"], 0)
            return text, {"query_variants": ["full DRES text"]}, {}, SimpleNamespace(provider="groq", protocol="openai", model="test")
        service._retrieve_text = retrieve
        def attach(values):
            order.append("ocr")
            self.assertEqual(len(values), 20)
            self.assertTrue(any(r["frame_id"] != 200 for r in values))
            for r in values:
                r["ocr"] = {"status": "ok", "text": "separate", "verified": False}
            return {"status": "attached"}
        service.remote_worker.attach_ocr.side_effect = attach
        def rerank(values, *a, **kw):
            order.append("qwen")
            self.assertEqual(kw["top_k"], 20)
            self.assertTrue(all("ocr" in r for r in values[:20]))
            return values
        @contextmanager
        def decode(*a, **kw):
            yield DecodedVideoQuery([object()], {})
        with patch.object(web_app, "decode_video_query", decode), \
                patch.object(web_app, "retrieve_video_query", return_value=video), \
                patch.object(web_app, "rerank_candidates_with_vlm", side_effect=rerank):
            result = service.search_video(b"video", official=True, hint_text="full DRES text")
        self.assertEqual(order, ["text", "ocr", "qwen"])
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(result["video_text_guidance"]["text_fusion"]["text_weight"], .7)

    def test_separate_ocr_endpoint_preserves_caption_and_rejects_shifted_provenance(self):
        client = self.client()
        pin = "a" * 64
        client.info = {"runtime": {"ocr_manifest_sha256": pin}}
        values = rows(2)
        evidence = [{**{k: r[k] for k in ("video_id", "frame_id", "keyframe_n")},
                     "ocr": {"status": "missing", "text": "", "source_format": "public_json",
                             "verified": False, "build_id": pin, "confidence": None, "engine": None,
                             "image_sha256": None}} for r in values]
        client.request = Mock(return_value={"candidates": evidence, "status": {"status": "attached"}})
        client.attach_ocr(values)
        self.assertTrue(all(r["retrieval_text"] == "original caption" for r in values))
        self.assertNotIn("retrieval_text", client.request.call_args.args[1]["candidates"][0])
        for field, value in (("verified", True), ("confidence", .9), ("build_id", "b" * 64)):
            evidence[0]["ocr"][field] = value
            with self.assertRaises(RuntimeError):
                client.attach_ocr(rows(2))
            evidence[0]["ocr"].update(verified=False, confidence=None, build_id=pin)
        evidence.reverse()
        with self.assertRaises(RuntimeError):
            client.attach_ocr(rows(2))

    def client(self):
        return RemoteWorker("http://127.0.0.1:18004", TOKEN, ("L01",))

    def test_only_explicit_loopback_ports_and_private_tokens(self):
        for url in ["http://example.com:1234", "https://127.0.0.1:1234", "http://127.0.0.1",
                    "http://user:pass@127.0.0.1:1234", "http://127.0.0.1:1234/path"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                RemoteWorker(url, TOKEN, ("L01",))
        with self.assertRaises(ValueError):
            RemoteWorker("http://127.0.0.1:1234", "short", ("L01",))

    def test_settings_do_not_enable_worker_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "missing.json"
            self.assertIsNone(worker_settings({}, path))
            self.assertEqual(worker_settings({"KIS_WORKER_URL": "http://127.0.0.1:12", "KIS_WORKER_TOKEN": TOKEN}, path)["token"], TOKEN)

    def test_full_pre_qwen_pool_not_truncated_and_query_not_shortened(self):
        client = self.client()
        client.request = Mock(return_value={"candidates": rows(), "diagnostics": {}})
        query = {"raw_query": "x" * 8000, "query_variants": ["full query"]}
        result = client.retrieve(query)
        self.assertEqual(len(result["candidates"]), 240)
        self.assertEqual(client.request.call_args.args[1]["structured_query"], query)
        self.assertNotIn(TOKEN, json.dumps(client.request.call_args.args[1]))

    def test_duplicate_short_nonfinite_and_out_of_scope_pools_fail(self):
        cases = [rows(199), rows() + [rows()[0]], [dict(row, video_id="L02_V001") for row in rows()],
                 [dict(row, score=float("nan")) for row in rows()], [dict(row, keyframe_n=None) for row in rows()]]
        for case in cases:
            with self.subTest(case=case[0].get("video_id") if case else None), self.assertRaises(RuntimeError):
                self.client().validate_candidates(case, complete=True)

    def test_config_and_model_mismatch_fail_closed(self):
        from kis.config import load_runtime_config, MODEL_MANIFEST
        config = load_runtime_config()
        client = self.client()
        client.request = Mock(return_value={"protocol": 1, "state": "ready", "runtime": {}, "models": MODEL_MANIFEST})
        with self.assertRaises(RuntimeError):
            client.connect(config, MODEL_MANIFEST)
        client.request.return_value["runtime"] = runtime_signature(config)
        client.request.return_value["result_limit"] = 200
        client.connect(config, MODEL_MANIFEST)
        client.request.return_value["result_limit"] = 100
        with self.assertRaises(RuntimeError):
            client.connect(config, MODEL_MANIFEST)

    def test_network_failure_is_distinct_from_a_missing_image(self):
        client = self.client()
        client._opener = Mock()
        client._opener.open.side_effect = URLError("tunnel closed")
        with self.assertRaises(RuntimeError):
            client.read("L01_V001", 1)
        self.assertTrue(client.transport_failed)

    def test_real_http_auth_image_404_and_no_redirect(self):
        runtime = SimpleNamespace(info={"state": "ready"}, image=Mock(side_effect=FileNotFoundError))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(runtime, TOKEN))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(url + "/status")
            self.assertEqual(error.exception.code, 401)
            client = RemoteWorker(url, TOKEN, ("L01",))
            self.assertEqual(client.request("/status")["state"], "ready")
            with self.assertRaises(FileNotFoundError):
                client.read("L01_V001", 1)
            with self.assertRaises(ValueError):
                client.read("../bad", 1)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_worker_retains_pool_before_vlm_and_keeps_auxiliary_temporal(self):
        from kis.config import load_runtime_config
        runtime = WorkerRuntime.__new__(WorkerRuntime)
        runtime.config = load_runtime_config()
        for name in ("visual", "encoder", "bm25", "captions", "lookup", "text_reranker", "ocr"):
            setattr(runtime, name, Mock())

        def solve(*args, **kwargs):
            self.assertEqual(kwargs["rerank_top_k"], 0)
            self.assertTrue(kwargs["temporal_as_auxiliary"])
            self.assertEqual(kwargs["result_top_k"], 200)
            self.assertEqual(kwargs["retrieval_top_k"], 200)
            kwargs["candidate_selector"].select(rows(240))
            kwargs["diagnostics"].update(stage_seconds={"visual": 1.0}, temporal_results=[])
            return rows(200)

        with patch("kis.solve_kis.solve_kis", side_effect=solve):
            result = runtime.retrieve({"structured_query": {"raw_query": "whole query"}})
        self.assertEqual(len(result["candidates"]), 240)
        self.assertIn("temporal_results", result["diagnostics"])

    def test_worker_limit_trims_pool_and_rejects_bad_limit(self):
        from kis.config import load_runtime_config
        runtime = WorkerRuntime.__new__(WorkerRuntime)
        runtime.config = load_runtime_config()
        for name in ("visual", "encoder", "bm25", "captions", "lookup", "text_reranker", "ocr"):
            setattr(runtime, name, Mock())

        def solve(*args, **kwargs):
            kwargs["candidate_selector"].select(rows(240))
            return rows(200)

        query = {"raw_query": "q"}
        with patch("kis.solve_kis.solve_kis", side_effect=solve):
            result = runtime.retrieve({"structured_query": query, "limit": 200})
            self.assertEqual(len(result["candidates"]), 200)
            for bad in (199, "200", True, 20001):
                with self.assertRaises(ValueError):
                    runtime.retrieve({"structured_query": query, "limit": bad})

    def test_remote_service_load_does_not_load_laptop_retrieval_models(self):
        from kis import web_app
        from kis.config import MODEL_MANIFEST
        with patch("kis.remote_worker.worker_settings", return_value={"url": "http://127.0.0.1:18004", "token": TOKEN}):
            service = web_app.SearchService()
        service.remote_worker.connect = Mock(return_value={})
        with patch("kis.config.load_caption_runtime") as local_load:
            service._load()
        local_load.assert_not_called()
        self.assertIs(service.visual_encoder, service.remote_worker)
        self.assertEqual(service.state, "ready")
        service.remote_worker.connect.assert_called_once_with(service.runtime_config, MODEL_MANIFEST)

    def test_remote_text_keeps_full_pool_for_local_qwen_and_twenty_budget(self):
        from kis import web_app
        with patch("kis.remote_worker.worker_settings", return_value={"url": "http://127.0.0.1:18004", "token": TOKEN}):
            service = web_app.SearchService()
        service._load = Mock()
        service.remote_worker.retrieve = Mock(return_value={"candidates": rows(240), "diagnostics": {"stage_seconds": {"visual": .1}}})
        structured = {"query_id": "test", "raw_query": "x" * 8000, "query_variants": ["full"]}
        service.vlm_reranker = Mock()
        with patch.object(web_app, "create_structured_query", return_value=structured), \
                patch("kis.structure_query.create_query_structurer", return_value=Mock()), \
                patch.object(web_app, "rerank_candidates_with_vlm", side_effect=lambda values, *a, **k: values) as rerank:
            response = service.search("x" * 8000)
        self.assertEqual(len(rerank.call_args.args[0]), 240)
        self.assertEqual(rerank.call_args.kwargs["top_k"], 20)
        self.assertEqual(len(response["results"]), 200)
        self.assertEqual(service.remote_worker.retrieve.call_args.args[0]["raw_query"], "x" * 8000)
        self.assertEqual(response["stage_seconds"]["visual"], .1)


if __name__ == "__main__":
    unittest.main()
