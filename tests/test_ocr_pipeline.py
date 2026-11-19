from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kis.config import load_runtime_config
from kis.ocr import OCRConfig
from kis.solve_kis import solve_kis
from tests.test_ocr import OCRFixture
from tests.test_public_ocr import PublicOCRFixture


class OCRPipelineTest(OCRFixture):
    def dependencies(self):
        return (Mock(search=Mock(return_value=[{"video_id": "L01_V002", "frame_id": 90, "score": .8}])),
                Mock(encode_text=Mock(return_value=[1, 0])), Mock(search_bm25=Mock(return_value=[])),
                Mock(search=Mock(return_value=[]), get_caption=Mock(return_value=None)))

    def query(self):
        return {"raw_query": 'Tìm cảnh bảng ghi "6A"', "query_variants": ["A classroom"],
                "needs_ocr": True, "entities": []}

    def test_independent_ocr_candidate_reaches_final_results_before_vlm(self):
        retriever, _, _ = self.index(["LỚP 6A"])
        v, e, b, c = self.dependencies()
        diagnostics = {}
        results = solve_kis(self.query(), v, e, b, c, None, ocr_config=OCRConfig(mode="retrieve"),
                            ocr_retriever=retriever, diagnostics=diagnostics, rerank_top_k=0)
        self.assertEqual(results[0]["video_id"], "L01_V001")
        self.assertEqual(results[0]["keyframe_n"], 1)
        self.assertEqual(results[0]["ocr"]["text"], "LỚP 6A")
        self.assertNotIn("retrieval_text", results[0])
        self.assertEqual(diagnostics["ocr"]["candidate_union_added"], 1)

    def test_disabled_diagnostics_and_unavailable_preserve_baseline(self):
        v, e, b, c = self.dependencies()
        base = solve_kis(self.query(), v, e, b, c, None, rerank_top_k=0)
        retriever, _, _ = self.index(["6A"])
        diag = {}
        result = solve_kis(self.query(), v, e, b, c, None, rerank_top_k=0, ocr_retriever=retriever,
                           ocr_config=OCRConfig(mode="diagnostics"), diagnostics=diag)
        self.assertEqual(result, base)
        self.assertEqual(diag["ocr"]["status"], "matched")
        broken = Mock(search=Mock(side_effect=ValueError("checksum")))
        result = solve_kis(self.query(), v, e, b, c, None, rerank_top_k=0, ocr_retriever=broken,
                           ocr_config=OCRConfig(mode="retrieve"), diagnostics=diag)
        self.assertEqual(result, base)
        self.assertEqual(diag["ocr"]["status"], "error")

    def test_temporal_ocr_seeds_new_video_but_does_not_satisfy_unsupported_event(self):
        retriever, _, _ = self.index(["6A"])
        v, e, b, c = self.dependencies()
        v.search.return_value = []
        actions = ["Students read a 6A sign", "Students walk upstairs"]
        query = dict(self.query(), events=[{"action": action} for action in actions],
                     temporal_constraints=[{"type": "sequence", "order": actions}])
        frames = [{k: f[k] for k in ("video_id", "frame_id", "keyframe_n", "pts_time")} for f in self.frames[:2]]
        scorer = Mock(score=Mock(return_value=(frames, np.array([[.1, 0], [0, .8]]),
            np.array([[False, False], [False, True]]), None, None, {"sampled": False})))
        diagnostics = {}
        result = solve_kis(query, v, e, b, c, None, event_scorer=scorer,
                           ocr_config=OCRConfig(mode="retrieve"), ocr_retriever=retriever, diagnostics=diagnostics)
        self.assertEqual(diagnostics["temporal"]["candidate_videos"], ["L01_V001"])
        self.assertEqual(scorer.score.call_args.args[-1], {10})
        self.assertEqual(result[0]["temporal"]["events"][0]["status"], "unknown")
        self.assertEqual(result[0]["temporal"]["coverage"], .5)
        self.assertFalse(result[0]["temporal"]["verified"])

    def test_runtime_config_defaults_off_and_explicit_mode_override(self):
        self.assertEqual(load_runtime_config(environ={}).ocr.mode, "off")
        config = load_runtime_config(environ={"KIS_OCR_MODE": "diagnostics", "KIS_OCR_INDEX": "custom/index.sqlite3"})
        self.assertEqual(config.ocr.mode, "diagnostics")
        self.assertTrue(config.ocr_index.is_absolute())
        with self.assertRaises(ValueError):
            load_runtime_config(environ={"KIS_OCR_MODE": "on"})


class OCRAPITest(unittest.TestCase):
    def test_manual_ocr_query_and_region_reach_service_and_bad_region_is_rejected(self):
        from fastapi.testclient import TestClient
        from kis import web_app

        service = SimpleNamespace(search=Mock(return_value={"results": []}))
        with patch.object(web_app, "service", service):
            client = TestClient(web_app.app)
            response = client.post("/api/search", json={"query": "a classroom", "ocr_text": "6A", "ocr_region": "content"})
            bad = client.post("/api/search", json={"query": "a classroom", "ocr_region": "guess_board"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(bad.status_code, 422)
        service.search.assert_called_once_with("a classroom", None, None, "6A", "content")

    def test_api_serialization_keeps_ocr_separate_from_caption(self):
        from kis.web_app import serialize_search_results

        row = {"video_id": "L01_V001", "frame_id": 10, "final_score": .5,
               "ocr": {"text": "6A", "source": "pixel_ocr", "verified": False}}
        result = serialize_search_results([row], None, lambda *_: None)[0]
        self.assertEqual(result["caption"], "")
        self.assertEqual(result["ocr"]["text"], "6A")
        self.assertFalse(result["ocr"]["verified"])


class PublicOCRPipelineTest(PublicOCRFixture):
    def test_real_visual_hit_shape_without_caption_resolves_csv_identity_before_qwen(self):
        for frames in ((101,), (101, 999)):
            with self.subTest(frames=frames):
                # Match SiglipRetriever.search's actual output: no n/image/caption.
                visual_rows = [{"video_id": "L04_V001", "frame_id": frame, "score": .8} for frame in frames]
                original_rows = [dict(row) for row in visual_rows]
                captions = Mock(search=Mock(return_value=[]), get_caption=Mock(return_value=None))
                objects = SimpleNamespace(get_keyframe_n=Mock(side_effect=AssertionError("CSV identity already resolved")))
                calls = []

                class Scorer:
                    supports_ocr = True

                    def score(self, image, query, caption=None, *, ocr=None):
                        calls.append((image, caption, ocr))
                        return 5

                diagnostics = {}
                with patch("kis.solve_kis.video_image_dir", return_value=self.images):
                    result = solve_kis({"raw_query": "classroom", "ocr_text": "6A", "entities": []},
                        Mock(search=Mock(return_value=visual_rows)), Mock(encode_text=Mock(return_value=[1, 0])),
                        Mock(search_bm25=Mock(return_value=[])), captions, objects, vlm_reranker=Scorer(),
                        ocr_config=self.config, ocr_retriever=self.adapter(), diagnostics=diagnostics,
                        rerank_top_k=4)
                self.assertEqual(diagnostics["ocr"]["status"], "matched")
                self.assertEqual(diagnostics["ocr"]["hits"], 1)
                self.assertNotIn("error", diagnostics["ocr"])
                self.assertIn("+ocr_rank", diagnostics["cross_fusion"])
                by_frame = {row["frame_id"]: row for row in result}
                self.assertEqual(set(by_frame), set(frames))
                self.assertIsNotNone(by_frame[101]["pre_ocr_score"])
                self.assertEqual(by_frame[101]["keyframe_n"], 1)
                self.assertEqual(by_frame[101]["image_path"], str(self.images / "0001.JPEG"))
                self.assertEqual(by_frame[101]["ocr"]["text"], "LỚP 6A x=1")
                self.assertTrue(all(row["vlm_status"] == "reranked" and "retrieval_text" not in row for row in result))
                self.assertEqual(len(calls), len(frames))
                self.assertTrue(all(caption == "" for _, caption, _ in calls))
                self.assertEqual(calls[0][0], self.images / "0001.JPEG")
                self.assertEqual(calls[0][2]["text"], "LỚP 6A x=1")
                self.assertEqual(calls[0][2]["keyframe_n"], 1)
                if 999 in by_frame:
                    self.assertEqual(by_frame[999]["keyframe_n"], 4)
                    self.assertEqual(by_frame[999]["image_path"], str(self.images / "0004.JPEG"))
                    self.assertEqual(by_frame[999]["ocr"]["status"], "missing")
                    self.assertFalse(by_frame[999]["ocr"]["absence_is_negative"])
                self.assertEqual(visual_rows, original_rows)
                objects.get_keyframe_n.assert_not_called()

    def run_public(self, adapter, scorer=None, query=None, mode="retrieve", visual_keys=None, metadata_updates=None):
        frames = [(1, 101), (2, 205), (3, 308), (4, 999)]
        rows = [{"video_id": "L04_V001", "frame_id": frame, "keyframe_n": n, "score": .8,
                 "retrieval_text": "original caption", "image_path": f"data/L04/L04_V001/{n:04d}.JPEG"}
                for n, frame in frames]
        metadata = {(r["video_id"], r["frame_id"]): r for r in rows}
        for frame, updates in (metadata_updates or {}).items():
            key = ("L04_V001", frame)
            metadata[key] = dict(metadata[key], **updates)
        visual_rows = rows if visual_keys is None else [r for r in rows if r["keyframe_n"] in visual_keys]
        v = Mock(search=Mock(return_value=visual_rows))
        c = Mock(search=Mock(return_value=[]), get_caption=lambda vid, frame: metadata.get((vid, frame)))
        image_source = SimpleNamespace(open_candidate=lambda *_: nullcontext(self.images / "0001.JPEG"))
        diag = {}
        result = solve_kis(query or {"raw_query": "a classroom", "entities": []}, v,
            Mock(encode_text=Mock(return_value=[1, 0])), Mock(search_bm25=Mock(return_value=[])), c, None,
            vlm_reranker=scorer, ocr_retriever=adapter, ocr_config=OCRConfig(mode=mode),
            image_source=image_source, diagnostics=diag, rerank_top_k=4)
        return result, diag

    def test_all_candidate_lookup_before_direct_qwen_without_literal_hit(self):
        calls = []

        class Scorer:
            supports_ocr = True

            def score(self, image, query, caption=None, *, ocr=None):
                calls.append((caption, ocr))
                return 5

        result, diag = self.run_public(self.adapter(), Scorer())
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(ocr is None for _, ocr in calls))
        self.assertEqual({r['ocr']['status'] for r in result}, {"ok", "empty", "error", "missing"})
        self.assertTrue(all(caption == "original caption" for caption, _ in calls))
        self.assertTrue(all(not r['ocr']['verified'] for r in result))
        self.assertEqual(diag["ocr"]["source_format"], "public_json")
        self.assertEqual(diag["ocr"]["hits"], 0)
        self.assertTrue(all(r["retrieval_text"] == "original caption" for r in result))

    def test_retrieval_independent_of_vlm_and_off_is_exact_legacy_three_args(self):
        query = {"raw_query": "a classroom", "ocr_text": "6A", "entities": []}
        result, diag = self.run_public(self.adapter(), query=query, visual_keys={2, 3, 4})
        self.assertEqual(diag["ocr"]["hits"], 1)
        self.assertEqual(diag["ocr"]["candidate_union_added"], 1)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["retrieval_text"], "original caption")
        self.assertEqual(result[0]["ocr"]["text"], "LỚP 6A x=1")
        self.assertEqual(result[0]["evidence"]["ocr"]["source"], "public_ocr")
        self.assertEqual(result[0]["ocr"]["matched_query"], "6A")
        old = SimpleNamespace(score=Mock(return_value=5))
        result, _ = self.run_public(self.adapter(), old, mode="off")
        self.assertEqual(old.score.call_count, 4)
        self.assertEqual(old.score.call_args.kwargs, {"caption": "original caption"})
        self.assertTrue(all("ocr" not in r for r in result))

    def test_lazy_forwarding_capability_and_inner_typeerror_not_swallowed(self):
        from kis.web_app import LazyVLMReranker

        inner = SimpleNamespace(supports_ocr=True, score=Mock(return_value=5))
        lazy = LazyVLMReranker()
        lazy.model = inner
        query = {'raw_query': 'a classroom', 'ocr_text': '6A'}
        self.run_public(self.adapter(), lazy, query=query)
        self.assertEqual(inner.score.call_count, 4)
        self.assertEqual(sum('ocr' in c.kwargs for c in inner.score.call_args_list), 1)
        inner.score.side_effect = TypeError("inner implementation failure")
        result, _ = self.run_public(self.adapter(), lazy, query=query)
        self.assertTrue(all(r["vlm_error"] == "inner implementation failure" for r in result))
        self.assertEqual(inner.score.call_count, 8)  # No compatibility retry.
        lazy.model = SimpleNamespace(score=Mock(return_value=5))
        result, _ = self.run_public(self.adapter(), lazy, query=query)
        self.assertEqual(sum('does not support' in r.get('vlm_error', '') for r in result), 1)
        self.assertEqual(lazy.model.score.call_count, 3)

    def test_unsupported_direct_scorer_is_explicit_and_does_not_remove_candidates(self):
        old = SimpleNamespace(score=Mock(return_value=5))
        result, _ = self.run_public(self.adapter(), old, query={'raw_query': 'classroom', 'ocr_text': '6A'})
        self.assertEqual(len(result), 4)
        self.assertEqual(sum('does not support' in r.get('vlm_error', '') for r in result), 1)
        self.assertEqual(old.score.call_count, 3)

    def test_public_diagnostics_does_not_feed_qwen_or_change_legacy_results(self):
        old = SimpleNamespace(score=Mock(return_value=5))
        result, diag = self.run_public(self.adapter(), old, mode="diagnostics",
            query={"raw_query": "classroom", "ocr_text": "6A", "entities": []})
        self.assertEqual(diag["ocr"]["hits"], 1)
        self.assertEqual(old.score.call_count, 4)
        self.assertTrue(all("ocr" not in r for r in result))
        self.assertEqual(old.score.call_args.kwargs, {"caption": "original caption"})

    def test_stale_assets_keep_candidates_but_no_stale_ocr(self):
        adapter = self.adapter()
        adapter.search("6A", self.config)
        self.rows[1]["ocr_text"] = self.rows[1]["retrieval_text"] = "changed"
        self.write(1)
        result, diag = self.run_public(adapter)
        self.assertEqual(len(result), 4)
        self.assertEqual(diag["ocr"]["status"], "error")
        self.assertTrue(all(r["ocr"]["status"] == "error" and not r["ocr"]["text"] for r in result))

    def test_source_change_between_fts_and_lookup_rolls_back_ocr_boost_and_union(self):
        adapter = self.adapter()
        original_attach = adapter.attach_candidates

        def changed_before_lookup(candidates, config):
            self.rows[1]["ocr_text"] = self.rows[1]["retrieval_text"] = "changed"
            self.write(1)
            original_attach(candidates, config)

        query = {"raw_query": "classroom", "ocr_text": "6A", "entities": []}
        baseline, _ = self.run_public(adapter, mode="off", query=query, visual_keys={2, 3, 4})
        with patch.object(adapter, "attach_candidates", side_effect=changed_before_lookup):
            result, diag = self.run_public(adapter, query=query, visual_keys={2, 3, 4})
        self.assertEqual([(r["frame_id"], r["final_score"]) for r in result],
                         [(r["frame_id"], r["final_score"]) for r in baseline])
        self.assertEqual(diag["ocr"]["status"], "error")
        self.assertEqual(diag["ocr"]["candidate_union_added"], 0)
        self.assertTrue(all(r["retrieval_text"] == "original caption" for r in result))

    def test_hydrated_identity_without_ocr_file_rolls_back_both_union_and_boost(self):
        query = {"raw_query": "classroom", "ocr_text": "6A", "entities": []}
        updates = {101: {"keyframe_n": 4, "image_path": "data/L04/L04_V001/0004.JPEG"}}
        for visual_keys in ({2, 3, 4}, {1, 2, 3, 4}):
            with self.subTest(visual_keys=visual_keys):
                adapter = self.adapter()
                baseline, _ = self.run_public(adapter, query=query, mode="off", visual_keys=visual_keys,
                                              metadata_updates=updates)
                result, diag = self.run_public(adapter, query=query, visual_keys=visual_keys, metadata_updates=updates)
                self.assertEqual([(r["frame_id"], r["final_score"]) for r in result],
                                 [(r["frame_id"], r["final_score"]) for r in baseline])
                self.assertEqual(diag["ocr"]["hits"], 1)
                self.assertEqual(diag["ocr"]["status"], "error")
                self.assertIn("candidate/mapping", diag["ocr"]["error"])
                self.assertEqual(diag["ocr"]["candidate_union_added"], 0)
                self.assertNotIn("+ocr_rank", diag["cross_fusion"])
                self.assertTrue(all(r["ocr"]["status"] == "error" for r in result))
                self.assertTrue(all("pre_ocr_score" not in r and "ocr" not in r.get("evidence", {}) for r in result))

    def test_legacy_flat_caption_path_keeps_valid_ocr_union_and_original_caption(self):
        result, diag = self.run_public(self.adapter(), visual_keys={2, 3, 4},
            query={"raw_query": "classroom", "ocr_text": "6A", "entities": []},
            metadata_updates={101: {"image_path": "data/L04_V001/00001.jPeG"}})
        self.assertEqual(diag["ocr"]["status"], "matched")
        self.assertEqual(diag["ocr"]["candidate_union_added"], 1)
        self.assertEqual(result[0]["frame_id"], 101)
        self.assertEqual(result[0]["image_path"], "data/L04_V001/00001.jPeG")
        self.assertEqual(result[0]["retrieval_text"], "original caption")
        self.assertEqual(result[0]["ocr"]["status"], "ok")
        self.assertFalse(result[0]["ocr"]["verified"])


if __name__ == "__main__":
    unittest.main()
