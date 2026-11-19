"""Offline SYNTHETIC safety tests, never benchmark accuracy measurements."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy  # Windows shared runtime must precede any Torch dependency.

from tools import l04_e2e_ocr_ablation as exp


def documents():
    return [{"video_id": "L04_V001", "frame_id": n * 137, "keyframe_n": n,
             "caption_text": "a display watch", "retrieval_text": "a display watch",
             "ocr_status": "ok", "ocr_text": f"SCHOOL token{n}", "image_path": f"data/L04/L04_V001/{n:03}.jpg",
             "image_sha256": "a" * 64, "score": 1 - n / 100} for n in range(1, 21)]


class SyntheticCaptions:
    def __init__(self):
        self.calls = Counter()
        self.docs = documents()

    def search_bm25(self, query, top_k):
        self.calls["bm25"] += 1
        return deepcopy(self.docs[:top_k])

    def search(self, query, top_k):
        self.calls["semantic"] += 1
        return deepcopy(self.docs[:top_k])

    def get_caption(self, video_id, frame_id):
        return next(d for d in self.docs if d["frame_id"] == frame_id)


class SyntheticImages:
    @contextmanager
    def open_candidate(self, candidate, lookup):
        yield Path("synthetic.jpg")


class ExperimentTests(unittest.TestCase):
    def query(self):
        return {"query_id": "synthetic", "raw_query": 'show "SCHOOL"', "query_variants": ["watch"],
                "stream_queries": {s: ["watch"] for s in ("visual", "bm25", "semantic")},
                "entities": [], "needs_ocr": True, "temporal_constraints": []}

    def providers(self):
        self.visual = Mock(search=Mock(return_value=documents()))
        self.encoder = exp.CachedVisualEncoder({"watch": "synthetic-vector"})
        self.captions = SyntheticCaptions()
        return (self.visual, self.encoder, self.captions, Mock(), SyntheticImages(), exp.PublicOCR(documents()))

    def test_real_solve_kis_di_called_all_providers_and_explicit_profile(self):
        from kis.solve_kis import solve_kis
        providers = self.providers()
        scorer = Mock(score=Mock(return_value=7), spec=["score", "arm", "query_id"])
        query = self.query()
        original = deepcopy(query)
        with patch("kis.solve_kis.solve_kis", wraps=solve_kis) as call:
            a = exp.run_arm(query, "A", providers, scorer)
            b = exp.run_arm(query, "B", providers, scorer)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(self.query(), original)
        self.assertEqual(query, original)
        self.assertEqual(scorer.score.call_count, 40)
        self.assertEqual(a["structured_sha256"], b["structured_sha256"])
        self.assertEqual(a["diagnostics"]["cross_fusion"], "rrf")
        self.assertIn("rrf", b["diagnostics"]["cross_fusion"])
        self.assertEqual(a["diagnostics"]["ocr"]["status"], "disabled")
        self.assertGreater(providers[-1].calls, 0)
        self.assertEqual(self.captions.calls, Counter(bm25=2, semantic=2))
        self.assertEqual(self.visual.search.call_count, 2)
        self.assertEqual(self.encoder.calls, 2)

    def test_solver_swallowed_vlm_errors_are_fatal(self):
        for score in (None, float("nan")):
            with self.subTest(score=score), self.assertRaises(exp.Blocked):
                exp.run_arm(self.query(), "A", self.providers(), Mock(score=Mock(return_value=score), spec=["score", "arm", "query_id"]))
        with self.assertRaises(exp.Blocked):
            exp.run_arm(self.query(), "A", self.providers(), Mock(score=Mock(side_effect=RuntimeError("invalid")), spec=["score", "arm", "query_id"]))

    def test_valid_zero_scores_demoted_outside_final100_are_not_failures(self):
        # SYNTHETIC: opposite stream ranks keep all 120 retrieval scores above
        # the twenty valid zero-score reranks. Two routed variants supply each
        # stream's interleaved halves; the real solver still truncates at100.
        docs = [dict(documents()[0], frame_id=n * 137, keyframe_n=n,
                     image_path=f"data/L04/L04_V001/{n:03}.jpg") for n in range(1, 121)]

        def hits(ordered, query):
            selected = ordered[0 if query == "first" else 1::2]
            return [dict(d, score=100 - i) for i, d in enumerate(selected)]

        captions = SyntheticCaptions()
        captions.docs = docs

        def text_hits(query, top_k, channel):
            captions.calls[channel] += 1
            return hits(list(reversed(docs)), query)[:top_k]

        captions.search_bm25 = lambda query, top_k: text_hits(query, top_k, "bm25")
        captions.search = lambda query, top_k: text_hits(query, top_k, "semantic")
        providers = (Mock(search=Mock(side_effect=lambda q, top_k: hits(docs, q)[:top_k])),
                     exp.CachedVisualEncoder({"first": "first", "second": "second"}),
                     captions, Mock(), SyntheticImages(), Mock())
        query = self.query()
        query["stream_queries"] = {s: ["first", "second"] for s in ("visual", "bm25", "semantic")}
        scorer = Mock(score=Mock(return_value=0), spec=["score", "arm", "query_id"])
        stage = exp.run_arm(query, "A", providers, scorer)
        expected = {(d["video_id"], d["frame_id"]) for d in stage["diagnostics"]["rerank_input"]}
        returned = {(d["video_id"], d["frame_id"]) for d in stage["results"]}
        self.assertEqual(len(stage["diagnostics"]["pre_selection_candidates"]), 120)
        self.assertEqual(len(stage["results"]), 100)
        self.assertEqual(len(expected), 20)
        self.assertTrue(expected.isdisjoint(returned))
        self.assertEqual(scorer.score.call_count, 20)

    def test_missing_image_is_fatal_not_negative(self):
        providers = list(self.providers())
        providers[4] = Mock(open_candidate=Mock(side_effect=FileNotFoundError("synthetic missing")))
        with self.assertRaises(exp.Blocked):
            exp.run_arm(self.query(), "A", providers, Mock(score=Mock(return_value=5), spec=["score", "arm", "query_id"]))

    def test_mandatory_retrieval_provider_unavailable(self):
        for index, method in ((0, "search"), (1, "encode_text"), (2, "search_bm25"), (2, "search")):
            providers = list(self.providers())
            with patch.object(providers[index], method, side_effect=exp.Blocked("unavailable")), self.assertRaises(exp.Blocked):
                exp.run_arm(self.query(), "A", providers, Mock(score=Mock(return_value=5), spec=["score", "arm", "query_id"]))

    def test_temporal_planner_output_blocked_not_rewritten(self):
        query = self.query()
        query["temporal_constraints"] = [{"type": "sequence", "order": ["x", "y"]}]
        with self.assertRaises(exp.Blocked):
            exp.run_arm(query, "A", self.providers(), Mock())
        self.assertTrue(query["temporal_constraints"])

    def test_ocr_missing_error_not_negative_and_unverified(self):
        for status in ("missing", "error"):
            docs = documents()
            docs[0]["ocr_status"] = status
            with self.assertRaises(exp.Blocked):
                exp.PublicOCR(docs)
        from kis.ocr import OCRConfig
        hits = exp.PublicOCR(documents()).search("SCHOOL", OCRConfig())
        self.assertTrue(hits)
        self.assertTrue(all(d["ocr"]["verified"] is False for d in hits))

    def test_ocr_failure_swallowed_by_solver_fatal(self):
        providers = list(self.providers())
        providers[-1] = Mock(search=Mock(side_effect=RuntimeError("OCR unavailable")))
        with self.assertRaises(exp.Blocked):
            exp.run_arm(self.query(), "B", providers, Mock(score=Mock(return_value=5), spec=["score", "arm", "query_id"]))

    def test_strict_scores_not_default_zero(self):
        for n in range(11):
            self.assertEqual(exp.strict_score(str(n)), n)
        for value in (None, True, 5, "", "!!!", "0.0", "11", "-1", "{\"score\":0}", "5\n6", "NaN"):
            with self.subTest(value=value), self.assertRaises(exp.Blocked):
                exp.strict_score(value)

    def test_prompt_same_caption_no_qrel_identity_or_mutation(self):
        d = documents()[0]
        before = deepcopy(d)
        a = exp.scoring_prompt("watch", d, "A")
        b = exp.scoring_prompt("watch", d, "B")
        self.assertTrue(b.startswith(a))
        self.assertNotIn("OCR_JSON", a)
        self.assertIn('"verified": false', b)
        self.assertNotIn(d["video_id"], b)
        self.assertNotIn(d["image_sha256"], b)
        self.assertNotIn(d["image_path"], b)
        self.assertEqual(d, before)
        self.assertIn("Return ONLY a single integer from 0 to 10", a)

    def test_selection_actual_kind_smallest_ids_not_assigned_kind(self):
        ids = [f"t{i:02}" for i in range(1, 19)]
        queries = [{"id": q, "kind": ("literal", "scene", "mixed")[i % 3], "assigned_kind": "scene"}
                   for i, q in enumerate(ids)]
        eligible = [q for q in ids if q not in {"t06", "t10"}]
        packet = {"queries": queries, "authoring": {"correction": {"subset": {"query_ids": eligible}}}}
        selected = exp.select_queries(packet, {"queries": deepcopy(queries)})
        self.assertEqual([q["id"] for q in selected], ["t01", "t02", "t03", "t04", "t05", "t09"])
        altered = deepcopy(queries)
        altered[0]["text"] = "altered"
        with self.assertRaises(exp.Blocked):
            exp.select_queries(packet, {"queries": altered})

    def test_readonly_allowlist_no_production_or_l05_server_query(self):
        raw = Mock()
        client = exp.ReadOnlyClient(raw, "kis_l04_l05_stg_20261108_02")
        for method in ("load_collection", "create_collection", "drop_collection", "insert", "upsert", "create_index", "alter_alias"):
            with self.assertRaises(exp.Blocked):
                getattr(client, method)
        for method in ("query", "search"):
            for filt in (None, 'video_id like "L04_%"', 'video_id == "L05_V001"'):
                with self.assertRaises(exp.Blocked):
                    getattr(client, method)(collection_name=client.collection, filter=filt)
        with self.assertRaises(exp.Blocked):
            client.describe_collection(collection_name="clip_keyframes")
        raw.assert_not_called()
        self.assertEqual(raw.method_calls, [])

    def test_real_visual_search_method_server_filter(self):
        raw = Mock(search=Mock(return_value=[[{"entity": {"video_id": "L04_V001", "frame_id": 137}, "distance": 0.7}]]))
        client = exp.ReadOnlyClient(raw, "kis_l04_l05_stg_20261108_02")
        retriever = exp.visual_provider(client)
        rows = retriever.search(numpy.ones(768), top_k=20)
        self.assertEqual(rows[0]["frame_id"], 137)
        kwargs = raw.search.call_args.kwargs
        self.assertEqual(kwargs["filter"], exp.FILTER)
        self.assertNotIn("L05", kwargs["filter"])
        self.assertEqual(kwargs["search_params"]["params"]["nprobe"], 128)

    def test_not_loaded_or_partial_index_block_without_query_load(self):
        for state, count in (("NotLoad", 14918), ("Loaded", 0)):
            raw = Mock()
            raw.describe_collection.return_value = {
                "description": "kis_l04_l05_local_milvus_staging_v1;owner=" + "a" * 32,
                "fields": [{"name": n, "params": {"dim": 768}} for n in ("id", "video_id", "frame_id", "embedding")]}
            raw.get_load_state.return_value = {"state": state}
            raw.describe_index.return_value = {"state": "Finished", "metric_type": "IP", "index_type": "IVF_FLAT",
                                               "field_name": "embedding", "total_rows": 14918,
                                               "indexed_rows": count, "pending_index_rows": 0}
            client = exp.ReadOnlyClient(raw, "kis_l04_l05_stg_20261108_02")
            with self.assertRaises(exp.Blocked):
                exp.readiness(client)
            raw.query.assert_not_called()
            self.assertNotIn("load_collection", [c[0] for c in raw.method_calls])

    def test_mapping_is_frame_idx_not_n_minus_one(self):
        record = {"video_id": "L04_V001", "keyframe_n": 1, "frame_id": 137}
        self.assertEqual(exp.adapter().identity(record, "L04_V001", {1: 137}), ("L04_V001", 1))
        with self.assertRaises(ValueError):
            exp.adapter().identity(dict(record, frame_id=0), "L04_V001", {1: 137})
        for path in ("dataset/data/L05/L05_V001/001.jpg", "dataset/data/L06/L06_V001/001.jpg"):
            with self.assertRaises(ValueError):
                exp.adapter().checked_path(path)

    def test_partial_comparison_forbids_metrics(self):
        with self.assertRaises(exp.Blocked):
            exp.evaluate([{"id": "synthetic"}], {"synthetic": {"A": {}}})

    def test_planner_one_request_cache_replay_and_no_secret_report(self):
        from kis.llm import QueryAPIConfig
        from kis.structure_query import QUERY_RESPONSE_EXAMPLE
        config = QueryAPIConfig("groq", "openai", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1",
                                api_key="SYNTHETIC_SECRET_NEVER_REPORT", response_format="json_schema", temperature=0)
        generator = Mock(generate=Mock(return_value=deepcopy(QUERY_RESPONSE_EXAMPLE)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exp.save(output / "freeze.json", {"synthetic": True})
            queries = [{"id": "synthetic", "text": "red shirt person bicycle"}]
            with patch("kis.llm.load_query_api_config", return_value=config), patch("kis.structure_query.create_query_structurer", return_value=generator):
                a = exp.planner_cache(output, queries)
                b = exp.planner_cache(output, queries)
            generator.generate.assert_called_once_with(queries[0]["text"])
            self.assertEqual(a, b)
            self.assertNotIn(config.api_key, (output / "plannercache.json").read_text())
            self.assertEqual(a["synthetic"]["requests"], 1)

    def test_planner_failed_call_intent_forbids_retry(self):
        from kis.llm import QueryAPIConfig
        config = QueryAPIConfig("groq", "openai", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1",
                                api_key="SYNTHETIC_SECRET", response_format="json_schema", temperature=0)
        generator = Mock(generate=Mock(side_effect=RuntimeError("synthetic transport failure")))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exp.save(output / "freeze.json", {"synthetic": True})
            with patch("kis.llm.load_query_api_config", return_value=config), patch("kis.structure_query.create_query_structurer", return_value=generator):
                with self.assertRaises(RuntimeError):
                    exp.planner_cache(output, [{"id": "synthetic", "text": "watch"}])
                with self.assertRaises(exp.Blocked):
                    exp.planner_cache(output, [{"id": "synthetic", "text": "watch"}])
            self.assertEqual(generator.generate.call_count, 1)

    def test_checkpoint_invalid_score_or_context_blocks_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exp.save(output / "checkpoint.json", {"x": {"context": {}, "key": "x", "status": "scored", "raw": "!!!", "score": 0}})
            with patch.object(exp, "gpu_gate", side_effect=AssertionError("must not reach GPU")), self.assertRaises(exp.Blocked):
                exp.QwenScorer(output, {}, documents())


if __name__ == "__main__":
    unittest.main()
