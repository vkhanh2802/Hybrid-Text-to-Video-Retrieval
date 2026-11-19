import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kis.ranking import finalize_vlm_ranking
from kis.text_reranker import (LazyCrossEncoder, TextRerankerConfig, ranking_query,
                               rerank_text_candidates, text_chunks)
from tools.check_text_reranker import check_cache


def candidate(frame, score, text=None):
    return {"video_id": "L01_V001", "frame_id": frame, "score": score, "retrieval_text": text}


def segment(frame, score, texts, partial=False):
    events = [dict(candidate(frame+i, score, text), action=f"event {i}", status="retrieved", pts_time=i+1)
              for i, text in enumerate(texts)]
    if partial:
        events[-1].update(status="unknown", frame_id=None, pts_time=None)
    return dict(candidate(frame, score), temporal={"events": events, "verified": False,
        "coverage": (len(events)-int(partial))/len(events), "score": score})


class TextRerankerTest(unittest.TestCase):
    def setUp(self):
        self.config = TextRerankerConfig(mode="rerank")
        self.query = {"raw_query": "nguồn gốc", "stream_queries": {"visual": ["faithful English query", "short"]}}

    def test_disabled_is_exact_noop_and_does_not_construct_model(self):
        values = [candidate(1, .9, "caption")]
        with patch("kis.text_reranker.LazyCrossEncoder") as factory:
            output, report = rerank_text_candidates(values, self.query, None)
        self.assertIs(output, values)
        self.assertEqual(report["status"], "disabled")
        factory.assert_not_called()

    def test_scored_slots_swap_but_missing_text_keeps_slot_and_original_scores(self):
        values = [candidate(1, .9, "less relevant"), candidate(2, .8), candidate(3, .7, "relevant")]
        original = copy.deepcopy(values)
        scorer = Mock(score_pairs=Mock(return_value=[-5, 2]))
        ranked, report = rerank_text_candidates(values, self.query, None, self.config, scorer)
        self.assertEqual([v["frame_id"] for v in ranked], [3, 2, 1])
        self.assertEqual([v["score"] for v in ranked], [.7, .8, .9])
        self.assertEqual([v.get("text_ranking_score", v["score"]) for v in ranked], [.9, .8, .7])
        self.assertEqual(values, original)
        self.assertEqual(report["candidate_trace"][0]["retrieval_rank"], 3)
        self.assertEqual(report["query_source"], "stream_queries.visual[0]")
        self.assertEqual(ranked[1]["text_rerank_status"], "missing_text")
        self.assertNotIn("text_logit", ranked[1])

    def test_text_ranking_survives_no_vlm_and_neutral_vlm_without_mutating_retrieval(self):
        ranked, _ = rerank_text_candidates([candidate(1, .9, "a"), candidate(2, .8, "b")],
            self.query, None, self.config, Mock(score_pairs=Mock(return_value=[-2, 3])))
        final, applied = finalize_vlm_ranking(ranked, top_k=0, base_score_field="text_ranking_score")
        self.assertFalse(applied)
        self.assertEqual([v["frame_id"] for v in final], [2, 1])
        for item in ranked:
            item["vlm_raw_score"] = 5
        neutral, applied = finalize_vlm_ranking(ranked, top_k=2, base_score_field="text_ranking_score")
        self.assertTrue(applied)
        self.assertEqual([v["frame_id"] for v in neutral], [2, 1])
        self.assertEqual(neutral[0]["score"], .8)

    def test_diagnostics_scores_without_reordering(self):
        values = [candidate(1, .9, "a"), candidate(2, .8, "b")]
        ranked, report = rerank_text_candidates(values, self.query, None,
            replace(self.config, mode="diagnostics"), Mock(score_pairs=Mock(return_value=[-2, 3])))
        self.assertEqual([v["frame_id"] for v in ranked], [1, 2])
        self.assertTrue(all("text_ranking_score" not in v for v in ranked))
        self.assertEqual(report["status"], "diagnostics")

    def test_invalid_scores_are_unknown_not_zero_and_json_remains_finite(self):
        values = [candidate(i, 1-i*.1, "a") for i in range(5)]
        ranked, report = rerank_text_candidates(values, self.query, None, self.config,
            Mock(score_pairs=Mock(return_value=[math.nan, math.inf, True, None, -3])))
        self.assertEqual([v["frame_id"] for v in ranked], list(range(5)))
        self.assertEqual(report["candidates_scored"], 1)
        self.assertEqual([v["text_rerank_status"] for v in ranked[:4]], ["invalid_score"]*4)
        json.dumps({"candidates": ranked, "diagnostics": report}, allow_nan=False)

    def test_failed_batch_and_wrong_output_count_preserve_ranking(self):
        values = [candidate(1, .9, "a"), candidate(2, .8, "b")]
        for result in (RuntimeError("error"), [1]):
            scorer = Mock(score_pairs=Mock(side_effect=result) if isinstance(result, Exception)
                          else Mock(return_value=result))
            ranked, report = rerank_text_candidates(values, self.query, None, self.config, scorer)
            self.assertEqual([v["frame_id"] for v in ranked], [1, 2])
            self.assertEqual(report["batch_errors"], 1)
            self.assertTrue(all(v["text_rerank_status"] == "inference_error" for v in ranked))

    def test_pool_pair_and_batch_limits(self):
        values = [candidate(i, 1-i*.1, "a") for i in range(4)]
        scorer = Mock(score_pairs=Mock(side_effect=lambda pairs: [-i for i in range(len(pairs))]))
        ranked, report = rerank_text_candidates(values, self.query, None,
            replace(self.config, pool_size=3, max_pairs=2, batch_size=1), scorer)
        self.assertEqual(report["pairs_requested"], 2)
        self.assertEqual(scorer.score_pairs.call_count, 2)
        self.assertEqual(ranked[2]["text_rerank_status"], "pair_budget_exhausted")
        self.assertEqual(ranked[3]["text_rerank_status"], "outside_pool")

    def test_chunking_reaches_tail_and_reports_omissions(self):
        text = "a" * 10000 + "TAIL_MARKER"
        chunks, omitted = text_chunks(text, self.config)
        self.assertEqual(len(chunks), 4)
        self.assertIn("TAIL_MARKER", chunks[-1])
        self.assertGreater(omitted, 0)
        self.assertTrue(all(len(c) <= self.config.chunk_chars for c in chunks))

    def test_complete_path_uses_separate_event_queries_and_weakest_event(self):
        values = [segment(1, .9, ["a", "b"]), segment(10, .8, ["c", "d"])]
        original = copy.deepcopy(values)
        scorer = Mock(score_pairs=Mock(return_value=[10, -5, 2, 1]))
        ranked, report = rerank_text_candidates(values, self.query, None, self.config, scorer)
        self.assertEqual([v["frame_id"] for v in ranked], [10, 1])
        self.assertEqual(ranked[0]["text_logit"], 1)
        self.assertEqual([q for q, _ in scorer.score_pairs.call_args.args[0]], ["event 0", "event 1"]*2)
        self.assertFalse(ranked[0]["temporal"]["verified"])
        self.assertEqual(values, original)

    def test_partial_or_missing_event_caption_not_scored_and_no_model_load(self):
        values = [segment(1, .9, ["a", "b"], partial=True), segment(10, .8, ["a", None])]
        with patch("kis.text_reranker.LazyCrossEncoder") as factory:
            ranked, report = rerank_text_candidates(values, self.query, None, self.config)
        factory.assert_not_called()
        self.assertEqual(report["pairs_requested"], 0)
        self.assertEqual(ranked[0]["text_rerank_status"], "partial_temporal_path")
        self.assertEqual(ranked[1]["text_rerank_status"], "missing_text")

    def test_support_count_groups_cannot_exchange_slots(self):
        values = [segment(1, .9, ["a", "b", "c"]), segment(10, .8, ["a", "b"])]
        ranked, _ = rerank_text_candidates(values, self.query, None, self.config,
                                          Mock(score_pairs=Mock(return_value=[-5]*3 + [10]*2)))
        self.assertEqual([v["frame_id"] for v in ranked], [1, 10])

    def test_long_query_truncation_empty_query_and_caption_lookup(self):
        config = replace(self.config, query_chars=128)
        scorer = Mock(score_pairs=Mock(return_value=[2]))
        caption_lookup = SimpleNamespace(get_caption=lambda *args: {"caption": "original caption"})
        _, report = rerank_text_candidates([candidate(1, .9)], {"raw_query": "q"*200},
                                          caption_lookup, config, scorer)
        self.assertTrue(report["query_truncated"])
        self.assertEqual(len(scorer.score_pairs.call_args.args[0][0][0]), 128)
        ranked, report = rerank_text_candidates([candidate(1, .9, "a")], {}, None, config, scorer)
        self.assertEqual(report["status"], "fallback")
        self.assertEqual(ranked[0]["text_rerank_status"], "missing_query")

    def test_model_construction_is_lazy_and_load_failure_cached(self):
        factory = Mock(side_effect=FileNotFoundError("not cached"))
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory),
                               AutoModelForSequenceClassification=Mock())
        with patch.dict("sys.modules", {"transformers": fake}):
            model = LazyCrossEncoder(self.config)
            factory.assert_not_called()
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "text_model_unavailable"):
                    model.score_pairs([("q", "text")])
            factory.assert_called_once()
        self.assertTrue(factory.call_args.kwargs["local_files_only"])
        self.assertFalse(factory.call_args.kwargs["trust_remote_code"])
        self.assertEqual(factory.call_args.kwargs["revision"], self.config.revision)

    def test_cpu_adapter_preserves_raw_negative_logits_and_uses_safetensors(self):
        import torch

        model = Mock()
        model.config.num_labels = 1
        model.to.return_value = model
        model.float.return_value = model
        model.eval.return_value = model
        model.return_value.logits = torch.tensor([[-3.0], [4.0]])
        tokenizer = Mock(return_value={"input_ids": torch.tensor([[1], [2]])})
        token_factory, model_factory = Mock(return_value=tokenizer), Mock(return_value=model)
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=token_factory),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_factory))
        with patch.dict("sys.modules", {"transformers": fake}):
            scorer = LazyCrossEncoder(self.config)
            for _ in range(2):
                self.assertEqual(scorer.score_pairs([("q", "a"), ("q", "b")]), [-3, 4])
        model_factory.assert_called_once()
        self.assertTrue(model_factory.call_args.kwargs["use_safetensors"])
        model.to.assert_called_once_with(device="cpu")
        self.assertEqual(tokenizer.call_args.kwargs["max_length"], 512)

    def test_config_and_query_anchor_validation(self):
        for kwargs in ({"mode": "on"}, {"pool_size": 301}, {"batch_size": True}, {"max_pairs": 0},
                       {"revision": "main"}, {"model": ""}, {"max_length": 513}):
            with self.assertRaises(ValueError):
                TextRerankerConfig(**kwargs)
        self.assertEqual(ranking_query(self.query), ("faithful English query", "stream_queries.visual[0]"))

    def test_cache_check_is_metadata_only_and_pins_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("fixture", encoding="utf-8")
            reader = Mock(return_value=str(path))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(try_to_load_from_cache=reader)}):
                report = check_cache(self.config)
            self.assertTrue(report["cache_ready"])
            self.assertFalse(report["model_loaded"])
            self.assertTrue(all(c.kwargs["revision"] == self.config.revision for c in reader.call_args_list))


if __name__ == "__main__":
    unittest.main()
