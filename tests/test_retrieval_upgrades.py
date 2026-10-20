import copy
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import json
import tempfile
from pathlib import Path

import numpy as np

from kis.text import BM25Retriever, tokenize_caption
from tools.evaluation import known_target_metrics, load_benchmark
from kis.ranking import FrameDiversitySelector, RetrievalPolicy, get_search_profile, queries_for_stream
from kis.ranking import fuse_query_variants, normalize_scores
from kis.solve_kis import rerank_candidates_with_vlm
from kis.config import milvus_model_properties
from kis.visual import SiglipRetriever
from tests.test_solve_kis import FakeObjectLookup, FakeVlmReranker


class RetrievalUpgradeTest(unittest.TestCase):
    def test_milvus_scope_and_ann_budget_are_sent_to_the_real_search_api(self):
        client = Mock()
        client.describe_collection.return_value = {"description": json.dumps(milvus_model_properties())}
        client.search.return_value = [[{"entity": {"video_id": "L19_V001", "frame_id": 1}, "distance": 0.5}]]
        with patch("pymilvus.MilvusClient", return_value=client):
            retriever = SiglipRetriever(video_groups=("L19", "L20"), nprobe=128)
            results = retriever.search(np.ones(768), top_k=3)
        kwargs = client.search.call_args.kwargs
        self.assertEqual(kwargs["filter"], 'video_id like "L19_%" or video_id like "L20_%"')
        self.assertEqual(kwargs["search_params"]["params"]["nprobe"], 128)
        self.assertEqual(kwargs["limit"], 3)
        self.assertEqual(results[0]["video_id"], "L19_V001")

    def test_runtime_profiles_preserve_the_baseline_and_select_the_measured_ann_setting(self):
        legacy, upgraded = get_search_profile("legacy"), get_search_profile("upgraded")
        self.assertEqual((legacy.tokenizer_mode, legacy.nprobe), ("legacy", 10))
        self.assertEqual((upgraded.tokenizer_mode, upgraded.nprobe), ("unicode", 128))
        self.assertEqual(upgraded.policy.variant_fusion, "normalized_max")
        self.assertTrue(get_search_profile("verified40").policy.verified_first)
        with self.assertRaises(ValueError):
            get_search_profile("misspelled")
    def test_benchmark_split_is_video_disjoint_and_qrels_include_anchor(self):
        queries = load_benchmark()["queries"]
        self.assertEqual(len(queries), 24)
        self.assertEqual(sum(query["split"] == "test" for query in queries), 16)
        self.assertTrue(all(query["keyframe_n"] in query["relevant_keyframes"] for query in queries))

    def test_metrics_do_not_accept_unjudged_neighbors_or_another_video(self):
        query = {"video_id": "A", "frame_id": 100, "relevant_keyframes": [5]}
        results = [
            {"video_id": "B", "frame_id": 100, "keyframe_n": 5},
            {"video_id": "A", "frame_id": 99, "keyframe_n": 4},
            {"video_id": "A", "frame_id": 100, "keyframe_n": 5},
        ]
        metrics = known_target_metrics(results, query)
        self.assertEqual(metrics["first_known_target_rank"], 3)
        self.assertAlmostEqual(metrics["mrr_at_20"], 1 / 3)
    def test_punctuation_no_longer_hides_the_discriminating_keyword(self):
        corpus = [
            {"video_id": "A", "frame_id": 1, "retrieval_text": "A boat. action: floating;"},
            {"video_id": "B", "frame_id": 2, "retrieval_text": "A car. action: driving;"},
            {"video_id": "C", "frame_id": 3, "retrieval_text": "A person. action: walking;"},
        ]
        legacy = BM25Retriever(corpus, tokenizer_mode="legacy")
        upgraded = BM25Retriever(corpus, tokenizer_mode="unicode")
        self.assertEqual(legacy.search_bm25("boat"), [])
        self.assertEqual(upgraded.search_bm25("boat")[0]["video_id"], "A")

    def test_unicode_equivalence_preserves_vietnamese_and_negation(self):
        self.assertEqual(tokenize_caption("mu\u0303, not_red"), ["mũ", "not", "red"])

    def test_flat_zero_scores_do_not_create_maximal_evidence(self):
        self.assertEqual([item["norm_score"] for item in normalize_scores([{"score": 0}, {"score": 0}], flat_score=0)], [0, 0])

    def test_routing_is_explicit_and_legacy_queries_remain_available(self):
        query = {"query_variants": ["Vietnamese", "English"], "stream_queries": {"bm25": ["English"]}}
        self.assertEqual(queries_for_stream(query, "bm25", RetrievalPolicy()), ["Vietnamese", "English"])
        self.assertEqual(queries_for_stream(query, "bm25", RetrievalPolicy(route_queries=True)), ["English"])

    def test_rank_fusion_is_invariant_to_bm25_query_scale(self):
        lists = [[{"video_id": "A", "frame_id": 1, "score": 10.0}], [{"video_id": "B", "frame_id": 2, "score": 1000.0}]]
        original = copy.deepcopy(lists)
        result = fuse_query_variants(lists, RetrievalPolicy(variant_fusion="rank_max"))
        self.assertEqual(result[0]["score"], result[1]["score"])
        self.assertEqual(lists, original)

    def test_diversity_requires_both_same_video_and_time_proximity(self):
        selector = FrameDiversitySelector()
        candidates = [
            {"video_id": "A", "frame_id": 1, "score": 0.9},
            {"video_id": "A", "frame_id": 2, "score": 0.8},
            {"video_id": "A", "frame_id": 3, "score": 0.7},
            {"video_id": "B", "frame_id": 4, "score": 0.6},
        ]
        descriptions = {1: (0, np.array([1, 0])), 2: (3, np.array([1, 0])), 3: (40, np.array([1, 0])), 4: (0, np.array([1, 0]))}
        with patch.object(selector, "describe", side_effect=lambda item: descriptions[item["frame_id"]]):
            results = selector.select(candidates)
        self.assertEqual([item["frame_id"] for item in results], [1, 3, 4])
        self.assertEqual(results[0]["suppressed_neighbors"][0]["frame_id"], 2)

    def test_verified_window_does_not_mix_unscored_scores_into_top_results(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "001.jpg"
            image.write_bytes(b"image fixture")
            candidates = [
                {"video_id": "V1", "frame_id": 1, "score": 0.8, "image_path": str(image)},
                {"video_id": "V2", "frame_id": 2, "score": 0.6},
            ]
            results = rerank_candidates_with_vlm(candidates, "query", FakeVlmReranker(), FakeObjectLookup(), top_k=1, verified_first=True)
        self.assertEqual(results[0]["frame_id"], 1)
        self.assertEqual(results[0]["vlm_status"], "reranked")
        self.assertEqual(results[1]["vlm_status"], "unscored")


if __name__ == "__main__":
    unittest.main()
