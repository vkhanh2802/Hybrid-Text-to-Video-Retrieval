import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import Mock, patch

from kis.solve_kis import solve_kis, rerank_candidates_with_vlm
from kis.paths import KIS_ROOT


class FakeEncoder:
    def encode_text(self, query):
        return query


class FakeVisualRetriever:
    def search(self, vector, top_k):
        return [
            {"video_id": "V1", "frame_id": 1, "score": 0.9},
            {"video_id": "V2", "frame_id": 2, "score": 0.8},
        ][:top_k]


class FakeBm25Retriever:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def search_bm25(self, query, top_k):
        return [
            {
                "video_id": "V2",
                "frame_id": 2,
                "score": 3.0,
                "retrieval_text": "second",
                "image_path": str(self.image_paths[1]),
            },
            {
                "video_id": "V1",
                "frame_id": 1,
                "score": 2.0,
                "retrieval_text": "first",
                "image_path": str(self.image_paths[0]),
            },
        ][:top_k]


class FakeSemanticRetriever:
    def search(self, query, top_k):
        return [
            {"video_id": "V1", "frame_id": 1, "score": 0.8},
            {"video_id": "V2", "frame_id": 2, "score": 0.7},
        ][:top_k]


class FakeObjectLookup:
    def get_keyframe_n(self, video_id, frame_id):
        return None


class FakeVlmReranker:
    def score(self, image_path, query, caption=None):
        return 10 if image_path.name == "002.jpg" else 0


class SolveKisIntegrationTest(unittest.TestCase):
    def test_async_ocr_join_after_streams_and_same_ranking(self):
        from kis.ocr import OCRConfig
        diagnostics = {}
        events = []
        class Ready:
            def result(self):
                events.append('ocr_join')
                return [], {'status': 'no_match', 'seconds': .01, 'queries': [], 'hits': 0}
        original = FakeSemanticRetriever.search
        def semantic(instance, *args, **kwargs):
            events.append('semantic')
            return original(instance, *args, **kwargs)
        with patch.object(FakeSemanticRetriever, 'search', semantic):
            result = self.run_pipeline(ocr_future=Ready(), diagnostics=diagnostics,
                                       ocr_config=OCRConfig(mode='retrieve'))
        self.assertLess(events.index('semantic'), events.index('ocr_join'))
        self.assertEqual(diagnostics['ocr']['execution'], 'overlapped_with_frame_retrieval')
        self.assertEqual([(r['video_id'], r['frame_id']) for r in result],
                         [(r['video_id'], r['frame_id']) for r in self.run_pipeline()])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image_paths = [Path(directory.name) / f"{n:03d}.jpg" for n in (1, 2)]
        for path in self.image_paths:
            path.write_bytes(b"image fixture")

    def run_pipeline(self, vlm_reranker=None, **kwargs):
        return solve_kis(
            structured_query={
                "raw_query": "test query",
                "query_variants": ["test variant"],
                "entities": [],
            },
            visual_retriever=FakeVisualRetriever(),
            visual_encoder=FakeEncoder(),
            text_bm25_retriever=FakeBm25Retriever(self.image_paths),
            text_semantic_retriever=FakeSemanticRetriever(),
            object_lookup=FakeObjectLookup(),
            vlm_reranker=vlm_reranker,
            retrieval_top_k=2,
            rerank_top_k=2,
            result_top_k=2,
            **kwargs,
        )

    def test_text_reranker_runs_before_vlm_and_survives_disabled_vlm(self):
        from kis.text_reranker import TextRerankerConfig

        diagnostics = {}
        reranker = Mock(score_pairs=Mock(return_value=[-3, 2]))
        results = self.run_pipeline(text_reranker_config=TextRerankerConfig(mode="rerank"),
                                    text_reranker=reranker, diagnostics=diagnostics)
        self.assertEqual([item["frame_id"] for item in results], [2, 1])
        self.assertEqual(diagnostics["rerank_input"][0]["frame_id"], 2)
        self.assertEqual(diagnostics["text_reranker"]["status"], "applied")
        self.assertIn("text_reranker", diagnostics["stage_seconds"])
        self.assertLess(results[0]["score"], results[1]["score"])
        self.assertGreater(results[0]["final_score"], results[1]["final_score"])

    def test_text_failure_keeps_original_pipeline_and_vlm_still_works(self):
        from kis.text_reranker import TextRerankerConfig

        config = TextRerankerConfig(mode="rerank")
        failed = Mock(score_pairs=Mock(side_effect=RuntimeError("not cached")))
        baseline = self.run_pipeline()
        result = self.run_pipeline(text_reranker_config=config, text_reranker=failed)
        self.assertEqual([(i["frame_id"], i["final_score"]) for i in result],
                         [(i["frame_id"], i["final_score"]) for i in baseline])
        with_vlm = self.run_pipeline(FakeVlmReranker(), text_reranker_config=config, text_reranker=failed)
        self.assertEqual(with_vlm[0]["frame_id"], 2)

    def test_retrieval_ranking_without_vlm(self):
        results = self.run_pipeline()

        self.assertEqual([item["frame_id"] for item in results], [1, 2])
        self.assertTrue(
            all(item["final_score"] == item["score"] for item in results)
        )

    def test_vlm_reranks_and_preserves_metadata(self):
        results = self.run_pipeline(FakeVlmReranker())

        self.assertEqual([item["frame_id"] for item in results], [2, 1])
        self.assertEqual(results[0]["retrieval_text"], "second")
        self.assertTrue(all("final_score" in item for item in results))

    def test_original_video_is_not_passed_to_reranker(self):
        directory = self.image_paths[0].parent
        self.image_paths[0].unlink()
        video = directory / "001.mp4"
        video.write_bytes(b"original video")
        candidate = {"video_id": "L15_V001", "frame_id": 10, "keyframe_n": 1,
                     "image_path": str(video), "score": 0.8}
        reranker = Mock()
        with patch("kis.solve_kis.video_image_dir", return_value=directory):
            results = rerank_candidates_with_vlm([candidate], "query", reranker, FakeObjectLookup())
        self.assertEqual(results[0]["vlm_status"], "missing_image")
        reranker.score.assert_not_called()
        self.assertEqual(video.read_bytes(), b"original video")

    def test_one_hundred_candidates_score_only_first_twenty_without_backfill(self):
        candidates = [{"video_id": "L01_V001", "frame_id": i, "score": 1 - i / 101} for i in range(100)]
        reranker = Mock(score=Mock(return_value=8))
        def resolve(candidate, lookup):
            return None if candidate["frame_id"] == 0 else Path(f"{candidate['frame_id']}.jpg")
        with patch("kis.solve_kis._resolve_candidate_image_path", side_effect=resolve):
            results = rerank_candidates_with_vlm(candidates, "query", reranker, None, top_k=20)
        self.assertEqual(len(results), 100)
        self.assertEqual(reranker.score.call_count, 19)
        self.assertEqual({int(call.args[0].stem) for call in reranker.score.call_args_list}, set(range(1, 20)))
        by_frame = {item["frame_id"]: item for item in results}
        self.assertEqual(by_frame[0]["vlm_status"], "missing_image")
        self.assertTrue(all(by_frame[i]["vlm_status"] == "unscored" for i in range(20, 100)))
        self.assertTrue(all(by_frame[i]["final_score"] == by_frame[i]["score"] for i in range(20, 100)))


class NotebookTest(unittest.TestCase):
    def test_code_cells_compile(self):
        notebook_path = KIS_ROOT / "notebooks" / "runbranch.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        for index, cell in enumerate(code_cells):
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")

        self.assertGreater(len(code_cells), 0)


if __name__ == "__main__":
    unittest.main()
