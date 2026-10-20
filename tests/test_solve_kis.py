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
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image_paths = [Path(directory.name) / f"{n:03d}.jpg" for n in (1, 2)]
        for path in self.image_paths:
            path.write_bytes(b"image fixture")

    def run_pipeline(self, vlm_reranker=None):
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
        )

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
