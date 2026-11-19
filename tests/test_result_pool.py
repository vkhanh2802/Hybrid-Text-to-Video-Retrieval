from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kis.result_pool import complete_frame_pool, require_frame_count, IncompleteResultPool


def frames(count, start=0):
    return [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1,
             "score": .9 - i / 10000, "final_score": .9 - i / 10000}
            for i in range(start, start + count)]


class FramePoolTest(unittest.TestCase):
    def test_two_hundred_contract_expands_without_padding_and_rejects_duplicates(self):
        fetch = Mock(return_value=frames(200))
        result, info = complete_frame_pool(frames(100), fetch, target=200, initial_k=200)
        require_frame_count(result, 200)
        fetch.assert_called_once_with(400)
        self.assertEqual(info['target'], 200)
        for invalid in (frames(199), frames(100) * 2):
            with self.assertRaises(IncompleteResultPool):
                require_frame_count(invalid, 200)

    def test_complete_pool_never_retrieves_again(self):
        fetch = Mock(side_effect=AssertionError("No unnecessary expansion"))
        result, info = complete_frame_pool(frames(100), fetch)
        require_frame_count(result)
        self.assertEqual(info["expansions"], 0)
        fetch.assert_not_called()

    def test_expands_real_retrieval_until_one_hundred_unique(self):
        initial = frames(10) * 10
        fetch = Mock(side_effect=[frames(70) * 2, frames(100)])
        result, info = complete_frame_pool(initial, fetch)
        require_frame_count(result)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [200, 400])
        self.assertEqual(info["expansions"], 2)

    def test_short_scope_fails_without_faking_or_looping_forever(self):
        rows = frames(5)
        before = deepcopy(rows)
        fetch = Mock(return_value=rows)
        with self.assertRaises(IncompleteResultPool) as error:
            complete_frame_pool(rows, fetch, max_k=400)
        self.assertEqual(error.exception.actual, 5)
        self.assertEqual(error.exception.target, 100)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(rows, before)

    def test_final_guard_rejects_repeated_frames_even_with_hundred_rows(self):
        with self.assertRaises(IncompleteResultPool):
            require_frame_count(frames(5) * 20)

    def test_final_guard_does_not_conflate_same_frame_id_across_videos(self):
        rows = frames(50) + [dict(row, video_id="L01_V002") for row in frames(50)]
        require_frame_count(rows)

    def test_nonfinite_score_is_not_a_real_candidate(self):
        rows = frames(100)
        rows[0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            complete_frame_pool(rows, Mock())


class FullFrameSolverTest(unittest.TestCase):
    def solve(self, count=100, temporal=False, scorer=None, auxiliary=True):
        from kis.solve_kis import solve_kis
        from kis.temporal import TemporalConfig
        diagnostics = {}
        visual = Mock(search=Mock(side_effect=lambda vector, top_k: frames(min(count, top_k))))
        bm25 = Mock(search_bm25=Mock(return_value=[]))
        captions = SimpleNamespace(search=lambda query, top_k: [], get_caption=lambda *_: {})
        query = {"raw_query": "pour then stir", "query_variants": ["pour then stir"], "entities": [],
                 "events": [{"action": "pour"}, {"action": "stir"}],
                 "temporal_constraints": [{"type": "sequence", "order": ["pour", "stir"]}] if temporal else []}
        result = solve_kis(query, visual, Mock(encode_text=Mock(return_value="vector")), bm25,
            captions, None, retrieval_top_k=50, result_top_k=100, diagnostics=diagnostics,
            temporal_config=TemporalConfig(enabled=temporal),
            temporal_as_auxiliary=auxiliary, require_full_results=True, vlm_reranker=scorer,
            rerank_top_k=20)
        return result, diagnostics, visual

    def test_temporal_five_segments_do_not_replace_hundred_frames(self):
        segments = [dict(row, temporal={"verified": False, "events": []}, vlm_status="skipped_temporal")
                    for row in frames(5, start=200)]

        def temporal(*args, **kwargs):
            args[6].update(status="applied", seconds=.01)
            return deepcopy(segments)

        with patch("kis.event_retrieval.retrieve_event_segments", side_effect=temporal):
            result, diag, _ = self.solve(temporal=True)
        require_frame_count(result)
        self.assertEqual(len(diag["temporal_results"]), 5)
        self.assertEqual(diag["temporal"]["presentation"], "auxiliary")
        self.assertEqual(diag["temporal_results"], segments)
        self.assertTrue(all("temporal" not in row for row in result))
        self.assertEqual(diag["result_pool"]["expansions"], 1)

    def test_full_result_contract_cannot_accidentally_early_return_temporal(self):
        def temporal(*args, **kwargs):
            args[6].update(status="applied", seconds=.01)
            return [dict(row, vlm_status="skipped_temporal") for row in frames(5, start=200)]

        with patch("kis.event_retrieval.retrieve_event_segments", side_effect=temporal):
            result, diag, _ = self.solve(temporal=True, auxiliary=False)
        require_frame_count(result)
        self.assertEqual(len(diag["temporal_results"]), 5)

    def test_small_index_fails_instead_of_success_with_five_frames(self):
        with self.assertRaises(IncompleteResultPool):
            self.solve(count=5)

    def test_vlm_errors_keep_hundred_frames_and_never_score_temporal_as_frames(self):
        scorer = Mock(score=Mock(side_effect=RuntimeError("test scoring error")))
        with patch("kis.solve_kis._resolve_candidate_image_path", return_value="fixture.jpg"):
            result, diag, _ = self.solve(scorer=scorer)
        require_frame_count(result)
        self.assertEqual(scorer.score.call_count, 20)
        self.assertEqual(sum(row["vlm_status"] == "error" for row in result), 20)


if __name__ == "__main__":
    unittest.main()
