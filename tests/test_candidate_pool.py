import unittest

from kis.candidate_pool import select_candidate_videos
from kis.temporal import TemporalConfig


def hits(*videos):
    return [{"video_id": video, "frame_id": i + 1, "score": 1.0} for i, video in enumerate(videos)]


def event(visual=(), bm25=(), semantic=()):
    return {"action": "event", "streams": {"visual": hits(*visual), "bm25": hits(*bm25), "semantic": hits(*semantic)}}


class CandidatePoolTest(unittest.TestCase):
    def test_legacy_round_robin_is_preserved(self):
        selected, seeds, _ = select_candidate_videos([event(visual=("A", "B")), event(visual=("C", "D"))], 2)
        self.assertEqual(selected, ["A", "C"])
        self.assertEqual(set(seeds), {"A", "B", "C", "D"})

    def test_repeated_frames_do_not_multiply_video_vote_or_lower_next_video_rank(self):
        _, _, repeated = select_candidate_videos([event(visual=("A", "A", "A", "B"))], 2, "event_rrf")
        _, _, plain = select_candidate_videos([event(visual=("A", "B"))], 2, "event_rrf")
        self.assertEqual([v["score"] for v in repeated["ranked_videos"]], [v["score"] for v in plain["ranked_videos"]])
        self.assertEqual(repeated["ranked_videos"][1]["streams"][0]["video_rank"], 2)
        self.assertEqual(repeated["ranked_videos"][1]["streams"][0]["frame_rank"], 4)

    def test_multi_event_evidence_outranks_single_stream_early_hit(self):
        events = [event(visual=("noise1", "target"), bm25=("target",)),
                  event(visual=("noise2", "target"), semantic=("target",))]
        selected, _, _ = select_candidate_videos(events, 1, "event_rrf")
        self.assertEqual(selected, ["target"])

    def test_reserves_rare_event_without_intersection_requirement(self):
        events = [event(visual=("broad", "common"), bm25=("broad", "common"), semantic=("broad", "common")),
                  event(visual=("rare", "common"), bm25=("rare",), semantic=("rare",))]
        selected, _, diagnostics = select_candidate_videos(events, 2, "balanced_rrf")
        self.assertEqual(set(selected), {"broad", "rare"})
        self.assertNotIn("common", selected)
        self.assertEqual(select_candidate_videos(events, 2, "event_rrf")[0][0], "common")
        self.assertEqual(next(item for item in diagnostics["ranked_videos"] if item["video_id"] == "rare")["reserved_for_events"], [1])
        self.assertFalse(diagnostics["rank_evidence_is_verification"])

    def test_budget_is_bounded_and_nested_for_rrf_policies(self):
        events = [event(visual=("A", "B", "C")), event(semantic=("D", "E", "F"))]
        for strategy in ("event_rrf", "balanced_rrf"):
            smaller, _, _ = select_candidate_videos(events, 3, strategy)
            bigger, _, _ = select_candidate_videos(events, 5, strategy)
            self.assertEqual(len(smaller), 3)
            self.assertTrue(set(smaller).issubset(bigger))
            self.assertEqual(len(bigger), len(set(bigger)))

    def test_event_permutation_and_score_scaling_do_not_change_rrf_selection(self):
        events = [event(visual=("A", "B"), semantic=("B",)), event(visual=("B", "C"))]
        for strategy in ("event_rrf", "balanced_rrf"):
            original, _, _ = select_candidate_videos(events, 2, strategy)
            self.assertEqual(original, select_candidate_videos(events[::-1], 2, strategy)[0])
            for e in events:
                for stream in e["streams"].values():
                    for hit in stream:
                        hit["score"] *= 1000
            self.assertEqual(original, select_candidate_videos(events, 2, strategy)[0])

    def test_empty_streams_and_invalid_config(self):
        self.assertEqual(select_candidate_videos([event()], 12, "balanced_rrf")[0], [])
        with self.assertRaises(ValueError):
            TemporalConfig(candidate_pooling="invalid")
        with self.assertRaises(ValueError):
            TemporalConfig(candidate_rrf_k=0)


if __name__ == "__main__":
    unittest.main()
