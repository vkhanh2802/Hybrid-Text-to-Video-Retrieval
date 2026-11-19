from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from kis.temporal import TemporalConfig, align_events
from tools.temporal_multipath import align_multiple, candidate_mask, compact_path


def frames(times):
    return [{"video_id": "test", "frame_id": i, "keyframe_n": i+1, "pts_time": t}
            for i, t in enumerate(times)]


def exhaustive(table, scores, mask, config):
    states = []
    for choice in product(range(-1, len(table)), repeat=len(scores)):
        path = tuple((e, f) for e, f in enumerate(choice) if f >= 0)
        if not path or any(not mask[e, f] for e, f in path):
            continue
        gaps = [table[b]["pts_time"] - table[a]["pts_time"]
                for (_, a), (_, b) in zip(path, path[1:])]
        if any(not 0 < gap <= config.max_gap_seconds for gap in gaps):
            continue
        objective = float(scores[path[0]])
        for index, (event, frame) in enumerate(path[1:]):
            objective += float(scores[event, frame])
            objective -= config.gap_penalty * gaps[index]
        states.append((len(path), objective, path))
    return sorted(states, key=lambda s: (-s[0], -s[1], s[2]))


class TemporalMultipathTest(unittest.TestCase):
    def test_random_tiny_dag_matches_exhaustive_with_pinned_ties(self):
        rng, config = np.random.default_rng(765), TemporalConfig()
        for _ in range(60):
            table = frames(sorted(rng.integers(0, 90, 4).tolist()))
            scores = rng.random((3, 4))
            mask = rng.random((3, 4)) > .35
            baseline = align_events(table, scores, mask, config)
            actual, _ = align_multiple(table, scores, mask, config, event_frame_budget=4,
                                       search_paths=8, max_paths=8)
            expected = exhaustive(table, scores, mask, config)
            paths = ([baseline["path"]] if baseline else [])
            paths += [s[2] for s in expected[:8] if s[2] not in paths]
            self.assertEqual([a["path"] for a in actual], paths[:8])
            if baseline:
                self.assertEqual(actual[0], baseline)

    def test_top1_exact_even_with_cap_ties_and_diversity(self):
        table, scores = frames([1, 1, 2, 3]), np.ones((3, 4)) * .8
        mask, config = np.ones((3, 4), bool), TemporalConfig()
        baseline = align_events(table, scores, mask, config)
        for diverse in (False, True):
            result, stats = align_multiple(table, scores, mask, config, event_frame_budget=1,
                                            max_paths=1, diverse=diverse)
            self.assertEqual(result, [baseline])
            self.assertLessEqual(max(stats["selected_nodes_by_event"]), 2)

    def test_skip_equal_times_gap_and_duplicate_paths(self):
        config = replace(TemporalConfig(), max_gap_seconds=2)
        table, scores = frames([1, 1, 3, 10]), np.ones((3, 4)) * .8
        mask = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]], bool)
        result, _ = align_multiple(table, scores, mask, config, max_paths=8)
        self.assertEqual(result[0]["coverage"], 2/3)
        self.assertEqual(len({a["path"] for a in result}), len(result))
        for aligned in result:
            for (_, a), (_, b) in zip(aligned["path"], aligned["path"][1:]):
                self.assertTrue(0 < table[b]["pts_time"] - table[a]["pts_time"] <= 2)
        compact = compact_path("test", table, result[0], 3)
        self.assertEqual(compact["temporal"]["events"][1]["status"], "unknown")
        self.assertFalse(compact["temporal"]["verified"])

    def test_alternative_valid_path_survives_and_gate_is_not_relaxed(self):
        table = frames([1, 2, 3, 4])
        scores = np.array([[.9, .8, .1, .1], [.1, .1, .9, .8]])
        support = scores > .5
        result, _ = align_multiple(table, scores, support, TemporalConfig(), max_paths=4)
        self.assertIn(((0, 1), (1, 3)), [a["path"] for a in result])
        mask = candidate_mask(scores, support, result[0], 32)
        self.assertFalse(np.any(mask & ~support))

    def test_empty_unsupported_and_validation(self):
        config = TemporalConfig()
        self.assertEqual(align_multiple([], np.zeros((2, 0)), np.zeros((2, 0)), config)[0], [])
        for kwargs in ({"max_paths": 0}, {"search_paths": True}, {"event_frame_budget": 129},
                       {"max_paths": 5, "search_paths": 4}, {"diverse": 1}):
            with self.assertRaises(ValueError):
                align_multiple([], np.zeros((2, 0)), np.zeros((2, 0)), config, **kwargs)
        with self.assertRaises(ValueError):
            align_multiple(frames([2, 1]), np.ones((2, 2)), np.ones((2, 2)), config)
        with self.assertRaises(ValueError):
            align_multiple(frames([1, 2]), np.ones((2, 2)), np.ones((2, 2)),
                           replace(config, max_frames_per_video=1))


if __name__ == "__main__":
    unittest.main()
