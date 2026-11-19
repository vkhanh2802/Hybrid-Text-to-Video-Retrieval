import json
from pathlib import Path
import unittest

from tools.benchmark_user_five_ocr import quality, validate_identities


class RandomImageQueryBenchmarkTests(unittest.TestCase):
    def test_frozen_execution_inputs_do_not_include_labels(self):
        path = Path(__file__).resolve().parents[1] / 'benchmarks/random_image_queries_20261118.json'
        queries = json.loads(path.read_text(encoding='utf-8'))['queries']
        self.assertEqual(len(queries), 5)
        self.assertEqual(len({q['id'] for q in queries}), 5)
        self.assertTrue(all(set(q) == {'id', 'query'} and q['query'].strip() for q in queries))

    def test_exact_source_frame_not_just_same_video(self):
        truth = {'video_id': 'L21_V025', 'keyframes': [85]}
        rows = [{'video_id': 'L21_V025', 'keyframe_n': 84},
                {'video_id': 'L20_V024', 'keyframe_n': 85},
                {'video_id': 'L21_V025', 'keyframe_n': 85}]
        metrics = quality(rows, truth)
        self.assertEqual(metrics['best_rank'], 3)
        self.assertEqual(metrics['hit_at_1'], 0)
        self.assertEqual(metrics['hit_at_5'], 1)
        self.assertAlmostEqual(metrics['mrr_at_100'], 1 / 3)
        self.assertIsNone(quality(rows[:2], truth)['best_rank'])

    def test_pool_rank_over_100_does_not_count_as_hit_at_100(self):
        truth = {'video_id': 'L21_V025', 'keyframes': [85]}
        rows = [{'video_id': 'L20_V024', 'keyframe_n': n} for n in range(101)]
        rows.append({'video_id': 'L21_V025', 'keyframe_n': 85})
        metrics = quality(rows, truth)
        self.assertEqual(metrics['best_rank'], 102)
        self.assertEqual(metrics['hit_at_100'], 0)
        self.assertEqual(metrics['mrr_at_100'], 0)

    def test_duplicate_or_wrong_frame_identity_is_rejected(self):
        row = {'video_id': 'L21_V025', 'keyframe_n': 85, 'frame_id': 7927}
        mapping = {('L21_V025', 85): 7927}
        validate_identities([row], mapping, 1)
        with self.assertRaises(ValueError):
            validate_identities([row, row], mapping)
        with self.assertRaises(ValueError):
            validate_identities([dict(row, frame_id=7928)], mapping)
