import unittest
from tools.benchmark_qwen_topk_cost import chunks, score_status, summarize, BUDGETS


class TopKCostTests(unittest.TestCase):
    def test_batch_partition_preserves_all_available_images(self):
        for count in (0, 19, 20, 37, 60, 93, 100):
            source = list(range(count))
            parts = chunks(source)
            self.assertEqual([v for part in parts for v in part], source)
            self.assertTrue(all(0 < len(part) <= 12 for part in parts))

    def test_invalid_scores_not_zero_or_valid(self):
        for value in (None, float('nan'), float('inf'), True, -1, 11, '7'):
            self.assertEqual(score_status(value, None), 'invalid_score')
        self.assertEqual(score_status(0, None), 'valid')
        self.assertEqual(score_status(7, 'cuda_out_of_memory'), 'error')

    def test_aggregation_uses_all_queries_and_real_ratios(self):
        rows = []
        for b in BUDGETS:
            for i in range(5):
                rows.append(dict(budget=b, image_fetch_seconds=1, qwen_seconds=b+i,
                    measured_total_seconds=b+i+1, images_scored=b-1, statuses={'valid': b-2, 'missing_image': 1, 'invalid_score': 1},
                    oom_splits=0, peak_allocated_bytes=1024**2, peak_reserved_bytes=2*1024**2))
        result = summarize(rows)
        self.assertEqual(result['20']['measured_total_seconds']['median'], 23)
        self.assertEqual(result['100']['total_median_ratio_to_20'], 103/23)
        self.assertEqual(result['100']['missing_images'], 5)


if __name__ == '__main__':
    unittest.main()
