import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from tools.benchmark_user_five_ocr import load_mapping
from tools.benchmark_user_five_ocr import execution_inputs, quality, strip_ocr, validate_identities


class UserFiveBenchmarkTests(unittest.TestCase):
    def test_forward_mapping_allows_frame_alias_not_duplicate_keyframe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'L21').mkdir()
            path = root / 'L21/L21_V006.csv'
            path.write_text('n,frame_idx\n1,0\n2,0\n', encoding='utf-8')
            with patch('tools.benchmark_user_five_ocr.ROOT', root):
                mapping, _ = load_mapping(root, ['L21'])
                self.assertEqual(mapping[('L21_V006', 1)], mapping[('L21_V006', 2)])
                path.write_text('n,frame_idx\n1,0\n1,1\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_mapping(root, ['L21'])

    def test_gt_not_in_execution_inputs(self):
        self.assertEqual(execution_inputs({'queries': [{'id': 'q1', 'query': 'typo quya',
                          'video_id': 'L01_V003', 'keyframes': [31]}]}),
                         [{'id': 'q1', 'query': 'typo quya'}])

    def test_multi_positive_keyframes_not_frame_idx(self):
        rows = [{'video_id': 'L01_V003', 'keyframe_n': n, 'frame_id': 31} for n in (30, 33, 31)]
        q = quality(rows, {'video_id': 'L01_V003', 'keyframes': [31, 32, 33]})
        self.assertEqual(q['positive_ranks'], [2, 3])
        self.assertEqual(q['mrr_at_100'], 0.5)
        self.assertEqual(q['hit_at_1'], 0)

    def test_unknown_truth_not_fabricated(self):
        q = quality([], {'video_id': 'L01_V003', 'keyframes': [31]})
        self.assertIsNone(q['best_rank'])
        self.assertEqual(q['mrr_at_100'], 0)

    def test_no_prompt_ocr_does_not_mutate_shared_pool(self):
        rows = [{'video_id': 'L01_V003', 'score': 0.1, 'ocr': {'text': 'ABC'}}]
        result = strip_ocr(rows)
        self.assertNotIn('ocr', result[0])
        self.assertEqual(rows[0]['ocr']['text'], 'ABC')
        self.assertEqual(result[0]['score'], rows[0]['score'])

    def test_mapping_and_distinct_checks(self):
        row = {'video_id': 'L01_V003', 'keyframe_n': 31, 'frame_id': 999}
        mapping = {('L01_V003', 31): 999}
        validate_identities([row], mapping, 1)
        for rows, expected in [([row, row], 2), ([row], 100)]:
            with self.assertRaises(ValueError):
                validate_identities(rows, mapping, expected)
        with self.assertRaises(ValueError):
            validate_identities([row], {('L01_V003', 31): 31})


if __name__ == '__main__':
    unittest.main()
