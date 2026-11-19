import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kis.llm import HTTPQueryStructurer, QueryAPIConfig, QueryAPIError
from tools.benchmark_image_accuracy import ROOT, RetryingPlanner, quality, stats


class ImageAccuracyBenchmarkTests(unittest.TestCase):
    def test_twenty_frozen_inputs_have_no_source_identity(self):
        data = json.loads((ROOT / 'benchmarks/image_accuracy_20_20261118.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['queries']), 20)
        self.assertEqual(len({q['id'] for q in data['queries']}), 20)
        self.assertTrue(all(set(q) == {'id', 'query'} for q in data['queries']))

    def test_matching_neighbor_counts_but_other_video_does_not(self):
        gt = {'video_id': 'L01_V028', 'keyframes': [32, 33]}
        rows = [{'video_id': 'L03_V026', 'keyframe_n': 32},
                {'video_id': 'L01_V028', 'keyframe_n': 32}]
        self.assertEqual(quality(rows, gt)['best_rank'], 2)
        self.assertEqual(quality(rows[:1], gt)['hit_at_100'], 0)

    def test_operational_failure_has_zero_hit_not_removed_from_denominator(self):
        metrics = quality([], {'video_id': 'L01_V028', 'keyframes': [32, 33]})
        self.assertEqual(metrics['hit_at_1'], 0)
        self.assertEqual(metrics['mrr_at_100'], 0)

    def generator(self):
        config = QueryAPIConfig(provider='ollama', protocol='openai', model='test', base_url='http://127.0.0.1:0/v1')
        generator = RetryingPlanner(config, 'instruction', {})
        generator.calls, generator.query_id = [], 'test_query'
        return generator

    def test_429_waits_and_retries_same_payload(self):
        generator = self.generator()
        response = {'choices': [], 'usage': {'total_tokens': 5}}
        payload = {'model': 'test', 'messages': []}
        with patch.object(HTTPQueryStructurer, '_post', side_effect=[QueryAPIError('groq', 429, 'try again in 4s.'), response]) as transport:
            with patch('tools.benchmark_image_accuracy.time.sleep') as sleep:
                result = generator._post('/chat/completions', payload)
        self.assertIs(result, response)
        self.assertEqual(transport.call_count, 2)
        sleep.assert_called_once_with(4.5)
        self.assertEqual(generator.calls[0]['http_status'], 429)
        self.assertTrue(generator.calls[1]['success'])

    def test_nonretryable_http_error_not_hidden(self):
        generator = self.generator()
        with patch.object(HTTPQueryStructurer, '_post', side_effect=QueryAPIError('groq', 401, 'unauthorized')):
            with patch('tools.benchmark_image_accuracy.time.sleep') as sleep:
                with self.assertRaises(QueryAPIError):
                    generator._post('/chat/completions', {})
                sleep.assert_not_called()
        self.assertEqual(len(generator.calls), 1)

    def test_stats_empty_and_small_batch(self):
        self.assertIsNone(stats([]))
        self.assertEqual(stats([1, 3])['median'], 2)
