import json
from pathlib import Path
import unittest

from tools.benchmark_groq_planner import ROOT, evaluate_checks, load_queries, retry_delay


class GroqPlannerBenchmarkTests(unittest.TestCase):
    def test_twenty_unique_execution_queries_and_all_labels(self):
        queries, files = load_queries(ROOT / 'benchmarks/groq_planner_20_20261118.json')
        self.assertEqual(len(queries), 20)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(set(q) == {'id', 'query'} for q in queries))
        labels = json.loads((ROOT / 'benchmarks/groq_planner_20_20261118.expectations.json').read_text(encoding='utf-8'))['queries']
        self.assertEqual(set(labels), {q['id'] for q in queries})

    def test_old_seven_queries_preserved(self):
        queries, _ = load_queries(ROOT / 'benchmarks/groq_planner_20_20261118.json')
        local = json.loads((ROOT / 'benchmarks/local_query_planner_20261118.json').read_text(encoding='utf-8'))
        previous = json.loads((ROOT / local['base_queries']).read_text(encoding='utf-8'))['queries'] + local['probes']
        self.assertEqual(queries[:7], previous)

    def test_rate_limit_delay_units_and_bounds(self):
        self.assertAlmostEqual(retry_delay('Please try again in 4.3725s.', 0), 4.8725)
        self.assertEqual(retry_delay('try again in 500ms.', 0), 1.)
        self.assertEqual(retry_delay('try again in 2m.', 0), 60.)
        self.assertEqual(retry_delay('no duration', 0), 15.)
        self.assertEqual(retry_delay('no duration', 1), 30.)

    def structured(self):
        return {'needs_ocr': True, 'needs_asr': False, 'question': None,
                'events': [], 'temporal_constraints': [], 'entities': [],
                'raw_query': 'SỬA XE MINH TÂM', 'query_variants': ['SỬA XE MINH TÂM'],
                'visual_description': 'a shop', 'stream_queries': {'visual': ['a shop']}}

    def test_literal_does_not_pass_by_copying_raw_query(self):
        value = self.structured()
        expected = {'needs_ocr': True, 'ocr_literal': 'SỬA XE MINH TÂM'}
        self.assertFalse(evaluate_checks(value, expected)['checks']['ocr_literal_preserved'])
        value['entities'] = [{'type': 'text', 'value': 'SỬA XE MINH TÂM', 'attributes': {}}]
        self.assertTrue(evaluate_checks(value, expected)['all_pass'])

    def test_sequence_flags_and_event_count_checked_separately(self):
        value = self.structured()
        value['events'] = [{'action': 'open door'}, {'action': 'enter car'}]
        value['temporal_constraints'] = [{'type': 'sequence', 'order': ['open door', 'enter car']}]
        self.assertTrue(evaluate_checks(value, {'has_sequence': True, 'minimum_events': 2})['all_pass'])
        self.assertFalse(evaluate_checks(value, {'has_sequence': True, 'minimum_events': 3})['all_pass'])

    def test_failed_query_is_not_evaluated_as_model_semantic_miss(self):
        self.assertIsNone(evaluate_checks(None, {'needs_ocr': True}))
