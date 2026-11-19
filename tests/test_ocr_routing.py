from copy import deepcopy
import unittest

from kis.ocr import OCRConfig, literal_queries, search_ocr
from kis.ocr_routing import select_prompt_ocr


class RoutingTests(unittest.TestCase):
    def test_visual_only_no_ocr_even_if_planner_says_needed(self):
        self.assertEqual(literal_queries({'raw_query': 'Hai xe xanh lá trên đường', 'needs_ocr': True}, OCRConfig()), [])

    def test_explicit_headline_survives_missing_entity(self):
        q = {'raw_query': 'Một cảnh có dòng chữ: UNICEF: bảo vệ trẻ em.', 'needs_ocr': False}
        self.assertEqual(literal_queries(q, OCRConfig())[0]['text'], 'UNICEF: bảo vệ trẻ em')

    def test_nested_text_cue_does_not_become_literal(self):
        q = {'raw_query': 'Bảng có ghi dòng chữ Lễ hội văn hóa', 'needs_ocr': True,
             'entities': [{'type': 'text', 'value': 'Lễ hội văn hóa'}]}
        self.assertEqual(literal_queries(q, OCRConfig()),
                         [{'text': 'Lễ hội văn hóa', 'source': 'literal_text_entity'}])

    def test_paraphrase_only_original_names(self):
        q = {'raw_query': 'Có dòng chữ liên quan đến việc video Youtube về tin giả xung dột Isreal-Hamas', 'needs_ocr': True}
        self.assertEqual([r['text'] for r in literal_queries(q, OCRConfig())], ['Youtube', 'Isreal-Hamas'])

    def test_no_translated_entity(self):
        q = {'raw_query': 'Bảng chữ tiếng Việt', 'needs_ocr': True, 'entities': [{'type': 'text', 'value': 'English invented text'}]}
        self.assertEqual(literal_queries(q, OCRConfig()), [])

    def test_compact_prompt_and_missing_no_prompt(self):
        q = {'raw_query': 'dòng chữ: UNICEF', 'needs_ocr': True}
        evidence = {'source_format': 'public_json', 'status': 'ok', 'text': 'noise ' * 200 + 'UNICEF protects children ' + 'noise ' * 200}
        before = deepcopy(evidence)
        selected = select_prompt_ocr(evidence, q)
        self.assertIn('UNICEF', selected['text'])
        self.assertLessEqual(len(selected['text']), 384)
        self.assertEqual(evidence, before)
        for status in ('missing', 'error', 'empty'):
            self.assertIsNone(select_prompt_ocr({**evidence, 'status': status}, q))
        self.assertIsNone(select_prompt_ocr(evidence, 'A person driving'))
        self.assertIsNone(select_prompt_ocr(evidence, 'dòng chữ: UNESCO'))

    def test_fallback_only_after_exact_miss_and_bounded(self):
        class Retriever:
            supports_candidate_ocr = True
            def __init__(self):
                self.calls = []
            def search(self, text, config):
                self.calls.append(text)
                if text == 'một hai ba bốn':
                    return [{'video_id': 'L01_V001', 'frame_id': 1, 'score': 1, 'ocr': {'text': text}}]
                return []
        r = Retriever()
        hits, diag = search_ocr({'raw_query': 'dòng chữ: một hai ba bốn năm sáu bảy tám'}, r, OCRConfig(mode='retrieve'))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['ocr']['match_kind'], 'original_subphrase')
        self.assertLessEqual(len(r.calls), 5)
        r = Retriever()
        search_ocr({'raw_query': 'dòng chữ: một hai ba bốn'}, r, OCRConfig(mode='retrieve'))
        self.assertEqual(r.calls, ['một hai ba bốn'])


if __name__ == '__main__':
    unittest.main()
