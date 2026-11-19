"""Geometry and orchestration regression only; no weights, GPU, or OCR install."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from tools.build_kaggle_ocr_vietnamese_notebook import ORIGINAL, build, worker_source


class LongLineTest(unittest.TestCase):
    def setUp(self):
        self.ns = {'__name__': 'long_line_test', '__file__': str(ORIGINAL)}
        exec(compile(worker_source()[3], '<long-line-worker>', 'exec'), self.ns)

    def crop(self, light_text=True):
        # Synthetic word-like blocks, separated by generous whitespace.
        value = np.zeros((32, 900, 3), dtype=np.uint8)
        for x in range(10, 880, 90):
            value[7:25, x:x + 65] = 255
        return value if light_text else 255 - value

    def result(self, text, **kwargs):
        return dict(text=text, confidence=kwargs.get('confidence', .9),
                    truncated=kwargs.get('truncated', False), width_capped=False,
                    crop_size=[900, 32], recognizer_size=[900, 32],
                    probability_kind='uncalibrated_mean_character_softmax')

    def test_word_gaps_cover_all_pixels_once_bounded_and_both_polarities(self):
        for light in (True, False):
            crop = self.crop(light)
            plan = self.ns['plan_line_segments'](crop)
            self.assertEqual(plan['status'], 'split')
            ranges = plan['ranges']
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], crop.shape[1])
            restored = np.concatenate([crop[:, a:b] for a, b in ranges], axis=1)
            self.assertEqual(restored.tobytes(), crop.tobytes())
            for i, (a, b) in enumerate(ranges):
                self.assertGreater(b, a)
                self.assertLessEqual(self.ns['recognition_width'](b - a, 32), 512)
                if i:
                    self.assertEqual(a, ranges[i-1][1])
                    # Boundary is in background, not inside a simulated word.
                    self.assertEqual(int(crop[:, a].max() if light else crop[:, a].min()), 0 if light else 255)

    def test_premature_eos_full_line_does_not_win_over_complete_segments(self):
        adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        emitted = iter([self.result('Diễn tập PCC', confidence=.99),
                        self.result('Diễn tập PCCC'), self.result('sân bay Tân Sơn Nhất')])
        seen = []
        def recognize(crop):
            seen.append(crop.shape[1])
            return next(emitted)
        adapter.recognize_crops = lambda crops: [recognize(c) for c in crops]
        result = adapter.recognize_line(self.crop())
        self.assertEqual(result['text'], 'Diễn tập PCCC sân bay Tân Sơn Nhất')
        self.assertEqual(result['full_line']['text'], 'Diễn tập PCC')
        self.assertEqual(result['recognition_strategy'], 'word_gap_segments')
        self.assertTrue(result['public_eligible'])
        self.assertEqual(len(seen), 3)
        self.assertEqual(result['completeness'], 'not_verified')

    def test_repeated_words_not_fuzzily_deduplicated(self):
        adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        emitted = iter([self.result('rất'), self.result('rất'), self.result('rất tốt')])
        adapter.recognize_crops = lambda crops: [next(emitted) for _ in crops]
        self.assertEqual(adapter.recognize_line(self.crop())['text'], 'rất rất tốt')

    def test_unsafe_gap_never_cuts_glyph_or_claims_complete(self):
        crop = np.zeros((32, 900, 3), dtype=np.uint8)
        crop[4:28, :] = 255  # continuous foreground, no word gap
        plan = self.ns['plan_line_segments'](crop)
        self.assertEqual(plan['status'], 'no_safe_gap')
        adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        adapter.recognize_crops = lambda crops: [self.result('prefix only') for _ in crops]
        line = adapter.recognize_line(crop)
        self.assertFalse(line['public_eligible'])
        self.assertEqual(line['full_line']['text'], 'prefix only')
        self.assertEqual(line['completeness'], 'not_verified')

    def test_any_bad_segment_blocks_whole_public_line_no_silent_partial(self):
        for bad in [dict(confidence=.1), dict(truncated=True), dict(text='')]:
            adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
            bad_text = bad.get('text', 'suffix')
            options = {k: v for k, v in bad.items() if k != 'text'}
            emitted = iter([self.result('prefix'), self.result('first'), self.result(bad_text, **options)])
            adapter.recognize_crops = lambda crops: [next(emitted) for _ in crops]
            line = adapter.recognize_line(self.crop())
            self.assertFalse(line['public_eligible'])
            record = dict(video_id='L01_V001', keyframe_n=1, relative_path='L01_V001/001.jpg',
                          status='ok', lines=[line])
            self.assertEqual(self.ns['public_record'](record)['ocr_text'], '')

    def test_short_line_unchanged_and_model_failure_propagates(self):
        adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        adapter.recognize_crops = lambda crops: [self.result('HOÀNG VĨNH') for _ in crops]
        line = adapter.recognize_line(self.crop()[:, :200])
        self.assertEqual(line['text'], 'HOÀNG VĨNH')
        self.assertEqual(line['recognition_strategy'], 'full_line')
        def fail(_):
            raise RuntimeError('GPU failure')
        adapter.recognize_crops = fail
        with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
            adapter.recognize_line(self.crop())

    def test_selected_frame_preview_strict_identity_order_no_fallback(self):
        items = [dict(id=f'L01_V001:{n}') for n in range(1, 6)]
        selected = self.ns['select_preview_items'](items, 3, ['L01_V001:5', 'L01_V001:2'])
        self.assertEqual([r['id'] for r in selected], ['L01_V001:5', 'L01_V001:2'])
        self.assertEqual([r['id'] for r in self.ns['select_preview_items'](items, 3, [])],
                         ['L01_V001:1', 'L01_V001:3', 'L01_V001:5'])
        for invalid in [['L01_V001:99'], ['L01_V001:2'] * 2, ['../bad'], 'L01_V001:1']:
            with self.assertRaises(ValueError):
                self.ns['select_preview_items'](items, 3, invalid)

    def test_standalone_new_names_and_profile_pin_segmentation(self):
        with tempfile.TemporaryDirectory() as d, patch('builtins.print'):
            doc = build(Path(d) / 'notebook.ipynb')
        setup = ''.join(doc['cells'][3]['source'])
        self.assertIn('ocr_vietnamese_v3_batch16.zip', setup)
        self.assertIn('ocr-vietnamese-lines-v3-', worker_source()[3])
        self.assertIn('PREVIEW_IDENTITIES', ''.join(doc['cells'][1]['source']))
        self.assertEqual(self.ns['OPTIONS']['line_segment_max_width'], 480)
        self.assertEqual(self.ns['OPTIONS']['line_segment_trigger_width'], 512)
        for c in doc['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), '<cell>', 'exec')

    def test_segment_diagnostics_survive_checkpoint_and_v3_export_without_touching_old_zip(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            image = base / 'input/L01_V001/001.png'
            image.parent.mkdir(parents=True)
            Image.fromarray(self.crop()).save(image)
            original = image.read_bytes()
            archive = base / 'working/ocr_vietnamese_v3.zip'
            archive.parent.mkdir()
            old = archive.with_name('ocr_vietnamese.zip')
            old.write_bytes(b'preserved-old-archive')
            adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
            adapter.engine = dict(test_only=True)
            adapter.detector = SimpleNamespace(detect=lambda _: (
                [[[32, 32], [932, 32], [932, 64], [32, 64]]], [self.crop()]))
            emitted = iter([self.result('prefix', confidence=.99), self.result('cầu cảng'), self.result('Quảng Ngãi')])
            adapter.recognize_crops = lambda crops: [next(emitted) for _ in crops]
            output, env, run_id = base / 'state', base / 'runtime', 'd' * 32
            with patch('builtins.print'):
                self.ns['run'](base / 'input', output, env, run_id, adapter=adapter, progress=lambda x: x)
                self.ns['export_current'](base / 'input', output, env, run_id, archive)
            with zipfile.ZipFile(archive) as z:
                public = json.loads(z.read('L01/L01_V001/001.json'))[0]
                self.assertEqual(len(public), 7)
                self.assertEqual(public['ocr_text'], 'cầu cảng Quảng Ngãi')
                self.assertEqual(public['retrieval_text'], public['ocr_text'])
            with zipfile.ZipFile(archive.with_name('ocr_vietnamese_v3_diagnostics.zip')) as z:
                record = json.loads(z.read('checkpoints/L01/L01_V001/001.json'))
                line = record['lines'][0]
                self.assertEqual(line['full_line']['text'], 'prefix')
                self.assertEqual(len(line['segments']), 2)
                self.assertTrue(line['public_included'])
                self.assertTrue(line['public_eligible'])
                self.assertEqual(line['completeness'], 'not_verified')
                self.assertEqual(len(line['segments'][0]['crop_bgr_sha256']), 64)
                profile = json.loads(z.read('profile.json'))
                self.assertEqual(profile['options']['line_segment_max_width'], 480)
            self.assertEqual(old.read_bytes(), b'preserved-old-archive')
            self.assertEqual(image.read_bytes(), original)

    def test_gap_planner_limits_and_invalid_images(self):
        for bad in [np.zeros((0, 9, 3), dtype=np.uint8), np.zeros((9, 9)),
                    np.zeros((9, 9, 3), dtype=float)]:
            with self.assertRaises(ValueError):
                self.ns['plan_line_segments'](bad)
        with patch.dict(self.ns['OPTIONS'], line_segment_max_count=1):
            self.assertEqual(self.ns['plan_line_segments'](self.crop())['status'], 'no_safe_gap')
        blank = np.zeros((32, 900, 3), dtype=np.uint8)
        self.assertEqual(self.ns['plan_line_segments'](blank)['status'], 'no_safe_gap')


if __name__ == '__main__':
    unittest.main()
