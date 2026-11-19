from io import BytesIO
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from PIL import Image

from tools.build_kaggle_ocr_vietnamese_notebook import ORIGINAL, build, worker_source


class FakeRecognizer:
    engine = {'models': {'recognition': 'test-VietOCR'}, 'weights_sha256': 'test-only'}
    def recognize(self, image):
        return [dict(text='Quảng Ngãi, MÌ TRỨNG', confidence=.95,
                     polygon_padded=[[32, 32], [52, 32], [52, 42], [32, 42]],
                     truncated=False, width_capped=False, crop_size=[20, 10], recognizer_size=[64, 32]),
                dict(text='unreliable', confidence=.1, polygon_padded=[[32, 32]] * 4),
                dict(text='unfinished', confidence=.99, polygon_padded=[[32, 32]] * 4, truncated=True)]


class VietnameseNotebookTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'input/L01'
        self.output, self.env = self.base / 'state', self.base / 'runtime'
        self.archive = self.base / 'working/ocr_vietnamese.zip'
        self.run_id = 'a' * 32
        self.ns = {'__name__': 'offline_vi_worker', '__file__': str(ORIGINAL)}
        self.worker = worker_source()[3]
        exec(compile(self.worker, '<vi-worker>', 'exec'), self.ns)
        self.original_hash = self.ns['file_digest'](ORIGINAL)

    def image(self, n=1):
        path = self.root / 'L01_V001' / f'{n:03d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 16), 'red').save(path)
        return path

    def run_full(self, adapter=None):
        with patch('builtins.print'):
            self.ns['run'](self.root, self.output, self.env, self.run_id,
                           adapter=adapter or FakeRecognizer(), progress=lambda rows: rows)

    def test_generated_notebook_standalone_compiles_and_preserves_original(self):
        with patch('builtins.print'):
            document = build(self.base / 'vi.ipynb')
        self.assertEqual(len(document['cells']), 10)
        for c in document['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), '<cell>', 'exec')
                self.assertIsNone(c['execution_count'])
                self.assertEqual(c['outputs'], [])
        import nbformat
        nbformat.validate(nbformat.from_dict(document))
        setup = ''.join(document['cells'][3]['source'])
        embedded = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'WORKER_SOURCE' for t in n.targets))
        self.assertEqual(ast.literal_eval(embedded.value), self.worker)
        self.assertIn("'torch': '2.8.0+cu126'", setup)
        self.assertIn('--constraint', setup)
        self.assertIn("'check'", setup)
        self.assertNotIn("'vietocr==", setup)
        self.assertIn('RUN_FULL = False', ''.join(document['cells'][1]['source']))
        self.assertEqual(self.ns['file_digest'](ORIGINAL), self.original_hash)
        self.assertNotIn('import kis', self.worker)
        self.assertNotIn('from tools', self.worker)

    def test_full_vietnamese_alphabet_checked_not_merely_lang_flag(self):
        required = self.ns['VI_REQUIRED']
        vocab = SimpleNamespace(chars=required, encode=lambda s: list(s), decode=lambda s: ''.join(s))
        self.assertTrue(self.ns['validate_alphabet'](vocab)['roundtrip'])
        vocab.chars = 'abcđ'
        with self.assertRaisesRegex(RuntimeError, 'Vietnamese'):
            self.ns['validate_alphabet'](vocab)

    def test_pinned_assets_cache_tamper_rejected_and_license_included(self):
        path = self.base / 'asset.txt'
        raw = b'known source'
        expected = self.ns['git_blob_digest'](raw)
        response = BytesIO(raw)
        response.geturl = lambda: 'https://example.com/source'
        with patch('urllib.request.urlopen', return_value=response) as opener:
            self.ns['fetch_asset']('https://example.com/source', path, 'git', expected, 100)
        opener.assert_called_once()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No refetch for unchanged cache')):
            self.ns['fetch_asset']('https://example.com/source', path, 'git', expected, 100)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Cached asset changed'):
            self.ns['fetch_asset']('https://example.com/source', path, 'git', expected, 100)
        self.assertIn('LICENSE', self.ns['VI_BLOBS'])

    def test_download_hash_error_no_partial_or_poisoned_cache_left(self):
        response = BytesIO(b'bad')
        response.geturl = lambda: 'https://example.com/model'
        path = self.base / 'model.pth'
        with patch('urllib.request.urlopen', return_value=response), self.assertRaisesRegex(ValueError, 'checksum'):
            self.ns['fetch_asset']('https://example.com/model', path, 'sha256', 'a' * 64, 100)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.base.glob('*.partial')), [])

    def test_public_filters_but_raw_confidence_geometry_and_unicode_remain(self):
        path = self.image()
        before = path.read_bytes()
        self.run_full()
        checkpoint = next((self.output / 'checkpoints').glob('*.json'))
        record = json.loads(checkpoint.read_text(encoding='utf-8'))['record']
        self.assertEqual(len(record['lines']), 3)
        self.assertEqual([r['public_included'] for r in record['lines']], [True, False, False])
        self.assertTrue(record['lines'][2]['truncated'])
        self.assertEqual(record['source_sha256'], self.ns['digest'](before))
        public = self.ns['public_record'](record)
        self.assertEqual(public['ocr_text'], 'Quảng Ngãi, MÌ TRỨNG')
        self.assertEqual(public['retrieval_text'], public['ocr_text'])
        self.assertIsNone(public['frame_id'])
        self.assertEqual(record['lines'][0]['polygon_original_clipped'][0], [0, 0])
        self.assertEqual(path.read_bytes(), before)

    def test_two_exports_public_schema_and_technical_provenance_separate(self):
        self.image()
        self.run_full()
        with patch('builtins.print'):
            self.ns['export_current'](self.root, self.output, self.env, self.run_id, self.archive)
        with zipfile.ZipFile(self.archive) as z:
            self.assertEqual(z.namelist(), ['L01/L01_V001/001.json'])
            row = json.loads(z.read(z.namelist()[0]))[0]
            self.assertEqual(len(row), 7)
            self.assertNotIn('confidence', row)
        technical = self.archive.with_name('ocr_vietnamese_diagnostics.zip')
        with zipfile.ZipFile(technical) as z:
            self.assertEqual(set(z.namelist()), {'profile.json', 'run.json', 'checkpoints/L01/L01_V001/001.json'})
            record = json.loads(z.read('checkpoints/L01/L01_V001/001.json'))
            self.assertIn('source_sha256', record)
            self.assertEqual(len(record['lines']), 3)
        self.assertFalse((self.archive.parent / 'ocr.zip').exists())

    def test_unowned_technical_archive_is_never_overwritten(self):
        self.image()
        self.run_full()
        self.archive.parent.mkdir()
        technical = self.archive.with_name('ocr_vietnamese_diagnostics.zip')
        technical.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            self.ns['export_current'](self.root, self.output, self.env, self.run_id, self.archive)
        self.assertEqual(technical.read_bytes(), b'user data')
        self.assertFalse(self.archive.exists())

    def test_preview_is_sampled_not_bulk_and_does_not_create_ready_or_checkpoints(self):
        for n in range(1, 6):
            self.image(n)
        adapter = FakeRecognizer()
        with patch.dict(self.ns, Adapter=lambda: adapter), patch('builtins.print'):
            self.ns['preview'](self.root, self.output, self.env, self.run_id, 3)
        report = json.loads((self.output / 'previews' / self.run_id / 'preview.json').read_text(encoding='utf-8'))
        self.assertEqual([r['keyframe_n'] for r in report['records']], [1, 3, 5])
        self.assertFalse((self.output / 'ready.json').exists())
        self.assertFalse((self.output / 'checkpoints').exists())

    def test_model_failure_not_mislabeled_as_empty(self):
        self.image()
        adapter = SimpleNamespace(engine={}, recognize=Mock(side_effect=RuntimeError('model failed')))
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            self.run_full(adapter)
        self.assertEqual(list((self.output / 'checkpoints').glob('*.json')), [])

    def test_recognize_preserves_bgr_crop_order_and_polygon_identity(self):
        cls = self.ns['Adapter']
        adapter = cls.__new__(cls)
        polygons = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]])
        def detect(image):
            self.assertEqual(image[0, 0].tolist(), [0, 0, 255])
            return polygons.tolist(), [image]
        adapter.detector = SimpleNamespace(detect=detect)
        def recognize(image):
            self.assertEqual(image[0, 0].tolist(), [0, 0, 255])
            return dict(text='ĐỦ DẤU', confidence=.9)
        adapter.recognize_crops = lambda crops: [recognize(c) for c in crops]
        lines = adapter.recognize(Image.new('RGB', (4, 4), 'red'))
        self.assertEqual(lines[0]['polygon_padded'], polygons[0].tolist())
        self.assertEqual(lines[0]['text'], 'ĐỦ DẤU')

    def test_greedy_decoder_vietnamese_eos_and_length_cap_without_gpu(self):
        import torch
        adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        adapter.vocab = SimpleNamespace(go=1, eos=2, decode=lambda ids: ''.join('ả' for i in ids if i == 4))
        choices = iter([4, 2])
        def decoder(tokens, memory):
            logits = torch.full((1, len(tokens), 5), -20.)
            logits[0, -1, next(choices)] = 20.
            return logits, memory
        adapter.model = SimpleNamespace(cnn=lambda image: image,
            transformer=SimpleNamespace(forward_encoder=lambda image: image, forward_decoder=decoder))
        tensor, to = torch.tensor, torch.Tensor.to
        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return tensor(*args, **kwargs)
        def cpu_to(value, *args, **kwargs):
            return value if args and args[0] == 'cuda:0' else to(value, *args, **kwargs)
        with patch('torch.tensor', side_effect=cpu_tensor), patch.object(torch.Tensor, 'to', cpu_to):
            result = adapter.recognize_crop(np.zeros((24, 960, 3), dtype=np.uint8))
        self.assertEqual(result['text'], 'ả')
        self.assertFalse(result['truncated'])
        self.assertEqual(result['recognizer_size'], [1280, 32])
        self.assertFalse(result['width_capped'])
        self.assertGreater(result['confidence'], .99)

        def endless(tokens, memory):
            logits = torch.full((1, len(tokens), 5), -20.)
            logits[0, -1, 4] = 20.
            return logits, memory
        adapter.model.transformer.forward_decoder = endless
        with patch('torch.tensor', side_effect=cpu_tensor), patch.object(torch.Tensor, 'to', cpu_to), \
                patch.dict(self.ns['OPTIONS'], recognition_max_tokens=2):
            result = adapter.recognize_crop(np.zeros((10, 3000, 3), dtype=np.uint8))
        self.assertTrue(result['truncated'])
        self.assertTrue(result['width_capped'])
        self.assertEqual(result['recognizer_size'], [2048, 32])

    def test_skipping_full_run_invalidates_previous_ready_state_in_kernel(self):
        with patch('builtins.print'):
            document = build(self.base / 'vi.ipynb')
            namespace = dict(RUN_OK=True, EXPORT_OK=True, RUN_FULL=False)
            exec(''.join(document['cells'][7]['source']), namespace)
        self.assertFalse(namespace['RUN_OK'])
        self.assertFalse(namespace['EXPORT_OK'])


if __name__ == '__main__':
    unittest.main()
