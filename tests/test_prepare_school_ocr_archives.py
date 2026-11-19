from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.prepare_school_ocr_archives import prepare


class PrepareOCRArchivesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'ocr'
        self.source.mkdir()
        self.mapping = self.root / 'mapping/L06'
        self.mapping.mkdir(parents=True)
        (self.mapping / 'L06_V001.csv').write_text('n,frame_idx\n1,10\n', encoding='utf-8')
        self.raw = json.dumps([{'video_id': 'L06_V001', 'frame_id': None, 'keyframe_n': 1,
                               'image_path': 'data/L06/L06_V001/001.jpg', 'ocr_text': 'TEST',
                               'retrieval_text': 'TEST', 'status': 'ok'}]).encode()

    def inner(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('L06/L06_V001/001.json', self.raw)
        return buffer.getvalue()

    def test_nested_main_preserves_zip_and_json_not_diagnostics(self):
        payload = self.inner()
        archive = self.source / 'L06.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('L06_OCR/', b'')
            z.writestr('L06_OCR/ocr_vietnamese_v3_batch32.zip', payload)
            z.writestr('L06_OCR/ocr_vietnamese_v3_batch32_diagnostics.zip', b'not search data')
        before = archive.read_bytes()
        result = prepare(self.source, self.root / 'mapping', self.root / 'out')
        staged = Path(result['validated_archives']) / 'L06.zip'
        self.assertEqual(staged.read_bytes(), payload)
        self.assertEqual(archive.read_bytes(), before)
        with zipfile.ZipFile(staged) as z:
            self.assertEqual(z.namelist(), ['L06/L06_V001/001.json'])
            self.assertEqual(z.read(z.namelist()[0]), self.raw)
        self.assertEqual(result['total_records'], 1)

    def test_plain_zip_unchanged(self):
        payload = self.inner()
        (self.source / 'L06.zip').write_bytes(payload)
        result = prepare(self.source, self.root / 'mapping', self.root / 'out')
        self.assertEqual((Path(result['validated_archives']) / 'L06.zip').read_bytes(), payload)
        self.assertIsNone(result['groups'][0]['nested_member'])

    def test_unknown_nested_zip_does_not_get_executed_or_imported(self):
        with zipfile.ZipFile(self.source / 'L06.zip', 'w') as z:
            z.writestr('L06_OCR/unexpected.zip', self.inner())
        with self.assertRaisesRegex(ValueError, 'exactly one known'):
            prepare(self.source, self.root / 'mapping', self.root / 'out')


if __name__ == '__main__':
    unittest.main()
