import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os
import subprocess
import sys
import zipfile

from PIL import Image
from tools.prepare_video_data import VideoDatasetBuilder
from tools.kaggle_prepare_keyframes import inspect_inputs, register_keyframes, archive_artifacts, run


class ExistingKeyframeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.images = self.root / 'input/data/L03/L03_V001'
        self.images.mkdir(parents=True)
        self.mapping_root = self.root / 'input/mapping'
        self.mapping_root.mkdir()
        self.csv = self.mapping_root / 'L03_V001.csv'
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n2,5.52,25,138\n', encoding='utf-8')
        for n in (1, 2):
            Image.new('RGB', (32, 24), 'red').save(self.images / f'{n:04d}.JPEG')
        self.builder = VideoDatasetBuilder(self.root / 'output')

    def plans(self):
        return inspect_inputs(self.root / 'input/data', self.mapping_root)

    def test_import_preserves_mapping_image_bytes_and_resume(self):
        plan = self.plans()['L03_V001']
        with patch.object(VideoDatasetBuilder, 'extract_keyframes', side_effect=AssertionError('no video')):
            register_keyframes(self.builder, 'L03_V001', plan)
            register_keyframes(self.builder, 'L03_V001', plan)
        paths = self.builder.artifact_paths('L03_V001')
        self.assertEqual(paths['mapping'].read_bytes(), self.csv.read_bytes())
        self.assertEqual((paths['images'] / '001.jpg').read_bytes(), (self.images / '0001.JPEG').read_bytes())
        self.assertEqual(self.builder.mapping_rows('L03_V001')[1]['frame_idx'], 138)

    def test_missing_csv_and_missing_keyframe_fail_without_guessing(self):
        self.csv.unlink()
        with self.assertRaisesRegex(ValueError, 'original mapping'):
            self.plans()
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.plans()

    def test_invalid_frame_ids_and_nonfinite_time_fail(self):
        for row in ('2,5.52,25,-1', '2,nan,25,138', '2,5.52,0,138', '3,5.52,25,138'):
            self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n' + row + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                self.plans()

    def test_frame_alias_import_preserves_both_images_csv_and_identity(self):
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,30,0\n2,0.0333333,30,0\n', encoding='utf-8')
        Image.new('RGB', (32, 24), 'blue').save(self.images / '0002.JPEG')
        original_csv = self.csv.read_bytes()
        plan = self.plans()['L03_V001']
        self.assertEqual(plan['frame_idx_aliases'], [{'frame_idx': 0, 'keyframe_ns': [1, 2]}])
        register_keyframes(self.builder, 'L03_V001', plan)
        register_keyframes(self.builder, 'L03_V001', plan)  # Resume keeps aliases.
        rows = self.builder.mapping_rows('L03_V001')
        self.assertEqual([(r['n'], r['frame_idx']) for r in rows], [(1, 0), (2, 0)])
        paths = self.builder.artifact_paths('L03_V001')
        self.assertEqual(paths['mapping'].read_bytes(), original_csv)
        self.assertEqual((paths['images'] / '002.jpg').read_bytes(), (self.images / '0002.JPEG').read_bytes())
        self.assertNotEqual((paths['images'] / '001.jpg').read_bytes(), (paths['images'] / '002.jpg').read_bytes())
        from tools.audit_artifact_formats import audit_video_formats
        audit = audit_video_formats(self.builder.root, 'L03_V001', allow_incomplete=True)
        mapping = next(r for r in audit if r['folder'] == 'mapping')
        self.assertEqual(mapping['status'], 'OK')
        self.assertIn('WARNING: 1 frame_idx aliases', mapping['detail'])

    def test_frame_alias_does_not_relax_video_extraction_checkpoint(self):
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,30,0\n2,0.0333333,30,0\n', encoding='utf-8')
        register_keyframes(self.builder, 'L03_V001', self.plans()['L03_V001'])
        state = self.builder._state('L03_V001')
        state['source'] = {'name': 'source.mp4'}
        self.builder._save_state('L03_V001', state)
        with self.assertRaisesRegex(ValueError, 'Invalid mapping order'):
            self.builder.mapping_rows('L03_V001')
        from tools.audit_artifact_formats import audit_video_formats
        audit = audit_video_formats(self.builder.root, 'L03_V001', allow_incomplete=True)
        self.assertEqual(next(r['status'] for r in audit if r['folder'] == 'mapping'), 'ERROR')

    def test_decreasing_frame_index_still_rejected(self):
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,30,10\n2,1,30,9\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            self.plans()

    @unittest.skipUnless(importlib.util.find_spec('faiss'), 'requires FAISS artifact verification')
    def test_aliases_survive_all_artifact_stages_with_fixture_models(self):
        import numpy as np
        import faiss  # Keep SWIG modules resident across the fake embedder context.
        from kis.config import VISUAL_CONFIG, CAPTION_CONFIG
        from kis.structured_caption import caption_generation_config
        from tools.audit_artifact_formats import audit_video_formats

        def vectors(count, dimension):
            result = np.zeros((count, dimension), dtype=np.float32)
            result[:, 0] = 1
            return result

        class Captioner:
            config = caption_generation_config('internvl', False, 'fp32')
            def describe_batch(self, images):
                return ['A red test image.', 'A blue test image.'][:len(images)]

        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,30,0\n2,0.0333333,30,0\n', encoding='utf-8')
        register_keyframes(self.builder, 'L03_V001', self.plans()['L03_V001'])
        self.builder.encode_visual_features(['L03_V001'], encoder=SimpleNamespace(
            encode_images=lambda images: vectors(len(images), int(VISUAL_CONFIG['dimension']))))
        self.builder.generate_objects(['L03_V001'], detector=SimpleNamespace(
            detect=lambda images, threshold: [{'detection_class_entities': [], 'detection_scores': [],
                                              'detection_boxes': []} for _ in images]))
        with patch('tools.structured_caption_generator.resolve_caption_backend_runtime',
                   return_value={'use_4bit': False, 'compute_dtype': 'fp32'}), \
             patch('kis.light_caption.InternVLCaptioner', return_value=Captioner()):
            self.builder.generate_captions(['L03_V001'], batch_size=2)
        embedder = SimpleNamespace(encode=lambda texts, **kw: vectors(len(texts), int(CAPTION_CONFIG['dimension'])))
        with patch.dict(sys.modules, {'sentence_transformers': SimpleNamespace(SentenceTransformer=lambda *a, **kw: embedder)}):
            self.builder.build_caption_indexes(['L03_V001'])
        self.assertEqual(self.builder.validate_video('L03_V001')['caption_vectors'], 2)
        audit = audit_video_formats(self.builder.root, 'L03_V001')
        self.assertTrue(all(r['status'] == 'OK' for r in audit), audit)
        mapping = json.loads(self.builder.artifact_paths('L03_V001')['caption_mapping'].read_text())
        self.assertEqual([(r['frame_id'], r['keyframe_n']) for r in mapping], [(0, 1), (0, 2)])
        self.assertNotEqual(mapping[0]['image_path'], mapping[1]['image_path'])

    def test_duplicate_image_number_and_corrupt_image_fail(self):
        Image.new('RGB', (32, 24)).save(self.images / '001.jpg')
        with self.assertRaisesRegex(ValueError, 'Duplicate keyframe'):
            self.plans()
        (self.images / '001.jpg').unlink()
        (self.images / '0001.JPEG').write_bytes(b'corrupt')
        with self.assertRaises(OSError):
            self.plans()

    def test_numbered_copy_suffixes_are_ignored_without_changing_originals(self):
        originals = self.plans()['L03_V001']
        copies = ['0001 (1).jpg', '0001 (2).JPEG', '0002 (12).JPG', 'L03_V001 0001 (1).jpg']
        for name in copies:
            # Ignored copies may have different or corrupt bytes; never infer
            # identity/content equality from the filename or replace the original.
            (self.images / name).write_bytes(b'ignored duplicate')
        plan = self.plans()['L03_V001']
        self.assertEqual(plan['images'], originals['images'])
        self.assertEqual(plan['image_sha256'], originals['image_sha256'])
        self.assertEqual(set(plan['ignored_duplicate_files']), set(copies))
        register_keyframes(self.builder, 'L03_V001', plan)
        self.assertEqual({p.name for p in self.builder.artifact_paths('L03_V001')['images'].iterdir()},
                         {'001.jpg', '002.jpg'})
        self.assertTrue(all((self.images / name).exists() for name in copies))

    def test_copy_does_not_fill_in_missing_original(self):
        (self.images / '0001.JPEG').rename(self.images / '0001 (1).JPEG')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.plans()

    def test_unrelated_non_numeric_filename_still_fails(self):
        (self.images / 'holiday (1).jpg').write_bytes(b'not a keyframe')
        with self.assertRaisesRegex(ValueError, 'Expected numbered keyframes'):
            self.plans()

    def test_changed_input_or_output_not_overwritten(self):
        plan = self.plans()['L03_V001']
        register_keyframes(self.builder, 'L03_V001', plan)
        path = self.builder.artifact_paths('L03_V001')['images'] / '001.jpg'
        path.write_bytes(b'user work')
        with self.assertRaises(FileExistsError):
            register_keyframes(self.builder, 'L03_V001', plan)
        self.assertEqual(path.read_bytes(), b'user work')
        self.csv.write_text(self.csv.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            register_keyframes(self.builder, 'L03_V001', self.plans()['L03_V001'])

    def test_check_only_never_loads_models_or_creates_output(self):
        with patch('tools.kaggle_prepare_keyframes.configure_device', side_effect=AssertionError('no load')):
            run(self.root / 'input/data', self.mapping_root, self.builder.root, check_only=True)
        self.assertFalse(self.builder.root.exists())
        with self.assertRaisesRegex(ValueError, 'separate'):
            run(self.root / 'input/data', self.mapping_root, self.root / 'input/data/output', check_only=True)

    def test_archive_excludes_images_by_default_and_keeps_mapping_checkpoint(self):
        register_keyframes(self.builder, 'L03_V001', self.plans()['L03_V001'])
        archive = archive_artifacts(self.builder, ['L03_V001'], 'partial')
        with zipfile.ZipFile(archive) as z:
            self.assertFalse(any(n.startswith('data/') for n in z.namelist()))
            self.assertTrue(any(n.startswith('mapping/') for n in z.namelist()))
            self.assertTrue(any(n.startswith('preparation/') for n in z.namelist()))
        archive = archive_artifacts(self.builder, ['L03_V001'], 'complete', True)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('data/L03/L03_V001/001.jpg', z.namelist())

    def test_notebook_code_compiles_and_uses_new_runner(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root / 'notebooks/keyframesprocessing.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        compile(code, 'notebook', 'exec')
        self.assertIn('tools.kaggle_prepare_keyframes', code)
        self.assertNotIn('--video-dir', code)
        self.assertNotIn('extract_keyframes', code)
        self.assertTrue(all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells']))

    def test_pipeline_keeps_artifact_stages_and_never_decodes_video(self):
        events = []
        def stage(name):
            def record(*args, **kwargs):
                events.append(name)
            return record
        with patch('tools.kaggle_prepare_keyframes.configure_device', return_value={}), \
             patch('tools.structured_caption_generator.resolve_caption_backend_runtime', return_value={'use_4bit': False, 'compute_dtype': 'fp32'}), \
             patch('tools.kaggle_prepare.release_models'), \
             patch('tools.kaggle_keyframes_caption.restart_caption_runtime_if_changed'), \
             patch.object(VideoDatasetBuilder, 'extract_keyframes', side_effect=AssertionError('no video')), \
             patch.object(VideoDatasetBuilder, 'encode_visual_features', side_effect=stage('features')), \
             patch.object(VideoDatasetBuilder, 'generate_objects', side_effect=stage('objects')), \
             patch.object(VideoDatasetBuilder, 'generate_captions', side_effect=stage('captions')), \
             patch.object(VideoDatasetBuilder, 'build_caption_indexes', side_effect=stage('index')), \
             patch.object(VideoDatasetBuilder, 'validate_video', return_value={'status': 'OK'}), \
             patch('tools.audit_artifact_formats.audit_video_formats', return_value=[{'status': 'OK'}]):
            archive = run(self.root / 'input/data', self.mapping_root, self.builder.root)
        self.assertEqual(events, ['features', 'objects', 'captions', 'index'])
        self.assertTrue(archive.exists())
        self.assertEqual(json.loads((self.builder.root / 'keyframes_run.json').read_text())['status'], 'complete')

    def test_model_failure_keeps_partial_zip_and_checkpoint(self):
        with patch('tools.kaggle_prepare_keyframes.configure_device', return_value={}), \
             patch('tools.structured_caption_generator.resolve_caption_backend_runtime', return_value={'use_4bit': False, 'compute_dtype': 'fp32'}), \
             patch('tools.kaggle_keyframes_caption.restart_caption_runtime_if_changed'), \
             patch.object(VideoDatasetBuilder, 'encode_visual_features', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                run(self.root / 'input/data', self.mapping_root, self.builder.root)
        self.assertEqual(json.loads((self.builder.root / 'keyframes_run.json').read_text())['status'], 'partial')
        self.assertTrue((self.root / 'output-from-keyframes-partial.zip').is_file())
        self.assertEqual(len(self.builder.mapping_rows('L03_V001')), 2)

    def test_packaged_runner_preflight_works_outside_repository(self):
        from tools.package_kaggle import package
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,30,0\n2,0.0333333,30,0\n', encoding='utf-8')
        archive = self.root / 'code.zip'
        package(archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.root / 'unpacked')
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('KIS_VIDEO_GROUPS', None)
        process = subprocess.run([sys.executable, '-B', '-m', 'tools.kaggle_prepare_keyframes',
            '--keyframes-root', str(self.root / 'input/data'), '--mapping-root', str(self.mapping_root),
            '--output-root', str(self.builder.root), '--check-only'],
            cwd=self.root / 'unpacked/kis_kaggle', env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('"preflight": "passed"', process.stdout)
        self.assertIn('WARNING: frame_idx aliases preserved', process.stdout)
        self.assertFalse(self.builder.root.exists())


if __name__ == '__main__':
    unittest.main()
