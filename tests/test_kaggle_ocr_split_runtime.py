import ast
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import venv
import zipfile

import numpy as np

from tools.build_kaggle_ocr_vietnamese_notebook import ROOT, build, worker_source


class SplitRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.worker = worker_source()[3]
        self.ns = {'__name__': 'split_test', '__file__': str(ROOT / 'tools/kaggle_ocr_vietnamese_extension.py')}
        exec(compile(self.worker, '<split-worker>', 'exec'), self.ns)

    def test_disjoint_torch_and_paddle_dependency_sets(self):
        torch, paddle = self.ns['PINS'], self.ns['PADDLE_PINS']
        self.assertNotIn('paddlepaddle-gpu', torch)
        self.assertNotIn('paddleocr', torch)
        self.assertNotIn('paddlex', torch)
        self.assertNotIn('torch', paddle)
        self.assertNotIn('torchvision', paddle)
        self.assertEqual(torch['torch'], '2.8.0+cu126')
        self.assertEqual(paddle['paddlepaddle-gpu'], '3.2.2')
        tree = ast.parse(self.worker)
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Adapter'][-1]
        imports = [n for n in ast.walk(cls) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(getattr(n, 'module', '') in {'paddle', 'paddlex', 'paddleocr'}
                             or any(x.name in {'paddle', 'paddlex', 'paddleocr'} for x in n.names) for n in imports))
        detector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'detector_worker')
        self.assertFalse(any(isinstance(n, ast.Import) and any(x.name == 'torch' for x in n.names) for n in ast.walk(detector)))

    def test_old_mixed_runtime_not_reused_and_all_workers_use_launcher(self):
        with tempfile.TemporaryDirectory() as d, patch('builtins.print'):
            document = build(Path(d) / 'notebook.ipynb')
        setup = ''.join(document['cells'][3]['source'])
        self.assertIn('ocr-vietnamese-lines-v3-', self.worker)
        self.assertIn("SCRATCH / 'runtime-paddle'", setup)
        self.assertIn("'-I', '-S', str(BOOTSTRAP)", setup)
        self.assertIn("prefix + ['check']", setup)
        self.assertNotIn("sys.executable, '-I', '-m', 'pip'", setup)
        for i in (5, 7, 9):
            source = ''.join(document['cells'][i]['source'])
            self.assertIn('WORKER_COMMAND', source)
            self.assertNotIn("[str(PYTHON), '-I'", source)

    def test_request_arrays_keep_bytes_shape_and_reject_pickle_nonimages(self):
        array = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        restored = self.ns['unpack_array'](self.ns['pack_array'](array))
        np.testing.assert_array_equal(array, restored)
        for invalid in (np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                self.ns['pack_array'](invalid)
        with self.assertRaises(ValueError):
            self.ns['unpack_array']('not-base64!')

    def test_detector_reply_alignment_and_child_errors_not_empty_results(self):
        detector = self.ns['DetectorProcess'].__new__(self.ns['DetectorProcess'])
        detector.sequence = 0
        detector.process = SimpleNamespace(stdin=StringIO())
        array = np.zeros((4, 6, 3), dtype=np.uint8)
        polygon = [[0, 0], [5, 0], [5, 3], [0, 3]]
        detector.read = Mock(return_value=dict(status='ok', id=1, polygons=[polygon], crops=[self.ns['pack_array'](array)]))
        polygons, crops = detector.detect(array)
        self.assertEqual(polygons, [polygon])
        np.testing.assert_array_equal(crops[0], array)
        self.assertEqual(json.loads(detector.process.stdin.getvalue())['id'], 1)
        detector.read.return_value = dict(status='error', id=2)
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            detector.detect(array)
        detector.read.return_value = dict(status='ok', id=3, polygons=[polygon], crops=[])
        with self.assertRaisesRegex(RuntimeError, 'count'):
            detector.detect(array)

    def test_detector_protocol_real_persistent_subprocess_no_gpu(self):
        child = """import sys,json
print(json.dumps({'status':'ready','profile':{'test_only':True}}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'status':'ok','id':r['id'],'polygons':[[[0,0],[1,0],[1,1],[0,1]]],'crops':[r['image']]}),flush=True)
"""
        real_popen = subprocess.Popen
        def factory(command, **kwargs):
            self.assertIn('-S', command)
            self.assertEqual(command[-1], 'detector-worker')
            return real_popen([sys.executable, '-I', '-u', '-c', child], **kwargs)
        with patch.dict(os.environ, KIS_VI_PADDLE_PYTHON=sys.executable,
                        KIS_VI_LAUNCHER=str(ROOT / 'tools/kaggle_ocr_isolated_launcher.py')), \
                patch('subprocess.Popen', side_effect=factory) as popen:
            detector = self.ns['DetectorProcess']()
            try:
                for size in (4, 7):
                    value = np.full((size, size, 3), size, dtype=np.uint8)
                    _, crops = detector.detect(value)
                    self.assertEqual(crops[0].shape, value.shape)
                    self.assertEqual(crops[0].tobytes(), value.tobytes())
                self.assertEqual(detector.sequence, 2)
                popen.assert_called_once()
            finally:
                detector.close()
            self.assertEqual(detector.process.returncode, 0)

    def test_real_detector_worker_handles_sorter_list_array_and_empty_output(self):
        # Execute the ACTUAL worker loop in a child, not a replacement JSON
        # server. Only GPU/models are stubbed. PaddleX's sorter returns a list.
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'worker-fixture.py'
            harness = r'''
import sys,types,os
import numpy as np
ns={'__name__':'fixture','__file__':__file__}
exec(compile(WORKER,'<actual-worker>','exec'),ns)
for name in ('paddle','paddleocr','paddlex','paddlex.inference','paddlex.inference.pipelines','paddlex.inference.pipelines.components'):
    sys.modules[name]=types.ModuleType(name)
polygon=np.array([[0,0],[2,0],[2,2],[0,2]],dtype=np.float32)
class Detection:
    def __init__(self,**kwargs):
        self.paddlex_predictor=types.SimpleNamespace(model_name='PP-OCRv5_server_det',
            pp_option=types.SimpleNamespace(device_type='gpu',device_id=0))
        os.write(1,b'native detector startup log\n')
    def predict(self,input):
        mode=int(input[0,0,0])
        current_mode[0]=mode
        return [{'dt_polys':np.array([polygon,polygon+3]) if mode in (1,2) else np.empty((0,4,2))}]
class Sort:
    def __call__(self,polys):
        if len(polys) and polys[0,0,0] == 0:
            # Each request mode sets which supported API form is returned.
            return list(reversed(polys)) if current_mode[0]==1 else polys[::-1]
        return []
class Crop:
    def __init__(self,**kwargs):pass
    def __call__(self,image,polygons):
        current_mode[0]=int(image[0,0,0])
        return [np.full((2,2,3),int(p[0,0])+10,dtype=np.uint8) for p in polygons]
current_mode=[1]
sys.modules['paddleocr'].TextDetection=Detection
components=sys.modules['paddlex.inference.pipelines.components']
components.SortQuadBoxes=Sort
components.CropByPolys=Crop
ns['require_gpu']=lambda _:dict(device='gpu:0',test_only=True)
ns['fetch_asset']=lambda *a,**kw:None
ns['file_digest']=lambda _: 'test-hash'
ns['importlib'].metadata.version=lambda name:ns['PADDLE_PINS'][name]
ns['detector_worker']()
'''
            source.write_text('WORKER = ' + repr(self.worker) + '\n' + harness, encoding='utf-8')
            requests = ''.join(json.dumps(dict(action='detect', id=i, image=self.ns['pack_array'](
                np.full((6, 6, 3), mode, dtype=np.uint8)))) + '\n' for i, mode in enumerate((1, 2, 0), 1))
            result = subprocess.run([sys.executable, '-I', str(source)], input=requests,
                                    capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('native detector startup log', result.stderr)
            replies = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(replies[0]['status'], 'ready')
            self.assertEqual([r['id'] for r in replies[1:]], [1, 2, 3])
            polygon = np.array([[0,0],[2,0],[2,2],[0,2]], dtype=float)
            expected = [(polygon + 3).tolist(), polygon.tolist()]
            for reply in replies[1:3]:
                self.assertEqual(reply['polygons'], expected)
                crops = [self.ns['unpack_array'](c) for c in reply['crops']]
                self.assertEqual([int(c[0,0,0]) for c in crops], [13, 10])
            self.assertEqual(replies[3]['polygons'], [])
            self.assertEqual(replies[3]['crops'], [])

    def test_launcher_real_venv_pip_check_and_host_hook_exclusion(self):
        # Actual Python subprocesses and real pip, no install/network/model/GPU.
        import pip
        launcher = ROOT / 'tools/kaggle_ocr_isolated_launcher.py'
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            environment = base / 'venv'
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            host = base / 'host-hooks'
            host.mkdir()
            (host / 'sitecustomize.py').write_text("raise RuntimeError('forbidden global wrapt hook')\n", encoding='utf-8')
            probe = base / 'probe.py'
            probe.write_text("import json,sys,importlib.util\nprint(json.dumps({'prefix':sys.prefix,'customize': 'sitecustomize' in sys.modules, 'numpy':importlib.util.find_spec('numpy') is not None}))\n", encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=str(host), PYTHONHOME=str(host))
            result = subprocess.run([str(python), '-I', '-S', str(launcher), 'worker', str(probe)],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            self.assertEqual(Path(state['prefix']), environment)
            self.assertFalse(state['customize'])
            self.assertFalse(state['numpy'])
            wheel = base / 'kis_isolation_probe-0.0.1-py3-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as z:
                z.writestr('kis_isolation_probe/__init__.py', 'VALUE = 42\n')
                z.writestr('kis_isolation_probe-0.0.1.dist-info/METADATA',
                           'Metadata-Version: 2.1\nName: kis-isolation-probe\nVersion: 0.0.1\n')
                z.writestr('kis_isolation_probe-0.0.1.dist-info/WHEEL',
                           'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
                z.writestr('kis_isolation_probe-0.0.1.dist-info/RECORD', '')
            result = subprocess.run([str(python), '-I', '-S', str(launcher), 'pip', str(Path(pip.__file__).parent),
                                     'install', '--no-index', str(wheel)],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            packages = environment / ('Lib/site-packages' if os.name == 'nt' else
                f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
            self.assertTrue((packages / 'kis_isolation_probe/__init__.py').is_file())
            result = subprocess.run([str(python), '-I', '-S', str(launcher), 'pip', str(Path(pip.__file__).parent), 'check'],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn('No broken requirements', result.stdout)
            # Reproduce the reported cuDNN metadata conflict without installing
            # either GPU package; the isolated pip gate MUST still fail.
            for name, version, requirement in [('paddle-test', '3.2.2', 'nvidia-cudnn-cu12==9.5.1.17'),
                                               ('nvidia-cudnn-cu12', '9.10.2.21', None)]:
                dist = packages / (name.replace('-', '_') + '-' + version + '.dist-info')
                dist.mkdir()
                (dist / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
                    + (f'Requires-Dist: {requirement}\n' if requirement else ''), encoding='utf-8')
            result = subprocess.run([str(python), '-I', '-S', str(launcher), 'pip', str(Path(pip.__file__).parent), 'check'],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
            self.assertIn('9.5.1.17', result.stdout)
            self.assertIn('9.10.2.21', result.stdout)
            # Paddle's own cuDNN version restores a valid independent graph.
            old = packages / 'nvidia_cudnn_cu12-9.10.2.21.dist-info'
            correct = packages / 'nvidia_cudnn_cu12-9.5.1.17.dist-info'
            old.rename(correct)
            (correct / 'METADATA').write_text('Metadata-Version: 2.1\nName: nvidia-cudnn-cu12\nVersion: 9.5.1.17\n', encoding='utf-8')
            result = subprocess.run([str(python), '-I', '-S', str(launcher), 'pip', str(Path(pip.__file__).parent), 'check'],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == '__main__':
    unittest.main()
