"""Notebook setup fragment; appended after WORKER_SOURCE/LAUNCHER_SOURCE literals."""

# Load stdlib helpers only. Models run in separate children, never in the kernel.
helpers = {'__name__': 'notebook_helpers'}
exec(compile(WORKER_SOURCE, '<visible-worker>', 'exec'), helpers)
root = helpers['safe_path'](INPUT_ROOT)
ARCHIVE = Path('/kaggle/working/ocr_vietnamese_v3_batch16.zip')
helpers['inventory'](root)
if sys.version_info[:2] != (3, 13) or sys.platform != 'linux':
    raise RuntimeError('This notebook requires Kaggle Linux Python 3.13')
SCRATCH, OUTPUT, ENV_ROOT = helpers['select_scratch'](root, ARCHIVE)
helpers['invalidate_ready'](OUTPUT)
child_env = helpers['isolated_environment'](SCRATCH, os.environ)
child_env.update(FLAGS_allocator_strategy='auto_growth', KIS_VI_OCR_MIN_CONFIDENCE=str(PUBLIC_MIN_CONFIDENCE))
BOOTSTRAP = SCRATCH / 'isolated-launcher.py'
WORKER = OUTPUT / 'worker.py'
helpers['atomic_bytes'](BOOTSTRAP, LAUNCHER_SOURCE.encode('utf-8'))
helpers['atomic_bytes'](WORKER, WORKER_SOURCE.encode('utf-8'))
import pip
PIP_SOURCE = Path(pip.__file__).resolve().parent  # Existing host pip, not its other packages.


def checked_step(command, label):
    print('\n' + label, flush=True)
    # Preserve installation progress in Save Version logs; dependency check
    # diagnostics must not be replaced by an opaque CalledProcessError alone.
    result = subprocess.run(command, env=child_env, cwd=SCRATCH, check=False)
    if result.returncode:
        raise RuntimeError(f'{label} failed (exit {result.returncode}); see complete pip output above. No dependency gate bypassed.')


def prepare_environment(path, pins, role):
    path = helpers['safe_path'](path)
    helpers['safe_runtime_tree'](path)
    if role == 'torch':
        marker, expected = helpers['runtime_identity'](path)
    else:
        marker = path / 'split-runtime.json'
        expected = dict(schema='vietnamese-paddle-runtime-v2', pins=pins)
        if path.exists() and any(path.iterdir()):
            if not marker.is_file() or helpers['read_json'](marker) != expected:
                raise RuntimeError('Unowned/incompatible Paddle venv; refusing modification')
    if not marker.exists():
        helpers['atomic_json'](marker, expected)
    python = path / 'bin/python'
    if not python.exists():
        checked_step([sys.executable, '-I', '-S', '-m', 'venv', '--without-pip', '--copies', str(path)],
                     f'Create {role} runtime (no ensurepip or global sitecustomize)')
    constraints = path / 'core-constraints.txt'
    helpers['atomic_bytes'](constraints, ('\n'.join(f'{k}=={v}' for k, v in pins.items()) + '\n').encode())
    prefix = [str(python), '-I', '-S', str(BOOTSTRAP), 'pip', str(PIP_SOURCE), '--isolated', '--disable-pip-version-check']
    if role == 'paddle':
        packages = [f'{k}=={v}' for k, v in pins.items() if k != 'paddlepaddle-gpu']
        packages += ['https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.2.2-cp313-cp313-linux_x86_64.whl']
        indexes = ['--index-url', 'https://pypi.org/simple']
    else:
        packages = [f'{k}=={v}' for k, v in pins.items()]
        indexes = ['--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cu126']
    checked_step(prefix + ['install', '--only-binary=:all:', '--no-cache-dir', '--constraint', str(constraints),
                          *indexes, *packages], f'Install {role} dependencies ONLY into {path}')
    checked_step(prefix + ['check'], f'Validate {role} dependencies with pip check')
    return python


PADDLE_ENV_ROOT = SCRATCH / 'runtime-paddle'
PADDLE_PYTHON = prepare_environment(PADDLE_ENV_ROOT, helpers['PADDLE_PINS'], 'paddle')
PYTHON = prepare_environment(ENV_ROOT, helpers['PINS'], 'torch')
child_env.update(KIS_VI_PADDLE_PYTHON=str(PADDLE_PYTHON), KIS_VI_LAUNCHER=str(BOOTSTRAP))
WORKER_COMMAND = [str(PYTHON), '-I', '-S', str(BOOTSTRAP), 'worker', str(WORKER)]
print('Two separate runtimes validated. NVIDIA dependencies are NOT shared. GPU/model verification follows in preview.', flush=True)
