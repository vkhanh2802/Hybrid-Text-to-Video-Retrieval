"""Build the deterministic, self-contained Kaggle OCR v4 notebook (stdlib only)."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import tempfile


BASE = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/kaggle_ocr_paddle_padding.ipynb"
MAX_BYTES = 1024 * 1024
PAYLOAD_FILES = tuple(sorted((
    "tools/kaggle_ocr.py",
    "config/ocr-kaggle.json",
    "config/ocr-kaggle-py313.json",
    "config/requirements-kaggle-ocr.txt",
    "config/requirements-kaggle-ocr-py313.txt",
    "schemas/ocr-sidecar-v1.schema.json",
)))

# This code is embedded verbatim, not imported from the repository at runtime.
BOOTSTRAP_HELPERS = r'''from pathlib import Path, PurePosixPath
import base64
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import zipfile

MAX_PAYLOAD_BYTES = 1024 * 1024
ALLOWED_FILES = {
    'tools/kaggle_ocr.py',
    'config/ocr-kaggle.json',
    'config/ocr-kaggle-py313.json',
    'config/requirements-kaggle-ocr.txt',
    'config/requirements-kaggle-ocr-py313.txt',
    'schemas/ocr-sidecar-v1.schema.json',
}


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def safe_member(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute()
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or any(p in ('', '.', '..') or p.endswith(('.', ' '))
                   for p in name.split('/'))):
        raise ValueError('Unsafe relative POSIX payload path')
    if name not in ALLOWED_FILES:
        raise ValueError('Payload path is not allowlisted: ' + name)


def validate_payload(payload, expected_hash):
    if not isinstance(payload, dict) or set(payload) != {'format', 'files'}:
        raise ValueError('Malformed embedded payload')
    if payload['format'] != 'kis-kaggle-ocr-v4' or not isinstance(payload['files'], list):
        raise ValueError('Unsupported embedded payload')
    if len(payload['files']) != len(ALLOWED_FILES):
        raise ValueError('Payload allowlist membership mismatch')
    decoded = {}
    for entry in payload['files']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'size', 'sha256', 'base64'}:
            raise ValueError('Malformed payload entry')
        name = entry['path']
        safe_member(name)
        if name in decoded:
            raise ValueError('Duplicate payload path')
        size, checksum, text = entry['size'], entry['sha256'], entry['base64']
        if (type(size) is not int or not 0 < size <= MAX_PAYLOAD_BYTES
                or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)
                or not isinstance(text, str) or len(text) > 4 * ((size + 2) // 3)):
            raise ValueError('Invalid payload size/hash/base64')
        data = base64.b64decode(text, validate=True)
        if (len(data) != size or base64.b64encode(data).decode('ascii') != text
                or hashlib.sha256(data).hexdigest() != checksum):
            raise ValueError('Payload bytes/hash/size mismatch: ' + name)
        if '\r' in data.decode('utf-8') or data.startswith(b'\xef\xbb\xbf') or not data.endswith(b'\n'):
            raise ValueError('Payload must use canonical UTF-8/LF')
        decoded[name] = data
    if set(decoded) != ALLOWED_FILES or list(decoded) != sorted(ALLOWED_FILES):
        raise ValueError('Payload allowlist/order mismatch')
    encoded = canonical_json(payload)
    if len(encoded) > MAX_PAYLOAD_BYTES or sum(map(len, decoded.values())) > MAX_PAYLOAD_BYTES:
        raise ValueError('Embedded payload exceeds 1 MiB')
    if not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash):
        raise ValueError('Malformed payload hash')
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError('Embedded payload hash mismatch')
    return decoded


def reject_links(path):
    # lstat also catches Windows junctions/reparse points on Python 3.10/3.11.
    path = Path(path).absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (part.is_symlink() or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400
                or (hasattr(part, 'is_junction') and part.is_junction())):
            raise ValueError('Payload/work symlink/junction/reparse point forbidden')
    return path


def materialize_payload(payload, expected_hash, working_base):
    files = validate_payload(payload, expected_hash)  # Validate ALL bytes before any write/import.
    base = reject_links(working_base)
    root = reject_links(base / expected_hash / 'code')
    if not root.resolve().is_relative_to(base.resolve()):
        raise ValueError('Payload target escapes working base')
    allowed_dirs = {str(PurePosixPath(n).parent) for n in files}
    if root.exists():
        for path in root.rglob('*'):
            reject_links(path)
            relative = path.relative_to(root).as_posix()
            if path.is_dir() and relative in allowed_dirs:
                continue
            if relative not in files or not path.is_file():
                raise ValueError('Unexpected materialized payload member')
    # Check all cached targets before creating/reusing any source file. Never overwrite tampering.
    for name, data in files.items():
        target = reject_links(root / name)
        if target.exists() and (not target.is_file() or target.stat().st_size != len(data)
                                or target.read_bytes() != data):
            raise ValueError('Materialized payload tamper: ' + name)
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = reject_links(root / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
        reject_links(target)
        if target.read_bytes() != data:
            raise ValueError('Materialized payload changed: ' + name)
    return root
'''

BOOTSTRAP_RUN = '''RUNTIME_READY = False
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError('DEPENDENCY_UNAVAILABLE: supported Python 3.10-3.13; ' + json.dumps({
        'python': sys.version, 'executable': sys.executable,
        'platform': sys.platform, 'profile': 'unsupported'}))
root = materialize_payload(EMBEDDED_PAYLOAD, PAYLOAD_SHA256,
                           Path('/kaggle/working/kis-kaggle-ocr-v4'))
work = root.parent
script = root / 'tools/kaggle_ocr.py'
spec = importlib.util.spec_from_file_location('portable_kaggle_ocr_v4', script)
ocr = importlib.util.module_from_spec(spec)
previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True  # Keep the materialized code tree limited to six files.
    spec.loader.exec_module(ocr)
finally:
    sys.dont_write_bytecode = previous_bytecode_setting
profile = ocr.select_profile()
config_path = root / profile['config']
print(json.dumps({'payload_sha256': PAYLOAD_SHA256,
                  'interpreter': ocr.interpreter_fingerprint(), 'profile': profile}, indent=2))
subprocess.run([sys.executable, '-B', str(script), 'validate-input',
                '--input-root', INPUT_ROOT, '--config', str(config_path)],
               env=ocr.runtime_environment(), check=True)
print('Embedded integrity and input layout validated. No package Dataset or OCR model used.')
'''

DEPENDENCIES = '''RUNTIME_READY = False
wheelhouses = [p for p in Path('/kaggle/input').rglob('ocr-wheelhouse') if p.is_dir()]
if len(wheelhouses) > 1:
    raise ValueError('Attach at most one ocr-wheelhouse')
reject_links(work / 'venv')
wheelhouse = reject_links(wheelhouses[0]) if wheelhouses else None
python, config_path, identity = ocr.prepare_runtime(work, wheelhouse)
print(json.dumps(identity, indent=2))
models = [p for p in Path('/kaggle/input').rglob('ocr-models') if p.is_dir()]
if len(models) > 1:
    raise ValueError('Attach at most one ocr-models folder')
runtime_env = ocr.runtime_environment(python, sys.executable)
runtime_env.pop('KIS_OCR_MODEL_ROOT', None)
runtime_env.pop('KIS_OCR_OFFLINE', None)
if models:
    runtime_env['KIS_OCR_MODEL_ROOT'] = str(reject_links(models[0]))
    runtime_env['KIS_OCR_OFFLINE'] = '1'
RUNTIME_READY = True
print('venv-host-pip: binary resolver, pip check and import smoke passed; no model initialized.')
'''

RUN = '''if not globals().get('RUNTIME_READY', False):
    raise RuntimeError('DEPENDENCY_UNAVAILABLE: dependency gate must pass before output')
output = reject_links(work / 'outputs/ocr-v1')
if output.exists():
    for path in output.rglob('*'):
        reject_links(path)
command = [str(python), '-B', str(script), 'run', '--input-root', INPUT_ROOT,
           '--output-root', str(output), '--config', str(config_path)]
if (output / 'manifest.json').exists():
    command += ['--resume']  # Matching-only resume; no silent error retry.
completed = subprocess.run(command, env=runtime_env, cwd=root, check=False)
if completed.returncode != 0:
    raise RuntimeError('OCR_STOPPED: inspect error above; unavailable models are not empty OCR.')
subprocess.run([str(python), '-B', str(script), 'validate-output', '--input-root', INPUT_ROOT,
                '--output-root', str(output)], cwd=root, env=runtime_env, check=True)
coverage = json.loads((output / 'coverage.json').read_text(encoding='utf-8'))
print(json.dumps(coverage, indent=2))
if not coverage['complete']:
    print('INCOMPLETE: preserve missing/error/unknown. Explicit retry: --resume --retry-errors.')
'''

DOWNLOAD = '''if not globals().get('RUNTIME_READY', False):
    raise RuntimeError('DEPENDENCY_UNAVAILABLE: no output package before dependency gate')
run_hash = hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()[:16]
archive = reject_links(work / ('ocr-sidecar-v1-' + run_hash + '.zip'))
if not archive.exists():
    subprocess.run([str(python), '-B', str(script), 'package', '--input-root', str(output),
                    '--output-root', str(archive)], cwd=root, env=runtime_env, check=True)
else:
    ocr.verify_zip(archive)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.read('manifest.json') != (output / 'manifest.json').read_bytes():
            raise ValueError('Existing ZIP belongs to a different run')
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
checksum_path = reject_links(Path(str(archive) + '.sha256'))
checksum_path.write_bytes((checksum + '  ' + archive.name + '\\n').encode('utf-8'))
print('ZIP SHA256:', checksum, 'bytes:', archive.stat().st_size)
from IPython.display import FileLink, display
display(FileLink(str(archive)))
display(FileLink(str(checksum_path)))
'''

OVERVIEW = '''# PaddleOCR + 32 px padding — self-contained v4 / sidecar v1

1. **Import this notebook** into Kaggle.
2. **Attach only your keyframes Dataset** (no code/package ZIP Dataset).
3. **Enable Internet** for pinned wheels/models; GPU is optional, CPU is the default.
4. **Edit only `INPUT_ROOT`** above, then **Run All**. Download the final output ZIP and checksum.

Video/group layouts are detected automatically, independent of Dataset slug:
`L99_V003/1.jpg`, `L99/L99_V003/1.jpg`, a group root, or a video root.
JPEG/PNG filenames must be numeric. Optional per-video `mapping.csv` uses
`n,frame_idx,pts_time`; `keyframe_n` is **not** `frame_idx`.

Six small source/config/schema files are embedded as JSON/base64 with SHA256.
No credentials, data, weights, docs or outputs are embedded. Bootstrap never reads
an external package ZIP/manifest or imports the production repo. Hashes detect
corruption, not malicious replacement of a notebook: import only a trusted copy.

Python 3.10–3.12 keeps NumPy 1.26.4; Python 3.13 automatically selects NumPy 2.2.6
with PaddlePaddle 3.3.1 and PaddleOCR/PaddleX 3.7.0. Venv inherits kernel Python.
Kaggle Python 3.13 may lack ensurepip: mode `venv-host-pip` creates the venv
without pip, then uses kernel `python -m pip --python <venv-python>` to target it.
Kernel pip must support `--python`, `--dry-run` and `--report`, otherwise stop.
No ensurepip, global/kernel install, pip upgrade or source build is attempted.
The 3.13 profile remains a **candidate pending a real Kaggle run**, not locally
inference/ABI verified. Unsupported Python, binary resolver or import failure
stops before models/output. No source build, unpinning or accuracy claim.

Advanced offline use only: optionally attach complete matching `ocr-wheelhouse`
and revision/hash-locked `ocr-models` caches (see `docs/KAGGLE_OCR.md` in source).
No caches are needed for the normal Internet-enabled, keyframes-only flow.
GPU requires a separately verified GPU profile/wheel, not just the Kaggle GPU switch.

Default: white 32 px on every side, no wrapper resize; exact inverse coordinates
and separate clipped drawing coordinates. Statuses `ok/empty/missing/error/unknown`
remain distinct. Run All again resumes only matching input/config/model/source
hashes; use a fresh session if input changes. Errors are not silently retried.
Outputs stay under `/kaggle/working`; download is the only export. No captions,
FAISS, Milvus or production artifacts are written. Future joins must verify
`(video_id,keyframe_n,image_sha256)`; incomplete coverage is not negative evidence.
'''


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def source_bytes(path):
    text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return (text.rstrip("\n") + "\n").encode("utf-8")


def helpers_namespace():
    namespace = {}
    exec(compile(BOOTSTRAP_HELPERS, "embedded-bootstrap-helpers", "exec"), namespace)
    return namespace


def build_notebook(base=BASE):
    files = []
    helpers = helpers_namespace()
    for name in PAYLOAD_FILES:
        path = Path(base) / name
        helpers["reject_links"](path)
        data = source_bytes(path)
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "base64": base64.b64encode(data).decode("ascii")})
    payload = {"format": "kis-kaggle-ocr-v4", "files": files}
    checksum = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    helpers["validate_payload"](payload, checksum)
    embedded = ("EMBEDDED_PAYLOAD = json.loads(" + repr(canonical_bytes(payload).decode("utf-8"))
                + ")\nPAYLOAD_SHA256 = " + repr(checksum) + "\n")

    def cell(kind, identity, source):
        result = {"cell_type": kind, "id": identity, "metadata": {},
                  "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    notebook = {
        "cells": [
            cell("code", "input-root", 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n'),
            cell("markdown", "overview-v4", OVERVIEW),
            cell("code", "embedded-bootstrap-v4", BOOTSTRAP_HELPERS + "\n" + embedded + BOOTSTRAP_RUN),
            cell("code", "isolated-dependencies", DEPENDENCIES),
            cell("code", "run-and-validate", RUN),
            cell("markdown", "download-help", "## Download output\n"
                 "Validate/package only sidecars. Review coverage, then download **both** links below.\n"),
            cell("code", "download-zip", DOWNLOAD),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    data = (json.dumps(notebook, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    verify_notebook(data)
    return data


def verify_notebook(data):
    if len(data) > MAX_BYTES or b"\r" in data:
        raise ValueError("Notebook must be UTF-8/LF and <= 1 MiB")
    notebook = json.loads(data.decode("utf-8"))
    codes = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if cell["outputs"] or cell["execution_count"] is not None:
                raise ValueError("Notebook must have empty outputs")
            code = "".join(cell["source"])
            compile(code, f"notebook-cell-{index}", "exec")
            codes.append(code)
    if codes[0] != 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n':
        raise ValueError("First cell must only assign INPUT_ROOT")
    tree = ast.parse(codes[1])
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "EMBEDDED_PAYLOAD":
                values[name] = json.loads(ast.literal_eval(node.value.args[0]))
            elif name == "PAYLOAD_SHA256":
                values[name] = ast.literal_eval(node.value)
    helpers_namespace()["validate_payload"](values["EMBEDDED_PAYLOAD"], values["PAYLOAD_SHA256"])
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Regenerate in temp and compare bytes; no repo writes")
    args = parser.parse_args(argv)
    data = build_notebook()
    target = BASE / NOTEBOOK
    if args.check:
        with tempfile.TemporaryDirectory(prefix="kis-ocr-notebook-") as temporary:
            generated = Path(temporary) / target.name
            generated.write_bytes(data)
            verify_notebook(target.read_bytes())
            if generated.read_bytes() != target.read_bytes():
                parser.exit(1, "FAIL: notebook is stale; run tools/build_kaggle_ocr_notebook.py\n")
        print(f"PASS: deterministic notebook, payload allowlist/hashes/size, code compile ({len(data)} bytes)")
    else:
        target.write_bytes(data)
        print(f"Wrote {NOTEBOOK} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
