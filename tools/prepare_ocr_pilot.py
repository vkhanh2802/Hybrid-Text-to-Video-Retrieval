"""Prepare <=12 local keyframes and an isolated CPU Kaggle OCR pilot, no inference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import zipfile

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.config import load_runtime_config
from kis.ocr import digest
from kis.paths import DATA_DIR, KIS_ROOT, MAPPING_DIR, discover_video_image_dirs, is_keyframe_image, video_group
from tools.build_ocr import mapping_rows


SOURCES = ("kis/__init__.py", "kis/paths.py", "kis/artifact_io.py", "kis/ocr.py",
           "tools/__init__.py", "tools/build_ocr.py")
MODELS = (("PP-OCRv5_mobile_det", "PaddlePaddle/PP-OCRv5_mobile_det", "0d63e78e2b680928f6b1747d76a08db6e645efb7"),
          ("latin_PP-OCRv5_mobile_rec", "PaddlePaddle/latin_PP-OCRv5_mobile_rec", "ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57"))
REQUIREMENTS = "paddleocr==3.7.0\npaddlex==3.7.0\npaddlepaddle==3.3.1\nnumpy==1.26.4\nhuggingface-hub==1.27.0\nPillow==11.1.0\n"


def notebook(path, manifest_sha256):
    cells = []
    def cell(kind, text):
        item = {"cell_type": kind, "id": f"ocr-{len(cells)}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell("markdown", "# KIS — pilot OCR độc lập, CPU\nBật **Internet**, không cần GPU. Thêm ZIP pilot làm **private Dataset**. Chỉ <=12 ảnh, không có caption/qrels/credentials. Notebook tạo venv riêng, không cài requirements Windows hoặc thay môi trường KIS. Đây là runtime/development pilot, chưa phải benchmark accuracy độc lập.\n")
    cell("code", """from pathlib import Path
import hashlib, json, sys, zipfile, subprocess, venv
assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'This pinned pilot requires Python 3.10-3.12; do not unpin NumPy to force it'
expected_manifest_sha = '__MANIFEST_SHA__'
code_files = ['kis/__init__.py','kis/paths.py','kis/artifact_io.py','kis/ocr.py','tools/__init__.py','tools/build_ocr.py']
inputs = list(Path('/kaggle/input').rglob('kis-ocr-pilot-20261104.zip'))
work = Path('/kaggle/working/kis-ocr-pilot')
assert not work.exists(), 'Use a fresh working directory/session; do not overwrite an earlier pilot'
work.mkdir()
root = work / 'code'
root.mkdir()
if inputs:
    assert len(inputs) == 1, 'Add only one pilot ZIP'
    with zipfile.ZipFile(inputs[0]) as z:
        assert len(z.namelist()) <= 200
        assert len(set(z.namelist())) == len(z.namelist()), 'Duplicate ZIP paths'
        assert sum(info.file_size for info in z.infolist()) < 100*1024*1024
        payload = z.read('manifest.json')
        assert hashlib.sha256(payload).hexdigest() == expected_manifest_sha, 'Wrong pilot manifest'
        m = json.loads(payload)
        assert set(m['source_sha256']) == set(code_files) and 1 <= len(m['frames']) <= 12
        allowed = {'manifest.json','requirements-ocr.txt',*code_files,*m['mapping_files']}
        allowed.update('images/'+frame['image_relative'] for frame in m['frames'])
        assert set(z.namelist()) == allowed, 'Unexpected pilot archive contents'
        for info in z.infolist():
            p = Path(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in info.filename
            assert info.file_size < 50*1024*1024
        z.extractall(root)
else:
    # Kaggle can expand uploaded ZIPs; accept only the manifest's allowlist.
    import shutil
    manifests = [p for p in Path('/kaggle/input').rglob('manifest.json')
                 if (p.parent / 'tools/build_ocr.py').is_file()]
    assert len(manifests) == 1, 'Add the pilot ZIP or its expanded contents'
    base = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == expected_manifest_sha
    assert set(manifest['source_sha256']) == set(code_files) and 1 <= len(manifest['frames']) <= 12
    names = ['manifest.json', 'requirements-ocr.txt', *manifest['source_sha256'], *manifest['mapping_files']]
    names += ['images/' + frame['image_relative'] for frame in manifest['frames']]
    for name in dict.fromkeys(names):
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        target = root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / p, target)
manifest = json.loads((root / 'manifest.json').read_text())
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() == expected_manifest_sha
assert hashlib.sha256((root/'requirements-ocr.txt').read_bytes()).hexdigest() == manifest['requirements_sha256']
for name, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
for frame in manifest['frames']:
    assert hashlib.sha256((root/'images'/frame['image_relative']).read_bytes()).hexdigest() == frame['image_sha256']
for name, expected in manifest['mapping_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
print('Frames:', len(manifest['frames']), 'CPU-only; no labels/captions')
""".replace('__MANIFEST_SHA__', manifest_sha256))
    cell("code", """env = work / 'venv'
venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env)
python = env / 'bin/python'
subprocess.run([str(python), '-m', 'pip', 'install', '-q', '-r', str(root/'requirements-ocr.txt')], check=True)
""")
    cell("code", """# Download the two pinned SMALL OCR checkpoints, not PaddleOCR-VL/PP-OCRv6 defaults.
setup = '''
import hashlib,json,importlib.metadata
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path.cwd()
m=json.loads((root/'manifest.json').read_text())
models=[]
for name,repo,revision in m['models']:
    directory=Path(snapshot_download(repo_id=repo,revision=revision,
        allow_patterns=['config.json','inference.json','inference.pdiparams','inference.yml']))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    models.append(dict(name=name,repo=repo,revision=revision,directory=str(directory),files_sha256=files))
settings=dict(backend='paddleocr',package_version=importlib.metadata.version('paddleocr'),
              paddle_version=importlib.metadata.version('paddlepaddle'),models=models)
assert settings['package_version']=='3.7.0' and settings['paddle_version']=='3.3.1'
(root/'engine.local.json').write_text(json.dumps(settings,indent=2))
'''
subprocess.run([str(python), '-c', setup], cwd=root, check=True)
""")
    cell("code", """# Bounded real inference; no generated captions and no automatic region guesses.
output = work / 'results'
output.mkdir()
command = [str(python), '-B', '-m', 'tools.build_ocr', 'extract',
           '--input', str(root/'manifest.json'), '--image-root', str(root/'images'),
           '--mapping-dir', str(root/'mapping'), '--engine', str(root/'engine.local.json'),
           '--output', str(output/'ocr_detections.json'), '--max-frames', '12']
completed = subprocess.run(command, cwd=root, check=False)
assert (output/'ocr_detections.json').is_file(), 'Model/runtime initialization failed; send the traceback'
results = json.loads((output/'ocr_detections.json').read_text())
assert completed.returncode in (0, 2)
assert (completed.returncode == 0) == results['complete']
if not results['complete']:
    print('INCOMPLETE PILOT: keep error records unknown; return the ZIP and logs, do not label them negative')
import shutil
shutil.copyfile(root/'manifest.json', output/'input_manifest.json')
packages = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
(output/'packages.txt').write_text(packages)
for name, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, 'Code changed during pilot'
(output/'run_context.json').write_text(json.dumps(dict(device='cpu', complete=results['complete'],
    source_sha256=manifest['source_sha256'], input_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
    frames=len(results['frames']), verified=False), indent=2))
archive = shutil.make_archive(str(work/'ocr-pilot-results'), 'zip', output)
from IPython.display import FileLink, display
display(FileLink(archive))
""")
    Path(path).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(output, archive, notebook_path, image_root=DATA_DIR, mapping_dir=MAPPING_DIR, max_frames=12):
    output, archive, notebook_path = map(Path, (output, archive, notebook_path))
    if any(path.exists() for path in (output, archive, notebook_path)):
        raise FileExistsError("Use versioned OCR pilot paths; never overwrite published packets")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 12:
        raise ValueError("Initial OCR pilot is limited to12 frames")
    scope = set(load_runtime_config(environ={}).video_groups)
    candidates, skipped = [], Counter()
    for vid, directory in discover_video_image_dirs(image_root).items():
        if video_group(vid) not in scope:
            continue
        rows, mapping = mapping_rows(vid, mapping_dir)
        for path in directory.iterdir():
            if not path.is_file() or not is_keyframe_image(path):
                continue
            row = rows.get(int(path.stem))
            if row is None:
                skipped["missing_or_ambiguous_mapping"] += 1
                continue
            candidates.append((vid, path, row, mapping))
    candidates.sort(key=lambda item: (item[0], item[2]["keyframe_n"]))
    random.Random(2026110401).shuffle(candidates)
    if not candidates:
        raise ValueError("No local uniquely mapped keyframes; no implicit Drive fetch is performed")
    output.mkdir(parents=True, exist_ok=False)
    frames, mappings = [], {}
    for vid, path, row, mapping in candidates:
        if len(frames) >= max_frames:
            break
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError):
            skipped["invalid_image"] += 1
            continue
        relative = f"{video_group(vid)}/{vid}/{path.name}"
        destination = output / "images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if digest(path) != digest(destination):
            raise ValueError("Pilot image changed during copy")
        frames.append({**row, "video_id": vid, "image_relative": relative, "image_sha256": digest(destination),
                       "width": width, "height": height})
        mapping_name = f"mapping/{video_group(vid)}/{vid}.csv"
        mappings[mapping_name] = digest(mapping)
        target = output / mapping_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mapping, target)
        if digest(target) != mappings[mapping_name]:
            raise ValueError("Pilot mapping changed during copy")
    if not frames:
        raise ValueError("No valid pilot images")
    for name in SOURCES:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(KIS_ROOT / name, target)
    (output / "requirements-ocr.txt").write_text(REQUIREMENTS, encoding="utf-8")
    manifest = {"schema": "kis_ocr_input_v1", "frames": frames, "models": MODELS,
        "source_sha256": {name: digest(output / name) for name in SOURCES}, "mapping_files": mappings,
        "requirements_sha256": digest(output / "requirements-ocr.txt"),
        "seed": 2026110401, "selection": "metadata-only fixed-seed local keyframes; no labels/captions read",
        "available_candidate_frames": len(candidates), "skipped": dict(skipped),
        "scope": "runtime/development pilot, not untouched accuracy holdout", "contains_qrels": False,
        "contains_credentials": False, "contains_generated_captions": False, "recognition_run_locally": False}
    atomic_write_json(output / "manifest.json", manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as target:
        names = ["manifest.json", "requirements-ocr.txt", *SOURCES, *mappings,
                 *("images/" + frame["image_relative"] for frame in frames)]
        for name in names:
            target.write(output / name, name)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook(notebook_path, digest(output / "manifest.json"))
    return {"frames": len(frames), "videos": len({f["video_id"] for f in frames}),
        "zip": str(archive.resolve()), "zip_sha256": digest(archive),
        "notebook": str(notebook_path.resolve()), "model_run_locally": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.zip, args.notebook, max_frames=args.max_frames), indent=2))
