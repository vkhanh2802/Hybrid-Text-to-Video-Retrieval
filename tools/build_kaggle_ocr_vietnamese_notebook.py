"""Build a NEW standalone notebook, reusing audited simple notebook IO guards.

No network, installs or weights at build time. Never changes the original .ipynb.
"""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'notebooks/kaggle_ocr_simple_gpu.ipynb'
DESTINATION = ROOT / 'notebooks/kaggle_ocr_vietnamese_gpu.ipynb'
EXTENSION = ROOT / 'tools/kaggle_ocr_vietnamese_extension.py'
LAUNCHER = ROOT / 'tools/kaggle_ocr_isolated_launcher.py'
SPLIT_SETUP = ROOT / 'tools/kaggle_ocr_split_setup.py'


def replace_assignment(source, name, value):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError('Expected one source assignment: ' + name)
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + name + ' = ' + repr(value) + '\n' + ''.join(lines[node.end_lineno:])


def worker_source():
    notebook = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    bootstrap = ''.join(notebook['cells'][3]['source'])
    node = next(n for n in ast.parse(bootstrap).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'WORKER_SOURCE' for t in n.targets))
    worker = ast.literal_eval(node.value)
    pins_node = next(n for n in ast.parse(worker).body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets))
    pins = ast.literal_eval(pins_node.value)
    paddle_pins = dict(pins)
    pins = {name: version for name, version in pins.items() if name not in {'paddlepaddle-gpu', 'paddleocr', 'paddlex'}}
    pins.update(torch='2.8.0+cu126', torchvision='0.23.0+cu126', einops='0.8.1', PyYAML='6.0.2')
    worker = replace_assignment(worker, 'PINS', pins)
    worker += '\nPADDLE_PINS = ' + repr(paddle_pins) + '\n'
    worker = replace_assignment(worker, 'SCHEMA', 'vietnamese-kaggle-ocr-v1')
    worker = worker.replace('simple-kaggle-runtime-v1', 'vietnamese-torch-runtime-v2')
    worker = worker.replace('simple-kaggle-scratch-v2', 'vietnamese-kaggle-lines-batch16-scratch-v3')
    worker = worker.replace('ocr-simple-gpu-', 'ocr-vietnamese-lines-v3-batch16-')
    main_guard = "if __name__ == '__main__':\n    main()"
    if worker.count(main_guard) != 1:
        raise ValueError('Original worker entry point changed')
    worker = worker.replace(main_guard, '') + '\n' + EXTENSION.read_text(encoding='utf-8')
    compile(worker, '<vietnamese-worker>', 'exec')
    return notebook, bootstrap, node, worker


def code(source, ident):
    return dict(cell_type='code', execution_count=None, outputs=[], metadata={}, id=ident,
                source=textwrap.dedent(source).lstrip('\n').splitlines(keepends=True))


def markdown(source, ident):
    return dict(cell_type='markdown', metadata={}, id=ident,
                source=textwrap.dedent(source).lstrip('\n').splitlines(keepends=True))


def build(destination=DESTINATION):
    old, bootstrap, node, worker = worker_source()
    lines = bootstrap.splitlines(keepends=True)
    setup = (''.join(lines[:node.lineno - 1]) + "WORKER_SOURCE = r'''" + worker + "'''\n"
             + 'LAUNCHER_SOURCE = ' + repr(LAUNCHER.read_text(encoding='utf-8')) + '\n'
             + SPLIT_SETUP.read_text(encoding='utf-8'))
    full = ''.join(old['cells'][5]['source'])
    full = full.replace("[str(PYTHON), '-I', '-u', str(WORKER), 'run',", "[*WORKER_COMMAND, 'run',")
    settings_guard = ("if helpers['safe_path'](INPUT_ROOT) != root or str(float(PUBLIC_MIN_CONFIDENCE)) != child_env['KIS_VI_OCR_MIN_CONFIDENCE']:\n"
                      "    raise RuntimeError('Input/confidence changed: rerun setup and preview first')\n")
    full = settings_guard + full
    full = "RUN_OK = False\nEXPORT_OK = False\nif not RUN_FULL:\n    print('Bulk OCR skipped. Inspect preview, set RUN_FULL=True, then rerun this cell. No ZIP published.')\nelse:\n" + textwrap.indent(full, '    ')
    export = ''.join(old['cells'][7]['source'])
    export = export.replace("[str(PYTHON), '-I', '-u', str(WORKER), 'export',", "[*WORKER_COMMAND, 'export',")
    export = export.replace('    display(FileLink(str(download)))  # The only downloadable output.\n', '')
    export = export.replace('    EXPORT_OK = True', "    technical = ARCHIVE.with_name(ARCHIVE.stem + '_diagnostics.zip')\n"
        "    receipt = helpers['read_json'](OUTPUT / 'technical-publication.json')\n"
        "    if receipt['run_id'] != RUN_ID or receipt['sha256'] != helpers['file_digest'](technical):\n"
        "        raise RuntimeError('Technical ZIP receipt mismatch')\n"
        "    display(FileLink(str(download)))\n    display(FileLink(str(technical)))\n    EXPORT_OK = True")
    export = "if not RUN_OK:\n    print('No completed bulk run: downloads intentionally unavailable.')\nelse:\n" + textwrap.indent(export, '    ')
    document = dict(nbformat=4, nbformat_minor=5, metadata=deepcopy(old['metadata']), cells=[
        markdown('''# Vietnamese keyframe OCR — Paddle detector + VietOCR (v3.1 — batch16 + long-line segmentation)
        **Python3.13 / Kaggle Linux / T4 GPU0 / Internet required.** New isolated runtime and new output names; original notebook/data untouched.

        Fixes the old Latin recognizer's missing Vietnamese alphabet. Uses pinned VietOCR VGG-Transformer inference source with full Vietnamese vocabulary, strict tensor-only checkpoint load, and native-image detection/crops. A complete alphabet is necessary, **not a guarantee of accuracy** on blurred video.

        Start with preview. Inspect boxes, raw lines/confidence and omitted/truncated lines before setting `RUN_FULL=True`. No captions, translations, guessed text correction, or generative upscaling. GPU/API/model/alphabet failures stop; no CPU/old-Latin fallback.

        **VietOCR now uses real tensor batches of up to16 crops.** Full OCR buffers up to16 uncached frames to fill exact-resized-width buckets, without extra image padding or mixed precision. Detector remains one image at a time on GPU0. Width buckets/final tails can be smaller than16. Inspect actual batch histogram in logs/run.json; no speedup or VRAM claim until measured on Kaggle. OOM stops instead of silently shrinking batches. Old v3 scalar ZIPs/checkpoints are preserved.
        ## 1. Input and quality controls''', 'vi-intro'),
        code("""INPUT_ROOT = '/kaggle/input/your-keyframes/L01'
RUN_FULL = False  # First Run All previews only. Set True after checking preview.
PREVIEW_COUNT = 8  # Evenly spaced real frames in this subtree, at most50.
PREVIEW_IDENTITIES = []  # Optional exact list, e.g. ['L01_V005:69', 'L01_V013:170'].
PUBLIC_MIN_CONFIDENCE = 0.5  # Uncalibrated; raw lines always kept in diagnostics.
RUN_OK = False
EXPORT_OK = False
""", 'vi-input'),
        markdown('''## 2. Isolated runtime
        No installs into the notebook kernel. TWO owned pip-less venvs/processes keep Paddle3.2 and Torch2.8 NVIDIA dependencies separate. Each gets its own resolver, core pins and mandatory pip check. Old mixed venv is preserved and not reused. No package-version fallback or compilation. Python starts with -I -S and an explicit venv-only launcher, avoiding Kaggle global sitecustomize/wrapt hooks without suppressing dependency checks.

        VietOCR PyPI's old Pillow pin is incompatible with Python3.13. Instead this worker downloads only an explicit inference-source allowlist from `pbcquoc/vietocr` commit`fe8c3a7...`, checks Git blob hashes and retains its license. No training/augmentation stack, `pip install vietocr --no-deps`, or gdown workaround. Model weights and detector assets are pinned and hash-checked; only the vgg checkpoint mirror at`doanhm/vgg_transformer` is used, with its provenance recorded.

        **Not GPU-validated on Kaggle yet.** Resolver, strict state load, both GPU smokes and alphabet gate must pass there before bulk runs. Downloading model weights happens only when preview/run is explicitly executed in this notebook. Expect the first run to take several minutes.''', 'vi-runtime-info'),
        code(setup, 'vi-runtime'),
        markdown('''## 3. Preview before bulk
        Samples are spread across the selected subtree unless PREVIEW_IDENTITIES explicitly selects exact VIDEO:n values. To retest diagnosed L01 cases use ['L01_V005:69', 'L01_V013:170', 'L01_V015:196', 'L01_V020:29', 'L01_V030:67']; use an additional unseen sample before bulk. Absent/duplicate requested identities fail, never silently substitute another image.

        Green boxes pass public filters, not accuracy verification. Red boxes are excluded and remain in diagnostics. Long crops whose normalized width exceeds512 are partitioned only at wide background gaps, targeting at most480px per segment. No overlap, fuzzy text correction, or confidence-based winner selection. Full-line baseline AND each segment remain in diagnostics. No safe gap or any blank/low-confidence/truncated segment excludes the whole long line rather than publishing a partial prefix. This may reduce coverage; inspect it.

        Detector settings/models/weights and v3 segmentation rules are unchanged. Crop jobs (full-line baselines and segments) are grouped by their exact normalized width and decoded in tensor batches up to16. Each crop has independent EOS/probabilities and results are restored to original frame/line order. EOS/truncated=false does NOT establish complete text; completeness always remains not_verified. Recheck text against the scalar v3 preview; batching can cause numerical differences on actual GPU.''', 'vi-preview-info'),
        code(settings_guard + """RUN_OK = False
EXPORT_OK = False
PREVIEW_ID = uuid.uuid4().hex
subprocess.run([*WORKER_COMMAND, 'preview', str(root), str(OUTPUT),
                str(ENV_ROOT), PREVIEW_ID, '--archive', str(ARCHIVE), '--preview-count', str(PREVIEW_COUNT),
                '--preview-identities', json.dumps(PREVIEW_IDENTITIES)],
               env=child_env, cwd=SCRATCH, check=True)
preview_root = Path(helpers['read_json'](OUTPUT / 'latest-preview.json')['directory'])
preview = helpers['read_json'](preview_root / 'preview.json')
print('Actual recognition batches:', preview.get('recognition_batches'))
from IPython.display import Image as DisplayImage, display
for i, record in enumerate(preview['records']):
    print(record['video_id'], record['keyframe_n'], 'status=', record['status'])
    image = preview_root / f'{i:02d}.png'
    if image.is_file():
        display(DisplayImage(filename=str(image), width=1000))
    for j, line in enumerate(record['lines']):
        print(j + 1, repr(line['text']), 'confidence=', round(line['confidence'], 4),
              'included=', line['public_included'], 'truncated=', line.get('truncated', False),
              'width_capped=', line.get('width_capped', False),
              'strategy=', line.get('recognition_strategy'), 'review=', line.get('review_reasons', []))
        if 'inference_batch_size' in line:
            print('  Actual tensor batch:', line['inference_batch_size'], 'width=', line['inference_batch_width'])
        if 'full_line' in line:
            print('  Full-line baseline:', repr(line['full_line']['text']))
        for segment in line.get('segments', []):
            print('  Segment', segment['x_range_crop'], repr(segment['text']),
                  'confidence=', round(segment['confidence'], 4), 'size=', segment['recognizer_size'],
                  'batch=', segment.get('inference_batch_size'))
print('Preview is NOT an accuracy benchmark. Inspect original words; do not correct source text by guessing.')
""", 'vi-preview'),
        markdown('''## 4. Full OCR / same-session resume
        Set `RUN_FULL=True` in cell1 after checking preview, then run this cell. Successful checkpoints are reused only with identical original-byte SHA256 and full model/config/source profile. No source image rename/overwrite, no `n-1` frame inference. Corrupt images have explicit error; model failures stop. Technical record keeps all raw line text, probabilities, polygons and per-image SHA; public text applies only the explicit quality filter.''', 'vi-full-info'),
        code(full, 'vi-full'),
        markdown('''## 5. Download two separate archives
        `ocr_vietnamese_v3_batch16.zip`: canonical `Lxx/Lxx_Vxxx/NNN.json`, each a one-record array with exactly the same seven public fields as before. `frame_id=null`, `retrieval_text==ocr_text`. Missing/filtered/empty is unknown, not negative evidence.

        `ocr_vietnamese_v3_batch16_diagnostics.zip`: profile, pinned source/model hashes, geometry, raw full-line and segment results/ranges, actual tensor batch sizes, confidences, EOS/token-cap/width flags, review reasons and original image SHA. run.json includes the actual recognition batch histogram. Keep this for quality evaluation/provenance. Do not merge it into captions or blindly replace production OCR. Scalar v3 ZIPs/checkpoints are preserved; batch16 uses a new scratch root and new archive names. An existing unrelated archive is refused.

        Resume works in the same Kaggle session; transient scratch disappears on session reset. Snapshot/export logs are not proof of OCR accuracy. Compare a small hand-transcribed validation set before running all groups or replacing school-worker data.''', 'vi-export-info'),
        code(export, 'vi-export'),
    ])
    for cell in document['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<' + cell['id'] + '>', 'exec')
    Path(destination).write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'notebook': str(destination), 'cells': len(document['cells']),
                      'original_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                      'worker_sha256': hashlib.sha256(worker.encode()).hexdigest()}))
    return document


if __name__ == '__main__':
    build()
