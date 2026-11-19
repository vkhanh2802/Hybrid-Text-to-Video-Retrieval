"""Build/check the deterministic standalone v5 notebook; never edit older versions."""

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import textwrap


BASE = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/kaggle_ocr_paddle_padding_v5.ipynb"
_spec = importlib.util.spec_from_file_location("_ocr_notebook_primitives", Path(__file__).with_name("build_kaggle_ocr_notebook.py"))
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
PAYLOAD_FILES = tuple(sorted((*shared.PAYLOAD_FILES, "tools/kaggle_ocr_v5.py", "config/ocr-kaggle-py313-v5.json")))
canonical_bytes = shared.canonical_bytes
source_bytes = shared.source_bytes

# Reuse audited stdlib-only verification, but extend its explicit allowlist.
# Neither builder is needed by or embedded in the notebook at runtime.
BOOTSTRAP_HELPERS = shared.BOOTSTRAP_HELPERS.replace(
    "    'tools/kaggle_ocr.py',",
    "    'tools/kaggle_ocr.py',\n    'tools/kaggle_ocr_v5.py',\n    'config/ocr-kaggle-py313-v5.json',"
).replace("kis-kaggle-ocr-v4", "kis-kaggle-ocr-v5")
BOOTSTRAP_RUN = shared.BOOTSTRAP_RUN.replace(
    "RUNTIME_READY = False\n",
    "RUNTIME_READY = False\nOCR_RUN_OK = False\n",
    1,
).replace("v4", "v5").replace(
    "tools/kaggle_ocr.py", "tools/kaggle_ocr_v5.py").replace("six files", "eight files").replace(
    "'--input-root', INPUT_ROOT, '--config', str(config_path)]",
    "'--input-root', INPUT_ROOT, '--config', str(config_path),\n"
    "                '--diagnostics-root', str(work / 'outputs/diagnostics-v5')]")
_DEPENDENCY_BODY = shared.DEPENDENCIES.removeprefix("RUNTIME_READY = False\n").replace(
    "python, config_path, identity = ocr.prepare_runtime(work, wheelhouse)",
    "setup_state['phase'] = 'runtime-setup'\n"
    "python, config_path, identity = ocr.prepare_runtime(work, wheelhouse, state=setup_state, input_root=INPUT_ROOT)\n"
    "setup_state['phase'] = 'notebook-setup'"
).replace("RUNTIME_READY = True\n", "")
DEPENDENCIES = ("RUNTIME_READY = False\nOCR_RUN_OK = False\n"
                "with ocr.invocation(work / 'outputs/diagnostics-v5', 'notebook-setup',\n"
                "                    input_root=INPUT_ROOT, output_root=work / 'outputs/ocr-v1') as setup_state:\n"
                + textwrap.indent(_DEPENDENCY_BODY, "    ") + "RUNTIME_READY = True\n") + '''print('Backend API has NOT yet been inference-verified. Run cell performs mandatory one-image smoke.')
print('A sitecustomize/wrapt warning alone is neither success nor failure. Do not install wrapt globally.')
'''
RUN = shared.RUN.replace(
    "if not globals().get('RUNTIME_READY', False):",
    "OCR_RUN_OK = False\nif not globals().get('RUNTIME_READY', False):"
).replace(
    "completed = subprocess.run(command, env=runtime_env, cwd=root, check=False)",
    "diagnostics = output.parent / 'diagnostics-v5'\n"
    "print('Current OCR diagnostics:', diagnostics / 'run-fatal.json')\n"
    "completed = subprocess.run(command, env=runtime_env, cwd=root, check=False,\n"
    "                           capture_output=True, text=True, encoding='utf-8', errors='replace')\n"
    "print('OCR child stdout:\\n' + ocr.safe_text(completed.stdout or ''))\n"
    "print('OCR child stderr:\\n' + ocr.safe_text(completed.stderr or ''), file=sys.stderr)"
).replace(
    "raise RuntimeError('OCR_STOPPED: inspect error above; unavailable models are not empty OCR.')",
    "raise RuntimeError('OCR_STOPPED: child exit ' + str(completed.returncode) + '; inspect ' + "
    "str(diagnostics / 'run-fatal.json') + '. Backend failures are fatal, not empty OCR.')"
) + "OCR_RUN_OK = True\n"
DOWNLOAD = shared.DOWNLOAD.replace(
    "if not globals().get('RUNTIME_READY', False):",
    "if not (globals().get('RUNTIME_READY', False) and globals().get('OCR_RUN_OK', False)):"
).replace(
    "run_hash = hashlib.sha256",
    "if 'output' not in globals() or not (output / 'manifest.json').is_file():\n"
    "    raise RuntimeError('OCR_STOPPED: no current validated output to package')\n"
    "run_hash = hashlib.sha256"
).replace("no output package before dependency gate", "no output package before successful run/validation")
OVERVIEW = '''# PaddleOCR CPU backend gate — self-contained v5 / sidecar v1

1. **Import this notebook** into Kaggle.
2. **Attach your keyframes Dataset**. No external code ZIP or manifest is needed.
3. **Enable Internet** for the exact pinned binary wheels and revision-pinned models.
4. **Edit only `INPUT_ROOT`** in cell 1, then **Run All**; download ZIP + `.sha256` at the end.

CPU is the default. Python 3.13 keeps NumPy 2.2.6, PaddlePaddle 3.3.1,
PaddleOCR/PaddleX 3.7.0; 3.10–3.12 keeps its existing NumPy 1.26.4 profile.
The pip-less isolated venv uses host pip `--python` (no ensurepip, `-S`,
global install, NumPy unpinning or source builds). Eight allowlisted source,
config and schema files are embedded with per-file and payload SHA256.
The unchanged shared contract implementation is embedded too; attached v3/v4
packages are never discovered or loaded. No data, credentials or weights embedded.

**v4 failure:** CPU oneDNN/PIR raised `ConvertPirAttribute2RuntimeAttribute`
on `pir::ArrayAttribute<pir::DoubleAttribute>`. v5 does not assume an env var fixes it.
In the exact installed stack it first verifies whether the constructor/common-args
API accepts `enable_mkldnn=False`, then runs the real model on the first existing
image in deterministic canonical order, with the same 32 px padding and adapter.
Required parser defaults are read from verified installed call-site semantics,
never guessed: only consistent inline `True`/`False` literals are accepted;
named/computed defaults fail closed. A parser `default_enable_hpi` TypeError is a verifier failure, not
proof of inference failure. Introspection/constructor evidence is retained.
Only if Paddle's `get_flags` confirms `FLAGS_use_mkldnn` exists may a **fresh child**
try `FLAGS_use_mkldnn=0` before importing paddle. Effective flags, API/source evidence,
model hashes and full bounded tracebacks are logged. `FLAGS_enable_pir_api=0` is
**not attempted**, because compatibility for this stack has not been verified.

No output/checkpoint is created until the mandatory inference smoke passes.
If no candidate passes, stop and inspect `outputs/diagnostics-v5/`: no thousands of
backend error records. A later backend inference failure also stops immediately.
Each invocation clears only v5-owned old reports and writes `invocation.json`;
setup/preflight/resume failures also write `run-fatal.json`.
Setup starts its invocation before cache/link checks. Unsafe containment or linked
input/output/diagnostic paths stop before any diagnostic mkdir, cleanup or write.
The run cell prints bounded redacted child stdout/stderr before `OCR_STOPPED` and its exact fatal path.
`OCR_STOPPED` is a wrapper; inspect current phase/type/message/traceback and candidate
reports, not an old payload directory. See `docs/KAGGLE_OCR_V5.md` for a diagnostic cell.
An individual corrupt image after the smoke remains `error`; a corrupt first image
stops smoke and must be corrected. A `sitecustomize`/`wrapt` startup warning alone
is neither success nor failure; do not install wrapt globally. Pip check and import
smoke remain separate from inference success. No new wrapt dependency is guessed.

This v5 is **offline/static/mock validated only, not Kaggle inference verified**.
The runtime API checks and smoke determine whether a workaround is safe on your session.
No accuracy, ABI or real-inference success claim is made by this notebook.

Supported layouts: video root `L99_V003/1.jpg`, group root `L99/L99_V003/1.jpg`,
or a folder of videos/groups; filenames must be numeric JPEG/PNG. Optional per-video
`mapping.csv`: `n,frame_idx,pts_time`; keyframe number is NOT frame index.
White padding 32 px per side, no wrapper resize, exact inverse and separate clipped
coordinates. Preserve `ok/empty/missing/error/unknown`; complete only without
error/missing/unknown. Join only on `(video_id,keyframe_n,image_sha256)`.

Outputs: `/kaggle/working/kis-kaggle-ocr-v5/<payload_hash>/outputs/ocr-v1`.
Run All again resumes only matching source/config/input/model/backend hashes; no
v3/v4 output/resume reuse, no implicit error retry. Future explicit retries use
`--resume --retry-errors`. No caption, production, FAISS or Milvus writes.
Optional offline caches retain the existing `ocr-wheelhouse` and revision/hash-locked
`ocr-models` conventions. Normal keyframes-only flow downloads models automatically.
Hashes protect corruption, not a maliciously replaced notebook; import a trusted copy.
'''


def helpers_namespace():
    namespace = {}
    exec(compile(BOOTSTRAP_HELPERS, "v5-bootstrap-helpers", "exec"), namespace)
    return namespace


def build_notebook(base=BASE):
    helpers = helpers_namespace()
    files = []
    for name in PAYLOAD_FILES:
        path = Path(base) / name
        helpers["reject_links"](path)
        data = source_bytes(path)
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "base64": base64.b64encode(data).decode("ascii")})
    payload = {"format": "kis-kaggle-ocr-v5", "files": files}
    checksum = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    helpers["validate_payload"](payload, checksum)
    embedded = ("EMBEDDED_PAYLOAD = json.loads(" + repr(canonical_bytes(payload).decode())
                + ")\nPAYLOAD_SHA256 = " + repr(checksum) + "\n")

    def cell(kind, identity, source):
        result = {"cell_type": kind, "id": identity, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    notebook = {"cells": [
        cell("code", "input-root", 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n'),
        cell("markdown", "overview-v5", OVERVIEW),
        cell("code", "embedded-bootstrap-v5", BOOTSTRAP_HELPERS + "\n" + embedded + BOOTSTRAP_RUN),
        cell("code", "isolated-dependencies", DEPENDENCIES),
        cell("code", "smoke-run-validate", RUN),
        cell("markdown", "download-help", "## Download\nReview coverage; download both ZIP and SHA256 links.\n"),
        cell("code", "download-zip", DOWNLOAD)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    data = (json.dumps(notebook, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode()
    verify_notebook(data)
    return data


def verify_notebook(data):
    # Shared verifier checks first-cell-only, JSON, compile, size, outputs and hash.
    # Its helper reference is replaced locally, never in the old builder module.
    import ast

    if len(data) > shared.MAX_BYTES or b"\r" in data:
        raise ValueError("Notebook must use UTF-8/LF and <= 1 MiB")
    notebook = json.loads(data)
    codes = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if cell["outputs"] or cell["execution_count"] is not None:
                raise ValueError("Notebook must have empty outputs")
            code = "".join(cell["source"])
            compile(code, f"v5-cell-{index}", "exec")
            codes.append(code)
    if codes[0] != 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n':
        raise ValueError("First cell must only assign INPUT_ROOT")
    values = {}
    for node in ast.parse(codes[1]).body:
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = build_notebook()
    target = BASE / NOTEBOOK
    if args.check:
        verify_notebook(target.read_bytes())
        if target.read_bytes() != data:
            parser.exit(1, "FAIL: v5 notebook is stale\n")
        print(f"PASS: deterministic v5 notebook, payload hashes, JSON/compile ({len(data)} bytes)")
    else:
        target.write_bytes(data)
        print(f"Wrote {NOTEBOOK} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
