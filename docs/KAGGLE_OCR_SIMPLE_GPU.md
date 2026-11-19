# Simple Kaggle GPU OCR

> **New Vietnamese notebook:** use
> [`notebooks/kaggle_ocr_vietnamese_gpu.ipynb`](../notebooks/kaggle_ocr_vietnamese_gpu.ipynb)
> for the new Paddle-detector + VietOCR recognizer path, including alphabet/hash
> gates, preview before bulk, and separate raw diagnostics. Original notebook is
> preserved. See [KAGGLE_OCR_VIETNAMESE.md](KAGGLE_OCR_VIETNAMESE.md). Actual Kaggle
> inference/accuracy is not yet validated; related unit tests are not a quality claim.

> **Quality warning from15/11/2026 audit:** the configured
> `latin_PP-OCRv5_mobile_rec` official inference alphabet inspected is missing
> many Vietnamese tone-marked letters (`ả/ấ/ứ/...`). `lang=vi` routing/name alone
> does not establish full Vietnamese coverage. Supplied67,882records show the
> same missing-character pattern. Do not attribute every error to blur or fix it
> merely by resizing/adding language flags. See
> [OCR_NOTEBOOK_DIAGNOSIS.md](OCR_NOTEBOOK_DIAGNOSIS.md) for evidence, provenance
> limits and a controlled follow-up plan. Notebook/source results remain unchanged.

Use **`notebooks/kaggle_ocr_simple_gpu.ipynb`** independently of the repository.
It does not change the existing notebooks or production OCR pipeline.

1. Import the notebook into Kaggle, attach keyframe data, enable **Internet** and
   **GPU T4 x2** (Python **3.13**).
2. Edit only `INPUT_ROOT` in cell 1, for example:
   - `/kaggle/input/my-keyframes/L04`, containing `L04_V001/001.jpg`;
   - `/kaggle/input/my-keyframes/L04/L04_V001`, containing `001.jpg` directly.
3. **Run All**. Cell 4 shows one link: **`/kaggle/working/ocr.zip`**.
   Download and extract it to browse the minimal per-image JSON records only.
   No summary/profile metadata is downloaded. No loose JSON, venv, caches,
   checkpoints, logs or readiness sidecars are published in working.

Only physical GPU **0** is exposed, and both OCR predictors must use `gpu:0`.
A CUDA build, exactly one visible T4, a real CUDA tensor operation, and a real
padded-image OCR smoke must succeed. There is no CPU, alternate API, or alternate
backend fallback. Installation, model downloads, and the first smoke can take
several minutes; subprocess output streams live.

## Files and resume

- Final output, in a **fresh session**:

  ```text
  /kaggle/working/
    ocr.zip
  ```

- Stable, owned transient scratch:
  `/kaggle/temp/ocr-simple-gpu-<sha256-of-absolute-input-root>/`, with a guarded
  writable `/tmp/ocr-simple-gpu-<same-hash>/` fallback if temp is not writable.
  Neither candidate may overlap input or the publication directory. Both failing
  is a hard error **before installation**; working is never a scratch fallback.
- Isolated pip-less venv: `<scratch>/runtime/`, reused non-destructively.
- State/worker: `<scratch>/state/`; integrity-protected checkpoints:
  `<scratch>/state/checkpoints/*.json` (hashed names are internal only).
- HOME, XDG cache/config/data/state, pip, Hugging Face hub/assets and legacy cache
  aliases, Paddle/PaddleX, ModelScope, Matplotlib configuration (`MPLCONFIGDIR`),
  and temporary-file paths are routed under scratch. Children inherit only an
  explicit allowlist: system search/locale settings, GPU library/driver settings,
  thread controls, proxies, read-only CA certificate locations, and supported
  model-service token variables. CUDA device visibility is still forced to `0`.
  Arbitrary write destinations are not inherited, including `SSLKEYLOGFILE`,
  cuDNN logging destinations/verbosity, glog directories/links, and loader debug
  output. Glog is explicitly directed to stderr. The notebook kernel's environment
  is not globally changed. Installer and worker CWDs are scratch, including
  model-library defaults such as `HOME/.paddlex`. These controls cover notebook
  files and known defaults in the pinned stack; they are not a filesystem sandbox
  preventing arbitrary external libraries/extensions from writing absolute paths.
- Loose export staging and ZIP construction stay at
  `<scratch>/state/downloads/<invocation-id>/`. The **ZIP contents** are:

  ```text
  L04/
    L04_V001/
      001.json
    L04_V002/
      002.json
  L05/
    L05_V001/
      001.json
  ```

`L04_V001/001.jpg` becomes `L04/L04_V001/001.json`, preserving the original stem
and zero padding, not `1.json` or a random/hash filename. Supported image
extensions share this rule. Each file contains a **one-element array** with the
seven-field public record described below, not the internal checkpoint envelope
or full technical record. The ZIP contains only verified current per-image JSONs
at these exact relative paths: no root summary/profile, readiness/ownership receipts,
error diagnostics, temporary files, environment, caches, itself or wrapper directory.
Summary statistics are printed in the cell 4 log and retained in private scratch
publication/readiness receipts; the runtime profile stays in scratch too.
The group (`L04`, `L05`, etc.) is derived deterministically from the canonical
`video_id`, not from input ancestors. This layout is identical when `INPUT_ROOT`
selects one video, one group, or a subtree containing multiple groups. Only that
selected subtree is enumerated; an outside `L06` sibling is never scanned or
invented in the output. JSONL is no longer the primary/download format.

Video folder names must be canonical (e.g. `L04_V001`, not `l4_v1`), and image
stems must be ASCII numeric. Duplicate export paths or canonical identities are
rejected before export writes, including same-stem JPEG/PNG, `001`/`1`, the same
keyframe in repeated video folders across subtrees, and alias video folders.
Malformed group/video identities are rejected. Input files are never renamed or
modified.

Scratch and output owners must match the selected input root. Unowned/nonempty
directories, input symlinks/reparse paths, traversal, nesting beyond eight levels,
and overlap in **either direction** between input, scratch/runtime/state and the
final publication directory are rejected before writes.
The input is never modified and siblings outside the selected subtree are not
enumerated. Existing venvs are reused, not removed. Runtime-generated internal
links (e.g. venv `lib64` and model-cache links) must resolve inside their runtime;
external runtime links are rejected before pip/cache writes. This assumes a normal,
single-user Kaggle filesystem, not concurrent adversarial filesystem mutation.

Rerun cell 3, then cell 4, to resume in the same Kaggle session. Successful
checkpoints are reused only when the original-byte SHA-256, full runtime/profile
hash, canonical identity/path and record checksum agree. The mandatory real-image
smoke still runs on resume. Error checkpoints are retried. Removing a keyframe
excludes its historical checkpoint from the next export. Kaggle session resets
lose the transient venv/checkpoints; **cross-session resume/import is not supported**.
This public-schema change modifies the worker source and therefore its profile
hash: checkpoints from earlier workers are not reused. When replacing an earlier
notebook version, rerun cell 2 to write the updated worker, then cell 3 to recompute
and cell 4 to export. Subsequent same-version reruns can resume normally; the
source/profile/record integrity checks have not been weakened.

Each invocation gets a fresh **scratch** export directory. Historical/partial
staging remains in transient storage, not Kaggle Output. If an export already
exists, rerun cell 3 for a fresh invocation before exporting again. No prior
working directories, v5 environments, sessions or unrelated files are enumerated
or deleted. Consequently, only-ZIP output is guaranteed for a fresh working
directory; existing unrelated outputs are intentionally left alone.

The archive is allowlist-built from current, integrity-verified records, CRC-tested,
then copied and fsynced to a unique `.ocr-publish-<uuid>.tmp` **beside** the final
ZIP. This narrowly scoped temporary publication file is permitted during packing;
`os.replace` atomically publishes it even when scratch is on another device.
Only that invocation's temporary file is removed in `finally`, on success or
handled failure/interruption. No working sidecar is created. As with any Python
cleanup, SIGKILL, a kernel crash or power loss cannot execute `finally`.

A scratch-only publication ownership receipt records the input root, invocation,
profile and archive SHA-256; the ready receipt is written **last** and reset on
rerun/export. Cell 4 requires successful export **and** matching receipts/profile/
archive digest before showing its single `FileLink`. Failure prints a warning:
an older ZIP may remain, but it is not presented as a current result. Prepublication
failures preserve the previous valid owned ZIP. A preexisting `ocr.zip` without
matching scratch ownership/digest is refused, never silently overwritten. After
a session reset, move that archive aside manually before starting a new export.
If publication succeeded but receipt persistence was interrupted, no link is
shown; an archive without matching ownership is likewise refused on the next run.
Late export output, worker completion, or final-cell link/display failures clear
only the matching owned **current** ready receipt. The valid published ZIP and its
scratch ownership receipt remain intact, and `checked_ready` rejects it as current
until a fresh invocation successfully exports. Another invocation's or dataset's
readiness receipt is not removed by this failure cleanup.

## Meaning of each image JSON

Example contents of `L04/L04_V001/001.json` (exactly these seven fields):

```json
[
  {
    "video_id": "L04_V001",
    "frame_id": null,
    "keyframe_n": 1,
    "image_path": "data/L04/L04_V001/001.jpg",
    "ocr_text": "HTVTHD\n18:29:55",
    "retrieval_text": "HTVTHD\n18:29:55",
    "status": "ok"
  }
]
```

This minimal public projection is deliberately **not** `ocr-sidecar-v1`.
There are no upload, merge, or production integration changes. No captions are
generated and nothing is merged into `caption_mapping`.

- `video_id`: canonical folder identity, e.g. `L04_V001`.
- `frame_id`: always null. No mapping is supplied or read, no video frame index or
  timestamp is inferred, and `keyframe_n - 1` is **not** used as a frame ID.
- `keyframe_n`: integer from the numeric image filename, **not** a video frame index.
- `image_path`: caption-mapping-**style logical reference**
  `data/<group>/<video_id>/<original filename>`. The group comes from `video_id`;
  filename zero padding and the actual extension/case are preserved, e.g.
  `data/L04/L04_V001/001.PNG`. This is not a claim that the notebook reads an actual
  caption mapping or writes a new image at that path. It is independent of whether
  `INPUT_ROOT` selects a video, group or multi-group subtree.
- `ocr_text`: nonblank OCR line texts joined with `\n` in model order. `.strip()`
  is used **only to test blankness**; retained text keeps its original whitespace
  and Unicode. No extra confidence filter, noise correction, sorting or caption
  generation is applied. Low-confidence nonblank lines are retained.
- `retrieval_text`: exactly identical to `ocr_text`; no added prose or normalization.
- `status`: `ok` if successful OCR retains any nonblank text, `empty` if none,
  or `error` for an image decode failure. Both text fields are `""` for `empty`
  and `error`; errors are never relabeled as empty and exception details are private.

Public records contain **no** schema tag, IDs beyond the agreed keys, lines,
geometry, model/confidence, hashes, profile, timing, provenance or error details.
Private checkpoints retain the **`simple-kaggle-ocr-v1`** technical schema unchanged:
original-byte SHA-256 (including corrupt files), profile/checksum, canonical
identity/path, mapping null, raw lines/confidences/polygons, geometry and decode
diagnostics. The private profile records actual model names, installed versions,
CUDA/cuDNN/GPU identity, explicit OCR options and worker-source hash, not model
file hashes. Projection happens only after existing integrity verification and
does not rewrite checkpoints. An internal `ok` with only blank lines projects to
public `empty`; private export summary `ok`/`empty`/`errors` counts reflect the
projected statuses. Internal `ok` and `empty` remain resumable successes.

The wrapper still applies 32 white pixels per side, no resize and no EXIF rotation.
Paddle internally resizes detector inputs and crops/resizes recognizer inputs;
private polygons refer to the padded input, with original/unclipped and clipped
coordinates (`[0, width]` / `[0, height]`) and translation matrices retained.

Decode failures, including Pillow lazy `SyntaxError`, are recorded and processing
continues after a valid smoke. Runtime/backend/API failures stop with a traceback,
not one error per remaining image. All-corrupt input fails because it cannot pass
a real-image smoke. The private/logged summary's `complete` is false if any current image has a decode
error or lacks a matching intact checkpoint. Inventory changes after a run require
rerunning cell 3; changed bytes cannot leak a stale row into an export.

## Dependency evidence and limitations

Core pins: **Paddle GPU 3.2.2 (CUDA 12.6), PaddleOCR 3.2.0, PaddleX 3.2.0,
NumPy 2.2.6**, Pillow 11.3.0, tqdm 4.67.1, OpenCV-contrib 4.10.0.84.

Official sources read before selecting these pins:

- [Paddle CUDA 12.6 index](https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/)
  lists `paddlepaddle_gpu-3.2.2-cp313-cp313-linux_x86_64.whl`.
  Its [wheel metadata](https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.2.2-cp313-cp313-linux_x86_64.whl)
  was also read through small HTTP byte ranges (no install/full wheel download):
  `numpy>=1.21`, CUDA runtime `12.6.77`, cuDNN `9.5.1.17`, and the other
  versioned NVIDIA runtime dependencies.
- [Paddle 3.2.2 requirements](https://github.com/PaddlePaddle/Paddle/blob/v3.2.2/python/requirements.txt)
  allow `numpy>=1.21`; no incompatible NumPy upper bound was found.
- [PaddleOCR 3.2.0 metadata](https://pypi.org/pypi/paddleocr/3.2.0/json)
  requires `paddlex[ocr-core]>=3.2.0,<3.3.0` and Python `>=3.8`.
- [PaddleX 3.2.0 metadata](https://pypi.org/pypi/paddlex/3.2.0/json)
  requires `numpy>=1.24`, `PyYAML==6.0.2`, and (for OCR core)
  `opencv-contrib-python==4.10.0.84`; optional document/IE extras are not requested.
- [NumPy 2.2.6 metadata](https://pypi.org/pypi/numpy/2.2.6/json)
  publishes a Linux x86_64 cp313 wheel. NumPy 1.x is not used on Python 3.13.
- [OpenCV metadata](https://pypi.org/pypi/opencv-contrib-python/4.10.0.84/json)
  has a Linux x86_64 `cp37-abi3` wheel and permits NumPy 2.2.6;
  [PyYAML 6.0.2](https://pypi.org/pypi/PyYAML/6.0.2/json) and
  [Pillow 11.3.0](https://pypi.org/pypi/Pillow/11.3.0/json) publish Linux cp313 wheels;
  [tqdm 4.67.1](https://pypi.org/pypi/tqdm/4.67.1/json) has a pure-Python wheel.
- [Versioned OCR API](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.2.0/paddleocr/_pipelines/ocr.py)
  supports `predict`, explicit model names and disabled orientation/unwarping;
  Vietnamese belongs to the Latin group using `latin_PP-OCRv5_mobile_rec`.
- [Versioned GPU arguments](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.2.0/paddleocr/_common_args.py)
  support explicit `device='gpu:0'`, `enable_hpi=False`, `use_tensorrt=False`.
- [Versioned PaddleX pipeline](https://github.com/PaddlePaddle/PaddleX/blob/v3.2.0/paddlex/inference/pipelines/ocr/pipeline.py)
  exposes detection/recognition predictors and `rec_texts`, `rec_scores`,
  `rec_polys` in padded-input coordinates.

The host pip targets a pip-less venv with `--python`, `--isolated` and
`--only-binary=:all:`; no ensurepip, global install, unpinning or source builds.
Core packages are pinned; transitive versions are resolver-selected and recorded
in the profile, not a full lockfile. A dependency/wheel/driver/model incompatibility
must fail clearly. **No local package installation or real GPU OCR was performed.
Metadata compatibility is not actual Kaggle validation**; the notebook's GPU and
real-image gates remain necessary. OCR accuracy for Vietnamese/video text is not
benchmarked by these offline tests.
