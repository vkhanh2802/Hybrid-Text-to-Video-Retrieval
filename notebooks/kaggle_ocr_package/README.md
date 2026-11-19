# KIS Kaggle PaddleOCR + padding — self-contained v4 / sidecar v1

1. Import `notebooks/kaggle_ocr_paddle_padding.ipynb` into Kaggle.
2. Attach **only the keyframes Dataset**; do not upload/attach any code/package ZIP.
3. Enable Internet for wheels/models. GPU is optional; CPU is the default.
4. Change only `INPUT_ROOT` in the first code cell, then Run All.
5. Download the final sidecar ZIP and its `.sha256` using the two notebook links.

The notebook embeds exactly six small source/config/schema files as JSON/base64
with per-file SHA256/size and a payload SHA256. Bootstrap validates all bytes,
allowlisted relative POSIX paths and the 1 MiB limit before materializing into
`/kaggle/working/kis-kaggle-ocr-v4/<payload_hash>/code`. Cached tamper, traversal,
symlinks/junctions/reparse points and extra files are rejected, not overwritten.
Only the exact embedded runner path is imported. External v1/v2/v3 ZIPs, package
Datasets and `MANIFEST.sha256` files are neither searched nor read for bootstrap.
No credentials, images, weights, docs or outputs are embedded in the notebook.

Accepted roots: `L<number>`, `L<number>_V<number>`, a folder of video directories,
or a folder of group directories. Images must have numeric JPEG/PNG filenames.
Folder IDs, not the dataset slug, determine identity. Ambiguous layouts stop.
Optional `mapping.csv` inside each video uses `n,frame_idx,pts_time`.

See `docs/KAGGLE_OCR.md` for offline wheels/models, GPU limitations, resume,
schema, coordinate semantics and future caption joins. Default border: 32 white
pixels on every side, no wrapper resize. Python 3.10–3.12 keeps the original
NumPy 1.26.4 profile; Python 3.13 selects NumPy 2.2.6 with PaddlePaddle 3.3.1,
PaddleOCR/PaddleX 3.7.0 (Pillow 11.1.0, Hub 1.27.0, jsonschema 4.23.0 unchanged).
The 3.13 profile is a **candidate pending verification by a Kaggle run**, not
local-inference verified; no Kaggle run/accuracy verification was done here.
Venv inherits kernel Python, never changes it. Unsupported Python stops.
Mode `venv-host-pip` creates an isolated venv with `with_pip=False` and
`system_site_packages=False`: no ensurepip is called, even on Kaggle Python 3.13.
Host `sys.executable -I -B -m pip --python <venv_python>` controls resolver,
install and check. Host pip must support `--python`, `--dry-run` and `--report`;
otherwise stop before models/output. No global/kernel install, pip upgrade,
source build or unpinning. Old-mode markers are rejected; marker-less partial
venvs are rebuilt, not reused.
Binary-only resolver preflight, pip check, exact versions and model-free import
smoke must pass before a dependency success marker is written. Enable Internet
or upload complete matching wheels/model cache. Wheel/ABI/CUDA availability on
your Kaggle image is **not guaranteed**. Existing v1/v2/v3 ZIPs are not overwritten
and are not required by this notebook. GPU needs a separately verified GPU
profile/wheel; merely enabling a Kaggle accelerator does not change CPU config.
Missing dependencies/models stop explicitly, never masquerade as empty OCR.

Optional `dist/kis_kaggle_ocr_paddle_padding_v4_py313.zip` is a source
distribution, not a required Kaggle Dataset. It contains code/config/schema/docs only: no images, weights,
credentials, captions, or retrieval indexes. `MANIFEST.sha256` covers every
payload file. ZIP metadata/member order are deterministic. This is an integrity
manifest, not a signature; use only a trusted package/notebook. Notebook payload
hashes likewise detect corruption, not a malicious replacement of all code/hashes.

Regenerate after source/config edits with `python -B tools/build_kaggle_ocr_notebook.py`.
Verify byte-identical generation, hashes/allowlist/size and cell compilation with
`python -B tools/build_kaggle_ocr_notebook.py --check`. Deterministic UTF-8/LF,
sorted files, fixed cell IDs, no timestamps/random IDs and empty cell outputs.
Advanced padding/GPU config changes require a regenerated notebook; the normal
workflow never requires editing package paths or choosing a Python profile.

Outputs are independent JSONL sidecars and provenance/checkpoint reports.
The canonical identity is `(video_id, keyframe_n)`; `keyframe_n` is **not**
`frame_idx`. A future merge must explicitly verify `image_sha256` as well.
Nothing here changes caption_generator or rebuilds FAISS/Milvus.
