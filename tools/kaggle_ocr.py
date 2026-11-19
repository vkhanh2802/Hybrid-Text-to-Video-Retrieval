"""Standalone portable PaddleOCR sidecars. No production/caption imports or writes.

The CLI imports Paddle only on `run`/`preflight`. Use a mock adapter for tests.
All durable payloads use relative paths; model directories stay process-local.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from uuid import uuid4
import zipfile


BASE = Path(__file__).resolve().parents[1]
SCHEMA = "ocr-sidecar-v1"
STATUSES = ("ok", "empty", "missing", "error", "unknown")
VIDEO = re.compile(r"(L[0-9]+)_V[0-9]+\Z")
GROUP = re.compile(r"L[0-9]+\Z")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_FILES = {"inference.json", "inference.pdiparams", "inference.yml"}
DIST_FILES = (
    "notebooks/kaggle_ocr_paddle_padding.ipynb",
    "tools/kaggle_ocr.py",
    "config/requirements-kaggle-ocr.txt",
    "config/ocr-kaggle.json",
    "config/requirements-kaggle-ocr-py313.txt",
    "config/ocr-kaggle-py313.json",
    "schemas/ocr-sidecar-v1.schema.json",
    "docs/KAGGLE_OCR.md",
    "notebooks/kaggle_ocr_package/README.md",
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ocr-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, encoded(value))


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(part in ("", ".", "..") or part.endswith((".", " "))
                   or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", part.split(".")[0])
                   for part in name.split("/"))):
        raise ValueError("Unsafe relative path")
    return path


def contained(root, name):
    safe_name(name)
    root = Path(root).resolve()
    candidate = root / name
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Symlinks/junctions are not accepted")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Path escapes the approved root")
    return candidate


def interpreter_fingerprint():
    return {"python": platform.python_version(), "version": list(sys.version_info[:2]),
            "executable": sys.executable, "implementation": platform.python_implementation(),
            "platform": sys.platform, "machine": platform.machine(),
            "sysconfig_platform": sysconfig.get_platform()}


def dependency_error(reason, profile="unsupported"):
    return RuntimeError("DEPENDENCY_UNAVAILABLE: " + reason + "; " +
                        json.dumps(dict(interpreter_fingerprint(), profile=profile), sort_keys=True))


def select_profile():
    """Select by the actual interpreter, never by a requested/venv Python label."""
    version = sys.version_info[:2]
    if (3, 10) <= version <= (3, 12):
        suffix, name = "", "py310-py312"
    elif version == (3, 13):
        suffix, name = "-py313", "py313-candidate"
    else:
        raise dependency_error("supported Python versions are 3.10-3.13; do not unpin NumPy")
    return {"name": name, "requirements": "config/requirements-kaggle-ocr" + suffix + ".txt",
            "config": "config/ocr-kaggle" + suffix + ".json"}


def check_profile_config(config):
    profile = select_profile()
    expected = read_json(BASE / profile["config"])["versions"]
    if config["versions"] != expected:
        raise dependency_error("config pins do not match interpreter profile", profile["name"])
    return profile


def runtime_environment(python=None, host_python=None):
    """A child-only environment: no kernel/user Python or pip overrides leak in."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("PYTHON", "PIP_", "_PIP_"))
                   and key.upper() not in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
                                           "__PYVENV_LAUNCHER__", "KIS_OCR_PIP_HOST"}}
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
    if python is not None:
        environment["VIRTUAL_ENV"] = str(Path(python).parent.parent)
        environment["PATH"] = str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
    if host_python is not None:
        environment["KIS_OCR_PIP_HOST"] = str(host_python)
    return environment


def pip_check_command():
    host = os.environ.get("KIS_OCR_PIP_HOST")
    if host:
        if sys.prefix == sys.base_prefix or not Path(host).is_absolute():
            raise ValueError("host pip controller requires an isolated runtime venv")
        return [host, "-I", "-B", "-m", "pip", "--python", sys.executable, "check"]
    return [sys.executable, "-m", "pip", "check"]


def runtime_preflight(config):
    """No model imports/initialization/downloads: versions, pip check, import smoke only."""
    profile = check_profile_config(config)
    versions = {}
    try:
        for name, expected in config["versions"].items():
            distribution = config["paddle_distribution"] if name == "paddlepaddle" else name
            versions[distribution] = importlib.metadata.version(distribution)
            if versions[distribution] != expected:
                raise ValueError(f"{distribution}: expected {expected}, got {versions[distribution]}")
        subprocess.run(pip_check_command(), check=True, env=runtime_environment(sys.executable))
        # PaddleX may otherwise probe remote model sources during import.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        for module in ("numpy", "paddle", "paddleocr", "paddlex", "PIL", "jsonschema"):
            importlib.import_module(module)
    except Exception as exc:
        raise dependency_error("OCR dependency unavailable: version/pip check/import smoke failed: " + str(exc),
                               profile["name"]) from exc
    return versions


def prepare_runtime(work, wheelhouse=None):
    """Pip-less same-interpreter venv controlled by the kernel's pip, never ensurepip.

    Called by the notebook using its kernel Python. No installs into that kernel.
    Offline callers must provide a complete binary wheelhouse; pip stays on the host.
    """
    import venv

    profile = select_profile()
    config = load_config(BASE / profile["config"])
    check_profile_config(config)
    requirements = (BASE / profile["requirements"]).read_bytes()
    pins = dict(line.split("==") for line in requirements.decode().splitlines()
                if line.strip() and not line.startswith("#"))
    if pins != config["versions"]:
        raise dependency_error("requirements/config pins differ", profile["name"])
    if config["paddle_distribution"] != "paddlepaddle":
        requirements = requirements.replace(b"paddlepaddle==", config["paddle_distribution"].encode() + b"==")
    work = Path(work).absolute()
    env = contained(work, "venv")
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = env / "requirements.sha256"
    host_python = str(Path(sys.executable).absolute())
    host_env = runtime_environment()
    child_env = runtime_environment(python, host_python)
    previous = marker.read_text() if marker.is_file() else None
    # Even cached environments must pass current checks. A failed recheck invalidates success.
    marker.unlink(missing_ok=True)
    try:
        controller = [host_python, "-I", "-B", "-m", "pip"]
        version = subprocess.run(controller + ["--version"], check=True, capture_output=True,
                                 text=True, env=host_env).stdout.strip()
        help_result = subprocess.run(controller + ["--help"], check=True, capture_output=True,
                                     text=True, env=host_env)
        if not version.startswith("pip ") or "--python" not in help_result.stdout:
            raise ValueError("host pip lacks --python support required by venv-host-pip; "
                             "use an image with a capable pip; no ensurepip or global install fallback")
        install_help = subprocess.run(controller + ["install", "--help"], check=True,
                                      capture_output=True, text=True, env=host_env)
        if "--dry-run" not in install_help.stdout or "--report" not in install_help.stdout:
            raise ValueError("host pip controller lacks --dry-run/--report; "
                             "use an image with a capable pip; no global install or fallback")
        identity = {"mode": "venv-host-pip", "interpreter": interpreter_fingerprint(),
                    "host_python": host_python, "target_python": str(python),
                    "pip_controller_version": version, "profile": profile,
                    "requirements_sha256": sha_bytes(requirements), "config_sha256": sha_bytes(encoded(config))}
        fingerprint = sha_bytes(encoded(identity))
        cached = previous is not None
        if cached and (previous != fingerprint or read_json(work / "interpreter.json") != identity):
            raise ValueError("dependency identity changed (including mode/controller); use a fresh Kaggle session")
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "interpreter.json", identity)
        if not cached:
            # A Python executable alone is not proof of a complete/successful environment.
            # Only this dedicated output venv is removed; never kernel/input/code directories.
            if env.exists():
                shutil.rmtree(env)
            venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env)
        venv_config = (env / "pyvenv.cfg").read_text(encoding="utf-8").lower()
        if not re.search(r"(?m)^include-system-site-packages\s*=\s*false\s*$", venv_config):
            raise ValueError("runtime venv must disable system-site-packages")
        probe = subprocess.run([str(python), "-I", "-B", "-c",
                                "import json,sys,site; print(json.dumps({'version':list(sys.version_info[:2]),"
                                "'base':sys.base_prefix,'prefix':sys.prefix,'executable':sys.executable,"
                                "'user_site':site.ENABLE_USER_SITE}))"], check=True, capture_output=True,
                               text=True, env=child_env)
        child = json.loads(probe.stdout)
        if (child["version"] != list(sys.version_info[:2])
                or Path(child["base"]).resolve() != Path(sys.base_prefix).resolve()
                or Path(child["prefix"]).resolve() != env.resolve()
                or Path(child["executable"]).absolute() != python
                or child["user_site"] is not False):
            raise ValueError("venv interpreter does not inherit the kernel interpreter with isolation")
        atomic_json(work / "venv-interpreter.json", child)
        runtime_requirements = work / "requirements-runtime.txt"
        atomic_bytes(runtime_requirements, requirements)
        if not cached:
            command = controller + ["--python", str(python), "install", "--only-binary=:all:",
                       "-r", str(runtime_requirements)]
            if wheelhouse is not None:
                command += ["--no-index", "--find-links", str(wheelhouse)]
            subprocess.run(command + ["--dry-run", "--report", str(work / "resolver-report.json")],
                           check=True, env=host_env)
            subprocess.run(command, check=True, env=host_env)
        subprocess.run(controller + ["--python", str(python), "check"], check=True, env=host_env)
        subprocess.run([str(python), "-B", str(Path(__file__).resolve()), "preflight",
                        "--config", str(BASE / profile["config"])], check=True, env=child_env)
        atomic_bytes(marker, fingerprint.encode())
    except Exception as exc:
        marker.unlink(missing_ok=True)
        raise dependency_error("binary-only resolver/install/runtime check failed; do not unpin: " + str(exc),
                               profile["name"]) from exc
    return python, BASE / profile["config"], identity


def load_config(path=None):
    config = read_json(path or BASE / select_profile()["config"])
    if config["schema"] != "kis_kaggle_ocr_config_v1" or config["output_schema"] != SCHEMA:
        raise ValueError("Unsupported OCR config/schema")
    if config["resize"] is not None:
        raise ValueError("v1 supports padding only; resize must be null")
    for side in ("left", "top", "right", "bottom"):
        value = config["padding"][side]
        if type(value) is not int or not 0 <= value <= 4096:
            raise ValueError("Padding must be integer pixels in [0, 4096]")
    color = config["padding"]["color"]
    if len(color) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
        raise ValueError("Padding color must be RGB bytes")
    if config["batch_size"] != 1 or config["checkpoint_every"] != 1:
        raise ValueError("v1 checkpoints every image and supports batch_size=1 only")
    if config["device"] not in ("cpu", "gpu:0"):
        raise ValueError("Device must be cpu or gpu:0")
    expected_dist = "paddlepaddle" if config["device"] == "cpu" else "paddlepaddle-gpu"
    if config["paddle_distribution"] != expected_dist:
        raise ValueError("GPU needs an explicitly provisioned compatible paddlepaddle-gpu wheel")
    if len(config["models"]) != 2:
        raise ValueError("Exactly two pinned detection/recognition models required")
    if [m["role"] for m in config["models"]] != ["detection", "recognition"]:
        raise ValueError("Model order must be detection, recognition")
    for model in config["models"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model["name"]):
            raise ValueError("Unsafe model name")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("A full model revision is required")
    return config


def _children(root, directory):
    result = sorted(directory.iterdir(), key=lambda p: p.name)
    for path in result:
        contained(root, path.relative_to(root).as_posix())
    return result


def discover(input_root, allow_empty=False):
    """Inspect only root, optional group, video and its immediate files (no repo scan)."""
    supplied = Path(input_root)
    if supplied.is_symlink() or (hasattr(supplied, "is_junction") and supplied.is_junction()):
        raise ValueError("Input root must not be a symlink/junction")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("INPUT_ROOT must be one keyframe folder")
    videos = []
    if VIDEO.fullmatch(root.name):
        videos = [root]
    else:
        for child in _children(root, root):
            if not child.is_dir():
                if child.suffix.lower() in IMAGE_EXTENSIONS:
                    raise ValueError("Images need a video directory named L<number>_V<number>")
                continue
            if VIDEO.fullmatch(child.name):
                if GROUP.fullmatch(root.name) and not child.name.startswith(root.name + "_"):
                    raise ValueError("Video/group folder identity mismatch")
                videos.append(child)
            elif GROUP.fullmatch(child.name) and not GROUP.fullmatch(root.name):
                for video in _children(root, child):
                    if not video.is_dir() or not VIDEO.fullmatch(video.name):
                        raise ValueError("Group must contain only named video directories")
                    if not video.name.startswith(child.name + "_"):
                        raise ValueError("Video/group folder identity mismatch")
                    videos.append(video)
            else:
                raise ValueError("Cannot infer video IDs: point INPUT_ROOT at a group, video, or folder of videos/groups")
    if (not videos and not allow_empty) or len({p.name for p in videos}) != len(videos):
        raise ValueError("No videos found or duplicate video IDs")
    entries, mapping_hashes = [], {}
    for video in sorted(videos, key=lambda p: p.name):
        video_id = video.name
        group = VIDEO.fullmatch(video_id).group(1)
        images = {}
        for image in _children(root, video):
            if image.is_dir():
                raise ValueError("Unexpected nested directory inside video")
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                if image.name != "mapping.csv":
                    raise ValueError("Video files must be numbered JPEG/PNG or optional mapping.csv")
                continue
            if not re.fullmatch(r"[0-9]+", image.stem):
                raise ValueError("Malformed keyframe filename: expected a numeric stem")
            n = int(image.stem)
            if n in images:
                raise ValueError("Duplicate canonical keyframe_n (including leading zeros/extensions)")
            images[n] = image
        mapping = {}
        mapping_path = video / "mapping.csv"
        if mapping_path.exists():
            mapping_hashes[video_id] = digest(mapping_path)
            with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"n", "frame_idx", "pts_time"}.issubset(reader.fieldnames or []):
                    raise ValueError("mapping.csv requires n,frame_idx,pts_time")
                for row in reader:
                    n, frame_idx, timestamp = int(row["n"]), int(row["frame_idx"]), float(row["pts_time"])
                    if n < 0 or frame_idx < 0 or timestamp < 0 or not math.isfinite(timestamp) or n in mapping:
                        raise ValueError("Invalid/duplicate mapping keyframe number")
                    mapping[n] = {"frame_idx": frame_idx, "timestamp_seconds": timestamp}
        if not images and not mapping and not allow_empty:
            raise ValueError("Video has no numbered JPEG/PNG images or mapping rows")
        for n in sorted(images.keys() | mapping.keys()):
            image = images.get(n)
            entries.append({"group": group, "video_id": video_id, "keyframe_n": n,
                            "image_name": image.name if image else None,
                            "source_path": image.relative_to(root).as_posix() if image else None,
                            "image_sha256": digest(image) if image else None,
                            "mapping": mapping.get(n)})
    return {"schema": "kis_ocr_inventory_v1", "records": entries, "mapping_sha256": mapping_hashes}


def compatible_inventory(saved, current):
    """A disappeared image is missing, never an empty OCR result. Additions/changes fail."""
    if saved["mapping_sha256"] != current["mapping_sha256"]:
        raise ValueError("Resume input mapping hash mismatch; start a new run")
    originals = {(e["video_id"], e["keyframe_n"]): e for e in saved["records"]}
    for entry in current["records"]:
        original = originals.get((entry["video_id"], entry["keyframe_n"]))
        if original is None or entry != original:
            # A mapping still declares a now-disappeared image.
            absent = dict(original or {}, image_name=None, source_path=None, image_sha256=None)
            if original is None or entry != absent:
                raise ValueError("Resume input/config hash mismatch; start a new run")


def preprocessing(size, padding):
    width, height = size
    left, top, right, bottom = (padding[s] for s in ("left", "top", "right", "bottom"))
    return {"padding": dict(padding), "original_size": [width, height],
            "padded_size": [width + left + right, height + top + bottom], "resize": None,
            "forward_transform": [[1, 0, left], [0, 1, top], [0, 0, 1]],
            "inverse_transform": [[1, 0, -left], [0, 1, -top], [0, 0, 1]],
            "coordinate_space": "Paddle predict rec_polys mapped back to supplied padded image; not model tensors"}


def normalize_lines(raw_lines, prep):
    lines = []
    width, height = prep["original_size"]
    padded_width, padded_height = prep["padded_size"]
    left, top = prep["padding"]["left"], prep["padding"]["top"]
    for raw in raw_lines:
        if not isinstance(raw["text"], str):
            raise ValueError("OCR text must be a string")
        text = raw["text"].strip()
        if not text:
            continue
        confidence = float(raw["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid OCR confidence")
        polygon = [[float(x), float(y)] for x, y in raw.get("polygon", raw.get("bbox", []))]
        if len(polygon) < 3 or any(not math.isfinite(v) for point in polygon for v in point):
            raise ValueError("Invalid OCR polygon")
        if any(not (0 <= x <= padded_width and 0 <= y <= padded_height) for x, y in polygon):
            raise ValueError("OCR coordinates outside supplied padded image")
        inverse = [[x - left, y - top] for x, y in polygon]
        clipped = [[min(width, max(0, x)), min(height, max(0, y))] for x, y in inverse]
        def bbox(points):
            return [min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points)]
        lines.append({"text": text, "confidence": confidence,
                      "language": raw.get("language"), "script": raw.get("script"),
                      "polygon_padded": polygon, "bbox_padded": bbox(polygon),
                      "polygon_original": inverse, "bbox_original": bbox(inverse),
                      "polygon_original_clipped": clipped, "bbox_original_clipped": bbox(clipped)})
    lines.sort(key=lambda line: (line["bbox_padded"][1], line["bbox_padded"][0],
                                 line["bbox_padded"][3], line["bbox_padded"][2], encoded(line)))
    return [dict(line, line_index=index) for index, line in enumerate(lines)]


class PaddleAdapter:
    """Paddle v3 predict adapter, following build_ocr.LocalPaddleOCR's JSON API.

    predict(rec_polys) is in the supplied input image coordinates, after Paddle's
    internal scale-back. This wrapper adds/subtracts only its own border.
    """

    def __init__(self, config):
        versions = runtime_preflight(config)
        directories, models = [], []
        local_root = os.environ.get("KIS_OCR_MODEL_ROOT")
        for model in config["models"]:
            if local_root:
                directory = contained(local_root, model["name"])
                lock = read_json(directory / "model-lock.json")
                if any(lock.get(k) != model[k] for k in ("repo", "revision")):
                    raise RuntimeError("Offline model-lock.json revision/repo mismatch")
                files = lock["files_sha256"]
                if not MODEL_FILES.issubset(files):
                    raise RuntimeError("Offline model lock needs inference.json/pdiparams/yml")
                actual_files = {name for name in MODEL_FILES | {"config.json"} if (directory / name).is_file()}
                if set(files) != actual_files:
                    raise RuntimeError("Offline model lock must cover all inference/config files exactly")
                for name, expected in files.items():
                    if digest(contained(directory, name)) != expected:
                        raise RuntimeError("Offline model file checksum mismatch")
            else:
                if os.environ.get("KIS_OCR_OFFLINE") == "1":
                    raise RuntimeError("Offline models unavailable; attach ocr-models with model-lock.json")
                from huggingface_hub import snapshot_download

                directory = Path(snapshot_download(repo_id=model["repo"], revision=model["revision"],
                                                   allow_patterns=sorted(MODEL_FILES | {"config.json"})))
                files = {p.name: digest(p) for p in directory.iterdir() if p.is_file() and p.name in MODEL_FILES | {"config.json"}}
                if not MODEL_FILES.issubset(files):
                    raise RuntimeError("Pinned model download incomplete")
            directories.append(str(directory))
            models.append({**model, "files_sha256": files})
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR

        self.model = PaddleOCR(text_detection_model_name=models[0]["name"], text_detection_model_dir=directories[0],
                               text_recognition_model_name=models[1]["name"], text_recognition_model_dir=directories[1],
                               ocr_version=config["ocr_version"], device=config["device"],
                               use_doc_orientation_classify=False, use_doc_unwarping=False,
                               use_textline_orientation=False)
        self.threshold = config["text_rec_score_thresh"]
        self.engine = {"name": "paddleocr", "versions": versions, "models": models,
                       "device": config["device"], "language": config["language"],
                       "config_sha256": sha_bytes(encoded(config))}
        self._model_files = [(Path(directory), model["files_sha256"])
                             for directory, model in zip(directories, models)]

    def check_models(self):
        for directory, files in self._model_files:
            for name, expected in files.items():
                if digest(directory / name) != expected:
                    raise RuntimeError("Model files changed during OCR run")

    def recognize(self, padded_image):
        import numpy as np

        # Paddle's ndarray input convention is BGR, not Pillow RGB.
        array = np.asarray(padded_image)[:, :, ::-1].copy()
        results = list(self.model.predict(array, text_rec_score_thresh=self.threshold))
        if len(results) != 1:
            raise ValueError("Expected one Paddle prediction per image")
        data = results[0].json
        if isinstance(data, str):
            data = json.loads(data)
        data = data.get("res", data)
        texts, scores, polygons = data["rec_texts"], data["rec_scores"], data["rec_polys"]
        if not len(texts) == len(scores) == len(polygons):
            raise ValueError("Paddle result lengths disagree")
        return [{"text": text, "confidence": float(score),
                 "polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon}
                for text, score, polygon in zip(texts, scores, polygons)]


def schema_validator():
    from jsonschema import Draft202012Validator

    schema = read_json(BASE / "schemas/ocr-sidecar-v1.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_record(record, validator=None):
    (validator or schema_validator()).validate(record)
    status, lines = record["status"], record["lines"]
    if (status == "ok") != bool(lines):
        raise ValueError("Only ok records have nonempty lines")
    if (status == "error") != (record["error"] is not None):
        raise ValueError("Error details must be separate and only on error records")
    if status in ("ok", "empty") and (record["preprocessing"] is None or record["image_sha256"] is None):
        raise ValueError("Successful OCR needs image provenance and transform")
    if record["preprocessing"] is not None:
        prep = record["preprocessing"]
        if prep != preprocessing(prep["original_size"], prep["padding"]):
            raise ValueError("Invalid padding/transform")
        raw = [{"text": line["text"], "confidence": line["confidence"], "polygon": line["polygon_padded"],
                "language": line["language"], "script": line["script"]} for line in lines]
        if normalize_lines(raw, prep) != lines:
            raise ValueError("Noncanonical line order/coordinates")
    if record["source_path"] is not None:
        safe_name(record["source_path"])
        if PurePosixPath(record["source_path"]).name != record["image_name"]:
            raise ValueError("Source basename/image_name mismatch")
    if record["group"] != VIDEO.fullmatch(record["video_id"]).group(1):
        raise ValueError("Group/video identity mismatch")
    if record["image_name"] is not None and int(Path(record["image_name"]).stem) != record["keyframe_n"]:
        raise ValueError("Image filename/keyframe_n mismatch")
    return record


def record_template(entry, engine):
    return {"schema": SCHEMA, "schema_version": 1, **entry,
            "engine": engine, "status": "unknown", "preprocessing": None, "lines": [], "error": None}


def _record_key(entry):
    return f"{entry['video_id']}/{entry['keyframe_n']}"


def _store_record(output, record, checkpoint):
    payload = encoded(record)
    checksum = sha_bytes(payload)
    name = f"records/{record['group']}/{record['video_id']}/{record['keyframe_n']}-{checksum}.json"
    path = contained(output, name)
    if path.exists():
        if digest(path) != checksum:
            raise ValueError("Immutable record checksum mismatch")
    else:
        atomic_bytes(path, payload)
    checkpoint["records"][_record_key(record)] = {"path": name, "sha256": checksum}
    atomic_json(output / "checkpoint.json", checkpoint)


def _load_record(output, reference, validator):
    path = contained(output, reference["path"])
    if digest(path) != reference["sha256"]:
        raise ValueError("Checkpoint record checksum mismatch")
    return validate_record(read_json(path), validator)


def _summarize(output, manifest, checkpoint, validator):
    by_video = defaultdict(list)
    records = []
    for entry in manifest["inventory"]["records"]:
        record = _load_record(output, checkpoint["records"][_record_key(entry)], validator)
        records.append(record)
        by_video[(record["group"], record["video_id"])].append(record)
    artifacts = {}
    for (group, video), frames in sorted(by_video.items()):
        name = f"ocr/{group}/{video}.jsonl"
        atomic_bytes(contained(output, name), b"".join(encoded(frame) for frame in frames))
        artifacts[name] = digest(output / name)
    counts = {status: sum(r["status"] == status for r in records) for status in STATUSES}
    coverage = {"schema": SCHEMA, "total": len(records), "status_counts": counts,
                "complete": not any(counts[s] for s in ("error", "unknown", "missing")),
                "processed": counts["ok"] + counts["empty"]}
    atomic_json(output / "coverage.json", coverage)
    atomic_json(output / "run_metadata.json", {"schema": SCHEMA, "config": manifest["config"],
                "engine": manifest["engine"], "input_sha256": manifest["input_sha256"],
                "config_sha256": manifest["config_sha256"], "model_sha256": manifest["model_sha256"],
                "source_sha256": manifest["source_sha256"], "verified_accuracy": False})
    atomic_bytes(output / "README.md", (
        "# Independent OCR sidecar v1\n\n"
        "Join only after an explicit future merge on (video_id, keyframe_n, image_sha256).\n"
        "keyframe_n is the numeric filename, NOT frame_idx. No captions/retrieval indexes were modified.\n"
        "missing/unknown/error are not negative OCR evidence; empty means successful OCR with no lines.\n"
        "Original polygons are exact inverse translations and may lie in the padding (negative/outside image).\n"
        "Use *_original_clipped fields for bounded drawing. Confidence is not calibrated.\n"
    ).encode("utf-8"))
    for name in ("coverage.json", "run_metadata.json", "README.md", "checkpoint.json"):
        artifacts[name] = digest(output / name)
    manifest["artifacts_sha256"] = artifacts
    manifest["complete"] = coverage["complete"]
    atomic_json(output / "manifest.json", manifest)
    return coverage


def run(input_root, output_root=None, config_path=None, resume=False, retry_errors=False, adapter=None):
    config = load_config(config_path)
    inventory = discover(input_root, allow_empty=resume)
    input_root = Path(input_root).resolve()
    output = Path(output_root or Path("outputs") / ("ocr-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
    if output.resolve().is_relative_to(input_root):
        raise ValueError("Output must be outside INPUT_ROOT")
    if output.is_symlink():
        raise ValueError("Output cannot be a symlink")
    config_hash = sha_bytes(encoded(config))
    source_hash = digest(__file__)
    if output.exists() and not resume:
        raise FileExistsError("Output exists; choose a new directory or explicitly --resume")
    if resume and not output.is_dir():
        raise ValueError("Resume needs an existing run directory")
    validator = schema_validator()
    if resume:
        manifest = read_json(contained(output, "manifest.json"))
        if manifest["config_sha256"] != config_hash or manifest["source_sha256"] != source_hash:
            raise ValueError("Resume config/code hash mismatch; start a new run")
        compatible_inventory(manifest["inventory"], inventory)
        checkpoint = read_json(contained(output, "checkpoint.json"))
    # Initialization failure is fatal; never manufacture empty records for absent dependencies/models.
    adapter = adapter if adapter is not None else PaddleAdapter(config)
    engine = dict(adapter.engine, config_sha256=config_hash)
    model_hash = sha_bytes(encoded(engine))
    if resume:
        if manifest["model_sha256"] != model_hash or manifest["engine"] != engine:
            raise ValueError("Resume model/runtime hash mismatch; start a new run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": SCHEMA, "schema_version": 1, "inventory": inventory,
                    "input_sha256": sha_bytes(encoded(inventory)), "config_sha256": config_hash,
                    "model_sha256": model_hash, "source_sha256": source_hash,
                    "config": config, "engine": engine, "complete": False, "artifacts_sha256": {}}
        checkpoint = {"schema": SCHEMA, "input_sha256": manifest["input_sha256"],
                      "config_sha256": config_hash, "model_sha256": model_hash, "records": {}}
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "checkpoint.json", checkpoint)
    if any(checkpoint[key] != manifest[key] for key in ("input_sha256", "config_sha256", "model_sha256")):
        raise ValueError("Checkpoint identity mismatch")
    expected_keys = {_record_key(e) for e in manifest["inventory"]["records"]}
    if not set(checkpoint["records"]).issubset(expected_keys):
        raise ValueError("Unexpected checkpoint identity")
    from PIL import Image, ImageOps

    for entry in manifest["inventory"]["records"]:
        reference = checkpoint["records"].get(_record_key(entry))
        if reference:
            previous = _load_record(output, reference, validator)
            if any(previous[k] != v for k, v in entry.items()) or previous["engine"] != engine:
                raise ValueError("Checkpoint record provenance mismatch")
            # Even an already processed image must still be present and hash-matching to skip.
            present = entry["source_path"] is not None and contained(input_root, entry["source_path"]).is_file()
            if present and previous["status"] in ("ok", "empty"):
                continue
            if present and previous["status"] == "error" and not retry_errors:
                continue
        record = record_template(entry, engine)
        _store_record(output, record, checkpoint)  # Interrupted/unattempted is unknown, not empty.
        path = contained(input_root, entry["source_path"]) if entry["source_path"] else None
        if path is None or not path.is_file():
            record["status"] = "missing"
        else:
            if digest(path) != entry["image_sha256"]:
                raise ValueError("Input changed during OCR; start a new run")
            try:
                with Image.open(path) as image:
                    if image.format not in ("JPEG", "PNG"):
                        raise ValueError("Input bytes are not JPEG/PNG")
                    image = image.convert("RGB")
                    record["preprocessing"] = preprocessing(image.size, config["padding"])
                    padding = config["padding"]
                    border = tuple(padding[s] for s in ("left", "top", "right", "bottom"))
                    padded = ImageOps.expand(image, border=border, fill=tuple(padding["color"]))
                    record["lines"] = normalize_lines(adapter.recognize(padded), record["preprocessing"])
                record["status"] = "ok" if record["lines"] else "empty"
            except Exception as exc:
                record.update(status="error", lines=[], error={"type": type(exc).__name__, "code": "OCR_FAILED"})
            if digest(path) != entry["image_sha256"]:
                raise ValueError("Input changed during OCR; start a new run")
        validate_record(record, validator)
        _store_record(output, record, checkpoint)
    if hasattr(adapter, "check_models"):
        adapter.check_models()
    # Recheck all discoverable input bytes before publishing the video sidecars.
    compatible_inventory(manifest["inventory"], discover(input_root, allow_empty=True))
    for entry in manifest["inventory"]["records"]:
        reference = checkpoint["records"][_record_key(entry)]
        record = _load_record(output, reference, validator)
        if (record["status"] in ("ok", "empty")
                and not contained(input_root, entry["source_path"]).is_file()):
            raise ValueError("Input disappeared during OCR; resume to record missing status")
    _summarize(output, manifest, checkpoint, validator)
    validate_output(output)
    return output


def validate_output(output_root, input_root=None):
    output = Path(output_root)
    manifest = read_json(contained(output, "manifest.json"))
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != 1:
        raise ValueError("Unsupported output schema")
    if sha_bytes(encoded(manifest["inventory"])) != manifest["input_sha256"]:
        raise ValueError("Input inventory checksum mismatch")
    if sha_bytes(encoded(manifest["config"])) != manifest["config_sha256"]:
        raise ValueError("Config checksum mismatch")
    if sha_bytes(encoded(manifest["engine"])) != manifest["model_sha256"]:
        raise ValueError("Engine/model checksum mismatch")
    if manifest["engine"]["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Engine config checksum mismatch")
    if input_root is not None:
        compatible_inventory(manifest["inventory"], discover(input_root, allow_empty=True))
    checkpoint = read_json(contained(output, "checkpoint.json"))
    if any(checkpoint[key] != manifest[key] for key in ("input_sha256", "config_sha256", "model_sha256")):
        raise ValueError("Checkpoint identity mismatch")
    validator = schema_validator()
    expected = manifest["inventory"]["records"]
    keys = [(entry["video_id"], entry["keyframe_n"]) for entry in expected]
    if not keys or keys != sorted(set(keys)):
        raise ValueError("Inventory must have unique, sorted canonical keys")
    if set(checkpoint["records"]) != {_record_key(e) for e in expected}:
        raise ValueError("Missing/extra checkpoint records")
    records, videos = [], defaultdict(list)
    for entry in expected:
        record = _load_record(output, checkpoint["records"][_record_key(entry)], validator)
        if any(record[k] != v for k, v in entry.items()) or record["engine"] != manifest["engine"]:
            raise ValueError("Output record identity/provenance mismatch")
        records.append(record)
        videos[f"ocr/{entry['group']}/{entry['video_id']}.jsonl"].append(record)
    required = {*videos, "checkpoint.json", "coverage.json", "run_metadata.json", "README.md"}
    if set(manifest["artifacts_sha256"]) != required:
        raise ValueError("Incomplete/extra output artifacts")
    for name, expected_hash in manifest["artifacts_sha256"].items():
        if digest(contained(output, name)) != expected_hash:
            raise ValueError("Output artifact checksum mismatch")
    for name, frames in videos.items():
        if contained(output, name).read_bytes() != b"".join(encoded(f) for f in frames):
            raise ValueError("Video JSONL differs from checkpoint")
    counts = {s: sum(r["status"] == s for r in records) for s in STATUSES}
    coverage = read_json(output / "coverage.json")
    complete = not any(counts[s] for s in ("error", "unknown", "missing"))
    if (coverage != {"schema": SCHEMA, "total": len(records), "status_counts": counts,
                     "complete": complete, "processed": counts["ok"] + counts["empty"]}
            or manifest["complete"] != complete):
        raise ValueError("Coverage semantics mismatch")
    return coverage


def write_zip(destination, files):
    """Exclusive deterministic ZIP with a manifest over every payload member."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("ZIP already exists; choose a new versioned filename")
    payload = dict(files)
    if "MANIFEST.sha256" in payload:
        raise ValueError("Reserved archive manifest name")
    for name in payload:
        safe_name(name)
    if len({n.casefold() for n in payload}) != len(payload):
        raise ValueError("Case-colliding ZIP members")
    payload["MANIFEST.sha256"] = "".join(f"{sha_bytes(data)}  {name}\n" for name, data in sorted(payload.items())).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build in memory/temp first, then publish with exclusive creation.
    with tempfile.TemporaryFile() as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data, compresslevel=9)
        stream.seek(0)
        with destination.open("xb") as target:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                target.write(block)
    verify_zip(destination)
    return {"sha256": digest(destination), "bytes": destination.stat().st_size,
            "members": sorted(payload)}


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) > 100000 or len({n.casefold() for n in names}) != len(names):
            raise ValueError("Duplicate/excessive archive members")
        if sum(info.file_size for info in infos) > 2 * 1024**3:
            raise ValueError("Archive too large")
        for info in infos:
            safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                raise ValueError("Archive links/directories not allowed")
        if "MANIFEST.sha256" not in names:
            raise ValueError("Archive must contain MANIFEST.sha256")
        lines = archive.read("MANIFEST.sha256").decode("utf-8").splitlines()
        expected = {}
        for line in lines:
            checksum, name = line.split("  ", 1)
            safe_name(name)
            if name in expected or not re.fullmatch("[0-9a-f]{64}", checksum):
                raise ValueError("Malformed archive manifest")
            expected[name] = checksum
        if set(names) != {*expected, "MANIFEST.sha256"}:
            raise ValueError("Archive manifest membership mismatch")
        for name, checksum in expected.items():
            if sha_bytes(archive.read(name)) != checksum:
                raise ValueError("Archive checksum mismatch")
        return names


def extract_zip(path, output):
    names = verify_zip(path)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Extraction needs a new directory")
    output.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        for name in names:
            destination = contained(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))


def package(input_root, output_zip, distribution=False):
    root = Path(input_root)
    if distribution:
        files = {name: contained(root, name).read_bytes() for name in DIST_FILES}
        # Convenient package entry README; original source README is also retained.
        files["README.md"] = files["notebooks/kaggle_ocr_package/README.md"]
    else:
        validate_output(root)
        manifest = read_json(root / "manifest.json")
        checkpoint = read_json(root / "checkpoint.json")
        names = {"manifest.json", *manifest["artifacts_sha256"],
                 *(ref["path"] for ref in checkpoint["records"].values())}
        files = {name: contained(root, name).read_bytes() for name in names}
    return write_zip(output_zip, files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-input", "preflight", "run", "validate-output", "package"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--distribution", action="store_true", help="Package only the portable source allowlist")
    args = parser.parse_args(argv)
    if args.command != "preflight" and args.input_root is None:
        parser.error("--input-root is required")
    try:
        if args.command == "preflight":
            result = {"status": "ok", "versions": runtime_preflight(load_config(args.config)),
                      "interpreter": interpreter_fingerprint(), "profile": select_profile()}
        elif args.command == "validate-input":
            check_profile_config(load_config(args.config))
            inventory = discover(args.input_root)
            result = {"status": "ok", "records": len(inventory["records"]),
                      "input_sha256": sha_bytes(encoded(inventory))}
        elif args.command == "run":
            output = run(args.input_root, args.output_root, args.config, args.resume, args.retry_errors)
            result = {"status": "ok", "output": str(output), **validate_output(output)}
        elif args.command == "validate-output":
            if args.output_root is None:
                parser.error("validate-output requires --output-root pointing to a run")
            result = validate_output(args.output_root, args.input_root)
        else:
            if args.output_root is None:
                parser.error("package requires --output-root pointing to a new ZIP")
            result = package(args.input_root, args.output_root, args.distribution)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
