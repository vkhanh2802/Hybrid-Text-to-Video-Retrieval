"""Versioned CPU backend gate for the portable OCR sidecar v1 contract.

Shared serialization/model/ZIP primitives are loaded from an exact sibling file,
embedded and hash-checked by the notebook, never discovered in attached packages.
No Paddle import occurs until a child backend environment has been selected.
"""

import argparse
import ast
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import traceback
import uuid


BASE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_ocr_v5_contract", Path(__file__).with_name("kaggle_ocr.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

encoded = core.encoded
digest = core.digest
sha_bytes = core.sha_bytes
read_json = core.read_json
atomic_json = core.atomic_json
contained = core.contained
load_config = core.load_config
verify_zip = core.verify_zip
interpreter_fingerprint = core.interpreter_fingerprint
runtime_environment = core.runtime_environment
TRACE_LIMIT = 65536
BACKEND_FLAGS = ("FLAGS_use_mkldnn", "FLAGS_enable_pir_api")


def select_profile():
    profile = core.select_profile()
    if sys.version_info[:2] == (3, 13):
        profile = dict(profile, name="py313-v5-candidate", config="config/ocr-kaggle-py313-v5.json")
    return profile


def config_for(path=None):
    return load_config(path or BASE / select_profile()["config"])


def source_hash():
    return sha_bytes(encoded({"v5": digest(__file__), "contract": digest(core.__file__)}))


def validate_output(output_root, input_root=None):
    coverage = core.validate_output(output_root, input_root)
    root = Path(output_root)
    manifest = read_json(contained(root, "manifest.json"))
    if digest(contained(root, "backend-v5.json")) != manifest["backend_sha256"]:
        raise ValueError("Backend checksum mismatch")
    return coverage


def safe_text(text, limit=TRACE_LIMIT):
    """Diagnostics, not records: redact common credentials and bound head + tail."""
    text = str(text)
    for key, value in os.environ.items():
        if len(value) >= 4 and re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY", key, re.I):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s'\"]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:token|password|api_key|secret|signature)=)[^\s&'\"]+", r"\1[REDACTED]", text)
    text = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)
    if len(text) > limit:
        marker = "\n[TRUNCATED: diagnostic head/tail retained]\n"
        room = limit - len(marker)
        text = text[:room // 2] + marker + text[-(room - room // 2):]
    return text


def exception_report(phase, exc):
    report = {"phase": phase, "type": type(exc).__name__, "message": safe_text(exc, 4096),
              "traceback": safe_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))}
    cause = exc
    while cause is not None:
        if isinstance(cause, subprocess.CalledProcessError):
            report.update(stdout=safe_text(cause.stdout or ""), stderr=safe_text(cause.stderr or ""))
            break
        cause = cause.__cause__
    return report


def diagnostic(path, report):
    atomic_json(path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)


def checked_path(path):
    """Read-only gate, including missing leaves and Windows reparse ancestors."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal is not accepted in input/output/diagnostic paths")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Input/output/diagnostic symlink/junction/reparse point forbidden")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Unsafe non-directory path ancestor")
    return path


def validate_write_paths(input_root, *paths):
    """Reject unsafe targets before any mkdir, cleanup or diagnostic write."""
    source = checked_path(input_root).resolve() if input_root is not None else None
    for path in paths:
        if path is not None:
            target = checked_path(path)
            if source is not None and target.resolve().is_relative_to(source):
                raise ValueError("Output and diagnostics must be outside INPUT_ROOT")


@contextmanager
def invocation(diagnostics, command, reset=True, input_root=None, output_root=None):
    """One current diagnostic set; never remove output data or unrelated files."""
    diagnostics = Path(diagnostics)
    validate_write_paths(input_root, output_root, diagnostics)
    owned = {"invocation.json", "api.json", "mkldnn-flag.json", "selected.json", "failed.json",
             "run-smoke.json", "run-fatal.json", "preflight.json"}
    existing = list(diagnostics.iterdir()) if diagnostics.exists() else []
    reports = [path for path in existing if path.name in owned or re.fullmatch(r"L\d+_V\d+-\d+\.json", path.name)]
    # Validate the entire set before cleanup; a linked report must not redirect IO.
    validate_write_paths(input_root, *(diagnostics / name for name in owned), *reports)
    diagnostics.mkdir(parents=True, exist_ok=True)
    if reset:
        for path in reports:
            path.unlink()
    metadata = {"id": uuid.uuid4().hex, "command": command, "pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat()}
    if not reset and (diagnostics / "invocation.json").is_file():
        metadata = read_json(diagnostics / "invocation.json")
    state = {"invocation": metadata, "phase": command}
    atomic_json(diagnostics / "invocation.json", metadata)
    try:
        yield state
    except Exception as exc:
        diagnostic(diagnostics / "run-fatal.json",
                   {**exception_report(state["phase"], exc), "invocation": metadata})
        raise


def backend_environment(mode, supported=False):
    environment = runtime_environment(sys.executable, os.environ.get("KIS_OCR_PIP_HOST"))
    # Do not inherit unverified kernel backend experiments, especially legacy PIR.
    for name in BACKEND_FLAGS:
        environment.pop(name, None)
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if mode == "mkldnn-flag":
        if not supported:
            raise RuntimeError("FLAGS_use_mkldnn support has not been verified")
        environment["FLAGS_use_mkldnn"] = "0"
    elif mode != "api":
        raise ValueError("Unsupported backend mode; no unverified PIR fallback")
    return environment


def apply_backend(mode, supported=False):
    if any(name == "paddle" or name.startswith("paddle.") for name in sys.modules):
        raise RuntimeError("Backend environment must be selected BEFORE importing paddle")
    environment = backend_environment(mode, supported)
    for name in (*BACKEND_FLAGS, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"):
        os.environ.pop(name, None)
        if name in environment:
            os.environ[name] = environment[name]


def flag_evidence():
    """Paddle's own get_flags is authoritative; an env var alone proves nothing."""
    import paddle

    result = {}
    for name in BACKEND_FLAGS:
        try:
            value = paddle.get_flags([name])[name]
            result[name] = {"supported": True, "effective": value}
        except (ValueError, KeyError, RuntimeError, AttributeError):
            result[name] = {"supported": False}
    return result


def constructor_backend(factory):
    """Verify the exact installed constructor/API, not a guessed **kwargs option.

    PaddleOCR 3.x can route common arguments through _common_args rather than
    expose them in the public signature. Execute that installed parser without
    initializing a pipeline; hash its source for the diagnostic audit trail.
    """
    evidence = {"accepted_by": None, "invocation": {"args": [{"enable_mkldnn": False}], "kwargs": {}}}
    try:
        signature = inspect.signature(factory)
        evidence["signature"] = safe_text(signature, 4096)
        parameter = signature.parameters.get("enable_mkldnn")
        if parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            evidence["accepted_by"] = "constructor-signature"
            return {"enable_mkldnn": False}, evidence
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            raise ValueError("constructor has no backend parameter or **kwargs")
        common = core.importlib.import_module("paddleocr._common_args")
        parser = common.parse_common_args
        evidence["parser_sha256"] = sha_bytes(inspect.getsource(parser).encode())
        parser_signature = inspect.signature(parser)
        evidence["parser_signature"] = safe_text(parser_signature, 4096)
        # The installed pipeline base must actually use this parser.
        base = core.importlib.import_module("paddleocr._pipelines.base")
        base_source = inspect.getsource(base)
        evidence["pipeline_base_sha256"] = sha_bytes(base_source.encode())
        if base.parse_common_args is not parser:
            raise ValueError("pipeline base does not use the inspected parser")
        calls = [node for node in ast.walk(ast.parse(base_source)) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "parse_common_args"]
        if not calls:
            raise ValueError("pipeline base does not call the common argument parser")
        kwargs = evidence["invocation"]["kwargs"]
        if "default_enable_hpi" in parser_signature.parameters:
            values = []
            for call in calls:
                value = next((kw.value for kw in call.keywords if kw.arg == "default_enable_hpi"), None)
                evidence["default_enable_hpi_source"] = safe_text(ast.unparse(value), 4096) if value is not None else None
                # Read literal semantics, never execute an arbitrary source expression.
                # Module attributes do not prove lexical binding at this call site.
                # Reject names, computed expressions and method calls conservatively.
                if not isinstance(value, ast.Constant) or type(value.value) is not bool:
                    raise ValueError("default_enable_hpi call-site semantics are not a verified literal boolean")
                values.append(value.value)
            if len(set(values)) != 1:
                raise ValueError("ambiguous default_enable_hpi call sites")
            kwargs["default_enable_hpi"] = values[0]
        parser_signature.bind({"enable_mkldnn": False}, **kwargs)
        parsed = parser({"enable_mkldnn": False}, **kwargs)
        if not isinstance(parsed, dict) or parsed.get("enable_mkldnn") is not False:
            raise ValueError("parser did not preserve enable_mkldnn=False")
        evidence["accepted_by"] = "installed-common-args-parser"
        return {"enable_mkldnn": False}, evidence
    except Exception as exc:
        evidence.update(reason=safe_text(exc, 2048), error=exception_report("backend-api-verification", exc))
        return {}, evidence


class PaddleAdapter(core.PaddleAdapter):
    def __init__(self, config, mode="api", report=None):
        versions = core.runtime_preflight(config)
        self.flags = flag_evidence()
        from paddleocr import PaddleOCR

        backend, self.api_evidence = constructor_backend(PaddleOCR)
        self.backend = {"mode": mode, "constructor": backend, "api": self.api_evidence, "flags": self.flags}
        if report is not None:
            report["backend"] = self.backend
        print(safe_text(json.dumps({"backend": self.backend}, sort_keys=True)))
        if mode == "api" and not backend:
            raise RuntimeError("BACKEND_API_UNAVAILABLE: installed API does not verify enable_mkldnn=False")
        if mode == "mkldnn-flag":
            flag = self.flags.get("FLAGS_use_mkldnn", {})
            if not flag.get("supported") or flag.get("effective") not in (False, 0):
                raise RuntimeError("FLAGS_use_mkldnn=0 is not effective in installed Paddle")
        directories, models = [], []
        local_root = os.environ.get("KIS_OCR_MODEL_ROOT")
        for model in config["models"]:
            if local_root:
                directory = contained(local_root, model["name"])
                lock = read_json(directory / "model-lock.json")
                if any(lock.get(k) != model[k] for k in ("repo", "revision")):
                    raise RuntimeError("Offline model-lock.json revision/repo mismatch")
                files = lock["files_sha256"]
                actual = {n for n in core.MODEL_FILES | {"config.json"} if (directory / n).is_file()}
                if not core.MODEL_FILES.issubset(files) or set(files) != actual:
                    raise RuntimeError("Offline model lock must cover all inference/config files exactly")
                for name, expected in files.items():
                    if digest(contained(directory, name)) != expected:
                        raise RuntimeError("Offline model file checksum mismatch")
            else:
                if os.environ.get("KIS_OCR_OFFLINE") == "1":
                    raise RuntimeError("Offline models unavailable; attach revision/hash-locked ocr-models")
                from huggingface_hub import snapshot_download

                directory = Path(snapshot_download(repo_id=model["repo"], revision=model["revision"],
                                                   allow_patterns=sorted(core.MODEL_FILES | {"config.json"})))
                files = {p.name: digest(p) for p in directory.iterdir()
                         if p.is_file() and p.name in core.MODEL_FILES | {"config.json"}}
                if not core.MODEL_FILES.issubset(files):
                    raise RuntimeError("Pinned model download incomplete")
            directories.append(str(directory))
            models.append({**model, "files_sha256": files})
        # No silent constructor retry: rejection is a failed backend candidate.
        self.model = PaddleOCR(
            text_detection_model_name=models[0]["name"], text_detection_model_dir=directories[0],
            text_recognition_model_name=models[1]["name"], text_recognition_model_dir=directories[1],
            ocr_version=config["ocr_version"], device=config["device"],
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            **backend)
        self.threshold = config["text_rec_score_thresh"]
        self.engine = {"name": "paddleocr", "versions": versions, "models": models,
                       "device": config["device"], "language": config["language"],
                       "config_sha256": sha_bytes(encoded(config))}
        self._model_files = [(Path(d), m["files_sha256"]) for d, m in zip(directories, models)]


def padded_input(path, config):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Input bytes are not JPEG/PNG")
        image = image.convert("RGB")
        prep = core.preprocessing(image.size, config["padding"])
        border = tuple(config["padding"][s] for s in ("left", "top", "right", "bottom"))
        padded = ImageOps.expand(image, border=border, fill=tuple(config["padding"]["color"]))
    return padded, prep


def smoke(input_root, inventory, config, adapter):
    # First existing image in canonical order. Never skip a corrupt first image
    # to manufacture a backend success on an unrelated/synthetic input.
    entry = next((e for e in inventory["records"] if e["source_path"] is not None
                  and contained(input_root, e["source_path"]).is_file()), None)
    if entry is None:
        raise RuntimeError("BACKEND_SMOKE_UNAVAILABLE: no existing deterministic image")
    path = contained(input_root, entry["source_path"])
    if digest(path) != entry["image_sha256"]:
        raise ValueError("Smoke input hash mismatch")
    padded, prep = padded_input(path, config)
    lines = core.normalize_lines(adapter.recognize(padded), prep)
    if digest(path) != entry["image_sha256"]:
        raise ValueError("Smoke input changed during inference")
    if hasattr(adapter, "check_models"):
        adapter.check_models()
    return {"video_id": entry["video_id"], "keyframe_n": entry["keyframe_n"],
            "image_sha256": entry["image_sha256"], "preprocessing": prep, "lines": len(lines)}


def smoke_worker(input_root, config_path, report_path, mode):
    validate_write_paths(input_root, report_path)
    report = {"schema": "kis_ocr_backend_diagnostic_v5", "success": False, "mode": mode}
    marker = Path(report_path).with_name("invocation.json")
    if marker.is_file():
        report["invocation"] = read_json(marker)
    phase = "backend-environment"
    try:
        verified_flag = False
        if mode == "mkldnn-flag":
            prior = read_json(Path(report_path).with_name("api.json"))
            verified_flag = prior.get("flags", {}).get("FLAGS_use_mkldnn", {}).get("supported") is True
        apply_backend(mode, supported=verified_flag)
        config = config_for(config_path)
        phase = "dependency-import"
        core.runtime_preflight(config)
        report["flags"] = flag_evidence()
        phase = "backend-initialization"
        adapter = PaddleAdapter(config, mode, report=report)
        report["backend"] = adapter.backend
        phase = "one-image-inference"
        report["smoke"] = smoke(input_root, core.discover(input_root), config, adapter)
        report.update(success=True, engine=adapter.engine)
    except Exception as exc:
        report["error"] = exception_report(phase, exc)
    diagnostic(report_path, report)
    return 0 if report["success"] else 2


def smoke_subprocess(input_root, config_path, diagnostics, state=None):
    validate_write_paths(input_root, diagnostics)
    if state is None:
        with invocation(diagnostics, "backend-smoke", input_root=input_root) as state:
            return smoke_subprocess(input_root, config_path, diagnostics, state)
    diagnostics.mkdir(parents=True, exist_ok=True)
    attempts = []
    for mode in ("api", "mkldnn-flag"):
        if mode == "mkldnn-flag" and not attempts[-1].get("flags", {}).get("FLAGS_use_mkldnn", {}).get("supported"):
            break
        report_path = diagnostics / (mode + ".json")
        report_path.unlink(missing_ok=True)
        environment = backend_environment(mode, supported=mode == "mkldnn-flag")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "backend-smoke",
                   "--input-root", str(input_root), "--config", str(config_path),
                   "--report", str(report_path), "--backend-mode", mode,
                   "--invocation-id", state["invocation"]["id"]]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
        stdout, stderr = safe_text(completed.stdout), safe_text(completed.stderr)
        print(stdout)
        print(stderr, file=sys.stderr)
        warning = "sitecustomize" in stderr.lower() and "wrapt" in stderr.lower()
        if warning:
            print("WRAPT_SITECUSTOMIZE_WARNING: host startup hook mentions wrapt. Warning alone is not "
                  "OCR success/failure. No global wrapt install; inspect dependency and inference gates.")
        report = read_json(report_path) if report_path.is_file() else {
            "success": False, "error": {"phase": "subprocess-startup", "type": "ChildProcessError",
                                        "message": "Smoke child exited without a diagnostic report"}}
        report.update(invocation=state["invocation"], returncode=completed.returncode, stdout=stdout, stderr=stderr,
                      wrapt_sitecustomize_warning=warning)
        atomic_json(report_path, report)
        attempts.append(report)
        if completed.returncode == 0 and report.get("success"):
            atomic_json(diagnostics / "selected.json", report)
            return report
    diagnostic(diagnostics / "failed.json", {"success": False, "attempts": attempts, "invocation": state["invocation"],
               "message": "BACKEND_SMOKE_FAILED: no verified workaround; dataset NOT processed. "
                          "FLAGS_enable_pir_api=0 was NOT attempted: compatibility is unverified."})
    raise RuntimeError("BACKEND_SMOKE_FAILED: inspect diagnostics-v5; no output/checkpoint created")


def prepare_runtime(work, wheelhouse=None, state=None, input_root=None):
    diagnostics = Path(work) / "outputs/diagnostics-v5"
    validate_write_paths(input_root, work, Path(work) / "venv", diagnostics)
    if state is not None:
        return _prepare_runtime(work, wheelhouse, state)
    with invocation(diagnostics, "runtime-setup", input_root=input_root) as state:
        return _prepare_runtime(work, wheelhouse, state)


def _prepare_runtime(work, wheelhouse, state):
    """Retain v4's pip-less, exact-pin host controller without running v4 preflight."""
    import venv

    profile = select_profile()
    config_path = BASE / profile["config"]
    config = config_for(config_path)
    core.check_profile_config(config)
    requirements = (BASE / profile["requirements"]).read_bytes()
    pins = dict(row.split("==") for row in requirements.decode().splitlines() if row and not row.startswith("#"))
    if pins != config["versions"] or config["paddle_distribution"] != "paddlepaddle":
        raise core.dependency_error("v5 requires matching exact CPU pins", profile["name"])
    work = Path(work).absolute()
    env = contained(work, "venv")
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = env / "requirements.sha256"
    previous = marker.read_text() if marker.is_file() else None
    marker.unlink(missing_ok=True)
    host = str(Path(sys.executable).absolute())
    controller = [host, "-I", "-B", "-m", "pip"]
    host_env = runtime_environment()
    child_env = runtime_environment(python, host)
    try:
        version = subprocess.run(controller + ["--version"], check=True, capture_output=True,
                                 text=True, env=host_env).stdout.strip()
        help_text = subprocess.run(controller + ["--help"], check=True, capture_output=True,
                                   text=True, env=host_env).stdout
        install_help = subprocess.run(controller + ["install", "--help"], check=True, capture_output=True,
                                      text=True, env=host_env).stdout
        if not version.startswith("pip ") or "--python" not in help_text or any(
                option not in install_help for option in ("--dry-run", "--report")):
            raise ValueError("host pip lacks --python/--dry-run/--report; no global install fallback")
        identity = {"mode": "venv-host-pip", "interpreter": interpreter_fingerprint(),
                    "host_python": host, "target_python": str(python), "pip_controller_version": version,
                    "profile": profile, "requirements_sha256": sha_bytes(requirements),
                    "config_sha256": sha_bytes(encoded(config)), "source_sha256": source_hash()}
        fingerprint = sha_bytes(encoded(identity))
        if previous is not None and (previous != fingerprint or read_json(work / "interpreter.json") != identity):
            raise ValueError("dependency identity changed; use a fresh Kaggle session")
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "interpreter.json", identity)
        if previous is None:
            if env.exists():
                core.shutil.rmtree(env)
            venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env)
        if not re.search(r"(?m)^include-system-site-packages\s*=\s*false\s*$",
                         (env / "pyvenv.cfg").read_text().lower()):
            raise ValueError("runtime venv must disable system-site-packages")
        probe = subprocess.run([str(python), "-I", "-B", "-c",
                                "import json,sys,site; print(json.dumps({'version':list(sys.version_info[:2]),"
                                "'base':sys.base_prefix,'prefix':sys.prefix,'executable':sys.executable,"
                                "'user_site':site.ENABLE_USER_SITE}))"], check=True, capture_output=True,
                               text=True, env=child_env)
        child = json.loads(probe.stdout)
        if (child["version"] != list(sys.version_info[:2]) or child["user_site"] is not False
                or Path(child["base"]).resolve() != Path(sys.base_prefix).resolve()
                or Path(child["prefix"]).resolve() != env.resolve()
                or Path(child["executable"]).absolute() != python):
            raise ValueError("runtime interpreter isolation mismatch")
        atomic_json(work / "venv-interpreter.json", child)
        core.atomic_bytes(work / "requirements-runtime.txt", requirements)
        if previous is None:
            command = controller + ["--python", str(python), "install", "--only-binary=:all:",
                                    "-r", str(work / "requirements-runtime.txt")]
            if wheelhouse is not None:
                command += ["--no-index", "--find-links", str(wheelhouse)]
            subprocess.run(command + ["--dry-run", "--report", str(work / "resolver-report.json")],
                           check=True, env=host_env)
            subprocess.run(command, check=True, env=host_env)
        subprocess.run(controller + ["--python", str(python), "check"], check=True, env=host_env)
        # Import preflight in a child, with backend env prepared before paddle.
        for name in BACKEND_FLAGS:
            child_env.pop(name, None)
        child_env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        state["phase"] = "runtime-preflight"
        subprocess.run([str(python), "-B", str(Path(__file__).resolve()), "preflight",
                        "--config", str(config_path), "--report", str(work / "outputs/diagnostics-v5/preflight.json"),
                        "--invocation-id", state["invocation"]["id"]],
                       check=True, env=child_env, capture_output=True, text=True)
        core.atomic_bytes(marker, fingerprint.encode())
    except Exception as exc:
        marker.unlink(missing_ok=True)
        raise core.dependency_error("binary-only runtime check failed; do not unpin: " + safe_text(exc),
                                    profile["name"]) from exc
    return python, config_path, identity


def run(input_root, output_root, config_path=None, resume=False, retry_errors=False, adapter=None):
    diagnostics = Path(output_root).parent / "diagnostics-v5"
    with invocation(diagnostics, "run", input_root=input_root, output_root=output_root) as state:
        return _run(input_root, output_root, config_path, resume, retry_errors, adapter, diagnostics, state)


def _run(input_root, output_root, config_path, resume, retry_errors, adapter, diagnostics, state):
    state["phase"] = "input-config-setup"
    from PIL import Image

    config_path = Path(config_path or BASE / select_profile()["config"])
    config = config_for(config_path)
    inventory = core.discover(input_root, allow_empty=resume)
    input_root = Path(input_root).resolve()
    output = Path(output_root)
    if output.exists() and not resume:
        raise FileExistsError("Output exists; choose a new directory or explicitly --resume")
    if resume and not output.is_dir():
        raise ValueError("Resume needs an existing run directory")
    config_hash = sha_bytes(encoded(config))
    validator = core.schema_validator()
    if resume:
        state["phase"] = "resume-validation"
        manifest = read_json(contained(output, "manifest.json"))
        if manifest["config_sha256"] != config_hash or manifest["source_sha256"] != source_hash():
            raise ValueError("Resume config/code hash mismatch; start a new run")
        core.compatible_inventory(manifest["inventory"], inventory)
        checkpoint = read_json(contained(output, "checkpoint.json"))
        if digest(contained(output, "backend-v5.json")) != manifest["backend_sha256"]:
            raise ValueError("Resume backend checksum mismatch")
    state["phase"] = "backend-smoke"
    selection = None
    if adapter is None:
        selection = smoke_subprocess(input_root, config_path, diagnostics, state)
        apply_backend(selection["mode"], supported=selection["mode"] == "mkldnn-flag")
        adapter = PaddleAdapter(config, selection["mode"])
        if adapter.engine != selection["engine"] or adapter.backend != selection["backend"]:
            raise RuntimeError("Backend/model identity changed since smoke subprocess")
    # Also smoke the actual run adapter before any output/checkpoint creation.
    result = smoke(input_root, inventory, config, adapter)
    atomic_json(diagnostics / "run-smoke.json", {"success": True, "smoke": result,
                "invocation": state["invocation"], "backend": getattr(adapter, "backend", {"mode": "offline-mock"})})
    state["phase"] = "resume-validation" if resume else "output-setup"
    engine = dict(adapter.engine, config_sha256=config_hash)
    model_hash = sha_bytes(encoded(engine))
    if resume:
        if manifest["model_sha256"] != model_hash or manifest["engine"] != engine:
            raise ValueError("Resume model/runtime hash mismatch; start a new run")
        saved_backend = read_json(contained(output, "backend-v5.json"))
        if saved_backend != getattr(adapter, "backend", {"mode": "offline-mock"}):
            raise ValueError("Resume backend hash mismatch; start a new run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": core.SCHEMA, "schema_version": 1, "inventory": inventory,
                    "input_sha256": sha_bytes(encoded(inventory)), "config_sha256": config_hash,
                    "model_sha256": model_hash, "source_sha256": source_hash(), "config": config,
                    "engine": engine, "complete": False, "artifacts_sha256": {}}
        checkpoint = {"schema": core.SCHEMA, "input_sha256": manifest["input_sha256"],
                      "config_sha256": config_hash, "model_sha256": model_hash, "records": {}}
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "checkpoint.json", checkpoint)
        atomic_json(output / "backend-v5.json", getattr(adapter, "backend", {"mode": "offline-mock"}))
        manifest["backend_sha256"] = digest(output / "backend-v5.json")
        atomic_json(output / "manifest.json", manifest)
    state["phase"] = "checkpoint-validation"
    if any(checkpoint[k] != manifest[k] for k in ("input_sha256", "config_sha256", "model_sha256")):
        raise ValueError("Checkpoint identity mismatch")
    expected = {core._record_key(e) for e in manifest["inventory"]["records"]}
    if not set(checkpoint["records"]).issubset(expected):
        raise ValueError("Unexpected checkpoint identity")
    try:
        for entry in manifest["inventory"]["records"]:
            reference = checkpoint["records"].get(core._record_key(entry))
            if reference:
                previous = core._load_record(output, reference, validator)
                if any(previous[k] != v for k, v in entry.items()) or previous["engine"] != engine:
                    raise ValueError("Checkpoint record provenance mismatch")
                present = entry["source_path"] is not None and contained(input_root, entry["source_path"]).is_file()
                if present and (previous["status"] in ("ok", "empty")
                                or previous["status"] == "error" and not retry_errors):
                    continue
            record = core.record_template(entry, engine)
            core._store_record(output, record, checkpoint)
            path = contained(input_root, entry["source_path"]) if entry["source_path"] else None
            if path is None or not path.is_file():
                record["status"] = "missing"
            else:
                if digest(path) != entry["image_sha256"]:
                    raise ValueError("Input changed during OCR; start a new run")
                state["phase"] = "image-decode"
                # Pillow can defer PNG decoding until convert()/load() and then raise SyntaxError.
                try:
                    padded, prep = padded_input(path, config)
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                    record.update(status="error", error={"type": type(exc).__name__, "code": "OCR_FAILED"})
                    diagnostic(diagnostics / (entry["video_id"] + "-" + str(entry["keyframe_n"]) + ".json"),
                               {**exception_report(state["phase"], exc), "invocation": state["invocation"]})
                else:
                    state["phase"] = "backend-inference"
                    record["preprocessing"] = prep
                    # Inference/adapter failures are fatal, never repeated per-frame errors.
                    record["lines"] = core.normalize_lines(adapter.recognize(padded), prep)
                    record["status"] = "ok" if record["lines"] else "empty"
                if digest(path) != entry["image_sha256"]:
                    raise ValueError("Input changed during OCR; start a new run")
            core.validate_record(record, validator)
            core._store_record(output, record, checkpoint)
        state["phase"] = "final-validation"
        if hasattr(adapter, "check_models"):
            adapter.check_models()
        core.compatible_inventory(manifest["inventory"], core.discover(input_root, allow_empty=True))
        for entry in manifest["inventory"]["records"]:
            record = core._load_record(output, checkpoint["records"][core._record_key(entry)], validator)
            if record["status"] in ("ok", "empty") and not contained(input_root, entry["source_path"]).is_file():
                raise ValueError("Input disappeared during OCR; resume to record missing status")
        core._summarize(output, manifest, checkpoint, validator)
        validate_output(output)
    except Exception:
        manifest["complete"] = False
        atomic_json(output / "manifest.json", manifest)
        (output / "coverage.json").unlink(missing_ok=True)
        raise
    return output


def package(input_root, output_zip):
    # backend-v5.json stays schema-external, but is included/hash-covered in ZIP.
    validate_output(input_root)
    root = Path(input_root)
    manifest = read_json(root / "manifest.json")
    checkpoint = read_json(root / "checkpoint.json")
    names = {"manifest.json", "backend-v5.json", *manifest["artifacts_sha256"],
             *(ref["path"] for ref in checkpoint["records"].values())}
    return core.write_zip(output_zip, {n: contained(root, n).read_bytes() for n in names})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-input", "preflight", "backend-smoke", "run", "validate-output", "package"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--diagnostics-root", type=Path,
                        help="CLI diagnostics directory (default: output parent or ./outputs/diagnostics-v5)")
    parser.add_argument("--invocation-id", help=argparse.SUPPRESS)
    parser.add_argument("--backend-mode", choices=("api", "mkldnn-flag"), default="api")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.command != "preflight" and args.input_root is None:
        parser.error("--input-root is required")
    diagnostics = args.diagnostics_root or (
        args.report.parent if args.report else
        args.output_root.parent / "diagnostics-v5" if args.output_root else Path.cwd() / "outputs/diagnostics-v5")
    if args.command in ("validate-output", "package"):
        diagnostics = diagnostics / ("cli-" + args.command)
    try:
        validate_write_paths(args.input_root, diagnostics, args.report,
                             args.output_root if args.command != "validate-output" else None)
        # Run owns its context. Explicit worker reports share their controller's
        # invocation and must not erase current candidate/preflight evidence.
        context = nullcontext() if args.command == "run" else invocation(
            diagnostics, "cli-" + args.command, reset=args.invocation_id is None, input_root=args.input_root)
        with context as state:
            if args.invocation_id and (state is None or state["invocation"]["id"] != args.invocation_id):
                raise ValueError("Diagnostic invocation identity mismatch")
            if args.command == "preflight":
                apply_backend("api")
                result = {"versions": core.runtime_preflight(config_for(args.config))}
                if args.report:
                    atomic_json(args.report, {**result, "success": True, "invocation": state["invocation"]})
            elif args.command == "backend-smoke":
                if args.report is None:
                    parser.error("backend-smoke requires --report")
                code = smoke_worker(args.input_root, args.config, args.report, args.backend_mode)
                return code
            elif args.command == "validate-input":
                core.check_profile_config(config_for(args.config))
                result = {"records": len(core.discover(args.input_root)["records"])}
            else:
                if args.output_root is None:
                    parser.error("--output-root is required")
                if args.command == "run":
                    run(args.input_root, args.output_root, args.config, args.resume, args.retry_errors)
                    result = validate_output(args.output_root)
                elif args.command == "validate-output":
                    result = validate_output(args.output_root, args.input_root)
                else:
                    result = package(args.input_root, args.output_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps(exception_report("cli-" + args.command, exc), ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
