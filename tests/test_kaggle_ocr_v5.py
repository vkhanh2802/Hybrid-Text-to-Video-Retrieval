"""Offline v5 backend gate contracts; no wheel/model installs or network/OCR."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile
import zlib

from PIL import Image

from tools import build_kaggle_ocr_notebook_v5 as builder
from tools import kaggle_ocr_v5 as ocr


class Adapter:
    def __init__(self, failure_at=None, failure=None):
        config = ocr.config_for()
        self.engine = {"name": "paddleocr", "versions": config["versions"],
                       "models": [{**m, "files_sha256": {n: "a" * 64 for n in ocr.core.MODEL_FILES}}
                                  for m in config["models"]], "device": "cpu",
                       "language": config["language"], "config_sha256": "0" * 64}
        self.calls = []
        self.failure_at = failure_at
        self.failure = failure or NotImplementedError("ConvertPirAttribute2RuntimeAttribute DoubleAttribute")

    def recognize(self, image):
        self.calls.append((image.size, image.getpixel((0, 0))))
        if len(self.calls) == self.failure_at:
            raise self.failure
        return [{"text": "sign", "confidence": 0.9, "polygon": [[34, 34], [40, 34], [40, 38], [34, 38]]}]


class KaggleOCRV5Test(unittest.TestCase):
    def setUp(self):
        ocr.interpreter_fingerprint()
        ocr.core.schema_validator()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.output = self.root / "out/ocr-v1"
        self.diagnostics = self.output.parent / "diagnostics-v5"
        if not (3, 10) <= sys.version_info[:2] <= (3, 13):
            kernel = patch.object(sys, "version_info", (3, 13, 0))
            kernel.start()
            self.addCleanup(kernel.stop)
        guard = patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def image(self, n=1):
        path = self.input / "L77_V001" / f"{n}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), (10, 20, 30)).save(path)
        return path

    def start(self, adapter=None, **kwargs):
        return ocr.run(self.input, self.output, adapter=adapter or Adapter(), **kwargs)

    def records(self):
        return [json.loads(row) for p in self.output.glob("ocr/*/*.jsonl")
                for row in p.read_text(encoding="utf-8").splitlines()]

    def test_smoke_failure_no_output_checkpoint_or_mass_error_records(self):
        for n in range(1, 11):
            self.image(n)
        adapter = Adapter(failure_at=1)
        with patch("builtins.print"), self.assertRaises(NotImplementedError):
            self.start(adapter)
        self.assertEqual(len(adapter.calls), 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.rglob("checkpoint.json")))
        self.assertFalse(list(self.root.rglob("records/*.json")))
        report = ocr.read_json(self.diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "backend-smoke")
        self.assertEqual(report["type"], "NotImplementedError")
        self.assertIn("Traceback (most recent call last)", report["traceback"])
        self.assertIn("ConvertPirAttribute2RuntimeAttribute", report["traceback"])

    def seed_stale_diagnostics(self):
        for name in ("api.json", "mkldnn-flag.json", "selected.json", "failed.json", "run-fatal.json",
                     "run-smoke.json", "preflight.json", "L77_V001-99.json"):
            ocr.atomic_json(self.diagnostics / name, {"success": True, "stale": True})
        ocr.atomic_json(self.diagnostics / "invocation.json", {"id": "old"})

    def test_setup_failure_replaces_stale_diagnostics_without_touching_other_data(self):
        self.image()
        self.seed_stale_diagnostics()
        unrelated = self.diagnostics / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        v4 = self.output.parent / "diagnostics-v4/selected.json"
        ocr.atomic_json(v4, {"keep": True})
        with patch("builtins.print"), self.assertRaises(FileNotFoundError):
            self.start(config_path=self.root / "missing-config.json")
        self.assertEqual({p.name for p in self.diagnostics.iterdir()},
                         {"invocation.json", "run-fatal.json", "notes.txt"})
        report = ocr.read_json(self.diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "input-config-setup")
        self.assertEqual(report["type"], "FileNotFoundError")
        self.assertIn("missing-config.json", report["message"])
        self.assertIn("Traceback", report["traceback"])
        self.assertEqual(report["invocation"], ocr.read_json(self.diagnostics / "invocation.json"))
        self.assertNotEqual(report["invocation"]["id"], "old")
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
        self.assertEqual(ocr.read_json(v4), {"keep": True})
        self.assertFalse(self.output.exists())

    def test_input_schema_and_output_setup_failures_have_current_fatal_report(self):
        self.image()
        for target in ("discover", "schema_validator"):
            with self.subTest(target=target), patch.object(ocr.core, target, side_effect=ValueError(target)), \
                    patch("builtins.print"), self.assertRaises(ValueError):
                self.start()
            report = ocr.read_json(self.diagnostics / "run-fatal.json")
            self.assertEqual(report["phase"], "input-config-setup")
            self.assertEqual(report["message"], target)
            self.assertFalse(self.output.exists())
        real_mkdir = Path.mkdir
        # Fail only output creation, after a successful mock backend smoke.
        def fail_output(path, *args, **kwargs):
            if path == self.output:
                raise PermissionError("output denied")
            return real_mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", new=fail_output), patch("builtins.print"), self.assertRaises(PermissionError):
            self.start()
        self.assertEqual(ocr.read_json(self.diagnostics / "run-fatal.json")["phase"], "output-setup")
        self.assertFalse(list(self.root.rglob("checkpoint.json")))

    def test_unsafe_containment_fails_before_any_filesystem_mutation(self):
        self.image()
        stale = self.input / "diagnostics-v5/selected.json"
        ocr.atomic_json(stale, {"success": True, "keep": "input bytes"})
        snapshot = {p.relative_to(self.input).as_posix(): p.read_bytes()
                    for p in self.input.rglob("*") if p.is_file()}
        calls = [lambda: ocr.run(self.input, self.input / "ocr-v1"),
                 lambda: ocr.run(self.input, self.input / "new/ocr-v1"),
                 lambda: ocr.run(self.input, self.input),
                 # Output is outside input, but its sibling diagnostics IS input.
                 lambda: ocr.run(self.input / "diagnostics-v5", self.input / "ocr-v1"),
                 lambda: ocr.smoke_subprocess(self.input, "config.json", stale.parent),
                 lambda: ocr.smoke_worker(self.input, None, stale, "api")]
        for call in calls:
            with self.subTest(call=call), patch.object(Path, "mkdir") as mkdir, \
                    patch.object(Path, "unlink") as unlink, patch.object(ocr, "atomic_json") as write, \
                    patch.object(ocr.core, "atomic_bytes") as write_bytes, self.assertRaisesRegex(ValueError, "outside INPUT_ROOT"):
                call()
            for mutation in (mkdir, unlink, write, write_bytes):
                mutation.assert_not_called()
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "unlink") as unlink, \
                patch.object(ocr, "atomic_json") as write, patch("builtins.print"):
            code = ocr.main(["validate-input", "--input-root", str(self.input),
                             "--diagnostics-root", str(stale.parent)])
        self.assertEqual(code, 2)
        for mutation in (mkdir, unlink, write):
            mutation.assert_not_called()
        self.assertEqual(snapshot, {p.relative_to(self.input).as_posix(): p.read_bytes()
                                    for p in self.input.rglob("*") if p.is_file()})

    def test_linked_paths_and_unsafe_ancestors_rejected_before_diagnostic_io(self):
        self.image()
        self.seed_stale_diagnostics()
        real_lstat = Path.lstat
        for linked in (self.input, self.output, self.output.parent, self.diagnostics,
                       self.diagnostics / "selected.json"):
            for kind in ("symlink", "junction"):
                def lstat(path):
                    if path == linked:
                        return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else stat.S_IFDIR,
                                               st_file_attributes=0x400 if kind == "junction" else 0)
                    return real_lstat(path)
                with self.subTest(linked=linked, kind=kind), patch.object(Path, "lstat", new=lstat), \
                        patch.object(Path, "mkdir") as mkdir, patch.object(Path, "unlink") as unlink, \
                        patch.object(ocr, "atomic_json") as write, self.assertRaisesRegex(ValueError, "symlink/junction/reparse"):
                    ocr.run(self.input, self.output)
                for mutation in (mkdir, unlink, write):
                    mutation.assert_not_called()
        blocker = self.root / "not-a-directory"
        blocker.write_bytes(b"keep")
        for target in (blocker / "ocr-v1", self.root / "unused/../out/ocr-v1"):
            with self.subTest(target=target), patch.object(Path, "mkdir") as mkdir, \
                    patch.object(Path, "unlink") as unlink, patch.object(ocr, "atomic_json") as write, \
                    self.assertRaises((ValueError, NotADirectoryError)):
                ocr.run(self.input, target)
            for mutation in (mkdir, unlink, write):
                mutation.assert_not_called()

    def test_runtime_setup_and_cli_preflight_failures_are_diagnostic(self):
        self.seed_stale_diagnostics()
        with patch.object(ocr, "select_profile", side_effect=RuntimeError("unsupported runtime")), \
                patch("builtins.print"), self.assertRaises(RuntimeError):
            ocr.prepare_runtime(self.output.parent.parent)
        # prepare_runtime uses <work>/outputs, independently of the run directory.
        runtime_diagnostics = self.output.parent.parent / "outputs/diagnostics-v5"
        report = ocr.read_json(runtime_diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "runtime-setup")
        self.assertIn("unsupported runtime", report["traceback"])
        with patch.object(ocr, "apply_backend"), \
                patch.object(ocr.core, "runtime_preflight", side_effect=ImportError("preflight root failure")), \
                patch("builtins.print"):
            code = ocr.main(["preflight", "--diagnostics-root", str(self.diagnostics)])
        self.assertEqual(code, 2)
        report = ocr.read_json(self.diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "cli-preflight")
        self.assertEqual(report["type"], "ImportError")
        self.assertIn("preflight root failure", report["traceback"])
        self.assertFalse((self.diagnostics / "selected.json").exists())
        self.assertFalse(self.output.exists())

    def test_successful_resume_clears_old_fatal_and_keeps_current_smoke(self):
        self.image()
        self.start()
        first = ocr.read_json(self.diagnostics / "invocation.json")
        self.seed_stale_diagnostics()
        self.start(resume=True)
        marker = ocr.read_json(self.diagnostics / "invocation.json")
        self.assertNotEqual(first["id"], marker["id"])
        self.assertFalse((self.diagnostics / "run-fatal.json").exists())
        self.assertFalse((self.diagnostics / "selected.json").exists())
        self.assertEqual(ocr.read_json(self.diagnostics / "run-smoke.json")["invocation"], marker)

    def lazy_decode_png(self, n=2):
        # Valid signature/IHDR/IEND, but Image.open() stops at an empty IDAT.
        # convert() then hits an invalid chunk id and Pillow 11.1 raises SyntaxError.
        path = self.image(n)
        data = bytearray(path.read_bytes())
        offset = 8
        while offset < len(data):
            length = int.from_bytes(data[offset:offset + 4], "big")
            if data[offset + 4:offset + 8] == b"IDAT":
                empty = struct.pack(">I", 0) + b"IDAT" + struct.pack(">I", zlib.crc32(b"IDAT") & 0xFFFFFFFF)
                broken = struct.pack(">I", 4) + b"X\x00XX" + b"junk" + struct.pack(">I", 0)
                data[offset:offset + 12 + length] = empty + broken
                path.write_bytes(data)
                return path
            offset += 12 + length
        raise AssertionError("PNG fixture has no IDAT chunk")

    def test_smoke_success_corrupt_later_image_continues_as_error(self):
        self.image(1)
        self.image(2).write_bytes(b"corrupt")
        self.image(3)
        adapter = Adapter()
        with patch("builtins.print"):
            self.start(adapter)
        self.assertEqual(adapter.calls, [((84, 74), (255, 255, 255))] * 3)
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["ok", "error", "ok"])
        self.assertEqual(records[0]["lines"][0]["polygon_original"][0], [2.0, 2.0])
        self.assertEqual(records[0]["preprocessing"]["resize"], None)
        self.assertFalse(ocr.validate_output(self.output)["complete"])
        report = ocr.read_json(self.diagnostics / "L77_V001-2.json")
        self.assertEqual(report["phase"], "image-decode")
        self.assertIn("Traceback", report["traceback"])

    def test_lazy_png_syntaxerror_is_bounded_error_and_processing_continues(self):
        self.image(1)
        self.lazy_decode_png(2)
        self.image(3)
        adapter = Adapter()
        with patch("builtins.print"):
            self.start(adapter)
        self.assertEqual(adapter.calls, [((84, 74), (255, 255, 255))] * 3)
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["ok", "error", "ok"])
        self.assertEqual(records[1]["error"], {"type": "SyntaxError", "code": "OCR_FAILED"})
        self.assertFalse(ocr.validate_output(self.output)["complete"])
        self.assertFalse((self.diagnostics / "run-fatal.json").exists())
        report = ocr.read_json(self.diagnostics / "L77_V001-2.json")
        self.assertEqual(report["phase"], "image-decode")
        self.assertEqual(report["type"], "SyntaxError")
        self.assertIn("broken PNG file", report["message"])
        self.assertIn("Traceback (most recent call last)", report["traceback"])

    def test_corrupt_first_image_fails_closed_before_output(self):
        self.image(1).write_bytes(b"invalid")
        self.image(2)
        adapter = Adapter()
        with patch("builtins.print"), self.assertRaises(OSError):
            self.start(adapter)
        self.assertEqual(adapter.calls, [])
        self.assertFalse(self.output.exists())

    def test_later_backend_failure_is_fatal_with_unknown_not_error(self):
        for n in range(1, 6):
            self.image(n)
        adapter = Adapter(failure_at=3)
        with patch("builtins.print"), self.assertRaises(NotImplementedError):
            self.start(adapter)
        self.assertEqual(len(adapter.calls), 3)  # smoke, first image, failing second image
        checkpoint = ocr.read_json(self.output / "checkpoint.json")
        self.assertEqual(len(checkpoint["records"]), 2)
        statuses = [ocr.read_json(self.output / r["path"])["status"] for r in checkpoint["records"].values()]
        self.assertEqual(statuses, ["ok", "unknown"])
        self.assertFalse(ocr.read_json(self.output / "manifest.json")["complete"])
        self.assertFalse((self.output / "coverage.json").exists())
        report = ocr.read_json(self.diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "backend-inference")

    def test_traceback_bounded_redacted_preserves_head_tail(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_private_test_token"}):
            try:
                raise RuntimeError("hf_private_test_token token=private https://user:pass@example.test/ " + "x" * 90000 + " END")
            except RuntimeError as exc:
                report = ocr.exception_report("test", exc)
        self.assertLessEqual(len(report["traceback"]), ocr.TRACE_LIMIT)
        self.assertIn("Traceback", report["traceback"])
        self.assertIn("TRUNCATED", report["traceback"])
        self.assertTrue(report["traceback"].endswith("END\n"))
        self.assertNotIn("hf_private_test_token", json.dumps(report))
        self.assertNotIn("token=private", json.dumps(report))
        self.assertNotIn("user:pass", json.dumps(report))

    def test_backend_env_is_set_before_paddle_import_and_pir_not_forced(self):
        events = []
        config = ocr.config_for()

        def imported(name):
            events.append(name)
            self.assertEqual(os.environ["FLAGS_use_mkldnn"], "0")
            self.assertNotIn("FLAGS_enable_pir_api", os.environ)
            self.assertEqual(os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"], "True")
            return SimpleNamespace()

        with patch.dict(sys.modules, {"paddle": None}), patch.dict(os.environ, {}, clear=False):
            # Remove even the mocked name: apply_backend rejects an already imported Paddle.
            sys.modules.pop("paddle")
            with patch.object(ocr.core.importlib.metadata, "version", side_effect=config["versions"].get), \
                    patch.object(ocr.core.subprocess, "run"), \
                    patch.object(ocr.core.importlib, "import_module", side_effect=imported):
                ocr.apply_backend("mkldnn-flag", supported=True)
                ocr.core.runtime_preflight(config)
        self.assertIn("paddle", events)
        with self.assertRaisesRegex(RuntimeError, "not been verified"):
            ocr.backend_environment("mkldnn-flag")
        with patch.dict(sys.modules, {"paddle": SimpleNamespace()}), self.assertRaisesRegex(RuntimeError, "BEFORE"):
            ocr.apply_backend("api")
        with patch.dict(os.environ, {"FLAGS_enable_pir_api": "0", "FLAGS_use_mkldnn": "1"}):
            env = ocr.backend_environment("api")
            self.assertFalse(any(n in env for n in ocr.BACKEND_FLAGS))

    def test_constructor_signature_and_common_parser_feature_detection(self):
        class Explicit:
            def __init__(self, enable_mkldnn=True):
                pass

        self.assertEqual(ocr.constructor_backend(Explicit)[0], {"enable_mkldnn": False})
        self.assertEqual(ocr.constructor_backend(lambda: None)[0], {})
        parser = Mock(return_value={"enable_mkldnn": False})
        modules = {"paddleocr._common_args": SimpleNamespace(parse_common_args=parser),
                   "paddleocr._pipelines.base": SimpleNamespace(parse_common_args=parser)}
        with patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                patch.object(ocr.inspect, "getsource", side_effect=["def parse_common_args(kwargs): pass", "parse_common_args(kwargs)"]):
            kwargs, evidence = ocr.constructor_backend(lambda **kwargs: None)
        self.assertEqual(kwargs, {"enable_mkldnn": False})
        self.assertEqual(evidence["accepted_by"], "installed-common-args-parser")
        self.assertEqual(len(evidence["parser_sha256"]), 64)
        parser.assert_called_once_with({"enable_mkldnn": False})
        for returned in ({}, {"enable_mkldnn": True}):
            parser.return_value = returned
            with patch.object(ocr.core.importlib, "import_module", side_effect=modules.get):
                self.assertEqual(ocr.constructor_backend(lambda **kwargs: None)[0], {})
        with patch.object(ocr.core.importlib, "import_module", side_effect=ImportError("missing parser")):
            self.assertEqual(ocr.constructor_backend(lambda **kwargs: None)[0], {})

    def test_parser_required_keyword_uses_verified_call_site_not_a_guessed_default(self):
        seen = []
        def parser(kwargs, *, default_enable_hpi):
            seen.append(default_enable_hpi)
            return kwargs
        class Factory:
            def __init__(self, **kwargs):
                pass
        modules = {"paddleocr._common_args": SimpleNamespace(parse_common_args=parser),
                   "paddleocr._pipelines.base": SimpleNamespace(parse_common_args=parser)}
        for expression, expected in (("True", True), ("False", False)):
            base_source = f"parse_common_args(kwargs, default_enable_hpi={expression})"
            sources = ["def parse_common_args(kwargs, *, default_enable_hpi): pass", base_source]
            with self.subTest(expression=expression), patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                    patch.object(ocr.inspect, "getsource", side_effect=sources):
                kwargs, evidence = ocr.constructor_backend(Factory)
            self.assertEqual(kwargs, {"enable_mkldnn": False})
            self.assertIs(seen[-1], expected)
            self.assertEqual(evidence["invocation"]["kwargs"], {"default_enable_hpi": expected})
            self.assertIn("default_enable_hpi", evidence["parser_signature"])
            self.assertEqual(evidence["default_enable_hpi_source"], expression)
            self.assertEqual(len(evidence["parser_sha256"]), 64)
        with patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                patch.object(ocr.inspect, "getsource", side_effect=["parser source", "parse_common_args(kwargs, default_enable_hpi=unknown())"]):
            kwargs, evidence = ocr.constructor_backend(Factory)
        self.assertEqual(kwargs, {})
        self.assertIn("not a verified literal boolean", evidence["reason"])
        self.assertIn("Traceback", evidence["error"]["traceback"])

    def test_parser_defaults_reject_computed_shadowed_and_ambiguous_bindings(self):
        seen = []
        def parser(kwargs, *, default_enable_hpi):
            seen.append(default_enable_hpi)
            return kwargs
        # Even a currently boolean module attribute is not lexical/source proof.
        modules = {"paddleocr._common_args": SimpleNamespace(parse_common_args=parser),
                   "paddleocr._pipelines.base": SimpleNamespace(parse_common_args=parser, DEFAULT=False)}
        sources = [
            "DEFAULT = bool(0)\nparse_common_args(kwargs, default_enable_hpi=DEFAULT)",
            "parse_common_args(kwargs, default_enable_hpi=bool(0))",
            "DEFAULT = False\ndef build():\n    DEFAULT = True\n    return parse_common_args(kwargs, default_enable_hpi=DEFAULT)",
            "DEFAULT = False\nDEFAULT = True\nparse_common_args(kwargs, default_enable_hpi=DEFAULT)",
            "if condition:\n    DEFAULT = True\nelse:\n    DEFAULT = False\nparse_common_args(kwargs, default_enable_hpi=DEFAULT)",
            "parse_common_args(kwargs, default_enable_hpi=True)\nparse_common_args(kwargs, default_enable_hpi=False)",
            "parse_common_args(kwargs, default_enable_hpi=self._get_default_enable_hpi())",
        ]
        for source in sources:
            with self.subTest(source=source), patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                    patch.object(ocr.inspect, "getsource", side_effect=["parser source", source]):
                kwargs, evidence = ocr.constructor_backend(lambda **kwargs: None)
            self.assertEqual(kwargs, {})
            self.assertIsNone(evidence["accepted_by"])
            self.assertEqual(evidence["invocation"]["kwargs"], {})
            self.assertIn("Traceback", evidence["error"]["traceback"])
        self.assertEqual(seen, [])

    def test_all_introspection_errors_are_evidence_and_api_stays_closed(self):
        for method in ("signature", "getsource"):
            with self.subTest(method=method), patch.object(ocr.core.importlib, "import_module", return_value=SimpleNamespace(
                    parse_common_args=lambda kwargs: kwargs)), \
                    patch.object(ocr.inspect, method, side_effect=RuntimeError("introspection failure")):
                kwargs, evidence = ocr.constructor_backend(lambda **kwargs: None)
            self.assertEqual(kwargs, {})
            self.assertIsNone(evidence["accepted_by"])
            self.assertEqual(evidence["error"]["type"], "RuntimeError")
            self.assertIn("introspection failure", evidence["error"]["traceback"])

    def test_parser_rejection_retains_signature_hash_kwargs_and_traceback(self):
        def parser(kwargs, *, default_enable_hpi):
            raise TypeError("installed parser rejected option")
        modules = {"paddleocr._common_args": SimpleNamespace(parse_common_args=parser),
                   "paddleocr._pipelines.base": SimpleNamespace(parse_common_args=parser)}
        with patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                patch.object(ocr.inspect, "getsource", side_effect=["parser source", "parse_common_args(kwargs, default_enable_hpi=False)"]):
            kwargs, evidence = ocr.constructor_backend(lambda **kwargs: None)
        self.assertEqual(kwargs, {})
        self.assertEqual(evidence["invocation"]["kwargs"], {"default_enable_hpi": False})
        self.assertIn("default_enable_hpi", evidence["parser_signature"])
        self.assertEqual(len(evidence["parser_sha256"]), 64)
        self.assertEqual(evidence["error"]["type"], "TypeError")
        self.assertIn("installed parser rejected option", evidence["error"]["traceback"])

    def subprocess_fixture(self, reports, stderr=""):
        def child(command, **kwargs):
            self.assertNotIn("-S", command)
            mode = command[command.index("--backend-mode") + 1]
            env = kwargs["env"]
            if mode == "mkldnn-flag":
                self.assertEqual(env["FLAGS_use_mkldnn"], "0")
            else:
                self.assertNotIn("FLAGS_use_mkldnn", env)
            self.assertNotIn("FLAGS_enable_pir_api", env)
            report = reports[mode]
            ocr.atomic_json(Path(command[command.index("--report") + 1]), report)
            return SimpleNamespace(returncode=0 if report["success"] else 2, stdout="", stderr=stderr)
        return child

    def test_subprocess_safe_order_fallback_and_wrapt_diagnostic(self):
        reports = {"api": {"success": False, "flags": {"FLAGS_use_mkldnn": {"supported": True}}},
                   "mkldnn-flag": {"success": True, "mode": "mkldnn-flag"}}
        with patch.object(ocr.subprocess, "run", side_effect=self.subprocess_fixture(
                reports, "Error in sitecustomize: No module named wrapt")) as child, patch("builtins.print") as output:
            result = ocr.smoke_subprocess(self.input, "config.json", self.diagnostics)
        self.assertEqual(result["mode"], "mkldnn-flag")
        self.assertEqual(child.call_count, 2)
        self.assertTrue(result["wrapt_sitecustomize_warning"])
        self.assertIn("WRAPT_SITECUSTOMIZE_WARNING", str(output.call_args_list))
        self.assertFalse(self.output.exists())

    def test_no_supported_flag_stops_and_child_without_report_cannot_succeed(self):
        for reports in ({"api": {"success": False, "flags": {"FLAGS_use_mkldnn": {"supported": False}}}},
                        {"api": {"success": False}}):
            with patch.object(ocr.subprocess, "run", side_effect=self.subprocess_fixture(reports)) as child, \
                    patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "BACKEND_SMOKE_FAILED"):
                ocr.smoke_subprocess(self.input, "config.json", self.diagnostics)
            self.assertEqual(child.call_count, 1)
        with patch.object(ocr.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
                patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "BACKEND_SMOKE_FAILED"):
            ocr.smoke_subprocess(self.input, "config.json", self.diagnostics)
        self.assertFalse(self.output.exists())

    def test_stale_candidate_success_cannot_pass_child_without_report(self):
        self.seed_stale_diagnostics()
        with patch.object(ocr.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
                patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "BACKEND_SMOKE_FAILED"):
            ocr.smoke_subprocess(self.input, "config.json", self.diagnostics)
        self.assertFalse((self.diagnostics / "selected.json").exists())
        self.assertFalse((self.diagnostics / "mkldnn-flag.json").exists())
        failed = ocr.read_json(self.diagnostics / "failed.json")
        self.assertEqual(len(failed["attempts"]), 1)
        self.assertFalse(failed["attempts"][0]["success"])
        self.assertEqual(failed["invocation"], ocr.read_json(self.diagnostics / "run-fatal.json")["invocation"])

    def test_flag_evidence_uses_get_flags_not_environment(self):
        def flags(names):
            if names[0] == "FLAGS_enable_pir_api":
                raise ValueError("unknown flag")
            return {names[0]: True}

        with patch.dict(sys.modules, {"paddle": SimpleNamespace(get_flags=flags)}), \
                patch.dict(os.environ, {"FLAGS_use_mkldnn": "0"}):
            evidence = ocr.flag_evidence()
        self.assertTrue(evidence["FLAGS_use_mkldnn"]["effective"])
        self.assertFalse(evidence["FLAGS_enable_pir_api"]["supported"])

    def test_actual_adapter_passes_verified_api_option_and_locked_models(self):
        config = ocr.config_for()
        models = self.root / "ocr-models"
        for model in config["models"]:
            directory = models / model["name"]
            directory.mkdir(parents=True)
            for name in ocr.core.MODEL_FILES:
                (directory / name).write_bytes(b"offline mock model")
            ocr.atomic_json(directory / "model-lock.json", {"repo": model["repo"], "revision": model["revision"],
                            "files_sha256": {n: ocr.digest(directory / n) for n in ocr.core.MODEL_FILES}})
        captured = {}

        class FakePaddleOCR:
            def __init__(self, enable_mkldnn=True, **kwargs):
                captured.update(enable_mkldnn=enable_mkldnn, **kwargs)

            def predict(self, array, **kwargs):
                captured["pixel"] = array[0, 0].tolist()
                return [SimpleNamespace(json={"res": {"rec_texts": [], "rec_scores": [], "rec_polys": []}})]

        flags = {"FLAGS_use_mkldnn": {"supported": False}, "FLAGS_enable_pir_api": {"supported": True, "effective": True}}
        with patch.object(ocr.core, "runtime_preflight", return_value=config["versions"]), \
                patch.object(ocr, "flag_evidence", return_value=flags), \
                patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=FakePaddleOCR)}), \
                patch.dict(os.environ, {"KIS_OCR_MODEL_ROOT": str(models), "KIS_OCR_OFFLINE": "1"}), \
                patch("builtins.print"):
            adapter = ocr.PaddleAdapter(config)
            self.assertIs(captured["enable_mkldnn"], False)
            self.assertFalse(captured["use_doc_unwarping"])
            self.assertEqual(adapter.recognize(Image.new("RGB", (84, 74), (10, 20, 30))), [])
            self.assertEqual(captured["pixel"], [30, 20, 10])
            self.assertNotIn(str(self.root), ocr.encoded(adapter.engine).decode())
            adapter.check_models()
            next(models.rglob("inference.json")).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                ocr.PaddleAdapter(config)
            with self.assertRaisesRegex(RuntimeError, "not effective"):
                ocr.PaddleAdapter(config, "mkldnn-flag")

            # API verification failure is NOT a veto of an independently verified flag.
            failed_api = {"accepted_by": None, "error": {"traceback": "original parser error"}}
            with patch.object(ocr, "constructor_backend", return_value=({}, failed_api)):
                with self.assertRaisesRegex(RuntimeError, "BACKEND_API_UNAVAILABLE"):
                    ocr.PaddleAdapter(config, "api")
                for flag in ({"supported": False}, {"supported": True, "effective": True}):
                    with patch.object(ocr, "flag_evidence", return_value={"FLAGS_use_mkldnn": flag}), \
                            self.assertRaisesRegex(RuntimeError, "not effective"):
                        ocr.PaddleAdapter(config, "mkldnn-flag")
                # Restore the changed mock model file before the flag-only initialization.
                next(models.rglob("inference.json")).write_bytes(b"offline mock model")
                with patch.object(ocr, "flag_evidence", return_value={"FLAGS_use_mkldnn": {"supported": True, "effective": False}}):
                    flag_adapter = ocr.PaddleAdapter(config, "mkldnn-flag")
                self.assertEqual(flag_adapter.backend["constructor"], {})
                self.assertEqual(flag_adapter.backend["api"], failed_api)
                image_path = self.image()
                smoke = ocr.smoke(self.input, ocr.core.discover(self.input), config, flag_adapter)
                self.assertEqual(smoke["image_sha256"], ocr.digest(image_path))
                self.assertEqual(smoke["lines"], 0)

            # Constructor failure must retain the verified API evidence and flags.
            def rejected(**kwargs):
                raise RuntimeError("constructor rejected backend")
            def parser(kwargs, *, default_enable_hpi):
                return kwargs
            modules = {"paddleocr._common_args": SimpleNamespace(parse_common_args=parser),
                       "paddleocr._pipelines.base": SimpleNamespace(parse_common_args=parser)}
            report_path = self.diagnostics / "api.json"
            with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=rejected)}), \
                    patch.object(ocr, "apply_backend"), \
                    patch.object(ocr.core.importlib, "import_module", side_effect=modules.get), \
                    patch.object(ocr.inspect, "getsource", side_effect=["parser source", "parse_common_args(kwargs, default_enable_hpi=False)"]):
                code = ocr.smoke_worker(self.input, None, report_path, "api")
            self.assertEqual(code, 2)
            report = ocr.read_json(report_path)
            self.assertEqual(report["backend"]["constructor"], {"enable_mkldnn": False})
            self.assertEqual(report["backend"]["flags"], flags)
            api = report["backend"]["api"]
            self.assertEqual(api["accepted_by"], "installed-common-args-parser")
            self.assertEqual(api["invocation"]["kwargs"], {"default_enable_hpi": False})
            self.assertEqual(len(api["parser_sha256"]), 64)
            self.assertIn("default_enable_hpi", api["parser_signature"])
            self.assertIn("constructor rejected backend", report["error"]["traceback"])
            self.assertFalse(self.output.exists())

    def test_unverified_constructor_stops_before_models_no_blind_kwargs_retry(self):
        config = ocr.config_for()
        factory = Mock()
        with patch.object(ocr.core, "runtime_preflight", return_value=config["versions"]), \
                patch.object(ocr, "flag_evidence", return_value={}), \
                patch.object(ocr, "constructor_backend", return_value=({}, {"accepted_by": None})), \
                patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=factory)}), \
                patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "BACKEND_API_UNAVAILABLE"):
            ocr.PaddleAdapter(config)
        factory.assert_not_called()

    def test_smoke_worker_captures_backend_phase_and_full_traceback(self):
        self.image()
        adapter = Adapter(failure_at=1)
        report_path = self.root / "worker.json"
        with patch.object(ocr, "apply_backend") as environment, \
                patch.object(ocr.core, "runtime_preflight"), \
                patch.object(ocr, "flag_evidence", return_value={}), \
                patch.object(ocr, "PaddleAdapter", return_value=adapter), patch("builtins.print"):
            adapter.backend = {"mode": "api"}
            result = ocr.smoke_worker(self.input, None, report_path, "api")
        self.assertEqual(result, 2)
        environment.assert_called_once_with("api", supported=False)
        report = ocr.read_json(report_path)
        self.assertFalse(report["success"])
        self.assertEqual(report["error"]["phase"], "one-image-inference")
        self.assertIn("NotImplementedError", report["error"]["traceback"])
        self.assertFalse(self.output.exists())

    def test_subprocess_smoke_failure_stops_real_run_before_adapter_or_output(self):
        self.image()
        with patch.object(ocr, "smoke_subprocess", side_effect=RuntimeError("BACKEND_SMOKE_FAILED")), \
                patch.object(ocr, "PaddleAdapter") as adapter, patch("builtins.print"), \
                self.assertRaisesRegex(RuntimeError, "BACKEND_SMOKE_FAILED"):
            ocr.run(self.input, self.output)
        adapter.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_profile_exact_pins_cpu_and_model_revision_unchanged(self):
        for minor, numpy in ((10, "1.26.4"), (11, "1.26.4"), (12, "1.26.4"), (13, "2.2.6")):
            with patch.object(sys, "version_info", (3, minor, 0)):
                config = ocr.config_for()
                profile = ocr.select_profile()
                pins = dict(row.split("==") for row in (ocr.BASE / profile["requirements"]).read_text().splitlines()
                            if row and not row.startswith("#"))
                self.assertEqual(pins, config["versions"])
                self.assertEqual(config["versions"]["numpy"], numpy)
                self.assertEqual(config["device"], "cpu")
                if minor == 13:
                    self.assertEqual(config, ocr.read_json(ocr.BASE / "config/ocr-kaggle-py313.json"))
                    self.assertTrue(profile["config"].endswith("-v5.json"))

    def test_resume_hash_backend_mismatch_and_zip_safety(self):
        image = self.image()
        self.start()
        original = {p.relative_to(self.output).as_posix(): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        adapter = Adapter()
        self.start(adapter, resume=True)
        self.assertEqual(len(adapter.calls), 1)  # mandatory smoke even on resume
        self.assertEqual(original, {p.relative_to(self.output).as_posix(): p.read_bytes()
                                    for p in self.output.rglob("*") if p.is_file()})
        first, second = self.root / "one.zip", self.root / "two.zip"
        self.assertEqual(ocr.package(self.output, first), ocr.package(self.output, second))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertIn("backend-v5.json", ocr.verify_zip(first))
        with zipfile.ZipFile(first) as archive:
            self.assertFalse(any("diagnostics" in name for name in archive.namelist()))
        for name in ("../x", "/x", "C:/x", "a\\b", "a/../b"):
            with self.assertRaises(ValueError):
                ocr.core.write_zip(self.root / "unsafe.zip", {name: b"bad"})
        image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.start(resume=True)
        image.write_bytes(image.read_bytes()[:-7])
        with patch.object(ocr, "source_hash", return_value="b" * 64), self.assertRaisesRegex(ValueError, "config/code hash"):
            self.start(resume=True)
        report = ocr.read_json(self.diagnostics / "run-fatal.json")
        self.assertEqual(report["phase"], "resume-validation")
        self.assertIn("config/code hash", report["message"])
        (self.output / "backend-v5.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "backend checksum"):
            self.start(resume=True)

    def test_resume_config_model_backend_identity_changes_fail_closed(self):
        self.image()
        self.start()
        config = ocr.config_for()
        config["padding"]["left"] = 33
        changed = self.root / "config.json"
        ocr.atomic_json(changed, config)
        with self.assertRaisesRegex(ValueError, "config/code hash"):
            self.start(resume=True, config_path=changed)
        adapter = Adapter()
        adapter.engine["models"][0]["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "model/runtime hash"):
            self.start(adapter, resume=True)
        adapter = Adapter()
        adapter.backend = {"mode": "different-backend"}
        with self.assertRaisesRegex(ValueError, "backend hash"):
            self.start(adapter, resume=True)

    def test_dependency_rerun_failure_clears_flags_and_blocks_download(self):
        notebook = json.loads((ocr.BASE / builder.NOTEBOOK).read_bytes())
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertIn("RUNTIME_READY = False\nOCR_RUN_OK = False\n", codes[1])
        self.assertTrue(codes[2].startswith("RUNTIME_READY = False\nOCR_RUN_OK = False\n"))
        self.assertIn("globals().get('RUNTIME_READY', False) and globals().get('OCR_RUN_OK', False)",
                      codes[4])
        attached = self.root / "attached"
        attached.mkdir()
        output = self.root / "outputs/ocr-v1"
        output.mkdir(parents=True)
        (output / "manifest.json").write_bytes(b"{}\n")
        namespace = {"RUNTIME_READY": True, "OCR_RUN_OK": True, "output": output,
                     "Path": Path, "reject_links": builder.helpers_namespace()["reject_links"],
                     "work": self.root / "work", "sys": sys, "json": json,
                     "INPUT_ROOT": str(self.input), "ocr": ocr}
        dependency = codes[2].replace("Path('/kaggle/input')", f"Path({str(attached)!r})")
        with patch.object(ocr, "prepare_runtime", side_effect=RuntimeError("simulated dependency rerun failure")), \
                patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "simulated dependency rerun failure"):
            exec(compile(dependency, "v5-dependencies", "exec"), namespace)
        self.assertFalse(namespace["RUNTIME_READY"])
        self.assertFalse(namespace["OCR_RUN_OK"])
        self.assertTrue((output / "manifest.json").is_file())
        with self.assertRaisesRegex(RuntimeError, "no output package before successful run/validation"):
            exec(compile(codes[4], "v5-download", "exec"), namespace)

    def test_notebook_setup_errors_replace_stale_reports_before_prepare_runtime(self):
        self.image()
        work = self.root / "work"
        diagnostics = work / "outputs/diagnostics-v5"
        for scenario in ("ocr-wheelhouse", "ocr-models", "reject-links"):
            attached = self.root / "attached" / scenario
            attached.mkdir(parents=True)
            if scenario != "reject-links":
                for name in ("a", "b"):
                    (attached / name / scenario).mkdir(parents=True)
            for name in ("selected.json", "api.json", "failed.json", "run-fatal.json"):
                ocr.atomic_json(diagnostics / name, {"success": True, "stale": True})
            ocr.atomic_json(diagnostics / "invocation.json", {"id": "old"})
            def reject(path):
                if scenario == "reject-links" and path == work / "venv":
                    raise ValueError("setup reject_links failure")
                return builder.helpers_namespace()["reject_links"](path)
            namespace = {"RUNTIME_READY": True, "OCR_RUN_OK": True, "Path": Path, "work": work,
                         "INPUT_ROOT": str(self.input), "ocr": ocr, "reject_links": reject,
                         "sys": sys, "json": json}
            code = builder.DEPENDENCIES.replace("Path('/kaggle/input')", f"Path({str(attached)!r})")
            with self.subTest(scenario=scenario), patch.object(ocr, "prepare_runtime", return_value=(
                    Path(sys.executable), Path("config.json"), {})) as prepare, \
                    patch("builtins.print"), self.assertRaises(ValueError):
                exec(compile(code, "v5-setup-error", "exec"), namespace)
            report = ocr.read_json(diagnostics / "run-fatal.json")
            marker = ocr.read_json(diagnostics / "invocation.json")
            self.assertNotEqual(marker["id"], "old")
            self.assertEqual(marker["command"], "notebook-setup")
            self.assertEqual(report["invocation"], marker)
            self.assertEqual(report["phase"], "notebook-setup")
            self.assertEqual(report["type"], "ValueError")
            self.assertIn("Traceback", report["traceback"])
            self.assertFalse((diagnostics / "selected.json").exists())
            self.assertFalse((diagnostics / "api.json").exists())
            self.assertFalse((diagnostics / "failed.json").exists())
            self.assertFalse(namespace["RUNTIME_READY"])
            self.assertFalse(namespace["OCR_RUN_OK"])
            self.assertFalse((work / "outputs/ocr-v1").exists())
            if scenario == "ocr-models":
                prepare.assert_called_once()
                self.assertEqual(prepare.call_args.kwargs["state"]["invocation"], marker)
            else:
                prepare.assert_not_called()

    def test_notebook_run_prints_redacted_child_streams_and_exact_fatal_path(self):
        namespace = {"RUNTIME_READY": True, "OCR_RUN_OK": True, "work": self.output.parent.parent,
                     "root": self.root, "reject_links": lambda p: p, "python": Path(sys.executable),
                     "script": Path(ocr.__file__), "config_path": Path("config.json"), "INPUT_ROOT": str(self.input),
                     "runtime_env": {}, "ocr": ocr, "subprocess": subprocess, "sys": sys, "json": json}
        completed = SimpleNamespace(returncode=2, stdout="candidate stdout token=private",
                                    stderr="Traceback (most recent call last):\nImportError: child root error hf_test_secret")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_secret"}), \
                patch.object(subprocess, "run", return_value=completed) as child, \
                patch("builtins.print") as printed, self.assertRaises(RuntimeError) as stopped:
            exec(compile(builder.RUN, "v5-run", "exec"), namespace)
        self.assertFalse(namespace["OCR_RUN_OK"])
        child.assert_called_once()
        self.assertTrue(child.call_args.kwargs["capture_output"])
        self.assertTrue(child.call_args.kwargs["text"])
        self.assertFalse(child.call_args.kwargs["check"])
        output = str(printed.call_args_list)
        self.assertIn("candidate stdout", output)
        self.assertIn("child root error", output)
        self.assertIn("Traceback", output)
        self.assertNotIn("token=private", output)
        self.assertNotIn("hf_test_secret", output)
        self.assertIn(str(namespace["output"].parent / "diagnostics-v5/run-fatal.json"), str(stopped.exception))
        self.assertIn("child exit 2", str(stopped.exception))
        with self.assertRaisesRegex(RuntimeError, "no output package"):
            exec(compile(builder.DOWNLOAD, "v5-download", "exec"), namespace)

    def test_notebook_only_bootstrap_in_fresh_process_without_repo_or_zip(self):
        self.image()
        notebook = self.root / "only-notebook.ipynb"
        notebook.write_bytes((ocr.BASE / builder.NOTEBOOK).read_bytes())
        probe = '''import json, sys
from pathlib import Path
document = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
code = ''.join(document['cells'][2]['source']).replace('/kaggle/working/kis-kaggle-ocr-v5', sys.argv[3])
namespace = {'INPUT_ROOT': sys.argv[2]}
exec(compile(code, 'notebook-only-v5', 'exec'), namespace)
assert namespace['script'].name == 'kaggle_ocr_v5.py'
assert len([p for p in namespace['root'].rglob('*') if p.is_file()]) == 8
assert not any(n.startswith(('paddle', 'kis', 'huggingface_hub', 'tools')) for n in sys.modules)
assert not (namespace['work'] / 'outputs/ocr-v1').exists()
assert (namespace['work'] / 'outputs/diagnostics-v5/invocation.json').is_file()
print('V5_STANDALONE_BOOTSTRAP_OK')
'''
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(notebook),
                                    str(self.input), (self.root / "k").as_posix()],
                                   cwd=self.root, capture_output=True, text=True, check=True, timeout=30)
        self.assertIn("V5_STANDALONE_BOOTSTRAP_OK", completed.stdout)

    def test_notebook_hash_payload_input_only_and_no_package_discovery(self):
        data = (ocr.BASE / builder.NOTEBOOK).read_bytes()
        self.assertEqual(data, builder.build_notebook())
        values = builder.verify_notebook(data)
        helpers = builder.helpers_namespace()
        files = helpers["validate_payload"](values["EMBEDDED_PAYLOAD"], values["PAYLOAD_SHA256"])
        self.assertEqual(set(files), set(builder.PAYLOAD_FILES))
        for name, content in files.items():
            self.assertEqual(content, builder.source_bytes(ocr.BASE / name))
            self.assertFalse(name.startswith(("kis/", "dataset/", "dist/")))
        notebook = json.loads(data)
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(codes[0], 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n')
        for index, code in enumerate(codes):
            compile(code, f"v5-{index}", "exec")
            self.assertNotIn("kis_kaggle_ocr_paddle_padding_v4", code)
            self.assertNotIn("kis_kaggle_ocr_paddle_padding_v3", code)
            self.assertNotIn("'-S'", code)
        self.image()
        working = self.root / "k"
        namespace = {"INPUT_ROOT": str(self.input)}
        with patch.object(ocr.core.subprocess, "run") as child, patch("builtins.print"), \
                patch("zipfile.ZipFile", side_effect=AssertionError("No external ZIP reads")):
            exec(compile(codes[1].replace("/kaggle/working/kis-kaggle-ocr-v5", working.as_posix()), "bootstrap", "exec"), namespace)
        self.assertEqual(namespace["script"].name, "kaggle_ocr_v5.py")
        self.assertEqual(child.call_count, 1)
        self.assertEqual(len(list(namespace["root"].rglob("*.py"))), 2)
        self.assertFalse((namespace["work"] / "outputs").exists())
        changed = deepcopy(values["EMBEDDED_PAYLOAD"])
        changed["files"][0]["path"] = "../escape"
        with self.assertRaises(ValueError):
            helpers["materialize_payload"](changed, values["PAYLOAD_SHA256"], self.root / "never")
        self.assertFalse((self.root / "never").exists())
        with patch.object(builder, "BASE", self.root), patch("builtins.print"):
            target = self.root / builder.NOTEBOOK
            target.parent.mkdir(parents=True)
            target.write_bytes(data + b"\n")
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                builder.main(["--check"])

    def test_host_pip_venv_only_binary_pins_no_ensurepip(self):
        work = self.root / "runtime"

        def create(directory):
            directory.mkdir(parents=True)
            (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n")

        def command(args, **kwargs):
            if "-c" in args:
                return SimpleNamespace(stdout=json.dumps({"version": list(sys.version_info[:2]),
                                       "base": sys.base_prefix, "prefix": str(Path(args[0]).parent.parent),
                                       "executable": args[0], "user_site": False}))
            if "--version" in args:
                return SimpleNamespace(stdout="pip 25.3 from host")
            if "--help" in args:
                return SimpleNamespace(stdout="--python --dry-run --report")
            return SimpleNamespace(returncode=0)

        with patch("venv.EnvBuilder") as venv, patch.object(ocr.subprocess, "run", side_effect=command) as child:
            venv.return_value.create.side_effect = create
            python, config, identity = ocr.prepare_runtime(work)
            venv.assert_called_once_with(with_pip=False, system_site_packages=False)
            self.assertEqual(identity["mode"], "venv-host-pip")
            installs = [c.args[0] for c in child.call_args_list if "install" in c.args[0] and "--help" not in c.args[0]]
            self.assertEqual(len(installs), 2)
            for args in installs:
                self.assertIn("--only-binary=:all:", args)
                self.assertEqual(args[args.index("--python") + 1], str(python))
            self.assertIn("preflight", child.call_args.args[0])
            self.assertTrue(child.call_args.args[0][2].endswith("kaggle_ocr_v5.py"))
            self.assertEqual(config, ocr.BASE / ocr.select_profile()["config"])
            self.assertTrue((work / "venv/requirements.sha256").exists())
            def fail_preflight(args, **kwargs):
                if "preflight" in args:
                    raise subprocess.CalledProcessError(2, args, output="preflight stdout", stderr="full child traceback")
                return command(args, **kwargs)
            child.side_effect = fail_preflight
            with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "binary-only"):
                ocr.prepare_runtime(work)
            fatal = ocr.read_json(work / "outputs/diagnostics-v5/run-fatal.json")
            self.assertEqual(fatal["phase"], "runtime-preflight")
            self.assertEqual(fatal["stderr"], "full child traceback")
            self.assertEqual(fatal["stdout"], "preflight stdout")
            self.assertFalse((work / "venv/requirements.sha256").exists())


if __name__ == "__main__":
    unittest.main()
