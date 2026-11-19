"""Offline portable OCR contract tests: tiny generated images and a mock adapter."""

import ast
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import venv
import zipfile

from PIL import Image

from tools import kaggle_ocr as ocr
from tools import build_kaggle_ocr_notebook as notebook_builder


class MockAdapter:
    def __init__(self, lines=None, failure=None):
        config = ocr.load_config()
        self.engine = {"name": "paddleocr", "versions": {"paddleocr": "3.7.0", "paddlepaddle": "3.3.1"},
                       "models": [{**m, "files_sha256": {name: "a" * 64 for name in ocr.MODEL_FILES}}
                                  for m in config["models"]],
                       "device": "cpu", "language": config["language"], "config_sha256": "0" * 64}
        self.lines = lines or []
        self.failure = failure
        self.calls = []

    def recognize(self, image):
        self.calls.append((image.size, image.getpixel((0, 0))))
        if self.failure:
            raise self.failure
        return deepcopy(self.lines)


def line(text="sign", x=34, y=34):
    return {"text": text, "confidence": 0.8,
            "polygon": [[x, y], [x + 6, y], [x + 6, y + 4], [x, y + 4]]}


class KaggleOCRTest(unittest.TestCase):
    def setUp(self):
        # Windows platform.machine() can launch a subprocess on its first call.
        # Prime its cache before mocking subprocess.run for venv/pip tests.
        ocr.interpreter_fingerprint()
        # Import schema dependencies against the real host before mocking kernel versions.
        ocr.schema_validator()
        self.actual_python = sys.version_info[:2]
        if not (3, 10) <= self.actual_python <= (3, 13):
            # Mock supported Kaggle kernels for offline contracts, not real inference.
            kernel = patch.object(sys, "version_info", (3, 13, 0))
            kernel.start()
            self.addCleanup(kernel.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "upload"
        self.output = self.root / "output"
        # The fixture must never accidentally start an actual network request.
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def image(self, relative="L77_V003/1.jpg", color=(10, 20, 30)):
        path = self.input / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), color).save(path)
        return path

    def records(self):
        paths = sorted(self.output.glob("ocr/*/*.jsonl"))
        return [json.loads(row) for path in paths for row in path.read_text(encoding="utf-8").splitlines()]

    def start(self, adapter=None, **kwargs):
        return ocr.run(self.input, self.output, adapter=adapter or MockAdapter(), **kwargs)

    def test_layouts_and_group_inference_are_not_group_whitelisted(self):
        for relative in ("L77/L77_V003/10.jpg", "L77/L77_V003/2.png", "L123/L123_V001/1.jpeg"):
            self.image(relative)
        inventory = ocr.discover(self.input)
        self.assertEqual([(e["video_id"], e["keyframe_n"]) for e in inventory["records"]],
                         [("L123_V001", 1), ("L77_V003", 2), ("L77_V003", 10)])
        self.assertEqual(len(ocr.discover(self.input / "L77")["records"]), 2)
        self.assertEqual(len(ocr.discover(self.input / "L77/L77_V003")["records"]), 2)
        self.assertEqual(inventory["records"][1]["group"], "L77")

    def test_root_video_and_numeric_keyframe_order(self):
        self.image("L88_V001/010.JPG")
        self.image("L88_V001/002.png")
        self.image("L88_V001/1.jpeg")
        entries = ocr.discover(self.input / "L88_V001")["records"]
        self.assertEqual([e["keyframe_n"] for e in entries], [1, 2, 10])
        self.assertEqual(entries[1]["source_path"], "002.png")
        self.assertIsNone(entries[0]["mapping"])

    def test_mapping_keyframe_n_is_never_frame_idx_and_declares_missing(self):
        self.image("L77_V003/002.png")
        (self.input / "L77_V003/mapping.csv").write_text(
            "n,frame_idx,pts_time\n2,570,19.0\n3,600,20.0\n", encoding="utf-8")
        self.start()
        records = self.records()
        self.assertEqual(records[0]["keyframe_n"], 2)
        self.assertEqual(records[0]["mapping"]["frame_idx"], 570)
        self.assertEqual(records[0]["mapping"]["timestamp_seconds"], 19.0)
        self.assertEqual(records[1]["status"], "missing")
        self.assertIsNone(records[1]["image_sha256"])
        self.assertIsNone(records[1]["preprocessing"])
        coverage = ocr.validate_output(self.output, self.input)
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["status_counts"]["empty"], 1)
        self.assertEqual(coverage["status_counts"]["missing"], 1)

    def test_padding_default_inverse_bounds_and_line_order(self):
        self.image()
        adapter = MockAdapter([line("bottom", x=35, y=38), line("right", x=40), line("left")])
        self.start(adapter)
        self.assertEqual(adapter.calls, [((84, 74), (255, 255, 255))])
        record = self.records()[0]
        self.assertEqual(record["status"], "ok")
        self.assertEqual([item["text"] for item in record["lines"]], ["left", "right", "bottom"])
        self.assertEqual(record["lines"][0]["polygon_original"][0], [2.0, 2.0])
        self.assertEqual(record["preprocessing"]["inverse_transform"], [[1, 0, -32], [0, 1, -32], [0, 0, 1]])
        self.assertIsNone(record["preprocessing"]["resize"])
        self.assertEqual(ocr.validate_output(self.output)["status_counts"]["ok"], 1)

    def test_exact_inverse_is_not_silently_clipped(self):
        prep = ocr.preprocessing((20, 10), ocr.load_config()["padding"])
        result = ocr.normalize_lines([line(x=1, y=2)], prep)[0]
        self.assertEqual(result["polygon_original"][0], [-31.0, -30.0])
        self.assertEqual(result["polygon_original_clipped"][0], [0, 0])
        with self.assertRaisesRegex(ValueError, "outside"):
            ocr.normalize_lines([line(x=83)], prep)

    def test_asymmetric_padding_and_invalid_resize(self):
        self.image()
        config = ocr.load_config()
        config["padding"].update(left=2, top=3, right=4, bottom=5)
        path = self.root / "config.json"
        ocr.atomic_json(path, config)
        adapter = MockAdapter([line(x=4, y=5)])
        self.start(adapter, config_path=path)
        self.assertEqual(adapter.calls[0][0], (26, 18))
        self.assertEqual(self.records()[0]["lines"][0]["polygon_original"][0], [2.0, 2.0])
        config["resize"] = [10, 5]
        ocr.atomic_json(path, config)
        with self.assertRaisesRegex(ValueError, "resize"):
            ocr.load_config(path)

    def test_line_sorting_is_deterministic_including_ties(self):
        prep = ocr.preprocessing((20, 10), ocr.load_config()["padding"])
        raw = [line("z"), line("a"), line("last", y=38), line(" ")]
        self.assertEqual(ocr.normalize_lines(raw, prep), ocr.normalize_lines(raw[::-1], prep))
        self.assertEqual([r["line_index"] for r in ocr.normalize_lines(raw, prep)], [0, 1, 2])
        for invalid in (float("nan"), float("inf"), 1.1, -0.1):
            bad = dict(line(), confidence=invalid)
            with self.assertRaises(ValueError):
                ocr.normalize_lines([bad], prep)

    def test_empty_error_unknown_are_not_collapsed(self):
        self.image()
        adapter = MockAdapter(failure=RuntimeError("secret absolute path C:/do-not-export"))
        self.start(adapter)
        record = self.records()[0]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"], {"type": "RuntimeError", "code": "OCR_FAILED"})
        self.assertNotIn("secret", ocr.encoded(record).decode())
        record.update(status="unknown", error=None, preprocessing=None)
        ocr.validate_record(record)
        checkpoint = ocr.read_json(self.output / "checkpoint.json")
        ocr._store_record(self.output, record, checkpoint)
        adapter = MockAdapter()
        self.start(adapter, resume=True)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(self.records()[0]["status"], "empty")

    def test_error_retry_requires_explicit_flag(self):
        self.image()
        self.start(MockAdapter(failure=RuntimeError("failed")))
        adapter = MockAdapter()
        self.start(adapter, resume=True)
        self.assertFalse(adapter.calls)
        self.assertEqual(self.records()[0]["status"], "error")
        self.start(adapter, resume=True, retry_errors=True)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(self.records()[0]["status"], "empty")

    def test_unavailable_runtime_is_fatal_without_output(self):
        self.image()
        with patch.object(ocr, "PaddleAdapter", side_effect=RuntimeError("dependency/model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                ocr.run(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_corrupt_image_is_error_not_empty(self):
        path = self.image()
        path.write_bytes(b"not an image")
        adapter = MockAdapter()
        self.start(adapter)
        self.assertFalse(adapter.calls)
        self.assertEqual(self.records()[0]["status"], "error")
        self.assertIsNone(self.records()[0]["preprocessing"])

    def test_resume_is_deterministic_and_never_overwrites_by_default(self):
        self.image("L77_V003/2.png")
        self.image("L77_V003/10.jpg")
        self.start(MockAdapter([line()]))
        before = {p.relative_to(self.output).as_posix(): p.read_bytes()
                  for p in self.output.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.start()
        adapter = MockAdapter([line()])
        self.start(adapter, resume=True)
        self.assertFalse(adapter.calls)
        after = {p.relative_to(self.output).as_posix(): p.read_bytes()
                 for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual([r["keyframe_n"] for r in self.records()], [2, 10])
        separate = self.root / "second-run"
        ocr.run(self.input, separate, adapter=MockAdapter([line()]))
        self.assertEqual(before, {p.relative_to(separate).as_posix(): p.read_bytes()
                                  for p in separate.rglob("*") if p.is_file()})

    def test_image_hash_config_and_model_mismatches_block_resume(self):
        image = self.image()
        self.start()
        original = image.read_bytes()
        image.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.start(resume=True)
        image.write_bytes(original)
        config = ocr.load_config()
        config["padding"]["left"] = 33
        path = self.root / "changed-config.json"
        ocr.atomic_json(path, config)
        with self.assertRaisesRegex(ValueError, "config/code hash mismatch"):
            self.start(resume=True, config_path=path)
        adapter = MockAdapter()
        adapter.engine["models"][0]["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "model/runtime hash mismatch"):
            self.start(adapter, resume=True)

    def test_new_image_and_changed_mapping_block_resume(self):
        self.image()
        mapping = self.input / "L77_V003/mapping.csv"
        mapping.write_text("n,frame_idx,pts_time\n1,100,1\n")
        self.start()
        extra = self.image("L77_V003/2.jpg")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.start(resume=True)
        extra.unlink()
        mapping.write_text("n,frame_idx,pts_time\n1,200,2\n")
        with self.assertRaisesRegex(ValueError, "mapping hash mismatch"):
            self.start(resume=True)

    def test_disappeared_known_image_becomes_missing_with_hash_retained(self):
        image = self.image()
        self.start()
        checksum = self.records()[0]["image_sha256"]
        image.unlink()
        self.start(resume=True)
        self.assertEqual(self.records()[0]["status"], "missing")
        self.assertEqual(self.records()[0]["image_sha256"], checksum)
        self.assertEqual(ocr.validate_output(self.output)["status_counts"]["missing"], 1)

    def test_crash_orphan_and_stale_summary_are_resume_safe(self):
        self.image()
        self.start()
        checkpoint = ocr.read_json(self.output / "checkpoint.json")
        record = self.records()[0]
        record.update(status="unknown", preprocessing=None)
        ocr._store_record(self.output, record, checkpoint)
        # Simulate a crash: checkpoint is newer than old video/manifest hashes.
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            ocr.validate_output(self.output)
        orphan = self.output / "records/orphan.json"
        orphan.write_text("unreferenced")
        self.start(resume=True)
        ocr.validate_output(self.output)
        archive = self.root / "result.zip"
        ocr.package(self.output, archive)
        self.assertNotIn("records/orphan.json", ocr.verify_zip(archive))

    def test_tampered_record_and_output_are_rejected(self):
        self.image()
        self.start()
        checkpoint = ocr.read_json(self.output / "checkpoint.json")
        record_path = self.output / next(iter(checkpoint["records"].values()))["path"]
        original = record_path.read_bytes()
        record_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.start(resume=True)
        record_path.write_bytes(original)
        (self.output / "ocr/L77/L77_V003.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            ocr.validate_output(self.output)

    def test_duplicates_malformed_and_unsupported_files(self):
        self.image()
        duplicate = self.image("L77_V003/001.png")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ocr.discover(self.input)
        duplicate.unlink()
        malformed = self.image("L77_V003/frame.jpg")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            ocr.discover(self.input)
        malformed.unlink()
        unsupported = self.input / "L77_V003/2.bmp"
        unsupported.write_bytes(b"bmp")
        with self.assertRaisesRegex(ValueError, "JPEG/PNG"):
            ocr.discover(self.input)

    def test_invalid_mapping_duplicates_nan_and_bad_headers(self):
        self.image()
        path = self.input / "L77_V003/mapping.csv"
        for data in ("n,frame_idx,pts_time\n1,1,0\n1,2,1\n", "n,frame_idx,pts_time\n1,1,nan\n",
                     "n,frame_idx,pts_time\n-1,1,0\n", "n,frame_id,time\n1,1,0\n"):
            path.write_text(data)
            with self.assertRaises(ValueError):
                ocr.discover(self.input)

    def test_ambiguous_and_conflicting_layouts_fail_actionably(self):
        self.image("mystery/1.jpg")
        with self.assertRaisesRegex(ValueError, "Cannot infer video IDs"):
            ocr.discover(self.input)
        self.image("L99/L77_V003/1.jpg")
        with self.assertRaises(ValueError):
            ocr.discover(self.input / "L99")
        with self.assertRaisesRegex(ValueError, "video directory"):
            ocr.discover(self.input / "mystery")

    def test_containment_and_output_inside_input(self):
        self.image()
        for name in ("../escape", "/absolute", "a\\b", "C:/absolute", "a/./b", "a//b",
                     "a\nb", "a\x00b", "con.txt", "a.", "a "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ocr.contained(self.input, name)
        with self.assertRaisesRegex(ValueError, "outside INPUT_ROOT"):
            ocr.run(self.input, self.input / "output", adapter=MockAdapter())

    def test_symlinks_are_rejected_without_following(self):
        self.image()
        outside = self.root / "outside"
        outside.mkdir()
        link = self.input / "L66_V001"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"OS cannot create symlink: {exc}")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            ocr.discover(self.input)
        with self.assertRaisesRegex(ValueError, "symlink"):
            ocr.discover(link)

    def test_mocked_symlink_detection_is_covered_on_windows(self):
        self.image()
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path.name == "1.jpg" or original(path)):
            with self.assertRaisesRegex(ValueError, "Symlinks"):
                ocr.discover(self.input)

    def test_schema_and_runtime_reject_invalid_status_and_geometry(self):
        self.image()
        self.start(MockAdapter([line()]))
        record = self.records()[0]
        for mutate in (
            lambda r: r.update(status="not_found"),
            lambda r: r.update(status="empty"),
            lambda r: r["preprocessing"]["inverse_transform"][0].__setitem__(2, -31),
            lambda r: r["lines"][0].__setitem__("line_index", 7),
            lambda r: r.update(keyframe_n=9),
            lambda r: r.update(source_path="../bad.jpg"),
        ):
            invalid = deepcopy(record)
            mutate(invalid)
            with self.assertRaises(Exception):
                ocr.validate_record(invalid)
        self.assertEqual(set(ocr.STATUSES), {"ok", "empty", "missing", "error", "unknown"})

    def test_package_zip_manifest_determinism_and_exclusions(self):
        self.image()
        self.start()
        # Even accidental unrelated files in run directory must not be exported.
        (self.output / "credentials.json").write_text("not for export")
        first, second = self.root / "one.zip", self.root / "two.zip"
        result = ocr.package(self.output, first)
        self.assertEqual(result, ocr.package(self.output, second))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertNotIn("credentials.json", result["members"])
        self.assertFalse(any(name.endswith((".jpg", ".png", ".pdiparams")) for name in result["members"]))
        with zipfile.ZipFile(first) as archive:
            self.assertTrue(all(i.date_time == (2020, 1, 1, 0, 0, 0) for i in archive.infolist()))
            self.assertTrue(all(i.create_system == 3 for i in archive.infolist()))
            self.assertNotIn(str(self.root).replace("\\", "/"), archive.read("manifest.json").decode())
        with self.assertRaises(FileExistsError):
            ocr.package(self.output, first)
        ocr.extract_zip(first, self.root / "expanded")
        ocr.validate_output(self.root / "expanded")

    def test_distribution_is_an_explicit_code_only_allowlist(self):
        result = ocr.package(ocr.BASE, self.root / "portable.zip", distribution=True)
        self.assertEqual(set(result["members"]), {*ocr.DIST_FILES, "README.md", "MANIFEST.sha256"})
        self.assertLess(result["bytes"], 200000)
        self.assertFalse(any(name.startswith(("kis/", "dataset/", "outputs/")) for name in result["members"]))
        self.assertFalse(any(name.endswith((".env", ".pdiparams", ".jpg")) for name in result["members"]))

    def test_zip_traversal_duplicate_symlink_and_hash_safety(self):
        for index, name in enumerate(("../x", "/x", "C:/x", "a\\b", "a/../b")):
            bad = self.root / f"bad-{index}.zip"
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr(name, b"x")
            with self.assertRaises(ValueError):
                ocr.verify_zip(bad)
        link_zip = self.root / "link.zip"
        with zipfile.ZipFile(link_zip, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
        with self.assertRaisesRegex(ValueError, "links"):
            ocr.verify_zip(link_zip)
        duplicate = self.root / "duplicate.zip"
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("a", b"x")
            archive.writestr("A", b"x")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ocr.verify_zip(duplicate)
        corrupt = self.root / "corrupt.zip"
        with zipfile.ZipFile(corrupt, "w") as archive:
            archive.writestr("a", b"changed")
            archive.writestr("MANIFEST.sha256", f"{'0' * 64}  a\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            ocr.extract_zip(corrupt, self.root / "never-created")
        self.assertFalse((self.root / "never-created").exists())

    def test_module_import_and_cli_validation_do_not_import_production_or_paddle(self):
        source = Path(ocr.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [node.module if isinstance(node, ast.ImportFrom) else alias.name
                 for node in imports for alias in (node.names if isinstance(node, ast.Import) else [None])]
        self.assertFalse(any(name.startswith(("kis", "pymilvus", "tools.build_ocr")) for name in names))
        self.assertNotIn("caption_generator", source)
        # A fresh module import and input validation must not touch optional backends.
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("paddle", "huggingface_hub", "kis", "pymilvus")):
                raise AssertionError(f"Forbidden import: {name}")
            return original_import(name, *args, **kwargs)

        self.image()
        with patch("builtins.__import__", side_effect=guarded_import):
            spec = importlib.util.spec_from_file_location("standalone_ocr_test", ocr.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for minor in (12, 13):
                with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)), \
                        patch("builtins.print"):
                    self.assertEqual(module.main(["validate-input", "--input-root", str(self.input)]), 0)
            with patch.object(sys, "version_info", (3, 14, 0)), patch("builtins.print") as output:
                self.assertEqual(module.main(["validate-input", "--input-root", str(self.input)]), 2)
                self.assertIn("DEPENDENCY_UNAVAILABLE", output.call_args.args[0])

    def test_paddle_v3_adapter_mocked_api_bgr_and_offline_model_lock(self):
        config = ocr.load_config()
        models = self.root / "ocr-models"
        for model in config["models"]:
            directory = models / model["name"]
            directory.mkdir(parents=True)
            for name in ocr.MODEL_FILES:
                (directory / name).write_bytes(b"mock weights")
            ocr.atomic_json(directory / "model-lock.json", {"repo": model["repo"], "revision": model["revision"],
                            "files_sha256": {name: ocr.digest(directory / name) for name in ocr.MODEL_FILES}})
        captured = {}

        class FakePaddle:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def predict(self, array, **kwargs):
                captured["pixel"] = array[0, 0].tolist()
                return [SimpleNamespace(json={"res": {"rec_texts": ["test"], "rec_scores": [0.9],
                                                      "rec_polys": [line()["polygon"]]}})]

        versions = lambda name: config["versions"]["paddlepaddle" if name.startswith("paddlepaddle") else name]
        with patch.dict("os.environ", {"KIS_OCR_MODEL_ROOT": str(models), "KIS_OCR_OFFLINE": "1"}), \
                 patch.object(ocr.importlib.metadata, "version", side_effect=versions), \
                 patch.object(ocr.subprocess, "run"), \
                 patch.object(ocr.importlib, "import_module"), \
                 patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=FakePaddle)}):
            adapter = ocr.PaddleAdapter(config)
            self.assertEqual(adapter.recognize(Image.new("RGB", (84, 74), (10, 20, 30)))[0]["text"], "test")
            self.assertEqual(captured["pixel"], [30, 20, 10])
            self.assertFalse(captured["kwargs"]["use_doc_unwarping"])
            self.assertNotIn(str(self.root), ocr.encoded(adapter.engine).decode())
            adapter.check_models()
            next(models.rglob("inference.json")).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                adapter.check_models()
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                ocr.PaddleAdapter(config)

    def test_paddle_dependency_absence_stops_before_download(self):
        with patch.object(ocr.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError("paddleocr")):
            with self.assertRaisesRegex(RuntimeError, "dependency unavailable"):
                ocr.PaddleAdapter(ocr.load_config())

    def test_profiles_select_actual_python_and_keep_exact_pins_in_sync(self):
        for minor, numpy in ((10, "1.26.4"), (11, "1.26.4"), (12, "1.26.4"), (13, "2.2.6")):
            with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)):
                profile = ocr.select_profile()
                config = ocr.load_config()
                self.assertEqual(config["versions"]["numpy"], numpy)
                requirements = (ocr.BASE / profile["requirements"]).read_text()
                pins = dict(row.split("==") for row in requirements.splitlines() if row and not row.startswith("#"))
                self.assertEqual(config["versions"], pins)
                self.assertEqual(ocr.check_profile_config(config), profile)
                other = deepcopy(config)
                other["versions"]["numpy"] = "2.2.6" if minor != 13 else "1.26.4"
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*config pins"):
                    ocr.check_profile_config(other)
        for version in ((3, 9), (3, 14), (4, 0)):
            with self.subTest(version=version), patch.object(sys, "version_info", version):
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*executable.*platform.*profile.*python"):
                    ocr.select_profile()

    def test_runtime_preflight_import_error_or_pip_conflict_stops_before_models_output(self):
        self.image()
        config = ocr.load_config()
        for failure in (ImportError("numpy ABI unavailable"), OSError("shared library unavailable")):
            with self.subTest(failure=failure), \
                    patch.object(ocr.importlib.metadata, "version", side_effect=config["versions"].get), \
                    patch.object(ocr.subprocess, "run") as command, \
                    patch.object(ocr.importlib, "import_module", side_effect=failure), \
                    patch.dict(sys.modules, {"huggingface_hub": None}):
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*import smoke"):
                    ocr.run(self.input, self.output)
                self.assertEqual(command.call_count, 1)
                self.assertEqual(command.call_args.args[0], [sys.executable, "-m", "pip", "check"])
                self.assertEqual(command.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
                self.assertFalse(self.output.exists())
        with patch.object(ocr.importlib.metadata, "version", side_effect=config["versions"].get), \
                patch.object(ocr.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pip check")), \
                patch.object(ocr.importlib, "import_module") as imports:
            with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                ocr.runtime_preflight(config)
            imports.assert_not_called()

    def test_runtime_smoke_checks_all_imports_and_versions_for_both_profiles(self):
        for minor in (12, 13):
            with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)):
                config = ocr.load_config()
                with patch.object(ocr.importlib.metadata, "version", side_effect=config["versions"].get), \
                        patch.object(ocr.subprocess, "run") as command, \
                        patch.object(ocr.importlib, "import_module") as imports:
                    self.assertEqual(ocr.runtime_preflight(config), config["versions"])
                    self.assertEqual(command.call_count, 1)
                    self.assertEqual(command.call_args.args[0], [sys.executable, "-m", "pip", "check"])
                    self.assertEqual(command.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
                    self.assertEqual([call.args[0] for call in imports.call_args_list],
                                     ["numpy", "paddle", "paddleocr", "paddlex", "PIL", "jsonschema"])
                with patch.object(ocr.importlib.metadata, "version", return_value="wrong-version"), \
                        patch.object(ocr.importlib, "import_module") as imports:
                    with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*expected"):
                        ocr.runtime_preflight(config)
                    imports.assert_not_called()

    def runtime_command(self, command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(stdout=json.dumps({"version": list(sys.version_info[:2]),
                                   "base": sys.base_prefix, "prefix": str(Path(command[0]).parent.parent),
                                   "executable": command[0], "user_site": False}))
        if "--version" in command:
            return SimpleNamespace(stdout="pip 25.3 from host (python 3.13)\n")
        if "--help" in command:
            return SimpleNamespace(stdout="--python --dry-run --report")
        return SimpleNamespace(returncode=0)

    def create_runtime_venv(self, directory):
        directory.mkdir(parents=True)
        (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")

    def test_binary_resolver_install_checks_marker_order_and_cached_recheck(self):
        overrides = {"_PIP_RUNNING_IN_SUBPROCESS": "1", "_pip_test_override": "poison",
                     "PIP_TARGET": "host-target", "PIP_PREFIX": "host-prefix"}
        for minor in (12, 13):
            work = self.root / f"runtime-{minor}"
            marker = work / "venv/requirements.sha256"
            calls = []

            def command(args, **kwargs):
                self.assertFalse(marker.exists(), "Success marker must not exist before all checks pass")
                environment = kwargs["env"]
                self.assertFalse(any(key.upper().startswith("_PIP_") for key in environment))
                self.assertNotIn("PIP_TARGET", environment)
                self.assertNotIn("PIP_PREFIX", environment)
                calls.append(args)
                return self.runtime_command(args, **kwargs)

            with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)), \
                    patch.dict("os.environ", overrides), \
                    patch("venv.EnvBuilder") as builder, patch.object(ocr.subprocess, "run", side_effect=command):
                parent_environment = dict(ocr.os.environ)
                builder.return_value.create.side_effect = self.create_runtime_venv
                python, config_path, identity = ocr.prepare_runtime(work, self.root / "ocr-wheelhouse")
                builder.assert_called_once_with(with_pip=False, system_site_packages=False)
                self.assertEqual(identity["mode"], "venv-host-pip")
                self.assertEqual(identity["interpreter"]["version"], [3, minor])
                self.assertEqual(config_path, ocr.BASE / ocr.select_profile()["config"])
                self.assertTrue(marker.is_file())
                self.assertEqual(marker.read_text(), ocr.sha_bytes(ocr.encoded(identity)))
                self.assertEqual(identity["requirements_sha256"],
                                 ocr.digest(work / "requirements-runtime.txt"))
                install = [c for c in calls if "install" in c and "--help" not in c]
                self.assertEqual(len(install), 2)
                self.assertIn("--dry-run", install[0])
                self.assertIn("--report", install[0])
                for args in install:
                    self.assertIn("--only-binary=:all:", args)
                    self.assertIn("--no-index", args)
                    self.assertIn("--find-links", args)
                    self.assertEqual(args[:7], [sys.executable, "-I", "-B", "-m", "pip", "--python", str(python)])
                checks = [c for c in calls if "check" in c]
                self.assertEqual(checks, [[sys.executable, "-I", "-B", "-m", "pip", "--python", str(python), "check"]])
                self.assertIn("preflight", calls[-1])
                self.assertEqual(ocr.read_json(work / "interpreter.json"), identity)
                calls.clear()
                ocr.prepare_runtime(work, self.root / "ocr-wheelhouse")
                self.assertFalse(any("install" in c and "--help" not in c for c in calls))
                self.assertIn("preflight", calls[-1])
                self.assertTrue(marker.is_file())
                self.assertEqual(dict(ocr.os.environ), parent_environment)

    def test_python313_venv_does_not_call_ensurepip(self):
        with patch.object(sys, "version_info", (3, 13, 0)), \
                patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip forbidden")) as setup_pip, \
                patch.object(ocr.subprocess, "run", side_effect=self.runtime_command):
            python, _, identity = ocr.prepare_runtime(self.root / "pipless")
            setup_pip.assert_not_called()
            self.assertTrue(python.is_file())
            self.assertEqual(identity["mode"], "venv-host-pip")

    def test_host_pip_without_python_support_fails_before_venv_models_output(self):
        work = self.root / "old-host-pip"

        def command(args, **kwargs):
            if args[-1] == "--help":
                return SimpleNamespace(stdout="--dry-run --report")
            return self.runtime_command(args, **kwargs)

        with patch("venv.EnvBuilder") as builder, \
                patch.object(ocr.subprocess, "run", side_effect=command) as calls, \
                patch.object(ocr, "PaddleAdapter") as adapter:
            with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*host pip lacks --python"):
                ocr.prepare_runtime(work)
            builder.assert_not_called()
            adapter.assert_not_called()
            self.assertEqual(len(calls.call_args_list), 2)
            self.assertFalse(work.exists())

    def test_preflight_pip_check_targets_venv_through_host(self):
        host = str(Path(sys.executable).absolute())
        python = self.root / "venv" / ("Scripts/python.exe" if ocr.os.name == "nt" else "bin/python")
        config = ocr.load_config()
        overrides = {"KIS_OCR_PIP_HOST": host, "_PIP_RUNNING_IN_SUBPROCESS": "1",
                     "_pip_test_override": "poison", "PIP_TARGET": "host-target"}
        with patch.dict("os.environ", overrides), \
                patch.object(sys, "executable", str(python)), patch.object(sys, "prefix", str(python.parent.parent)), \
                patch.object(ocr.importlib.metadata, "version", side_effect=config["versions"].get), \
                patch.object(ocr.subprocess, "run") as command, patch.object(ocr.importlib, "import_module"):
            ocr.runtime_preflight(config)
            self.assertEqual(command.call_args.args[0], [host, "-I", "-B", "-m", "pip", "--python", str(python), "check"])
            environment = command.call_args.kwargs["env"]
            self.assertNotIn("KIS_OCR_PIP_HOST", environment)
            self.assertFalse(any(key.upper().startswith("_PIP_") for key in environment))
            self.assertNotIn("PIP_TARGET", environment)
            for key, value in overrides.items():
                self.assertEqual(ocr.os.environ[key], value)

    def test_partial_venv_without_marker_is_rebuilt_not_reused(self):
        work = self.root / "partial-runtime"
        env = work / "venv"
        env.mkdir(parents=True)
        partial = env / "partial-install"
        partial.write_text("not reusable", encoding="utf-8")
        with patch("venv.EnvBuilder") as builder, patch.object(ocr.subprocess, "run", side_effect=self.runtime_command):
            builder.return_value.create.side_effect = self.create_runtime_venv
            ocr.prepare_runtime(work)
            builder.return_value.create.assert_called_once_with(env)
            self.assertFalse(partial.exists())
            self.assertTrue((env / "requirements.sha256").is_file())

    def test_resolver_missing_wheel_install_or_import_failure_never_writes_marker(self):
        for stage in ("--dry-run", "install", "preflight"):
            work = self.root / stage.lstrip("-")

            def command(args, **kwargs):
                if stage in args and "--help" not in args and (stage != "install" or "--dry-run" not in args):
                    raise subprocess.CalledProcessError(1, args, stderr="missing wheel/import failed")
                return self.runtime_command(args, **kwargs)

            with self.subTest(stage=stage), patch("venv.EnvBuilder") as builder, \
                    patch.object(ocr.subprocess, "run", side_effect=command) as run_command:
                builder.return_value.create.side_effect = self.create_runtime_venv
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                    ocr.prepare_runtime(work)
                self.assertFalse((work / "venv/requirements.sha256").exists())
                self.assertFalse((work / "outputs").exists())
                if stage == "--dry-run":
                    self.assertFalse(any("preflight" in call.args[0] for call in run_command.call_args_list))

    def test_runtime_identity_changes_and_cached_failures_are_fail_closed(self):
        work = self.root / "runtime"
        marker = work / "venv/requirements.sha256"
        with patch("venv.EnvBuilder") as builder, patch.object(ocr.subprocess, "run", side_effect=self.runtime_command):
            builder.return_value.create.side_effect = self.create_runtime_venv
            ocr.prepare_runtime(work)
            fingerprint = marker.read_bytes()
            for key in ("python", "platform", "executable"):
                marker.write_bytes(fingerprint)
                changed = dict(ocr.interpreter_fingerprint(), **{key: "changed"})
                with patch.object(ocr, "interpreter_fingerprint", return_value=changed):
                    with self.assertRaisesRegex(RuntimeError, "identity changed"):
                        ocr.prepare_runtime(work)
                self.assertFalse(marker.exists())
            original_identity = ocr.read_json(work / "interpreter.json")
            old_identity = dict(original_identity, mode="venv-ensurepip")
            ocr.atomic_json(work / "interpreter.json", old_identity)
            marker.write_text(ocr.sha_bytes(ocr.encoded(old_identity)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                ocr.prepare_runtime(work)
            self.assertFalse(marker.exists())
            ocr.atomic_json(work / "interpreter.json", original_identity)
            marker.write_bytes(fingerprint)
            with patch.object(ocr.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "failed")):
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                    ocr.prepare_runtime(work)
            self.assertFalse(marker.exists())
            self.assertTrue(fingerprint)

    def test_wrong_venv_python_or_old_pip_cannot_fall_back_to_source_build(self):
        for stage in ("python", "pip"):
            def command(args, **kwargs):
                if stage == "python" and "-c" in args:
                    return SimpleNamespace(stdout=json.dumps({"version": [3, 9], "base": sys.base_prefix}))
                if stage == "pip" and "--help" in args:
                    return SimpleNamespace(stdout="old pip without resolver report")
                return self.runtime_command(args, **kwargs)

            work = self.root / stage
            with self.subTest(stage=stage), patch("venv.EnvBuilder") as builder, \
                    patch.object(ocr.subprocess, "run", side_effect=command) as calls:
                builder.return_value.create.side_effect = self.create_runtime_venv
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                    ocr.prepare_runtime(work)
                self.assertFalse((work / "venv/requirements.sha256").exists())
                self.assertFalse(any("--only-binary=:all:" in call.args[0] for call in calls.call_args_list))

    def notebook_codes(self):
        notebook = ocr.read_json(ocr.BASE / "notebooks/kaggle_ocr_paddle_padding.ipynb")
        self.assertEqual(notebook["nbformat"], 4)
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        for index, code in enumerate(codes):
            compile(code, f"notebook-cell-{index}", "exec")
        first = ast.parse(codes[0])
        self.assertEqual(len(first.body), 1)
        self.assertEqual(first.body[0].targets[0].id, "INPUT_ROOT")
        self.assertEqual(codes[0], 'INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"\n')
        return codes

    def local_bootstrap(self):
        # Leave room for the payload hash and content-addressed records under Windows TEMP.
        return self.notebook_codes()[1].replace(
            "/kaggle/working/kis-kaggle-ocr-v4", (self.root / "k").as_posix())

    def test_notebook_determinism_payload_allowlist_and_canonical_sources(self):
        path = ocr.BASE / notebook_builder.NOTEBOOK
        data = path.read_bytes()
        self.assertEqual(notebook_builder.build_notebook(), data)
        self.assertEqual(notebook_builder.build_notebook(), data)
        self.assertLessEqual(len(data), 1024 * 1024)
        values = notebook_builder.verify_notebook(data)
        files = notebook_builder.helpers_namespace()["validate_payload"](
            values["EMBEDDED_PAYLOAD"], values["PAYLOAD_SHA256"])
        self.assertEqual(sorted(files), list(notebook_builder.PAYLOAD_FILES))
        for name, content in files.items():
            self.assertEqual(content, notebook_builder.source_bytes(ocr.BASE / name))
            # Windows source checkout line endings must not affect generation.
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.replace(b"\n", b"\r\n"))
        self.assertEqual(notebook_builder.build_notebook(self.root), data)

    def test_generator_check_detects_stale_notebook_without_writing(self):
        target = self.root / notebook_builder.NOTEBOOK
        target.parent.mkdir(parents=True)
        data = notebook_builder.build_notebook()
        target.write_bytes(data)
        with patch.object(notebook_builder, "BASE", self.root), patch("builtins.print"):
            self.assertEqual(notebook_builder.main(["--check"]), 0)
        stale = data + b"\n"
        target.write_bytes(stale)
        with patch.object(notebook_builder, "BASE", self.root), patch("sys.stderr"), \
                self.assertRaises(SystemExit) as stopped:
            notebook_builder.main(["--check"])
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(target.read_bytes(), stale)

    def test_selfcontained_bootstrap_import_materialize_and_ignore_v1_v2_packages(self):
        # Deliberately corrupt/poison external packages: neither discovery nor reads are needed.
        package_root = self.root / "kaggle-input/old-package"
        package_root.mkdir(parents=True)
        for version in ("v1", "v2_py313"):
            (package_root / f"kis_kaggle_ocr_paddle_padding_{version}.zip").write_bytes(b"not a zip")
        (package_root / "MANIFEST.sha256").write_text("poison manifest")
        (package_root / "tools").mkdir()
        (package_root / "tools/kaggle_ocr.py").write_text("raise AssertionError('external package imported')")
        self.image()
        code = self.local_bootstrap().replace("/kaggle/input", package_root.parent.as_posix())
        namespace = {"INPUT_ROOT": str(self.input)}
        original_read = Path.read_bytes
        original_import = __import__

        def guarded_read(path):
            self.assertFalse(path.is_relative_to(package_root), "External package was read")
            return original_read(path)

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("paddle", "huggingface_hub", "kis", "pymilvus", "tools")):
                raise AssertionError(f"Forbidden import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("subprocess.run") as command, patch("builtins.print"), \
                patch.object(Path, "read_bytes", guarded_read), \
                patch("zipfile.ZipFile", side_effect=AssertionError("No ZIP bootstrap allowed")), \
                patch("builtins.__import__", side_effect=guarded_import):
            exec(compile(code, "bootstrap", "exec"), namespace)
            exec(compile(code, "bootstrap-rerun", "exec"), namespace)
            self.assertEqual(command.call_count, 2)
            self.assertEqual(command.call_args.args[0][2], str(namespace["script"]))
        root = namespace["root"]
        self.assertEqual(root.parent.name, namespace["PAYLOAD_SHA256"])
        self.assertEqual(root.name, "code")
        self.assertEqual({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()},
                         set(notebook_builder.PAYLOAD_FILES))
        self.assertEqual(namespace["ocr"].BASE, root)
        self.assertEqual(len(namespace["ocr"].discover(self.input)["records"]), 1)
        self.assertFalse((namespace["work"] / "outputs").exists())
        for name in ("../bad", "a\\b", "C:/bad", "/bad", "a/./b", "a//b", "tools/../x", "a\x00b",
                     "MANIFEST.sha256", "docs/KAGGLE_OCR.md", "config/ocr-kaggle.json/"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                namespace["safe_member"](name)

    def test_selfcontained_bootstrap_profile_matrix_and_unsupported_no_output(self):
        self.image()
        code = self.local_bootstrap()
        for minor in (10, 11, 12, 13):
            config_name = "ocr-kaggle-py313.json" if minor == 13 else "ocr-kaggle.json"
            namespace = {"INPUT_ROOT": str(self.input)}
            with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)), \
                    patch("subprocess.run") as command, patch("builtins.print"):
                exec(compile(code, "bootstrap-profile", "exec"), namespace)
                self.assertEqual(namespace["config_path"].name, config_name)
                self.assertIn(str(namespace["config_path"]), command.call_args.args[0])
        for version in ((3, 9), (3, 14), (4, 0)):
            target = self.root / ("unsupported-" + str(version[1]))
            unsupported = code.replace((self.root / "k").as_posix(), target.as_posix())
            with self.subTest(version=version), patch.object(sys, "version_info", (*version, 0)), \
                    patch("subprocess.run") as command:
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE.*profile.*unsupported"):
                    exec(compile(unsupported, "bootstrap-unsupported", "exec"), {"INPUT_ROOT": str(self.input)})
                command.assert_not_called()
                self.assertFalse(target.exists())

    def test_notebook_only_bootstrap_in_isolated_stdlib_interpreter(self):
        if not (3, 10) <= self.actual_python <= (3, 13):
            self.skipTest("Real notebook bootstrap requires Python 3.10-3.13; host is unsupported")
        self.image()
        # Copy ONLY the notebook, not any source package/config/manifest, into the temp session.
        notebook = self.root / "imported.ipynb"
        notebook.write_bytes((ocr.BASE / notebook_builder.NOTEBOOK).read_bytes())
        working = self.root / "i"
        probe = r'''
import json, sys
from pathlib import Path
document = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
codes = [''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code']
source = codes[1].replace('/kaggle/working/kis-kaggle-ocr-v4', sys.argv[3])
namespace = {'INPUT_ROOT': sys.argv[2]}
exec(compile(source, 'notebook-only-bootstrap', 'exec'), namespace)
assert not any(n.startswith(('kis', 'paddle', 'huggingface_hub', 'pymilvus')) for n in sys.modules)
assert not (namespace['work'] / 'outputs').exists()
assert len([p for p in namespace['root'].rglob('*') if p.is_file()]) == 6
print('SELF_CONTAINED_STDLIB_ONLY_OK')
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", probe,
                                 str(notebook), str(self.input), working.as_posix()],
                                cwd=self.root, check=True, capture_output=True, text=True, timeout=30)
        self.assertIn("SELF_CONTAINED_STDLIB_ONLY_OK", result.stdout)
        self.assertEqual(result.stderr, "")

    def embedded_fixture(self):
        namespace = notebook_builder.helpers_namespace()
        values = notebook_builder.verify_notebook((ocr.BASE / notebook_builder.NOTEBOOK).read_bytes())
        return namespace, values["EMBEDDED_PAYLOAD"], values["PAYLOAD_SHA256"]

    def test_embedded_payload_tamper_traversal_size_and_membership_before_write(self):
        namespace, original, checksum = self.embedded_fixture()
        mutations = (
            lambda p: p["files"][0].update(base64="AAAA"),
            lambda p: p["files"][0].update(sha256="0" * 64),
            lambda p: p["files"][0].update(size=1),
            lambda p: p["files"][0].update(size=True),
            lambda p: p["files"][0].update(size=1024 * 1024 + 1),
            lambda p: p["files"][0].update(path="../escape.py"),
            lambda p: p["files"][0].update(path="/absolute.py"),
            lambda p: p["files"][0].update(path="config\\ocr-kaggle.json"),
            lambda p: p["files"][0].update(path="kis/config.py"),
            lambda p: p["files"][0].update(mode="symlink"),
            lambda p: p["files"].pop(),
            lambda p: p["files"].__setitem__(0, p["files"][1]),
            lambda p: p["files"].reverse(),
            lambda p: p.update(format="v2"),
        )
        for index, mutate in enumerate(mutations):
            payload = deepcopy(original)
            mutate(payload)
            target = self.root / f"bad-payload-{index}"
            with self.subTest(index=index), self.assertRaises(ValueError):
                namespace["materialize_payload"](payload, checksum, target)
            self.assertFalse(target.exists())
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            namespace["materialize_payload"](original, "0" * 64, self.root / "bad-hash")
        self.assertFalse((self.root / "bad-hash").exists())
        with patch.dict(namespace, MAX_PAYLOAD_BYTES=100), self.assertRaisesRegex(ValueError, "size|MiB"):
            namespace["materialize_payload"](original, checksum, self.root / "oversize")

    def test_materialized_payload_tamper_extra_file_and_links_rejected(self):
        namespace, payload, checksum = self.embedded_fixture()
        base = self.root / "materialized"
        root = namespace["materialize_payload"](payload, checksum, base)
        runner = root / "tools/kaggle_ocr.py"
        original = runner.read_bytes()
        runner.write_bytes(original + b"# tamper\n")
        with self.assertRaisesRegex(ValueError, "tamper"):
            namespace["materialize_payload"](payload, checksum, base)
        self.assertEqual(runner.read_bytes(), original + b"# tamper\n")
        runner.write_bytes(original)
        extra = root / "credentials.json"
        extra.write_text("not allowed")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            namespace["materialize_payload"](payload, checksum, base)
        extra.unlink()
        original_symlink = Path.is_symlink
        for link in (base, root.parent, root, root / "tools", runner):
            with self.subTest(link=link), \
                    patch.object(Path, "is_symlink", lambda p: p == link or original_symlink(p)), \
                    self.assertRaisesRegex(ValueError, "symlink/junction"):
                namespace["materialize_payload"](payload, checksum, base)
        original_lstat = Path.lstat

        def reparse_lstat(path, *args, **kwargs):
            if path == root / "config":
                return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=0x400)
            return original_lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", reparse_lstat), self.assertRaisesRegex(ValueError, "reparse"):
            namespace["materialize_payload"](payload, checksum, base)

    def test_notebook_dependency_failure_blocks_run_and_download_without_outputs(self):
        self.image()
        codes = self.notebook_codes()
        namespace = {"INPUT_ROOT": str(self.input)}
        with patch("subprocess.run"), patch("builtins.print"):
            exec(compile(self.local_bootstrap(), "bootstrap", "exec"), namespace)
        dependencies = codes[2].replace("/kaggle/input", self.input.as_posix())
        for minor in (10, 11, 12, 13):
            with self.subTest(minor=minor), patch.object(sys, "version_info", (3, minor, 0)), \
                    patch.object(namespace["ocr"], "prepare_runtime",
                                 side_effect=RuntimeError("DEPENDENCY_UNAVAILABLE: mocked ABI failure")), \
                    patch("subprocess.run") as command:
                namespace["RUNTIME_READY"] = True  # A failed rerun must clear a stale success flag.
                with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                    exec(compile(dependencies, "dependencies", "exec"), namespace)
                self.assertFalse(namespace["RUNTIME_READY"])
                for code in codes[3:]:
                    with self.assertRaisesRegex(RuntimeError, "DEPENDENCY_UNAVAILABLE"):
                        exec(compile(code, "blocked-after-dependencies", "exec"), namespace)
                command.assert_not_called()
                self.assertFalse((namespace["work"] / "outputs").exists())
                self.assertFalse(list(namespace["work"].glob("*.zip")))

    def test_selfcontained_notebook_mock_run_resume_package_and_download_checksums(self):
        self.image()
        codes = self.notebook_codes()
        namespace = {"INPUT_ROOT": str(self.input)}
        with patch("subprocess.run"), patch("builtins.print"):
            exec(compile(self.local_bootstrap(), "bootstrap", "exec"), namespace)
        standalone = namespace["ocr"]
        adapter = MockAdapter([line()])
        display = Mock()
        links = SimpleNamespace(FileLink=lambda path: path, display=display)
        commands = []

        def local_command(args, **kwargs):
            self.assertEqual(Path(args[2]), namespace["script"])
            commands.append(args)
            status = standalone.main(args[3:])
            if kwargs.get("check") and status:
                raise subprocess.CalledProcessError(status, args)
            return SimpleNamespace(returncode=status)

        dependencies = codes[2].replace("/kaggle/input", self.input.as_posix())
        with patch.object(standalone, "prepare_runtime", return_value=(
                Path(sys.executable), namespace["config_path"], {"mock": True})), \
                patch.object(standalone, "PaddleAdapter", return_value=adapter), \
                patch("subprocess.run", side_effect=local_command), patch("builtins.print"), \
                patch.dict(sys.modules, {"IPython.display": links}):
            exec(compile(dependencies, "mocked-dependencies", "exec"), namespace)
            exec(compile(codes[3], "mocked-run", "exec"), namespace)
            before = (namespace["output"] / "manifest.json").read_bytes()
            exec(compile(codes[3], "mocked-resume", "exec"), namespace)
            self.assertEqual(before, (namespace["output"] / "manifest.json").read_bytes())
            self.assertEqual(len(adapter.calls), 1)
            self.assertTrue(any("--resume" in args for args in commands))
            exec(compile(codes[4], "mocked-download", "exec"), namespace)
            archive = namespace["archive"]
            original = archive.read_bytes()
            exec(compile(codes[4], "mocked-download-rerun", "exec"), namespace)
        self.assertEqual(archive.read_bytes(), original)
        self.assertTrue(archive.is_relative_to(self.root / "k"))
        self.assertEqual(namespace["checksum_path"].read_text(encoding="utf-8"),
                         ocr.digest(archive) + "  " + archive.name + "\n")
        self.assertEqual(display.call_count, 4)
        self.assertEqual(display.call_args_list[0].args[0], str(archive))
        self.assertEqual(display.call_args_list[1].args[0], str(namespace["checksum_path"]))
        members = standalone.verify_zip(archive)
        self.assertNotIn("tools/kaggle_ocr.py", members)
        self.assertFalse(any(name.endswith((".jpg", ".png", ".pdiparams")) for name in members))


if __name__ == "__main__":
    unittest.main()
