"""Offline contracts for the new standalone notebook; no pip, Paddle or network."""

import errno
import io
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from PIL import Image


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/kaggle_ocr_simple_gpu.ipynb"
DOCUMENT = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(cell["source"]) for cell in DOCUMENT["cells"] if cell["cell_type"] == "code"]
VISIBLE = {}
exec(compile(CELLS[1].split("# Load just stdlib")[0], "<worker-definition>", "exec"), VISIBLE)
WORKER = VISIBLE["WORKER_SOURCE"]
PUBLIC_FIELDS = {"video_id", "frame_id", "keyframe_n", "image_path",
                 "ocr_text", "retrieval_text", "status"}


class FakeAdapter:
    def __init__(self, fail_at=None, failure=None, empty=False):
        self.engine = {"device": "gpu:0", "models": {"recognition": "fake-latin"}}
        self.calls = []
        self.fail_at = fail_at
        self.failure = failure or RuntimeError("backend failed")
        self.empty = empty

    def recognize(self, image):
        self.calls.append((image.size, image.getpixel((0, 0)), image.getpixel((32, 32))))
        if len(self.calls) == self.fail_at:
            raise self.failure
        if self.empty:
            return []
        return [{"text": "Tiếng Việt", "confidence": 0.95,
                 "polygon_padded": [[30, 31], [54, 31], [54, 44], [30, 44]]}]


class SimpleKaggleOCRTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.input = self.base / "input/L04"
        self.output = self.base / "output"
        self.env = self.base / "env"
        self.working = self.base / "working"
        self.archive = self.working / "ocr.zip"
        self.run_id = "a" * 32
        self.ns = {"__name__": "offline_worker", "__file__": str(NOTEBOOK)}
        exec(compile(WORKER, "<worker>", "exec"), self.ns)
        guard = patch.object(socket, "create_connection", side_effect=AssertionError("No network"))
        guard.start()
        self.addCleanup(guard.stop)

    def image(self, number=1, root=None, name=None):
        path = (root or self.input) / "L04_V001" / (name or f"{number:03d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), (10, 20, 30)).save(path)
        return path

    def run_ocr(self, adapter=None, run_id=None):
        with patch("builtins.print"):
            self.ns["run"](self.input, self.output, self.env, run_id or self.run_id,
                           adapter=adapter or FakeAdapter(), progress=lambda items: items)

    def export(self, run_id=None):
        with patch("builtins.print"):
            return self.ns["export_current"](self.input, self.output, self.env, run_id or self.run_id,
                                             self.archive)

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def checkpoints(self):
        return sorted((self.output / "checkpoints").glob("*.json"))

    def exported_rows(self, destination):
        return [self.public_row(path) for path in sorted(destination.glob("L*/L*_V*/*.json"))]

    def public_row(self, path):
        rows = self.read(path)
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), PUBLIC_FIELDS)
        self.assertIsNone(rows[0]["frame_id"])
        self.assertEqual(rows[0]["retrieval_text"], rows[0]["ocr_text"])
        return rows[0]

    def summary(self):
        # Private publication/ready receipts retain summary stats; not downloaded.
        return self.read(self.output / "ready.json")

    def download_bytes(self, destination):
        return {path.relative_to(destination).as_posix(): path.read_bytes()
                for path in destination.rglob("*") if path.is_file()}

    def test_notebook_json_four_cells_compile_and_optional_nbformat(self):
        self.assertEqual(len(CELLS), 4)
        for index, cell in enumerate(CELLS):
            compile(cell, f"<cell-{index}>", "exec")
        compile(WORKER, "<worker>", "exec")
        self.assertEqual(CELLS[0].splitlines()[0].split("=")[0].strip(), "INPUT_ROOT")
        for forbidden in ("base64", "import ast", "from tools", "import kis", "ensurepip", "rmtree"):
            self.assertNotIn(forbidden, WORKER)
        for cell in DOCUMENT["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        try:
            import nbformat
        except ImportError:
            return  # JSON and all code are still compiled above.
        nbformat.validate(nbformat.from_dict(DOCUMENT))

    def test_padding_coordinates_identity_and_named_schema(self):
        path = self.image()
        adapter = FakeAdapter()
        self.run_ocr(adapter)
        self.assertEqual(adapter.calls, [((84, 74), (255, 255, 255), (10, 20, 30))])
        record = self.read(self.checkpoints()[0])["record"]
        self.assertEqual(record["id"], "L04_V001:1")
        self.assertEqual(record["keyframe_n"], 1)
        self.assertNotIn("frame_idx", record)
        self.assertIsNone(record["mapping"])
        self.assertEqual(record["schema"], "simple-kaggle-ocr-v1")
        self.assertEqual(record["source_sha256"], self.ns["digest"](path.read_bytes()))
        geometry = record["geometry"]
        self.assertEqual(geometry["original_size"], [20, 10])
        self.assertEqual(geometry["padded_size"], [84, 74])
        self.assertEqual(geometry["padded_to_original"], [[1, 0, -32], [0, 1, -32], [0, 0, 1]])
        self.assertEqual(geometry["original_to_padded"], [[1, 0, 32], [0, 1, 32], [0, 0, 1]])
        self.assertFalse(geometry["wrapper_resize"])
        line = record["lines"][0]
        self.assertEqual(line["text"], "Tiếng Việt")
        self.assertEqual(line["polygon_original_unclipped"], [[-2, -1], [22, -1], [22, 12], [-2, 12]])
        self.assertEqual(line["polygon_original_clipped"], [[0, 0], [20, 0], [20, 10], [0, 10]])

    def test_input_scope_root_video_and_duplicate_canonical_identity(self):
        self.image()
        sibling = self.base / "input/L06/L06_V001"
        sibling.mkdir(parents=True)
        (sibling / "001.jpg").write_bytes(b"irrelevant")
        self.assertEqual(len(self.ns["inventory"](self.input)), 1)
        self.assertEqual(len(self.ns["inventory"](self.input / "L04_V001")), 1)
        self.image(name="1.jpg")
        with self.assertRaisesRegex(ValueError, "Duplicate canonical identity"):
            self.run_ocr()
        self.assertEqual(self.checkpoints(), [])

    def test_noncanonical_image_names_fail_before_bulk(self):
        self.image(name="frame_1.jpg")
        adapter = FakeAdapter()
        with self.assertRaisesRegex(ValueError, "numeric keyframe"):
            self.run_ocr(adapter)
        self.assertEqual(adapter.calls, [])

    def test_export_collisions_and_alias_videos_rejected_before_writes(self):
        self.image()
        self.run_ocr()
        cases = [(self.input / "L04_V001/001.jpg", "Duplicate canonical identity"),
                 (self.input / "L04_V001/1.png", "Duplicate canonical identity"),
                 (self.input / "nested/L04_V001/001.jpeg", "Duplicate canonical identity"),
                 (self.input / "l4_v1/002.png", "canonical video folder"),
                 (self.input / "L004_V0001/002.png", "canonical video folder"),
                 (self.input / "L4_V001/002.png", "canonical video folder"),
                 (self.input / "L04_V1/002.png", "canonical video folder"),
                 (self.input / "Lxx_V001/002.png", "video folder / numeric keyframe"),
                 (self.input / "L04_V001/００２.png", "numeric keyframe")]
        for path, message in cases:
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 10)).save(path)
                before = {p: p.read_bytes() for p in self.input.rglob("*") if p.is_file()}
                with patch.dict(self.ns, atomic_json=Mock()) as patched:
                    with self.assertRaisesRegex(ValueError, message):
                        self.export()
                    patched["atomic_json"].assert_not_called()
                self.assertFalse((self.output / "downloads").exists())
                self.assertEqual({p: p.read_bytes() for p in before}, before)
                path.unlink()

    def test_export_original_stems_multiple_videos_extensions_and_exact_records(self):
        self.image(name="001.jpg")
        self.image(name="002.jpeg")
        self.image(name="010.PNG")
        second_video = self.input / "L04_V002"
        second_video.mkdir()
        Image.new("RGB", (20, 10)).save(second_video / "0001.webp")
        self.run_ocr()
        destination = self.export()
        expected = {"L04/L04_V001/001.json", "L04/L04_V001/002.json",
                    "L04/L04_V001/010.json", "L04/L04_V002/0001.json"}
        checkpoints = {path: path.read_bytes() for path in self.checkpoints()}
        filenames = {"001": "001.jpg", "002": "002.jpeg", "010": "010.PNG", "0001": "0001.webp"}
        files = self.download_bytes(destination)
        self.assertEqual(set(files), expected | {"ocr.zip"})
        self.assertNotIn("ocr.jsonl", files)
        for name in expected:
            group, video, filename = name.split("/")
            self.assertEqual(self.public_row(destination / name), {
                "video_id": video, "frame_id": None, "keyframe_n": int(Path(filename).stem),
                "image_path": f"data/{group}/{video}/{filenames[Path(filename).stem]}",
                "ocr_text": "Tiếng Việt", "retrieval_text": "Tiếng Việt", "status": "ok"})
            self.assertIn("Tiếng Việt".encode("utf-8"), files[name])
            self.assertIn(b'\n  {\n', files[name])  # Readable UTF-8, not escaped text.
        self.assertEqual({path: path.read_bytes() for path in checkpoints}, checkpoints)
        with zipfile.ZipFile(destination / "ocr.zip") as archive:
            self.assertEqual(set(archive.namelist()), expected)
            for name in archive.namelist():
                self.assertEqual(archive.read(name), files[name])
        self.assertEqual(self.read(self.output / "ready.json")["archive_sha256"],
                         self.ns["file_digest"](self.archive))
        self.assertEqual(list(self.working.iterdir()), [self.archive])

    def test_video_input_root_still_exports_video_folder_and_original_stem(self):
        path = self.image(name="0001.jpg")
        self.input = path.parent
        self.run_ocr()
        destination = self.export()
        self.assertEqual(self.public_row(destination / "L04/L04_V001/0001.json")["image_path"],
                         "data/L04/L04_V001/0001.jpg")
        self.assertEqual(self.read(self.checkpoints()[0])["record"]["relative_path"], "0001.jpg")
        with zipfile.ZipFile(destination / "ocr.zip") as archive:
            self.assertEqual(archive.namelist(), ["L04/L04_V001/0001.json"])

    def test_multiple_selected_group_roots_no_sibling_scan_or_zip_wrapper(self):
        self.input = self.base / "input/selected-subtree"
        paths = [self.image(root=self.input / "L04", name="001.jpg")]
        for folder, stem in (("L04/L04_V002", "002"), ("L05/L05_V001", "001")):
            path = self.input / folder / (stem + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 10)).save(path)
            paths.append(path)
        # The ancestor name is deliberately not a group: identity determines output.
        path = self.input / "arbitrary-container/L05_V002/0002.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (20, 10)).save(path)
        paths.append(path)
        sibling = self.base / "input/L06/not-a-video/001.jpg"
        sibling.parent.mkdir(parents=True)
        sibling.write_bytes(b"must not enumerate or decode this sibling")
        before = {path: path.read_bytes() for path in [*paths, sibling]}
        adapter = FakeAdapter()
        self.run_ocr(adapter)
        self.assertEqual(len(adapter.calls), 4)
        destination = self.export()
        expected = {"L04/L04_V001/001.json", "L04/L04_V002/002.json",
                    "L05/L05_V001/001.json", "L05/L05_V002/0002.json"}
        files = self.download_bytes(destination)
        self.assertEqual(set(files), expected | {"ocr.zip"})
        self.assertEqual(self.summary()["total"], 4)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        with zipfile.ZipFile(destination / "ocr.zip") as archive:
            self.assertEqual(set(archive.namelist()), expected)
            for name in expected:
                self.assertEqual(archive.read(name), files[name])
                record = self.public_row(destination / name)
                group, video, _ = name.split("/")
                self.assertEqual(video, record["video_id"])
                self.assertEqual(group, video.split("_V")[0])
                original = next(path for path in paths
                                if path.parent.name == video and path.stem == Path(name).stem)
                self.assertEqual(record["image_path"], f"data/{group}/{video}/{original.name}")

    def test_bidirectional_overlap_rejected_before_any_creation(self):
        self.image()
        cases = [(self.input, self.input, self.env),
                 (self.input, self.input / "new/out", self.env),
                 (self.input, self.input.parent, self.env),
                 (self.input, self.output, self.input / "venv"),
                 (self.input, self.output, self.input.parent),
                 (self.input, self.output, self.output / "venv")]
        for root, output, env in cases:
            with self.subTest(paths=(root, output, env)), patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "BOTH directions"):
                    self.ns["validate_paths"](root, output, env)
                mkdir.assert_not_called()
        with self.assertRaisesRegex(ValueError, "traversal"):
            self.ns["safe_path"](self.input / ".." / "L06")

    def test_symlink_and_reparse_paths_rejected_without_creation(self):
        image = self.image()
        real_lstat = Path.lstat
        for target in (image, self.input, self.output, self.env, self.output.parent):
            for reparse in (False, True):
                def fake_lstat(path, *args, **kwargs):
                    if path == target:
                        return SimpleNamespace(st_mode=stat.S_IFDIR if reparse else stat.S_IFLNK,
                                               st_file_attributes=0x400 if reparse else 0)
                    return real_lstat(path, *args, **kwargs)
                with self.subTest(target=target, reparse=reparse), patch.object(Path, "lstat", fake_lstat):
                    with patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "Symlink/reparse"):
                        self.ns["validate_paths"](self.input, self.output, self.env)
                    mkdir.assert_not_called()

    def test_unrelated_output_and_other_dataset_never_overwritten(self):
        self.image()
        self.output.mkdir()
        note = self.output / "notes.txt"
        note.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrelated files"):
            self.run_ocr()
        self.assertEqual(note.read_text(), "keep")
        note.unlink()
        self.run_ocr()
        other = self.base / "other"
        self.image(root=other)
        with self.assertRaisesRegex(ValueError, "another dataset"):
            self.ns["claim_output"](other, self.output)
        self.assertEqual(self.read(self.output / "owner.json")["input_root"], str(self.input))

    def test_existing_runtime_cannot_redirect_pip_or_cache_to_input(self):
        self.image()
        linked = self.env / "lib"
        linked.mkdir(parents=True)
        real_lstat, real_resolve = Path.lstat, Path.resolve

        def fake_lstat(path, *args, **kwargs):
            if path == linked:
                return SimpleNamespace(st_mode=stat.S_IFLNK)
            return real_lstat(path, *args, **kwargs)

        def fake_resolve(path, *args, **kwargs):
            if path == linked:
                return self.input
            return real_resolve(path, *args, **kwargs)

        with patch.object(Path, "lstat", fake_lstat), patch.object(Path, "resolve", fake_resolve):
            with patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "External runtime"):
                self.ns["validate_paths"](self.input, self.output, self.env)
            mkdir.assert_not_called()

    def test_resume_requires_bytes_profile_and_checkpoint_integrity(self):
        self.image(1)
        second = self.image(2)
        self.run_ocr()
        again = FakeAdapter()
        self.run_ocr(again)
        self.assertEqual(len(again.calls), 1)  # Mandatory smoke still happens on resume.
        self.assertEqual(self.read(self.output / "run.json")["skipped"], 2)
        Image.new("RGB", (20, 10), "red").save(second)
        changed = FakeAdapter()
        self.run_ocr(changed)
        self.assertEqual(len(changed.calls), 2)
        self.assertEqual(self.read(self.output / "run.json")["skipped"], 1)
        new_profile = FakeAdapter()
        new_profile.engine["revision"] = 2
        self.run_ocr(new_profile)
        self.assertEqual(self.read(self.output / "run.json")["skipped"], 0)
        for path in self.checkpoints():
            envelope = self.read(path)
            envelope["record"]["lines"][0]["text"] = "tampered"
            path.write_text(json.dumps(envelope), encoding="utf-8")
        repair = FakeAdapter()
        repair.engine["revision"] = 2
        self.run_ocr(repair)
        self.assertEqual(self.read(self.output / "run.json")["skipped"], 0)

    def test_decode_corruption_continues_including_lazy_syntax_error(self):
        bad = self.image(1)
        bad.write_bytes(b"broken image")
        self.image(2)
        adapter = FakeAdapter()
        self.run_ocr(adapter)
        self.assertEqual(len(adapter.calls), 1)
        rows = [self.read(p)["record"] for p in self.checkpoints()]
        self.assertEqual(sorted(r["status"] for r in rows), ["error", "ok"])
        self.assertFalse(self.read(self.output / "run.json")["complete"])
        output = self.export()
        self.assertFalse(self.summary()["complete"])
        self.assertEqual(self.summary()["errors"], 1)
        self.assertEqual(sorted(r["status"] for r in self.exported_rows(output)), ["error", "ok"])
        self.assertTrue(next(row for row in rows if row["status"] == "error")["error"])
        self.assertEqual(self.public_row(output / "L04/L04_V001/001.json"), {
            "video_id": "L04_V001", "frame_id": None, "keyframe_n": 1,
            "image_path": "data/L04/L04_V001/001.png", "ocr_text": "",
            "retrieval_text": "", "status": "error"})
        opened = Mock()
        opened.__enter__ = Mock(return_value=opened)
        opened.__exit__ = Mock(return_value=False)
        opened.load.side_effect = SyntaxError("lazy Pillow decoder error")
        with patch("PIL.Image.open", return_value=opened):
            decoded, error = self.ns["decode_or_error"](b"image")
        self.assertIsNone(decoded)
        self.assertEqual(error["type"], "SyntaxError")

    def test_all_corrupt_cannot_pass_real_image_smoke(self):
        self.image().write_bytes(b"bad")
        with patch("sys.stderr", new=io.StringIO()), self.assertRaisesRegex(RuntimeError, "No decodable"):
            self.run_ocr()
        self.assertEqual(self.checkpoints(), [])

    def test_empty_is_success_not_decode_error(self):
        self.image()
        self.run_ocr(FakeAdapter(empty=True))
        record = self.read(self.checkpoints()[0])["record"]
        self.assertEqual(record["status"], "empty")
        self.assertEqual(record["lines"], [])
        self.assertIsNone(record["error"])
        destination = self.export()
        self.assertTrue(self.summary()["complete"])
        self.assertEqual(self.public_row(destination / "L04/L04_V001/001.json")["status"], "empty")

    def test_empty_checkpoints_resume_without_extra_inference_and_export_as_empty(self):
        for number in range(1, 4):
            self.image(number)
        initial = FakeAdapter(empty=True)
        self.run_ocr(initial)
        self.assertEqual(len(initial.calls), 3)
        before = {path: path.read_bytes() for path in self.checkpoints()}
        self.assertEqual(len(before), 3)
        for path in before:
            self.assertEqual(self.read(path)["record"]["status"], "empty")

        resumed = FakeAdapter(empty=True, fail_at=2)
        self.run_ocr(resumed)
        self.assertEqual(len(resumed.calls), 1)  # Only the mandatory real-image smoke.
        state = self.read(self.output / "run.json")
        self.assertEqual(state["skipped"], 3)
        self.assertTrue(state["complete"])
        self.assertEqual({path: path.read_bytes() for path in self.checkpoints()}, before)

        destination = self.export()
        rows = self.exported_rows(destination)
        self.assertEqual([row["status"] for row in rows], ["empty"] * 3)
        self.assertTrue(all(row["ocr_text"] == row["retrieval_text"] == "" for row in rows))
        self.assertEqual([row["keyframe_n"] for row in rows], [1, 2, 3])
        self.assertEqual({path: path.read_bytes() for path in self.checkpoints()}, before)
        summary = self.summary()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["missing"], [])
        self.assertEqual(summary["exported"], 3)
        self.assertEqual(summary["empty"], 3)
        self.assertEqual(summary["ok"], 0)

    def test_projection_preserves_nonblank_unicode_order_low_confidence_and_private_checkpoints(self):
        for number in range(1, 5):
            self.image(number)
        (self.input / "L04_V001/003.png").write_bytes(b"corrupt image")
        blank_texts = ["", " \t\n", "\u2003"]
        mixed_texts = ["", " HTVTHD ", "\t", "18:29:55", "  Tiếng Việt 日本語 \t", "!", "\u2003"]

        def lines(texts):
            return [{"text": text, "confidence": 0.0,
                     "polygon_padded": [[32, 32], [40, 32], [40, 40], [32, 40]]}
                    for text in texts]

        adapter = FakeAdapter()
        adapter.recognize = Mock(side_effect=[lines(blank_texts), lines(mixed_texts), []])
        self.run_ocr(adapter)
        self.assertEqual(adapter.recognize.call_count, 3)
        before = {path: path.read_bytes() for path in self.checkpoints()}
        records = {self.read(path)["record"]["keyframe_n"]: self.read(path)["record"] for path in before}
        # Internal status still counts raw model lines; projection must not rewrite it.
        self.assertEqual(records[1]["status"], "ok")
        self.assertEqual([line["text"] for line in records[1]["lines"]], blank_texts)
        self.assertEqual([line["text"] for line in records[2]["lines"]], mixed_texts)
        self.assertTrue(all(line["confidence"] == 0.0 for line in records[2]["lines"]))
        self.assertTrue(records[3]["error"]["type"])
        self.assertTrue(records[3]["error"]["message"])
        for path in before:
            envelope = self.read(path)
            self.assertEqual(envelope["record_sha256"],
                             self.ns["digest"](self.ns["encoded"](envelope["record"])))
            self.assertTrue(envelope["record"]["source_sha256"])
            self.assertTrue(envelope["record"]["profile_sha256"])

        with patch("builtins.print") as log:
            destination = self.ns["export_current"](self.input, self.output, self.env,
                                                     self.run_id, self.archive)
        rows = self.exported_rows(destination)
        self.assertEqual([row["status"] for row in rows], ["empty", "ok", "error", "empty"])
        self.assertEqual([row["ocr_text"] for row in rows],
                         ["", " HTVTHD \n18:29:55\n  Tiếng Việt 日本語 \t\n!", "", ""])
        summary = json.loads(log.call_args_list[0].args[0])
        self.assertEqual({key: summary[key] for key in ("total", "exported", "ok", "empty", "errors")},
                         {"total": 4, "exported": 4, "ok": 1, "empty": 2, "errors": 1})
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing"], [])
        self.assertEqual({key: self.summary()[key] for key in summary}, summary)
        self.assertEqual({path: path.read_bytes() for path in self.checkpoints()}, before)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.namelist(), [f"L04/L04_V001/{n:03d}.json" for n in range(1, 5)])
            for name in archive.namelist():
                self.assertEqual(json.loads(archive.read(name)), [self.public_row(destination / name)])

    def test_worker_source_change_invalidates_profile_without_weakening_resume(self):
        self.image(1)
        self.image(2)
        worker = self.base / "worker.py"
        worker.write_text(WORKER, encoding="utf-8")
        with patch.dict(self.ns, __file__=str(worker)):
            self.run_ocr()
            old_profile = self.read(self.output / "run.json")["profile_sha256"]
            worker.write_text(WORKER + "\n# Changed worker source\n", encoding="utf-8")
            changed = FakeAdapter()
            self.run_ocr(changed)
            state = self.read(self.output / "run.json")
            self.assertNotEqual(state["profile_sha256"], old_profile)
            self.assertEqual(state["skipped"], 0)
            self.assertEqual(len(changed.calls), 2)
            resumed = FakeAdapter()
            self.run_ocr(resumed)
            self.assertEqual(self.read(self.output / "run.json")["skipped"], 2)
            self.assertEqual(len(resumed.calls), 1)

    def test_smoke_failure_creates_no_processed_results_and_no_links(self):
        self.image()
        adapter = FakeAdapter(fail_at=1, failure=SyntaxError("backend API, not decoder"))
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            with self.assertRaisesRegex(SyntaxError, "backend API"):
                self.ns["run"](self.input, self.output, self.env, self.run_id,
                               adapter=adapter, progress=lambda items: items)
        self.assertIn("Traceback", stderr.getvalue())
        self.assertEqual(self.checkpoints(), [])
        self.assertEqual(self.read(self.output / "run.json")["phase"], "fatal")
        with self.assertRaisesRegex(RuntimeError, "stale downloads"):
            self.export()
        self.assertFalse((self.output / "downloads").exists())

    def test_backend_failure_midrun_is_fatal_not_mass_decode_errors(self):
        for number in range(1, 6):
            self.image(number)
        adapter = FakeAdapter(fail_at=2)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.run_ocr(adapter)
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(len(self.checkpoints()), 1)
        self.assertEqual(self.read(self.checkpoints()[0])["record"]["status"], "ok")
        with self.assertRaisesRegex(RuntimeError, "stale downloads"):
            self.export()

    def test_export_sorted_current_only_zip_allowlist_and_failure_invalidates_old(self):
        first = self.image(1)
        self.image(10)
        second = self.image(2)
        self.run_ocr()
        old_download = self.export()
        old_raw = self.download_bytes(old_download)
        self.assertEqual([r["keyframe_n"] for r in self.exported_rows(old_download)], [1, 2, 10])
        with zipfile.ZipFile(old_download / "ocr.zip") as archive:
            self.assertEqual(archive.namelist(), ["L04/L04_V001/001.json", "L04/L04_V001/002.json",
                                                 "L04/L04_V001/010.json"])
        second.unlink()
        new_id = "b" * 32
        self.run_ocr(run_id=new_id)
        current = self.export(new_id)
        self.assertEqual([r["keyframe_n"] for r in self.exported_rows(current)], [1, 10])
        self.assertEqual(self.download_bytes(old_download), old_raw)
        self.assertFalse((current / "L04/L04_V001/002.json").exists())
        self.assertEqual(len(self.checkpoints()), 3)  # Historical checkpoint is retained, not exported.
        with zipfile.ZipFile(current / "ocr.zip") as archive:
            self.assertNotIn("L04/L04_V001/002.json", archive.namelist())
        with self.assertRaisesRegex(RuntimeError, "stale downloads"):
            self.export(self.run_id)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(RuntimeError):
            self.run_ocr(FakeAdapter(fail_at=1), run_id="c" * 32)
        with self.assertRaisesRegex(RuntimeError, "stale downloads"):
            self.export(new_id)
        self.assertFalse((self.output / "downloads" / ("c" * 32)).exists())
        self.assertTrue(first.is_file())

    def test_export_invalid_or_missing_checkpoint_complete_false_no_stale_row(self):
        for number in range(1, 5):
            self.image(number)
        self.run_ocr()
        paths = self.checkpoints()
        paths[0].write_text("not json", encoding="utf-8")
        paths[1].unlink()
        tampered = self.read(paths[2])
        tampered["record"]["lines"][0]["text"] = "unverified text"
        paths[2].write_text(json.dumps(tampered), encoding="utf-8")
        destination = self.export()
        summary = self.summary()
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["missing"]), 3)
        self.assertEqual(summary["exported"], 1)
        self.assertEqual(len(self.exported_rows(destination)), 1)
        surviving_stem = Path(self.read(paths[3])["record"]["relative_path"]).stem
        with zipfile.ZipFile(destination / "ocr.zip") as archive:
            self.assertEqual(archive.namelist(), [f"L04/L04_V001/{surviving_stem}.json"])

    def test_unchanged_inventory_exports_identical_json_records_across_invocations(self):
        self.image(10)
        self.image(2)
        self.run_ocr()
        first = {name: raw for name, raw in self.download_bytes(self.export()).items()
                 if name.startswith("L04/")}
        next_id = "d" * 32
        self.run_ocr(run_id=next_id)
        second = {name: raw for name, raw in self.download_bytes(self.export(next_id)).items()
                  if name.startswith("L04/")}
        self.assertEqual(first, second)
        rows = [json.loads(row)[0] for row in second.values()]
        self.assertEqual(len(rows), len({(row["video_id"], row["keyframe_n"]) for row in rows}))

    def test_export_changed_inventory_rejected_and_changed_bytes_not_exported(self):
        first = self.image(1)
        self.run_ocr()
        self.image(2)
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            self.export()
        self.assertFalse((self.output / "downloads").exists())
        self.run_ocr()
        first.write_bytes(b"changed")
        destination = self.export()
        summary = self.summary()
        self.assertEqual(summary["missing"], ["L04_V001:1"])
        self.assertFalse(summary["complete"])
        self.assertFalse((destination / "L04/L04_V001/001.json").exists())

    def test_zip_explicit_allowlist_excludes_unexpected_files_even_during_export(self):
        self.image()
        self.run_ocr()
        original = self.ns["atomic_bytes"]
        injected = []

        def inject(path, raw):
            original(path, raw)
            if path.name == "001.json":
                for name in ("L04/L04_V001/999.json", "env/private.json", "checkpoints/hash.json", "record.tmp",
                             "summary.json", "profile.json", "ready.json"):
                    extra = path.parents[2] / name
                    extra.parent.mkdir(parents=True, exist_ok=True)
                    extra.write_text("not a current record", encoding="utf-8")
                    injected.append(extra)

        with patch.dict(self.ns, atomic_bytes=inject):
            destination = self.export()
        self.assertEqual(len(injected), 7)  # Prove the real per-image write hook was reached.
        self.assertTrue(all(path.is_file() for path in injected))
        with zipfile.ZipFile(destination / "ocr.zip") as archive:
            self.assertEqual(archive.namelist(), ["L04/L04_V001/001.json"])

    def test_interrupted_exports_never_ready_or_reused_and_keep_historical_downloads(self):
        self.image(1)
        self.image(2)
        self.run_ocr()
        historical = self.export()
        before = self.download_bytes(historical)
        old_zip = self.archive.read_bytes()
        original_replace = os.replace
        for index, target in enumerate(("002.json", "ocr.zip"), start=1):
            run_id = str(index) * 32
            self.run_ocr(run_id=run_id)

            def interrupt(source, destination):
                if Path(destination).name == target:
                    raise OSError("interrupted export")
                original_replace(source, destination)

            with self.subTest(target=target), patch("os.replace", side_effect=interrupt):
                with self.assertRaisesRegex(OSError, "interrupted export"):
                    self.export(run_id)
            destination = self.output / "downloads" / run_id
            self.assertFalse((self.output / "ready.json").exists())
            self.assertFalse((destination / target).exists())
            if target == "002.json":
                self.assertTrue((destination / "L04/L04_V001/001.json").exists())
                self.assertFalse((destination / "L04/L04_V001/002.json").exists())
            if target == "ocr.zip":
                self.assertTrue((destination / "ocr.zip.tmp").exists())
            partial = self.download_bytes(destination)
            with self.assertRaisesRegex(RuntimeError, "Export already exists"):
                self.export(run_id)
            self.assertEqual(self.download_bytes(destination), partial)
            self.assertEqual(self.download_bytes(historical), before)
            self.assertEqual(self.archive.read_bytes(), old_zip)
            self.assertEqual(list(self.working.iterdir()), [self.archive])
        next_id = "e" * 32
        self.run_ocr(run_id=next_id)
        self.export(next_id)
        self.assertTrue((self.output / "ready.json").is_file())
        self.assertEqual(self.download_bytes(historical), before)

    def test_final_cell_links_only_after_export_and_current_ready_marker(self):
        self.image()
        self.run_ocr()
        display = Mock()
        file_link = Mock()
        namespace = {"RUN_OK": True, "RUN_ID": self.run_id, "OUTPUT": self.output,
                      "PYTHON": Path("python"), "WORKER": Path("worker.py"), "root": self.input,
                      "ENV_ROOT": self.env, "child_env": {}, "helpers": self.ns,
                      "ARCHIVE": self.archive, "SCRATCH": self.base,
                      "subprocess": SimpleNamespace(run=Mock())}
        fake_module = SimpleNamespace(FileLink=file_link, display=display)
        with patch.dict("sys.modules", {"IPython.display": fake_module}), patch("builtins.print"):
            with self.assertRaises(FileNotFoundError):
                exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
            display.assert_not_called()
            self.export()
            ready = self.read(self.output / "ready.json")
            ready["run_id"] = "f" * 32
            def stale_export(*args, **kwargs):
                self.ns["atomic_json"](self.output / "ready.json", ready)
            namespace["subprocess"].run.side_effect = stale_export
            with self.assertRaisesRegex(RuntimeError, "invocation mismatch"):
                exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
            display.assert_not_called()
            ready["run_id"] = self.run_id
            namespace["subprocess"].run.side_effect = RuntimeError("export failed")
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
            display.assert_not_called()  # Even a historical ready marker cannot bypass subprocess failure.
            namespace["subprocess"].run.side_effect = stale_export
            exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
        self.assertEqual(display.call_count, 1)
        self.assertEqual([call.args[0] for call in file_link.call_args_list],
                          [str(self.archive)])

    def paddle(self):
        result = SimpleNamespace(place=SimpleNamespace(is_gpu_place=lambda: True),
                                 numpy=lambda: [[4.0]])
        cuda = SimpleNamespace(device_count=lambda: 1, get_device_name=lambda n: "Tesla T4",
                               synchronize=Mock())
        return SimpleNamespace(is_compiled_with_cuda=lambda: True, device=SimpleNamespace(cuda=cuda),
                               set_device=Mock(), to_tensor=Mock(return_value="tensor"),
                               CUDAPlace=Mock(return_value="cuda0"), matmul=Mock(return_value=result),
                               version=SimpleNamespace(cuda=lambda: "12.6", cudnn=lambda: "9.5"))

    def test_gpu_fail_closed_cuda_count_name_actual_operation(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            for failure in ("cpu", "none", "multi", "wrong_gpu", "cpu_tensor", "cuda_exception"):
                paddle = self.paddle()
                if failure == "cpu":
                    paddle.is_compiled_with_cuda = lambda: False
                elif failure in {"none", "multi"}:
                    paddle.device.cuda.device_count = lambda: 0 if failure == "none" else 2
                elif failure == "wrong_gpu":
                    paddle.device.cuda.get_device_name = lambda n: "A100"
                elif failure == "cpu_tensor":
                    paddle.matmul.return_value.place.is_gpu_place = lambda: False
                else:
                    paddle.matmul.side_effect = RuntimeError("driver mismatch")
                with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                    self.ns["require_gpu"](paddle)
            paddle = self.paddle()
            with patch("builtins.print"):
                self.assertEqual(self.ns["require_gpu"](paddle)["device"], "gpu:0")
            paddle.set_device.assert_called_once_with("gpu:0")
            paddle.to_tensor.assert_called_once_with([[2.0]], place="cuda0")
            paddle.matmul.assert_called_once_with("tensor", "tensor")
            paddle.device.cuda.synchronize.assert_called_once()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), self.assertRaises(RuntimeError):
            self.ns["require_gpu"](self.paddle())

    def test_adapter_gpu_options_actual_models_bgr_and_api_failures(self):
        self.assertEqual(self.ns["OPTIONS"]["device"], "gpu:0")
        self.assertFalse(self.ns["OPTIONS"]["enable_hpi"])
        self.assertFalse(self.ns["OPTIONS"]["use_tensorrt"])
        model = lambda name: SimpleNamespace(model_name=name, pp_option=SimpleNamespace(device_type="gpu", device_id=0))
        pipeline = SimpleNamespace(text_det_model=model(self.ns["MODELS"]["detection"]),
                                   text_rec_model=model(self.ns["MODELS"]["recognition"]))
        engine = Mock(paddlex_pipeline=pipeline)
        factory = Mock(return_value=engine)
        modules = {"paddle": self.paddle(), "paddleocr": SimpleNamespace(PaddleOCR=factory)}
        with patch.dict("sys.modules", modules), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            with patch("importlib.metadata.version", side_effect=lambda name: self.ns["PINS"][name]), \
                    patch("importlib.metadata.distributions", return_value=[]), patch("builtins.print"):
                adapter = self.ns["Adapter"]()
            factory.assert_called_once_with(**self.ns["OPTIONS"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        engine.predict.return_value = [{"rec_texts": ["test"], "rec_scores": [0.9],
                                        "rec_polys": [np.array([[32, 32], [40, 32], [40, 40], [32, 40]])]}]
        adapter.recognize(Image.new("RGB", (84, 74), (10, 20, 30)))
        bgr = engine.predict.call_args.kwargs["input"]
        self.assertEqual(bgr[0, 0].tolist(), [30, 20, 10])
        engine.predict.return_value = [{"error": "bad API"}]
        with self.assertRaisesRegex(RuntimeError, "Unexpected OCR"):
            adapter.recognize(Image.new("RGB", (10, 10)))
        engine.predict.return_value = [{"rec_texts": ["a"], "rec_scores": [], "rec_polys": []}]
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            adapter.recognize(Image.new("RGB", (10, 10)))

    def test_bootstrap_commands_mocked_pipless_binary_isolated_and_streamed(self):
        self.image()
        source = CELLS[1].replace("/kaggle/working", self.working.as_posix())
        source = source.replace("'/kaggle/temp'", repr((self.base / "transient").as_posix()))
        source = source.replace("'/tmp'", repr((self.base / "fallback").as_posix()))
        namespace = {"INPUT_ROOT": str(self.input)}
        leaked = {"PIP_TARGET": str(self.input), "PIP_NO_BINARY": ":all:",
                   "PYTHONPATH": str(self.input), "PADDLE_PDX_CACHE_HOME": str(self.input),
                   "MODELSCOPE_CACHE": str(self.input), "HF_HOME": str(self.input),
                   "HOME": str(self.working), "TMPDIR": str(self.working),
                   "TEMP": str(self.working), "TMP": str(self.working),
                   "HF_HUB_CACHE": str(self.input), "HUGGINGFACE_HUB_CACHE": str(self.input),
                   "HF_ASSETS_CACHE": str(self.input), "XDG_CONFIG_HOME": str(self.working),
                   "XDG_DATA_HOME": str(self.input), "TRANSFORMERS_CACHE": str(self.input),
                   "SSLKEYLOGFILE": str(self.working / "tls.keys"),
                   "CUDNN_LOGDEST_DBG": str(self.working / "cudnn.log"), "CUDNN_LOGLEVEL_DBG": "3",
                   "GLOG_log_dir": str(self.working / "glog"), "GLOG_logtostderr": "0",
                   "MPLCONFIGDIR": str(self.working / "matplotlib")}
        with patch.dict(os.environ, leaked), patch("sys.platform", "linux"), \
                patch("sys.version_info", (3, 13, 0)), patch("subprocess.run") as command, patch("builtins.print"):
            exec(compile(source, "<bootstrap>", "exec"), namespace)
        self.assertEqual(command.call_count, 2)
        venv, pip = command.call_args_list
        self.assertIn("--without-pip", venv.args[0])
        self.assertIn("--copies", venv.args[0])
        self.assertIn("--python", pip.args[0])
        self.assertIn("--isolated", pip.args[0])
        self.assertIn("--only-binary=:all:", pip.args[0])
        self.assertIn("numpy==2.2.6", pip.args[0])
        self.assertTrue(any("3.2.2-cp313-cp313-linux_x86_64.whl" in arg for arg in pip.args[0]))
        env = pip.kwargs["env"]
        for key in ("PIP_TARGET", "PIP_NO_BINARY", "PYTHONPATH", "SSLKEYLOGFILE",
                    "CUDNN_LOGDEST_DBG", "CUDNN_LOGLEVEL_DBG", "GLOG_log_dir"):
            self.assertNotIn(key, env.keys())
        for key in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HF_HOME",
                    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "MODELSCOPE_CACHE",
                    "PADDLE_PDX_CACHE_HOME", "PADDLE_HOME", "PADDLEX_HOME", "PIP_CACHE_DIR",
                    "TMPDIR", "TEMP", "TMP", "TRANSFORMERS_CACHE", "MPLCONFIGDIR"):
            self.assertTrue(Path(env[key]).is_relative_to(namespace["SCRATCH"]), key)
            self.assertTrue(Path(env[key]).is_dir(), key)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["GLOG_logtostderr"], "1")
        for call in command.call_args_list:
            self.assertNotIn("capture_output", call.kwargs)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["cwd"], namespace["SCRATCH"])
        self.assertFalse(self.working.exists())
        with patch("subprocess.run") as runner:
            exec(compile(CELLS[2], "<run-cell>", "exec"), namespace)
        self.assertEqual(runner.call_args.args[0][0], str(namespace["PYTHON"]))
        self.assertIn("-u", runner.call_args.args[0])
        self.assertNotIn("stdout", runner.call_args.kwargs)
        self.assertEqual(runner.call_args.kwargs["cwd"], namespace["SCRATCH"])
        with patch("subprocess.run", side_effect=RuntimeError("failed")), self.assertRaises(RuntimeError):
            exec(compile(CELLS[2], "<run-cell>", "exec"), namespace)
        self.assertFalse(namespace["RUN_OK"])
        with patch("subprocess.run") as runner, self.assertRaisesRegex(RuntimeError, "no current downloads"):
            exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
        runner.assert_not_called()

    def test_transient_scratch_stable_resume_and_guarded_permission_fallback(self):
        self.image()
        primary, fallback = self.base / "transient", self.base / "fallback"
        probe = self.ns["writable_probe"]

        def denied_primary(folder):
            if primary in folder.parents:
                raise PermissionError("read-only Kaggle temp")
            probe(folder)

        with patch.dict(self.ns, writable_probe=denied_primary):
            scratch, output, runtime = self.ns["select_scratch"](
                self.input, self.archive, (primary, fallback))
            again = self.ns["select_scratch"](self.input, self.archive, (primary, fallback))
        self.assertEqual(again, (scratch, output, runtime))
        self.assertTrue(scratch.is_relative_to(fallback))
        self.assertFalse(self.working.exists())
        self.assertFalse(primary.exists())
        self.assertFalse(list(scratch.glob(".write-probe-*")))
        self.output, self.env = output, runtime
        self.run_ocr()
        resumed = FakeAdapter()
        self.run_ocr(resumed, run_id="b" * 32)
        self.assertEqual(len(resumed.calls), 1)
        self.export("b" * 32)
        self.assertEqual(list(self.working.iterdir()), [self.archive])
        with patch.dict(self.ns, writable_probe=denied_primary):
            self.assertEqual(self.ns["select_scratch"](self.input, self.archive, (primary, fallback)),
                             (scratch, output, runtime))

    def test_no_writable_scratch_never_falls_back_to_working_or_installs(self):
        self.image()
        source = CELLS[1].replace("/kaggle/working", self.working.as_posix())
        source = source.replace("'/kaggle/temp'", repr((self.base / "transient").as_posix()))
        source = source.replace("'/tmp'", repr((self.base / "fallback").as_posix()))
        original_open = Path.open

        def denied_probe(path, *args, **kwargs):
            if path.name.startswith(".write-probe-"):
                raise PermissionError("scratch not writable")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", denied_probe), patch("sys.platform", "linux"), \
                patch("sys.version_info", (3, 13, 0)), patch("subprocess.run") as install:
            with self.assertRaisesRegex(RuntimeError, "No writable transient scratch"):
                exec(compile(source, "<bootstrap>", "exec"), {"INPUT_ROOT": str(self.input)})
        install.assert_not_called()
        self.assertFalse(self.working.exists())
        self.assertFalse(list(self.base.rglob("*owner.json")))

    def test_scratch_overlap_unowned_and_unsafe_fallback_rejected_before_writes(self):
        self.image()
        cases = [(self.input, self.base / "fallback"),
                 (self.working, self.base / "fallback"),
                 (self.base / "transient", self.input),
                 (self.base / "transient", self.base / ".." / "fallback")]
        for candidates in cases:
            with self.subTest(candidates=candidates), patch.object(Path, "mkdir") as mkdir:
                with patch.dict(self.ns, writable_probe=Mock(side_effect=PermissionError("denied"))):
                    with self.assertRaisesRegex(ValueError, "BOTH directions|traversal"):
                        self.ns["select_scratch"](self.input, self.archive, candidates)
                mkdir.assert_not_called()
        candidate = self.base / "transient"
        scratch = candidate / ("ocr-simple-gpu-" + self.ns["digest"](str(self.input).encode()))
        scratch.mkdir(parents=True)
        note = scratch / "keep.txt"
        note.write_text("unrelated", encoding="utf-8")
        with patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "Unowned scratch"):
            self.ns["select_scratch"](self.input, self.archive, (candidate,))
        mkdir.assert_not_called()
        self.assertEqual(note.read_text(), "unrelated")

    def test_publish_overlap_and_symlink_reparse_guard_before_writes(self):
        self.image()
        archives = [self.input / "ocr.zip", self.input.parent / "ocr.zip",
                    self.output / "ocr.zip", self.env / "ocr.zip",
                    self.base / "ocr.zip", self.working / ".." / "ocr.zip"]
        for archive in archives:
            with self.subTest(archive=archive), patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "BOTH directions|traversal"):
                    self.ns["validate_publish"](self.input, self.output, self.env, archive)
                mkdir.assert_not_called()
        original = Path.lstat
        for target in (self.archive, self.working, self.base / "transient"):
            for reparse in (False, True):
                def linked(path, *args, **kwargs):
                    if path == target:
                        return SimpleNamespace(st_mode=stat.S_IFDIR if reparse else stat.S_IFLNK,
                                               st_file_attributes=0x400 if reparse else 0)
                    return original(path, *args, **kwargs)
                with self.subTest(target=target, reparse=reparse), patch.object(Path, "lstat", linked):
                    with patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "Symlink/reparse"):
                        self.ns["select_scratch"](self.input, self.archive, (self.base / "transient",))
                    mkdir.assert_not_called()

    def test_cache_routes_host_environment_unchanged_and_nested_defaults_in_scratch(self):
        scratch = self.base / "transient"
        hostile = {key: str(self.working) for key in (
            "HOME", "USERPROFILE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
            "TMPDIR", "TMP", "TEMP", "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE",
            "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE", "TRANSFORMERS_CACHE",
            "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "MODELSCOPE_CACHE",
            "PADDLE_PDX_CACHE_HOME", "PADDLE_HOME", "PADDLEX_HOME", "PIP_CACHE_DIR",
            "UNRECOGNIZED_CACHE", "PADDLE_UNKNOWN_OVERRIDE", "XDG_RUNTIME_DIR")}
        hostile.update(PIP_TARGET=str(self.input), PYTHONPATH=str(self.input), CUDA_VISIBLE_DEVICES="7")
        before = dict(hostile)
        with patch.dict(os.environ, hostile):
            parent = dict(os.environ)
            child = self.ns["isolated_environment"](scratch, os.environ)
            self.assertEqual(dict(os.environ), parent)
        self.assertEqual(hostile, before)
        for key in ("PIP_TARGET", "PYTHONPATH", "UNRECOGNIZED_CACHE", "PADDLE_UNKNOWN_OVERRIDE", "XDG_RUNTIME_DIR"):
            self.assertNotIn(key, child)
        for key in hostile.keys() - {"PIP_TARGET", "PYTHONPATH", "UNRECOGNIZED_CACHE",
                                     "PADDLE_UNKNOWN_OVERRIDE", "XDG_RUNTIME_DIR", "CUDA_VISIBLE_DEVICES"}:
            self.assertTrue(Path(child[key]).is_relative_to(scratch), key)
        for default in (".paddleocr", ".paddlex/official_models", ".cache/huggingface/hub"):
            self.assertTrue((Path(child["HOME"]) / default).is_relative_to(scratch))
        self.assertFalse(self.working.exists())

    def test_crossdevice_publication_copies_fsyncs_validates_then_same_directory_replace(self):
        self.image()
        self.run_ocr()
        original = os.replace
        publications = []

        def crossdevice(source, destination):
            source, destination = Path(source), Path(destination)
            if destination == self.archive:
                if source.parent != destination.parent:
                    raise OSError(errno.EXDEV, "cross-device rename")
                with zipfile.ZipFile(source) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertIn("L04/L04_V001/001.json", archive.namelist())
                publications.append(source)
            return original(source, destination)

        with patch("os.replace", side_effect=crossdevice), patch("os.fsync", wraps=os.fsync) as fsync:
            destination = self.export()
        self.assertTrue(fsync.called)
        self.assertEqual(len(publications), 1)
        self.assertFalse(publications[0].exists())
        self.assertEqual(self.archive.read_bytes(), (destination / "ocr.zip").read_bytes())
        self.assertEqual(list(self.working.iterdir()), [self.archive])
        self.assertEqual(self.ns["checked_ready"](self.input, self.output, self.env,
                                                self.run_id, self.archive), self.archive)

    def test_publication_failure_cleans_only_own_temp_and_preserves_owned_old_zip(self):
        self.image()
        self.run_ocr()
        self.export()
        old_zip = self.archive.read_bytes()
        historical_temp = self.working / ".ocr-publish-user-file.tmp"
        historical_temp.write_bytes(b"leave this alone")
        unrelated = self.working / "unrelated-session"
        unrelated.mkdir()
        (unrelated / "keep.json").write_text("keep", encoding="utf-8")
        original_replace, original_copy = os.replace, self.ns["shutil"].copyfileobj
        for index, failure in enumerate(("copy", "fsync", "replace", "interrupt"), start=1):
            run_id = str(index) * 32
            self.run_ocr(run_id=run_id)

            def copy(source, target, *args, **kwargs):
                if Path(str(getattr(target, "name", ""))).parent == self.working:
                    if failure == "copy":
                        target.write(b"partial")
                        raise OSError("publication failure")
                    if failure == "interrupt":
                        target.write(b"partial")
                        raise KeyboardInterrupt("publication failure")
                return original_copy(source, target, *args, **kwargs)

            def replace(source, destination):
                if Path(destination) == self.archive and failure == "replace":
                    raise OSError("publication failure")
                return original_replace(source, destination)

            original_fsync = os.fsync
            def fsync(fd):
                if failure == "fsync" and list(self.working.glob(".ocr-publish-*.tmp")) != [historical_temp]:
                    raise OSError("publication failure")
                return original_fsync(fd)

            with self.subTest(failure=failure), patch("shutil.copyfileobj", side_effect=copy), \
                    patch("os.replace", side_effect=replace), patch("os.fsync", side_effect=fsync):
                with self.assertRaisesRegex((OSError, KeyboardInterrupt), "publication failure"):
                    self.export(run_id)
            self.assertEqual(self.archive.read_bytes(), old_zip)
            self.assertFalse((self.output / "ready.json").exists())
            self.assertEqual(set(self.working.iterdir()), {self.archive, historical_temp, unrelated})
            self.assertEqual(historical_temp.read_bytes(), b"leave this alone")
            self.assertEqual((unrelated / "keep.json").read_text(), "keep")

    def test_first_publication_failure_leaves_working_empty(self):
        self.image()
        self.run_ocr()
        original = self.ns["shutil"].copyfileobj

        def copy(source, target, *args, **kwargs):
            if Path(str(getattr(target, "name", ""))).parent == self.working:
                raise OSError("copy failed")
            return original(source, target, *args, **kwargs)

        with patch("shutil.copyfileobj", side_effect=copy):
            with self.assertRaisesRegex(OSError, "copy failed"):
                self.export()
        self.assertEqual(list(self.working.iterdir()), [])
        self.assertFalse((self.output / "ready.json").exists())

    def test_unrelated_or_changed_zip_collision_refused_without_writes(self):
        self.image()
        self.run_ocr()
        self.working.mkdir()
        self.archive.write_bytes(b"unrelated ZIP, do not touch")
        original = self.archive.read_bytes()
        with patch.dict(self.ns, atomic_json=Mock()) as namespace:
            with self.assertRaisesRegex(ValueError, "Unrelated preexisting"):
                self.export()
            namespace["atomic_json"].assert_not_called()
        self.assertEqual(self.archive.read_bytes(), original)
        self.archive.unlink()
        self.export()
        self.archive.write_bytes(b"changed externally")
        with self.assertRaisesRegex(ValueError, "changed preexisting"):
            self.ns["checked_ready"](self.input, self.output, self.env, self.run_id, self.archive)
        self.assertEqual(self.archive.read_bytes(), b"changed externally")

    def test_receipt_digest_profile_and_rerun_block_stale_success(self):
        self.image()
        self.run_ocr()
        self.export()
        ready = self.read(self.output / "ready.json")
        for key, value in (("archive_sha256", "0" * 64), ("profile_sha256", "0" * 64)):
            with self.subTest(key=key):
                self.ns["atomic_json"](self.output / "ready.json", dict(ready, **{key: value}))
                with self.assertRaisesRegex(RuntimeError, "receipt/digest mismatch"):
                    self.ns["checked_ready"](self.input, self.output, self.env, self.run_id, self.archive)
        self.ns["atomic_json"](self.output / "ready.json", ready)
        self.run_ocr(run_id="b" * 32)
        self.assertFalse((self.output / "ready.json").exists())
        with self.assertRaises(FileNotFoundError):
            self.ns["checked_ready"](self.input, self.output, self.env, self.run_id, self.archive)
        self.assertTrue(self.archive.is_file())

    def test_working_directory_never_enumerated_or_broadly_cleaned(self):
        self.image()
        self.run_ocr()
        self.working.mkdir()
        unrelated = self.working / "v5-runtime-do-not-touch"
        unrelated.mkdir()
        note = unrelated / "owner.json"
        note.write_text("keep", encoding="utf-8")
        original_walk, original_iterdir = os.walk, Path.iterdir

        def walk(folder, *args, **kwargs):
            self.assertFalse(Path(folder).is_relative_to(self.working))
            return original_walk(folder, *args, **kwargs)

        def iterdir(folder):
            self.assertFalse(folder.is_relative_to(self.working))
            return original_iterdir(folder)

        with patch("os.walk", side_effect=walk), patch.object(Path, "iterdir", iterdir):
            self.export()
        self.assertEqual(note.read_text(), "keep")
        self.assertEqual(set(self.working.iterdir()), {self.archive, unrelated})

    def test_runtime_or_state_owner_rejected_before_scratch_probe(self):
        self.image()
        candidate = self.base / "transient"
        scratch, output, runtime = self.ns["select_scratch"](self.input, self.archive, (candidate,))
        runtime.mkdir()
        note = runtime / "unrelated.txt"
        note.write_text("keep", encoding="utf-8")
        with patch.dict(self.ns, writable_probe=Mock()) as namespace:
            with self.assertRaisesRegex(RuntimeError, "Unowned/incompatible venv"):
                self.ns["select_scratch"](self.input, self.archive, (candidate,))
            namespace["writable_probe"].assert_not_called()
        self.assertEqual(note.read_text(), "keep")
        note.unlink()
        self.ns["atomic_json"](output / "owner.json", {"input_root": "another dataset"})
        with patch.dict(self.ns, writable_probe=Mock()) as namespace:
            with self.assertRaisesRegex(ValueError, "another dataset"):
                self.ns["select_scratch"](self.input, self.archive, (candidate,))
            namespace["writable_probe"].assert_not_called()
        self.assertTrue((scratch / "scratch-owner.json").is_file())
        self.assertFalse(self.working.exists())

    def test_exclusive_publication_collision_never_deletes_existing_temp(self):
        self.image()
        self.run_ocr()
        destination = self.export()
        old_zip = self.archive.read_bytes()
        existing = self.working / ".ocr-publish-fixed.tmp"
        existing.write_bytes(b"not ours")
        summary = self.summary()
        with patch("uuid.uuid4", return_value=SimpleNamespace(hex="fixed")):
            with self.assertRaises(FileExistsError):
                self.ns["publish_zip"](self.input, self.output, self.env, self.run_id,
                                       destination / "ocr.zip", self.archive, summary)
        self.assertEqual(existing.read_bytes(), b"not ours")
        self.assertEqual(self.archive.read_bytes(), old_zip)

    def test_copy_digest_failure_never_replaces_owned_zip(self):
        self.image()
        self.run_ocr()
        self.export()
        old_zip = self.archive.read_bytes()
        next_id = "b" * 32
        self.run_ocr(run_id=next_id)
        original = self.ns["shutil"].copyfileobj

        def corrupt(source, target, *args, **kwargs):
            original(source, target, *args, **kwargs)
            if Path(str(getattr(target, "name", ""))).parent == self.working:
                target.write(b"tampered copy")

        with patch("shutil.copyfileobj", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "copy integrity failure"):
                self.export(next_id)
        self.assertEqual(self.archive.read_bytes(), old_zip)
        self.assertEqual(list(self.working.iterdir()), [self.archive])
        self.assertFalse((self.output / "ready.json").exists())

    def test_postpublish_ready_failure_has_no_success_receipt_and_no_temp(self):
        self.image()
        self.run_ocr()
        original = self.ns["atomic_json"]

        def interrupt(path, value):
            if path == self.output / "ready.json":
                raise OSError("ready interrupted")
            return original(path, value)

        with patch.dict(self.ns, atomic_json=interrupt):
            with self.assertRaisesRegex(OSError, "ready interrupted"):
                self.export()
        self.assertEqual(list(self.working.iterdir()), [self.archive])
        self.assertFalse((self.output / "ready.json").exists())
        with self.assertRaises(FileNotFoundError):
            self.ns["checked_ready"](self.input, self.output, self.env, self.run_id, self.archive)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertIsNone(archive.testzip())
        next_id = "b" * 32
        self.run_ocr(run_id=next_id)
        self.export(next_id)
        self.assertEqual(self.ns["checked_ready"](self.input, self.output, self.env,
                                                next_id, self.archive), self.archive)

    def test_environment_allowlist_rejects_logging_paths_but_keeps_driver_and_network_settings(self):
        scratch = self.base / "transient"
        preserved = {"PATH": "/usr/local/bin:/usr/bin", "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
                     "CUDA_HOME": "/usr/local/cuda", "CUDA_MODULE_LOADING": "LAZY",
                     "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                     "HTTPS_PROXY": "http://proxy.invalid:8080", "https_proxy": "http://proxy.invalid:8080",
                     "NO_PROXY": "localhost", "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
                     "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt", "LANG": "C.UTF-8",
                     "HF_TOKEN": "offline-placeholder-not-a-credential"}
        rejected = {"SSLKEYLOGFILE": str(self.working / "tls.keys"),
                    "CUDNN_LOGDEST_DBG": str(self.working / "cudnn.log"), "CUDNN_LOGLEVEL_DBG": "3",
                    "CUDNN_LOGINFO_DBG": "1", "CUDNN_LOGWARN_DBG": "1", "CUDNN_LOGERR_DBG": "1",
                    "GLOG_log_dir": str(self.working / "glog"), "GLOG_log_link": str(self.working / "glog-link"),
                    "GLOG_logtostderr": "0", "MPLCONFIGDIR": str(self.working / "matplotlib"),
                    "UNRECOGNIZED_WRITE_DESTINATION": str(self.working / "unknown.log"),
                    "LD_DEBUG_OUTPUT": str(self.working / "loader.log"),
                    "CUDA_INJECTION64_PATH": str(self.working / "injection.so")}
        inherited = dict(preserved, **rejected)
        before = dict(inherited)
        child = self.ns["isolated_environment"](scratch, inherited)
        self.assertEqual(inherited, before)
        for key, value in preserved.items():
            self.assertEqual(child[key], value, key)
        for key in rejected.keys() - {"MPLCONFIGDIR", "GLOG_logtostderr"}:
            self.assertNotIn(key, child.keys())
        self.assertEqual(child["GLOG_logtostderr"], "1")
        self.assertTrue(Path(child["MPLCONFIGDIR"]).is_relative_to(scratch))
        self.assertTrue(Path(child["MPLCONFIGDIR"]).is_dir())
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "0")
        self.assertFalse(self.working.exists())

    def test_late_export_print_failure_invalidates_ready_but_preserves_zip_and_ownership(self):
        self.image()
        for index, failure in enumerate((OSError("summary output failed"),
                                         KeyboardInterrupt("download output interrupted")), start=1):
            run_id = str(index) * 32
            self.run_ocr(run_id=run_id)
            effects = [failure] if index == 1 else [None, failure]
            with self.subTest(failure=type(failure).__name__), patch("builtins.print", side_effect=effects):
                with self.assertRaises(type(failure)):
                    self.ns["export_current"](self.input, self.output, self.env, run_id, self.archive)
            self.assertFalse((self.output / "ready.json").exists())
            self.assertEqual(self.read(self.output / "publication.json")["run_id"], run_id)
            with zipfile.ZipFile(self.archive) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("L04/L04_V001/001.json", archive.namelist())
            self.assertEqual(list(self.working.iterdir()), [self.archive])
            with self.assertRaises(FileNotFoundError):
                self.ns["checked_ready"](self.input, self.output, self.env, run_id, self.archive)

    def test_late_final_cell_failures_clear_current_ready_and_keep_publication(self):
        self.image()
        for index, failure in enumerate(("worker", "filelink", "display"), start=1):
            run_id = str(index) * 32
            self.run_ocr(run_id=run_id)
            file_link, display = Mock(), Mock()
            namespace = {"RUN_OK": True, "RUN_ID": run_id, "OUTPUT": self.output,
                         "PYTHON": Path("python"), "WORKER": Path("worker.py"), "root": self.input,
                         "ENV_ROOT": self.env, "child_env": {}, "helpers": self.ns,
                         "ARCHIVE": self.archive, "SCRATCH": self.base}

            def export_then_finish(*args, **kwargs):
                self.export(run_id)
                if failure == "worker":
                    raise RuntimeError("late worker completion failure")

            namespace["subprocess"] = SimpleNamespace(run=Mock(side_effect=export_then_finish))
            if failure == "filelink":
                file_link.side_effect = RuntimeError("late filelink completion failure")
            if failure == "display":
                display.side_effect = RuntimeError("late display completion failure")
            module = SimpleNamespace(FileLink=file_link, display=display)
            with self.subTest(failure=failure), patch.dict("sys.modules", {"IPython.display": module}), \
                    patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "late .* completion failure"):
                    exec(compile(CELLS[3], "<export-cell>", "exec"), namespace)
            self.assertFalse(namespace["EXPORT_OK"])
            self.assertFalse((self.output / "ready.json").exists())
            self.assertEqual(self.read(self.output / "publication.json")["run_id"], run_id)
            self.assertEqual(list(self.working.iterdir()), [self.archive])
            with self.assertRaises(FileNotFoundError):
                self.ns["checked_ready"](self.input, self.output, self.env, run_id, self.archive)
            if failure != "display":
                display.assert_not_called()

    def test_failure_cleanup_removes_only_owned_current_ready(self):
        self.image()
        self.run_ocr()
        self.export()
        ready = self.read(self.output / "ready.json")
        ownership = (self.output / "publication.json").read_bytes()
        archive = self.archive.read_bytes()
        self.ns["invalidate_current_ready"](self.input, self.output, self.env, "b" * 32)
        self.assertEqual(self.read(self.output / "ready.json"), ready)
        foreign = dict(ready, input_root=str(self.base / "other-input"))
        self.ns["atomic_json"](self.output / "ready.json", foreign)
        self.ns["invalidate_current_ready"](self.input, self.output, self.env, self.run_id)
        self.assertEqual(self.read(self.output / "ready.json"), foreign)
        self.ns["atomic_json"](self.output / "ready.json", ready)
        self.ns["invalidate_current_ready"](self.input, self.output, self.env, self.run_id)
        self.assertFalse((self.output / "ready.json").exists())
        self.assertEqual((self.output / "publication.json").read_bytes(), ownership)
        self.assertEqual(self.archive.read_bytes(), archive)


if __name__ == "__main__":
    unittest.main()
