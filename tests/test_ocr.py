from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image

from kis.ocr import (OCRConfig, OCRRetriever, SCHEMA, contains_literal, digest, fuse_ocr,
                     literal_queries, normalize_text, search_ocr, validate_frame)
from tools.build_ocr import LocalPaddleOCR, build_index, extract
from tools.prepare_ocr_pilot import prepare


def engine():
    # Synthetic provenance ONLY for temp unit fixtures; no real OCR inference.
    return {"backend": "paddleocr", "package_version": "test", "models": [
        {"name": "unit_fixture", "revision": "fixture", "files_sha256": {"fixture": "0"*64}}]}


class OCRFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.images, self.mapping = self.root / "images", self.root / "mapping"
        folder = self.images / "L01/L01_V001"
        folder.mkdir(parents=True)
        target = self.mapping / "L01/L01_V001.csv"
        target.parent.mkdir(parents=True)
        target.write_text("n,frame_idx,pts_time\n1,10,1.0\n2,20,2.0\n3,30,3.0\n", encoding="utf-8")
        self.frames = []
        for i in (1, 2, 3):
            path = folder / f"{i:03d}.jpg"
            Image.new("RGB", (100, 60), "white").save(path)
            self.frames.append({"video_id": "L01_V001", "frame_id": i*10, "keyframe_n": i, "pts_time": float(i),
                "image_relative": f"L01/L01_V001/{i:03d}.jpg", "image_sha256": digest(path),
                "width": 100, "height": 60, "status": "empty", "lines": []})

    def line(self, text, confidence=.95, region="unknown"):
        return {"text": text, "confidence": confidence, "region": region,
                "bbox": [[1, 1], [99, 1], [99, 30], [1, 30]]}

    def index(self, texts, *, confidence=.95, regions=None):
        frames = deepcopy(self.frames[:len(texts)])
        for i, text in enumerate(texts):
            if text is not None:
                frames[i].update(status="ok", lines=[self.line(text, confidence, regions[i] if regions else "unknown")])
        source = self.root / "detections.json"
        source.write_text(json.dumps({"schema": SCHEMA, "source_kind": "pixel_ocr", "engine": engine(), "frames": frames}), encoding="utf-8")
        destination = self.root / "index"
        manifest = build_index(source, self.images, self.mapping, destination)
        return OCRRetriever(destination / "index.sqlite3", ("L01",), self.mapping), manifest, source


class OCRIndexTest(OCRFixture):
    def test_independent_index_literal_numbers_and_no_caption_fallback(self):
        retriever, manifest, _ = self.index(["LỚP 6A", "LỚP 16A", "LỚP 6B"])
        hits = retriever.search("6A", OCRConfig())
        self.assertEqual([h["keyframe_n"] for h in hits], [1])
        self.assertEqual(hits[0]["ocr"]["text"], "LỚP 6A")
        self.assertFalse(hits[0]["ocr"]["verified"])
        self.assertNotIn("retrieval_text", hits[0])
        self.assertEqual(manifest["lines"], 3)

    def test_vietnamese_diacritics_and_math_operators_preserved(self):
        retriever, _, _ = self.index(["Tiếng Anh: x=1", "Tieng Anh: x=10", "Tiếng Anh: x+1"])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("x=1", OCRConfig())], [1])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("Tiếng Anh", OCRConfig())], [1, 3])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("Tieng Anh", OCRConfig())], [2])
        self.assertIn("+", normalize_text("X+1"))
        self.assertFalse(contains_literal("notebook", "not"))

    def test_regions_remain_separate_and_unknown_is_not_guessed_content(self):
        retriever, manifest, _ = self.index(["English", "English", "English"], regions=["content", "overlay", "unknown"])
        self.assertEqual(len(retriever.search("English", OCRConfig())), 3)
        self.assertEqual([h["keyframe_n"] for h in retriever.search("English", OCRConfig(region="content"))], [1])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("English", OCRConfig(region="overlay"))], [2])
        self.assertEqual(manifest["regions"], {"content": 1, "overlay": 1, "unknown": 1})

    def test_superscripts_subscripts_and_plain_digits_do_not_collapse(self):
        retriever, _, _ = self.index(["x²=1", "x2=1", "x₂=1"])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("x²=1", OCRConfig())], [1])
        self.assertEqual([h["keyframe_n"] for h in retriever.search("x2=1", OCRConfig())], [2])

    def test_low_confidence_kept_in_artifact_not_accepted_as_match(self):
        retriever, manifest, _ = self.index(["English"], confidence=.2)
        self.assertEqual(manifest["lines"], 1)
        self.assertEqual(retriever.search("English", OCRConfig()), [])
        self.assertEqual(len(retriever.search("English", OCRConfig(min_confidence=.1))), 1)

    def test_empty_ocr_is_not_negative_or_invented_text(self):
        retriever, manifest, _ = self.index([None])
        self.assertEqual(manifest["frame_status_counts"], {"empty": 1})
        self.assertEqual(retriever.search("English", OCRConfig()), [])

    def test_fts_operator_input_is_literal_and_scope_is_filtered(self):
        retriever, _, _ = self.index(["English", "hello"])
        self.assertEqual(retriever.search('English" OR "hello', OCRConfig()), [])
        other = OCRRetriever(retriever.index_path, ("L02",), self.mapping)
        self.assertEqual(other.search("English", OCRConfig()), [])

    def test_mapping_and_index_tampering_are_rejected(self):
        retriever, _, _ = self.index(["English"])
        self.assertTrue(retriever.search("English", OCRConfig()))
        path = self.mapping / "L01/L01_V001.csv"
        path.write_text("n,frame_idx,pts_time\n1,99,1.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mapping"):
            retriever.search("English", OCRConfig())
        with retriever.index_path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            OCRRetriever(retriever.index_path, ("L01",), self.mapping).search("English", OCRConfig())

    def test_build_checks_image_hash_and_rejects_caption_artifact(self):
        frame = deepcopy(self.frames[0])
        frame["image_sha256"] = "0"*64
        source = self.root / "bad.json"
        source.write_text(json.dumps({"schema": SCHEMA, "source_kind": "pixel_ocr", "engine": engine(), "frames": [frame]}))
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_index(source, self.images, self.mapping, self.root / "bad_index")
        self.assertFalse((self.root / "bad_index").exists())
        source.write_text(json.dumps({"schema": SCHEMA, "source_kind": "generated_caption", "frames": []}))
        with self.assertRaises(ValueError):
            build_index(source, self.images, self.mapping, self.root / "caption_index")

    def test_validation_rejects_nan_and_invalid_bbox(self):
        for updates in ({"confidence": float("nan")}, {"confidence": True}, {"bbox": [[0,0]]*4},
                        {"region": "board_guessed_by_caption"}, {"bbox": [[0,0],[999,0],[99,10],[0,10]]}):
            frame = dict(self.frames[0], status="ok", lines=[dict(self.line("text"), **updates)])
            with self.assertRaises(ValueError):
                validate_frame(frame)


class OCRRoutingTest(unittest.TestCase):
    def test_literal_routing_never_translates_or_invents_visible_text(self):
        query = {"raw_query": 'Bảng ghi "Tiếng Anh" và lớp 6A', "needs_ocr": True,
                 "entities": [{"type": "text", "value": "English"}, {"type": "text", "value": "6A"}]}
        result = literal_queries(query, OCRConfig())
        self.assertEqual([r["text"] for r in result], ["Tiếng Anh", "6A"])
        self.assertEqual(literal_queries(dict(query, needs_ocr=False), OCRConfig()), [])
        self.assertEqual(literal_queries({"ocr_text": "English"}, OCRConfig())[0]["source"], "user_ocr_text")

    def test_off_and_error_fallback_never_change_base_candidates(self):
        retriever = Mock()
        query = {"ocr_text": "6A"}
        hits, diag = search_ocr(query, retriever, OCRConfig())
        self.assertEqual(diag["status"], "disabled")
        retriever.search.assert_not_called()
        retriever.search.side_effect = FileNotFoundError("absent")
        hits, diag = search_ocr(query, retriever, OCRConfig(mode="retrieve"))
        self.assertEqual(diag["status"], "unavailable")
        base = [{"video_id": "v", "frame_id": 1, "score": .8}]
        self.assertIs(fuse_ocr(base, hits, OCRConfig(mode="retrieve")), base)

    def test_ocr_union_can_add_an_unseen_frame_without_fabricating_caption(self):
        base = [{"video_id": "v", "frame_id": 1, "score": .5, "evidence": {}}]
        original = deepcopy(base)
        hit = {"video_id": "other", "frame_id": 2, "score": 1.0,
               "ocr": {"text": "6A", "source": "pixel_ocr", "verified": False}}
        ranked = fuse_ocr(base, [hit], OCRConfig(mode="retrieve"))
        self.assertEqual(ranked[0]["video_id"], "other")
        self.assertNotIn("retrieval_text", ranked[0])
        self.assertEqual(base, original)
        self.assertIs(fuse_ocr(base, [hit], OCRConfig(mode="diagnostics")), base)

    def test_config_limits(self):
        for kwargs in ({"mode": "on"}, {"weight": float("nan")}, {"min_confidence": True},
                       {"top_k": 101}, {"max_queries": False}, {"region": "guessed"}):
            with self.assertRaises(ValueError):
                OCRConfig(**kwargs)


class OCRExtractionTest(OCRFixture):
    def test_bounded_extract_distinguishes_errors_from_empty_detections(self):
        manifest = self.root / "input.json"
        manifest.write_text(json.dumps({"schema": "kis_ocr_input_v1", "frames": self.frames[:2]}))
        settings = self.root / "engine.json"
        settings.write_text(json.dumps(engine()))
        backend = Mock(recognize=Mock(side_effect=[[self.line("text")], RuntimeError("injected")]))
        result = extract(manifest, settings, self.images, self.mapping, self.root / "out.json", backend=backend)
        self.assertFalse(result["complete"])
        self.assertEqual([f["status"] for f in result["frames"]], ["ok", "error"])
        self.assertEqual(result["frames"][1]["lines"], [])
        self.assertNotIn("caption", result["frames"][0])
        with self.assertRaises(ValueError):
            extract(manifest, settings, self.images, self.mapping, self.root / "bound.json", max_frames=1, backend=backend)

    def test_local_paddle_adapter_pins_versions_paths_and_cpu(self):
        settings = {"backend": "paddleocr", "package_version": "3.7.0", "paddle_version": "3.3.1", "models": []}
        for name in ("det", "rec"):
            directory = self.root / name
            directory.mkdir()
            files = {}
            for filename in ("inference.json", "inference.pdiparams", "inference.yml"):
                path = directory / filename
                path.write_text("fixture")
                files[filename] = digest(path)
            settings["models"].append({"name": name, "revision": "fixture", "directory": str(directory), "files_sha256": files})
        prediction = SimpleNamespace(json={"res": {"rec_texts": ["6A"], "rec_scores": [.9],
                                                   "rec_polys": [[[1,1],[10,1],[10,10],[1,10]]]}})
        factory = Mock(return_value=Mock(predict=Mock(return_value=[prediction])))
        with patch("tools.build_ocr.importlib.metadata.version", side_effect=lambda p: "3.7.0" if p == "paddleocr" else "3.3.1"), \
             patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory)}), \
             patch.dict("os.environ", {}, clear=False):
            backend = LocalPaddleOCR(settings)
            self.assertEqual(backend.recognize("fixture.jpg")[0]["region"], "unknown")
        self.assertEqual(factory.call_args.kwargs["device"], "cpu")
        self.assertFalse(factory.call_args.kwargs["use_doc_unwarping"])
        self.assertEqual(factory.call_args.kwargs["text_detection_model_dir"], str(self.root / "det"))

    def test_pilot_bundle_is_bounded_label_free_and_notebook_compiles(self):
        packet, archive, book = self.root / "packet", self.root / "pilot.zip", self.root / "pilot.ipynb"
        result = prepare(packet, archive, book, self.images, self.mapping, max_frames=2)
        self.assertEqual(result["frames"], 2)
        self.assertFalse(result["model_run_locally"])
        with zipfile.ZipFile(archive) as z:
            self.assertFalse(any("caption" in p or "qrels" in p or "credential" in p for p in z.namelist()))
            manifest = json.loads(z.read("manifest.json"))
            self.assertFalse(manifest["contains_generated_captions"])
        document = json.loads(book.read_text(encoding="utf-8"))
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "ocr-pilot", "exec")
        with self.assertRaises(FileExistsError):
            prepare(packet, archive, book, self.images, self.mapping, max_frames=2)

    def test_pilot_bootstrap_accepts_zip_and_kaggle_expanded_input_without_model(self):
        import shutil

        packet, archive, book = self.root / "packet", self.root / "pilot.zip", self.root / "pilot.ipynb"
        prepare(packet, archive, book, self.images, self.mapping, max_frames=2)
        document = json.loads(book.read_text(encoding="utf-8"))
        bootstrap = "".join(next(c["source"] for c in document["cells"] if c["cell_type"] == "code"))
        for mode in ("zip", "expanded"):
            inputs = self.root / (mode + "_input")
            inputs.mkdir()
            work = self.root / (mode + "_work")
            if mode == "zip":
                shutil.copyfile(archive, inputs / "kis-ocr-pilot-20261104.zip")
            else:
                shutil.copytree(packet, inputs / "expanded")
            code = bootstrap.replace("'/kaggle/input'", repr(str(inputs))).replace(
                "'/kaggle/working/kis-ocr-pilot'", repr(str(work)))
            scope = {}
            exec(compile(code, "ocr-bootstrap-fixture", "exec"), scope)
            self.assertEqual(len(scope["manifest"]["frames"]), 2)
            self.assertTrue((scope["root"] / "tools/build_ocr.py").is_file())


if __name__ == "__main__":
    unittest.main()
