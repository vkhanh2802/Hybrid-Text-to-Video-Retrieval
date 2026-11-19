import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import shutil
import sys

from PIL import Image

from kis.artifact_io import atomic_write_json
from tools.human_holdout import sha
from tools.kaggle_vlm_benchmark import run, validate_packet
from tools.pipeline_ablation import PROTOCOL, metrics
from tools.vlm_benchmark_packet import export, package, replay


class VLMBenchmarkPacketTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_run = self.root / "cpu"
        self.source_run.mkdir()
        self.data = {}
        frames, judgments = [], []
        for n, color in enumerate(("blue", "red", "gray"), 1):
            image = Image.new("RGB", (24, 24), color)
            raw = BytesIO()
            image.save(raw, format="PNG")
            image.close()
            self.data[n] = raw.getvalue()
            frame = {"video_id": "L01_V001", "frame_id": n*10, "keyframe_n": n, "pts_time": n,
                     "score": 0.9-n/10}
            frames.append(frame)
            judgments.append({**frame, "image_sha256": hashlib.sha256(self.data[n]).hexdigest(),
                              "judgment": ("positive", "negative", "unknown")[n-1]})
        unit = {"query_id": "dev_q", "role": "development", "split": "dev", "task": "image",
                "event_text": "A blue object.", "raw_query": "A blue object.", "video_id": "L01_V001", "judgments": judgments}
        self.case = {"case_id": "dev_q__full", "unit": unit, "arm": "full", "results": frames[:2],
                     "metrics": metrics(frames[:2], unit)}
        atomic_write_json(self.source_run / "protocol.json", {**PROTOCOL, "source_sha256": {}})
        atomic_write_json(self.source_run / "cases/dev_q__full.json", self.case)
        atomic_write_json(self.source_run / "case_index.json", [{"case_id": "dev_q__full"}])
        self.packet_root = self.root / "packet_root"

    def export_fixture(self):
        data = self.data
        class Source:
            def read(self, video_id, number):
                return data[number], "image/png"
        def reference(candidate):
            return {"status": "requested", **{key: candidate[key] for key in ("video_id", "frame_id", "keyframe_n", "pts_time")}}
        with patch("tools.vlm_benchmark_packet.image_reference", side_effect=reference), \
             patch("tools.vlm_benchmark_packet.ArtifactCaptions") as caption:
            caption.return_value.get_caption.return_value = None
            export(self.source_run, self.packet_root, source=Source())

    def test_blind_packet_excludes_ground_truth_and_archive_excludes_private_bindings(self):
        self.export_fixture()
        manifest = validate_packet(self.packet_root / "vlm_packet")
        text = json.dumps(manifest)
        for forbidden in ('"judgment"', '"judgments"', '"video_id"', '"credentials_file"', '"drive_folder_id"'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(len(manifest["tasks"]), 6)  # 3 images x 2 caption modes, candidate/pair reuse.
        bundle, notebook = self.root / "kis-vlm-benchmark.zip", self.root / "run.ipynb"
        package(self.packet_root, bundle, notebook)
        with zipfile.ZipFile(bundle) as archive:
            self.assertFalse(any("private" in name or "/cases/" in name for name in archive.namelist()))
        book = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertTrue(all(cell.get("execution_count") is None for cell in book["cells"] if cell["cell_type"] == "code"))
        # Run only the notebook's input extraction cell, no pip/model/GPU calls.
        for mode in ("zip", "expanded"):
            input_dir, work_dir = self.root / ("input_" + mode), self.root / ("work_" + mode)
            input_dir.mkdir()
            if mode == "zip":
                shutil.copy2(bundle, input_dir / bundle.name)
            else:
                with zipfile.ZipFile(bundle) as archive:
                    archive.extractall(input_dir)
            code = "".join(book["cells"][1]["source"]).replace("/kaggle/input", input_dir.as_posix()).replace(
                "/kaggle/working/benchmark_code", work_dir.as_posix())
            old_path = list(sys.path)
            try:
                namespace = {}
                exec(compile(code, "synthetic_notebook_input_cell", "exec"), namespace)
                self.assertTrue((namespace["root"] / "vlm_packet/manifest.json").is_file())
            finally:
                sys.path[:] = old_path
        with self.assertRaises(FileExistsError):
            export(self.source_run, self.packet_root, source=None)

    def test_replay_rejects_fake_execution_then_uses_production_formula_on_fixture_scores(self):
        self.export_fixture()
        results = self.root / "gpu_fixture"
        run(self.packet_root / "vlm_packet", results, scorer=lambda task: 8)
        with self.assertRaisesRegex(ValueError, "real Kaggle run"):
            replay(self.source_run, self.packet_root, results, self.root / "bad")
        # Simulate the real-result schema only in an isolated unit fixture. No GPU or actual benchmark scores.
        context = json.loads((results / "run_context.json").read_text(encoding="utf-8"))
        context["execution"] = "kaggle_cuda_qwen_nf4"
        atomic_write_json(results / "run_context.json", context)
        replay(self.source_run, self.packet_root, results, self.root / "replayed")
        summary = json.loads((self.root / "replayed/summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["automatically_promoted"])
        self.assertEqual(summary["pair_diagnostics"]["development"]["paired_valid_pairs"], 3)
        with self.assertRaises(FileExistsError):
            replay(self.source_run, self.packet_root, results, self.root / "replayed")

    def test_missing_scores_block_final_comparison(self):
        self.export_fixture()
        results = self.root / "gpu_fixture"
        run(self.packet_root / "vlm_packet", results, max_tasks=1, scorer=lambda task: 8)
        context = json.loads((results / "run_context.json").read_text(encoding="utf-8"))
        context["execution"] = "kaggle_cuda_qwen_nf4"
        atomic_write_json(results / "run_context.json", context)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay(self.source_run, self.packet_root, results, self.root / "replayed")

    def test_incomplete_sequence_does_not_get_a_single_image_sequence_score(self):
        case = self.case
        case["unit"].update(task="sequence", actions=["First event.", "Second event."],
                            event_relevant_keyframes=[[1], [2]])
        first, second = case["results"]
        case["results"] = [
            {**first, "temporal": {"coverage": 0.5, "events": [dict(first, action="First event.", status="retrieved"),
                {"action": "Second event.", "status": "unknown"}]}},
            {**first, "temporal": {"coverage": 1, "events": [dict(first, action="First event.", status="retrieved"),
                dict(second, action="Second event.", status="retrieved")]}}]
        case["metrics"] = metrics(case["results"], case["unit"])
        atomic_write_json(self.source_run / "cases/dev_q__full.json", case)
        self.export_fixture()
        manifest = validate_packet(self.packet_root / "vlm_packet")
        self.assertEqual(len(manifest["tasks"]), 2)  # One full path x two caption modes, not the incomplete result.
        self.assertTrue(all(task["kind"] == "path" and len(task["images"]) == 2 for task in manifest["tasks"]))
        private = json.loads((self.packet_root / "bindings_private.json").read_text(encoding="utf-8"))
        self.assertEqual(set(private["cases"]["dev_q__full"]["modes"]["image_only"]), {"1"})


if __name__ == "__main__":
    unittest.main()
