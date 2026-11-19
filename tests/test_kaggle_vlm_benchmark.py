from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.config import VLM_CONFIG
from tools.kaggle_vlm_benchmark import PROMPT_VERSION, path_prompt, run, score_path, sha, task_id, validate_packet


def synthetic_vlm_packet(root):
    root = Path(root)
    (root / "images").mkdir(parents=True)
    image = Image.new("RGB", (24, 24), "blue")
    raw = BytesIO()
    image.save(raw, format="PNG")
    image.close()
    import hashlib
    digest = hashlib.sha256(raw.getvalue()).hexdigest()
    relative = "images/" + digest + ".png"
    (root / relative).write_bytes(raw.getvalue())
    reference = {"path": relative, "sha256": digest, "status": "ready", "pts_time": 1,
                 "action": "A blue rectangle.", "caption": "Untrusted caption says red."}
    tasks = [{"kind": "image", "caption_mode": "with_caption", "query": "A blue rectangle.", "images": [reference]},
             {"kind": "path", "caption_mode": "image_only", "query": "First blue then blue.",
              "images": [reference, {**reference, "pts_time": 2}]}]
    for task in tasks:
        task["task_id"] = task_id(task, VLM_CONFIG)
    manifest = {"schema": "kis_blind_vlm_packet_v1", "prompt_version": PROMPT_VERSION,
                "model": VLM_CONFIG, "tasks": tasks}
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


class KaggleVLMBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "packet"
        self.manifest = synthetic_vlm_packet(self.root)

    def test_packet_validates_and_rejects_labels_or_changed_pixels(self):
        self.assertEqual(len(validate_packet(self.root)["tasks"]), 2)
        bad = {**self.manifest, "judgments": []}
        atomic_write_json(self.root / "manifest.json", bad)
        with self.assertRaisesRegex(ValueError, "Labels or credentials"):
            validate_packet(self.root)
        atomic_write_json(self.root / "manifest.json", self.manifest)
        (self.root / self.manifest["tasks"][0]["images"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed candidate image"):
            validate_packet(self.root)

    def test_missing_inputs_never_become_valid_scores_and_resume_skips_completed(self):
        output = Path(self.temp.name) / "scores"
        score = Mock(return_value=8)
        run(self.root, output, max_tasks=1, scorer=score)
        self.assertFalse(json.loads((output / "summary.json").read_text())["complete"])
        run(self.root, output, scorer=score)
        self.assertEqual(score.call_count, 2)
        run(self.root, output, scorer=score)
        self.assertEqual(score.call_count, 2)
        self.assertEqual(json.loads((output / "run_context.json").read_text())["execution"], "injected_test_scorer")

    def test_invalid_score_is_unknown_not_a_fabricated_zero(self):
        output = Path(self.temp.name) / "scores"
        run(self.root, output, scorer=lambda task: float("nan"))
        rows = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["score"] is None and row["status"] == "invalid_score" for row in rows))

    def test_heavy_execution_stops_on_local_machine_before_model_load(self):
        with patch("tools.kaggle_vlm_benchmark.Path.is_dir", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "reserved.*Kaggle"):
                run(self.root, Path(self.temp.name) / "scores")

    def test_path_prompt_is_ordered_and_image_only_omits_caption(self):
        task = self.manifest["tasks"][1]
        prompt = path_prompt(task)
        self.assertIn("1.000s", prompt)
        self.assertIn("2.000s", prompt)
        self.assertNotIn("Untrusted caption says red", prompt)
        self.assertIn("Do not infer continuous motion", prompt)

    def test_multi_image_call_uses_all_images_and_strict_score_parser(self):
        class Tensor:
            shape = (1, 2)
            def to(self, device):
                return self
        processor = Mock(return_value={"input_ids": Tensor()})
        processor.apply_chat_template.return_value = "chat"
        processor.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        processor.batch_decode.return_value = ["8"]
        reranker = SimpleNamespace(processor=processor, model=SimpleNamespace(device="test", generate=Mock(return_value=[[0, 1, 2, 3]])))
        self.assertEqual(score_path(reranker, self.root, self.manifest["tasks"][1]), 8)
        self.assertEqual(len(processor.call_args.kwargs["images"]), 2)
        processor.batch_decode.return_value = ["8 because good"]
        self.assertIsNone(score_path(reranker, self.root, self.manifest["tasks"][1]))

    def test_unsafe_path_and_non_increasing_times_rejected(self):
        for change in ("path", "times"):
            manifest = json.loads(json.dumps(self.manifest))
            task = manifest["tasks"][1]
            if change == "path":
                task["images"][0]["path"] = "../secret.png"
            else:
                task["images"][1]["pts_time"] = 1
            task["task_id"] = task_id(task, VLM_CONFIG)
            atomic_write_json(self.root / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                validate_packet(self.root)


if __name__ == "__main__":
    unittest.main()
