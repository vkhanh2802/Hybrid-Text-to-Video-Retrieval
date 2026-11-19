import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from kis.structured_caption import parse_description, render_description, validate_description
from tools.structured_caption_generator import (
    FiniteCaptionLogits, QwenCaptioner, RepeatedCaptionTokenStop,
    describe_image_batches, generate_structured_captions_with_mapping, resolve_caption_runtime,
)
from kis.caption_schema import normalize_caption_record


def description():
    return {
        "summary": "Two presenters hold cue cards in a television studio.",
        "setting": "studio", "background": "a sunset city skyline backdrop",
        "people": [
            {"id": "p1", "position": "left", "description": "adult man with glasses", "clothing": "black shirt", "headwear": "none", "action": "standing", "holding": "red cue card"},
            {"id": "p2", "position": "right", "description": "adult woman", "clothing": "dark blue dress with a neck bow", "headwear": "unknown", "action": "standing", "holding": "red cue card"},
        ],
        "objects": [], "relations": ["p1 stands beside p2"],
    }


class StructuredCaptionTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("faiss"), "requires artifact audit dependencies")
    def test_artifact_audit_accepts_batch_metadata_and_rejects_invalid_batch_size(self):
        from kis.artifact_io import atomic_write_json
        from tools.audit_artifact_formats import audit_video_formats
        from kis.caption_schema import file_sha256
        from kis.structured_caption import caption_generation_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data" / "L19" / "L19_V001" / "001.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image)
            (root / "mapping" / "L19").mkdir(parents=True)
            (root / "mapping" / "L19" / "L19_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0,30,0\n", encoding="utf-8")
            record = {
                "video_id": "L19_V001", "frame_id": 0, "keyframe_n": 1, "image_path": "data/L19/L19_V001/001.jpg",
                "caption": "A scene", "details": "", "retrieval_text": "A scene", "vqa_answers": {},
                "visual_description": description(), "caption_generation": caption_generation_config(),
                "image_sha256": file_sha256(image), "generation_seconds": 1.0, "generation_batch_size": 16,
            }
            path = root / "caption_generator" / "L19" / "L19_V001.json"
            atomic_write_json(path, [record])
            report = next(item for item in audit_video_formats(root, "L19_V001") if item["folder"] == "caption_generator")
            self.assertEqual(report["status"], "OK", report)
            record["generation_batch_size"] = True
            atomic_write_json(path, [record])
            report = next(item for item in audit_video_formats(root, "L19_V001") if item["folder"] == "caption_generator")
            self.assertEqual(report["status"], "ERROR")
            self.assertIn("generation_batch_size", report["detail"])

    def test_describe_batch_preserves_order_and_retries_only_invalid_items(self):
        captioner = QwenCaptioner.__new__(QwenCaptioner)
        captioner.config = {"format_attempts": 2}
        first, second = description(), description()
        first["summary"], second["summary"] = "First image", "Second image"
        images = [object(), object()]
        captioner._generate_batch = Mock(return_value=[json.dumps(first), "!" * 64])
        captioner._generate = Mock(return_value=json.dumps(second))
        self.assertEqual(captioner.describe_batch(images), [first, second])
        captioner._generate_batch.assert_called_once()
        self.assertIs(captioner._generate.call_args.args[0], images[1])
        self.assertEqual(captioner._generate.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_generate_batch_uses_two_prompts_and_slices_after_padded_width(self):
        import numpy
        import torch

        captioner = QwenCaptioner.__new__(QwenCaptioner)
        captioner.config = {"max_new_tokens": 640}
        captioner.processor = Mock()
        captioner.processor.tokenizer.pad_token_id = 9
        captioner.processor.tokenizer.eos_token_id = 2
        captioner.processor.apply_chat_template.side_effect = ["prompt A", "prompt B"]
        input_ids = torch.tensor([[9, 9, 1, 2], [3, 4, 5, 6]])
        captioner.processor.return_value = {"input_ids": input_ids, "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])}
        captioner.processor.batch_decode.return_value = ["result A", "result B"]
        captioner.model = Mock(device=torch.device("cpu"))
        captioner.model.generate.return_value = torch.tensor([[9, 9, 1, 2, 101, 102], [3, 4, 5, 6, 201, 202]])
        images = [object(), object()]
        self.assertEqual(captioner._generate_batch(images, ["A", "B"]), ["result A", "result B"])
        self.assertEqual(captioner.processor.call_args.kwargs["text"], ["prompt A", "prompt B"])
        self.assertEqual(captioner.processor.call_args.kwargs["images"], images)
        decoded = captioner.processor.batch_decode.call_args.args[0]
        self.assertEqual(decoded.tolist(), [[101, 102], [201, 202]])
        self.assertEqual(captioner.model.generate.call_args.kwargs["stopping_criteria"][0].prompt_length, 4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_oom_split_preserves_input_order_and_actual_subbatch_sizes(self):
        import numpy
        import torch

        class Captioner:
            sizes = []

            def describe_batch(self, images):
                self.sizes.append(len(images))
                if len(images) > 2:
                    raise torch.cuda.OutOfMemoryError("test OOM")
                return list(images)

        captioner = Captioner()
        with patch("torch.cuda.empty_cache"):
            groups = list(describe_image_batches(captioner, list(range(5))))
        self.assertEqual([item for descriptions, _, _ in groups for item in descriptions], list(range(5)))
        self.assertTrue(all(size <= 2 for _, _, size in groups))
        self.assertEqual(captioner.sizes, [5, 2, 3, 1, 2])

    def test_batch_16_reaches_backend_and_keeps_frame_identity(self):
        class Captioner:
            sizes = []

            def describe_batch(self, images):
                self.sizes.append(len(images))
                result = []
                for image in images:
                    item = description()
                    item["summary"] = f"Image level {image.getpixel((0, 0))[0]}"
                    result.append(item)
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "data" / "L19" / "L19_V001"
            image_dir.mkdir(parents=True)
            (root / "mapping").mkdir()
            (root / "mapping" / "L19_V001.csv").write_text(
                "n,frame_idx\n" + "".join(f"{n},{n * 10}\n" for n in range(1, 19)), encoding="utf-8",
            )
            for n in range(1, 19):
                Image.new("RGB", (8, 8), (n * 10,) * 3).save(image_dir / f"{n:03d}.jpg")
            captioner = Captioner()
            generate_structured_captions_with_mapping(
                image_root=root / "data", csv_root=root / "mapping", output_dir=root / "captions",
                video_ids=["L19_V001"], path_root=root, captioner=captioner, batch_size=16,
            )
            records = json.loads((root / "captions" / "L19" / "L19_V001" / "L19_V001.json").read_text())
            self.assertEqual(captioner.sizes, [16, 2])
            self.assertEqual([item["frame_id"] for item in records], [n * 10 for n in range(1, 19)])
            self.assertEqual([item["generation_batch_size"] for item in records], [16] * 16 + [2] * 2)
            for n, record in enumerate(records, 1):
                with Image.open(image_dir / f"{n:03d}.jpg") as source:
                    self.assertEqual(record["caption"], f"Image level {source.getpixel((0, 0))[0]}")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_successful_oom_subbatch_is_checkpointed_before_later_failure(self):
        import numpy
        import torch

        class Captioner:
            calls = 0

            def describe_batch(self, images):
                self.calls += 1
                if self.calls == 1:
                    raise torch.cuda.OutOfMemoryError("test OOM")
                if self.calls == 3:
                    raise ValueError("bad remaining images")
                return [description() for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "data" / "L19" / "L19_V001"
            images.mkdir(parents=True)
            (root / "mapping").mkdir()
            (root / "mapping" / "L19_V001.csv").write_text("n,frame_idx\n1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
            for n in range(1, 5):
                Image.new("RGB", (8, 8), "blue").save(images / f"{n:03d}.jpg")
            with patch("torch.cuda.empty_cache"), self.assertRaisesRegex(RuntimeError, "L19_V001/003"):
                generate_structured_captions_with_mapping(
                    image_root=root / "data", csv_root=root / "mapping", output_dir=root / "captions",
                    video_ids=["L19_V001"], path_root=root, captioner=Captioner(), batch_size=4,
                )
            saved = json.loads((root / "captions" / "L19" / "L19_V001" / "L19_V001.json").read_text())
            self.assertEqual([item["keyframe_n"] for item in saved], [1, 2])

    def test_failed_numerical_generation_does_not_publish_a_placeholder(self):
        class BadCaptioner:
            def describe(self, image):
                raise FloatingPointError("non-finite logits")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "data" / "L19" / "L19_V001"
            images.mkdir(parents=True)
            (root / "mapping").mkdir()
            (root / "mapping" / "L19_V001.csv").write_text("n,frame_idx\n1,10\n", encoding="utf-8")
            Image.new("RGB", (32, 32), "blue").save(images / "001.jpg")
            with self.assertRaisesRegex(RuntimeError, "L19_V001/001.*non-finite logits"):
                generate_structured_captions_with_mapping(
                    image_root=root / "data", csv_root=root / "mapping", output_dir=root / "captions",
                    video_ids=["L19_V001"], path_root=root, captioner=BadCaptioner(),
                )
            self.assertFalse((root / "captions" / "L19" / "L19_V001" / "L19_V001.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_runtime_uses_native_bf16_or_fp32_not_emulated_bf16_or_fp16(self):
        import numpy
        import torch
        from types import SimpleNamespace

        for capability, memory_gib, expected in (
            ((7, 5), 16, {"compute_dtype": "fp32", "use_4bit": True}),
            ((6, 0), 16, {"compute_dtype": "fp32", "use_4bit": True}),
            ((8, 0), 16, {"compute_dtype": "bf16", "use_4bit": False}),
            ((8, 6), 6, {"compute_dtype": "bf16", "use_4bit": True}),
        ):
            with self.subTest(capability=capability), patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device"), patch("torch.cuda.get_device_capability", return_value=capability), \
                 patch("torch.cuda.is_bf16_supported", return_value=True), \
                 patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=memory_gib * 1024**3)):
                self.assertEqual(resolve_caption_runtime(), expected)
                with self.assertRaises(ValueError):
                    resolve_caption_runtime(compute_dtype="fp16")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_nonfinite_logits_are_caught_before_greedy_argmax(self):
        import numpy
        import torch

        guard = FiniteCaptionLogits()
        scores = torch.full((1, 8), float("nan"))
        self.assertEqual(scores.argmax(-1).item(), 0)
        with self.assertRaisesRegex(FloatingPointError, "non-finite logits"):
            guard(None, scores)
        for invalid in (torch.full((1, 8), float("-inf")), torch.tensor([[1.0, float("inf")]])):
            with self.assertRaises(FloatingPointError):
                guard(None, invalid)
        masked = torch.tensor([[1.0, float("-inf"), 2.0]])
        self.assertIs(guard(None, masked), masked)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_repetition_stop_counts_only_generated_tokens_not_image_padding(self):
        import numpy
        import torch

        stop = RepeatedCaptionTokenStop(prompt_length=40)
        self.assertFalse(stop(torch.full((1, 71), 7), None).item())
        self.assertTrue(stop(torch.full((1, 72), 7), None).item())
        self.assertFalse(stop(torch.tensor([[7] * 40 + list(range(32))]), None).item())

    def test_exclamation_output_is_rejected_and_never_treated_as_caption(self):
        captioner = QwenCaptioner.__new__(QwenCaptioner)
        captioner.config = {"format_attempts": 2, "compute_dtype": "fp32"}
        captioner._generate = Mock(return_value="!" * 640)
        with self.assertRaisesRegex(ValueError, "Degenerate repetitive Qwen output") as error:
            captioner.describe(object())
        self.assertEqual(captioner._generate.call_count, 2)
        self.assertLess(len(str(error.exception)), 500)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires transformers")
    def test_loader_keeps_weights_and_quantized_compute_in_safe_dtype(self):
        import numpy
        import torch

        for dtype, quantized in (("bf16", False), ("fp32", False), ("fp32", True)):
            with self.subTest(dtype=dtype, quantized=quantized), \
                  patch("tools.structured_caption_generator.resolve_caption_runtime", return_value={"compute_dtype": dtype, "use_4bit": quantized}), \
                 patch("torch.cuda.get_device_name", return_value="test GPU"), \
                 patch("transformers.AutoProcessor.from_pretrained"), \
                 patch("transformers.BitsAndBytesConfig") as bnb, \
                 patch("transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained") as load:
                captioner = QwenCaptioner()
                expected = torch.bfloat16 if dtype == "bf16" else torch.float32
                self.assertEqual(load.call_args.kwargs["torch_dtype"], expected)
                if quantized:
                    self.assertEqual(bnb.call_args.kwargs["bnb_4bit_compute_dtype"], expected)
                self.assertEqual(captioner.config["compute_dtype"], dtype)
                self.assertEqual(captioner.processor.tokenizer.padding_side, "left")

    def test_index_normalization_refreshes_text_after_attribute_edits(self):
        data = description()
        data["people"][1]["clothing"] = "pink dress"
        record = {
            "video_id": "L19_V001", "frame_id": 10, "keyframe_n": 1,
            "image_path": "data/L19/L19_V001/001.jpg", "caption": "stale caption",
            "details": "stale details", "retrieval_text": "stale retrieval text", "visual_description": data,
        }
        normalized = normalize_caption_record(record, "L19_V001", {1: 10})
        self.assertIn("pink dress", normalized["retrieval_text"])
        self.assertNotIn("stale", normalized["retrieval_text"])

    @unittest.skipUnless(importlib.util.find_spec("faiss") and importlib.util.find_spec("sentence_transformers"), "requires caption index dependencies")
    def test_separate_artifact_roots_do_not_mix_caption_versions(self):
        import numpy as np
        from tools.ingest import write_caption_artifacts
        from kis.text import CaptionRetriever

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in (("v1", "Old caption"), ("v2", "New detailed caption")):
                record = {"video_id": "L19_V001", "keyframe_n": 1, "frame_id": 10,
                          "image_path": "data/L19/L19_V001/001.jpg", "caption": text, "retrieval_text": text}
                write_caption_artifacts("L19_V001", [record], np.eye(1, 384, dtype=np.float32), project_root=root / name)
            with patch("kis.text.SentenceTransformer"):
                old = CaptionRetriever(video_groups=("L19",), artifact_root=root / "v1", caption_subdir="caption_generator")
                new = CaptionRetriever(video_groups=("L19",), artifact_root=root / "v2", caption_subdir="caption_generator")
            self.assertEqual(old.caption_data[0]["caption"], "Old caption")
            self.assertEqual(new.caption_data[0]["caption"], "New detailed caption")
            self.assertEqual((old.index.ntotal, new.index.ntotal), (1, 1))

    def test_format_retry_does_not_publish_a_truncated_description(self):
        captioner = QwenCaptioner.__new__(QwenCaptioner)
        captioner.config = {"format_attempts": 2}
        captioner._generate = Mock(side_effect=['{"summary":', json.dumps(description())])
        self.assertEqual(captioner.describe(object()), description())
        self.assertEqual(captioner._generate.call_count, 2)
        self.assertIn("FORMAT CORRECTION", captioner._generate.call_args.args[1])

    def test_rendering_binds_each_person_to_their_own_clothing(self):
        _, details = render_description(description())
        self.assertIn("Person p1 on the left: adult man with glasses, wearing black shirt", details)
        self.assertIn("Person p2 on the right: adult woman, wearing dark blue dress with a neck bow", details)
        self.assertNotIn("headwear: unknown", details)
        self.assertIn("Setting: studio", details)

    def test_json_fences_supported_but_truncation_and_prose_rejected(self):
        self.assertEqual(parse_description("```json\n" + json.dumps(description()) + "\n```"), description())
        with self.assertRaises(ValueError):
            parse_description(json.dumps(description())[:-10])
        with self.assertRaises(ValueError):
            parse_description(json.dumps(description()) + " explanation")

    def test_invalid_relations_and_duplicate_person_ids_rejected(self):
        invalid = description()
        invalid["relations"] = ["p1 stands next to p4"]
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_description(invalid)
        invalid = description()
        invalid["people"][1]["id"] = "p1"
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_description(invalid)

    def test_unexpected_fields_and_invalid_setting_rejected(self):
        invalid = description()
        invalid["query"] = "leaked query"
        with self.assertRaises(ValueError):
            validate_description(invalid)
        invalid = description()
        invalid["setting"] = "new york"
        with self.assertRaisesRegex(ValueError, "setting"):
            validate_description(invalid)

    def test_optional_arrays_default_to_empty_and_overlay_noise_is_not_indexed(self):
        data = description()
        del data["relations"]
        del data["objects"]
        self.assertEqual(validate_description(data)["relations"], [])
        data["objects"] = [{"name": "TV screen", "attributes": "lower third banner", "position": "bottom"}]
        _, details = render_description(data)
        self.assertNotIn("TV screen", details)

    def test_literal_attribute_lists_are_canonicalized_without_new_facts(self):
        data = description()
        data["people"][0]["clothing"] = ["black shirt", "gray tie"]
        data["objects"] = [{"name": "engine", "attributes": ["rusty", "red pipes"]}]
        normalized = validate_description(data)
        self.assertEqual(normalized["people"][0]["clothing"], "black shirt, gray tie")
        self.assertEqual(normalized["objects"][0]["attributes"], "rusty, red pipes")
        self.assertEqual(normalized["objects"][0]["position"], "unknown")

    def test_checkpoint_resume_and_changed_source_are_checked_before_inference(self):
        class Captioner:
            calls = 0

            def describe(self, image):
                self.calls += 1
                return copy.deepcopy(description())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "data" / "L19" / "L19_V001"
            images.mkdir(parents=True)
            (root / "mapping" / "L19").mkdir(parents=True)
            (root / "mapping" / "L19" / "L19_V001.csv").write_text("n,frame_idx\n1,10\n2,20\n", encoding="utf-8")
            Image.new("RGB", (32, 32), "blue").save(images / "001.jpg")
            Image.new("RGB", (32, 32), "green").save(images / "002.jpg")
            captioner = Captioner()
            kwargs = {
                "image_root": root / "data", "csv_root": root / "mapping",
                "output_dir": root / "v2" / "caption_generator", "video_ids": ["L19_V001"],
                "path_root": root, "keyframes_by_video": {"L19_V001": [2]}, "captioner": captioner,
            }
            generate_structured_captions_with_mapping(**kwargs)
            generate_structured_captions_with_mapping(**kwargs)
            self.assertEqual(captioner.calls, 1)
            path = root / "v2" / "caption_generator" / "L19" / "L19_V001" / "L19_V001.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([record["keyframe_n"] for record in saved], [2])
            self.assertEqual(saved[0]["frame_id"], 20)
            self.assertIn("visual_description", saved[0])
            Image.new("RGB", (32, 32), "red").save(images / "002.jpg")
            with self.assertRaisesRegex(ValueError, "source image changed"):
                generate_structured_captions_with_mapping(**kwargs)
            self.assertEqual(captioner.calls, 1)


if __name__ == "__main__":
    unittest.main()
