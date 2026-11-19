import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from kis.light_caption import InternVLCaptioner, light_caption_generation_config, validate_dense_caption
from kis.config import FAST_CAPTION_CONFIG
from kis.structured_caption import caption_generation_config
from tools.structured_caption_generator import generate_structured_captions_with_mapping, resolve_caption_backend_runtime


class InternVLCaptionTest(unittest.TestCase):
    def test_pinned_model_and_short_output_budget(self):
        config = light_caption_generation_config("fp32")
        self.assertEqual(config["model"], "OpenGVLab/InternVL3_5-1B-HF")
        self.assertEqual(config["revision"], "9191dbccf312b537016f041b25d61c72e7c5c9f3")
        self.assertEqual(config["max_new_tokens"], 96)
        self.assertEqual(config["max_patches"], 2)
        self.assertEqual(config["backend"], "internvl_dense_v3")
        self.assertEqual(config, caption_generation_config("internvl", False, "fp32"))

    def test_dense_caption_validation_rejects_padding_and_reasoning(self):
        self.assertEqual(validate_dense_caption(" A woman  in a blue shirt. "), "A woman in a blue shirt.")
        for text in ("", "!" * 100, "Caption <pad>", "<think>reasoning</think>caption", "Caption<|im_end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_dense_caption(text)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_runtime_t4_uses_fp32_and_no_automatic_quantization(self):
        import numpy
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device"), \
             patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(resolve_caption_backend_runtime("internvl"), {"compute_dtype": "fp32", "use_4bit": False})
            self.assertTrue(resolve_caption_backend_runtime("internvl", use_4bit=True)["use_4bit"])
            with self.assertRaises(ValueError):
                resolve_caption_backend_runtime("internvl", compute_dtype="fp16")

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires transformers")
    def test_loader_uses_native_hf_class_and_safe_dtype(self):
        import numpy
        import torch

        with patch("tools.structured_caption_generator.resolve_caption_backend_runtime", return_value={"compute_dtype": "bf16", "use_4bit": False}), \
             patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.InternVLForConditionalGeneration.from_pretrained") as load:
            captioner = InternVLCaptioner()
        self.assertEqual(load.call_args.args[0], FAST_CAPTION_CONFIG["model"])
        self.assertEqual(load.call_args.kwargs["torch_dtype"], torch.bfloat16)
        self.assertFalse(load.call_args.kwargs["trust_remote_code"])
        self.assertEqual(captioner.processor.tokenizer.padding_side, "left")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_batch_slices_padded_prompt_and_preserves_image_order(self):
        import numpy
        import torch

        captioner = InternVLCaptioner.__new__(InternVLCaptioner)
        captioner.config = light_caption_generation_config("bf16")
        captioner.device, captioner.dtype = torch.device("cpu"), torch.bfloat16
        captioner.processor = Mock()
        captioner.processor.tokenizer.pad_token_id = 9
        captioner.processor.tokenizer.unk_token_id = None
        captioner.processor.tokenizer.convert_tokens_to_ids.return_value = 99
        captioner.processor.apply_chat_template.side_effect = ["prompt A", "prompt B"]
        captioner.processor.return_value = {
            "input_ids": torch.tensor([[9, 1, 2], [3, 4, 5]]),
            "pixel_values": torch.ones((2, 3, 2, 2)),
        }
        captioner.processor.batch_decode.return_value = ["A man holds a bowl.", "A woman teaches English."]
        captioner.model = Mock()
        captioner.model.generate.return_value = torch.tensor([[9, 1, 2, 11, 12], [3, 4, 5, 21, 22]])
        images = [object(), object()]
        result = captioner.describe_batch(images)
        self.assertEqual(result, ["A man holds a bowl.", "A woman teaches English."])
        self.assertEqual(captioner.processor.call_args.kwargs["images"], images)
        self.assertEqual(captioner.processor.call_args.kwargs["max_patches"], 2)
        self.assertTrue(captioner.processor.call_args.kwargs["crop_to_patches"])
        self.assertEqual(captioner.model.generate.call_args.kwargs["input_ids"].dtype, torch.int64)
        self.assertEqual(captioner.model.generate.call_args.kwargs["pixel_values"].dtype, torch.bfloat16)
        self.assertEqual(captioner.processor.batch_decode.call_args.args[0].tolist(), [[11, 12], [21, 22]])
        self.assertTrue(captioner.processor.batch_decode.call_args.kwargs["skip_special_tokens"])
        captioner.model.generate.return_value = torch.tensor([[9, 1, 2, 99, 12], [3, 4, 5, 21, 22]])
        captioner.processor.apply_chat_template.side_effect = ["prompt A", "prompt B"]
        captioner.processor.batch_decode.reset_mock()
        with self.assertRaisesRegex(ValueError, "reasoning"):
            captioner.describe_batch(images)
        captioner.processor.batch_decode.assert_not_called()

    def test_dense_generation_checkpoint_schema_and_model_switch_backup(self):
        from kis.artifact_io import atomic_write_json
        from kis.caption_schema import file_sha256
        from kis.caption_schema import normalize_caption_record
        from tools.kaggle_keyframes_caption import restart_caption_runtime_if_changed
        from tools.prepare_video_data import VideoDatasetBuilder

        class Captioner:
            calls = 0
            config = light_caption_generation_config("bf16")

            def describe_batch(self, images):
                self.calls += 1
                return ["A cook pours eggs into a pot.", "A cook cuts tofu above a pot."][:len(images)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "data" / "L19" / "L19_V001"
            images.mkdir(parents=True)
            (root / "mapping").mkdir()
            (root / "mapping" / "L19_V001.csv").write_text("n,frame_idx\n1,10\n2,20\n", encoding="utf-8")
            for n in (1, 2):
                Image.new("RGB", (8, 8), "blue").save(images / f"{n:03d}.jpg")
            captioner = Captioner()
            kwargs = dict(image_root=root / "data", csv_root=root / "mapping", output_dir=root / "caption_generator",
                          video_ids=["L19_V001"], path_root=root, captioner=captioner, backend="internvl", batch_size=16)
            generate_structured_captions_with_mapping(**kwargs)
            generate_structured_captions_with_mapping(**kwargs)
            self.assertEqual(captioner.calls, 1)
            path = root / "caption_generator" / "L19" / "L19_V001" / "L19_V001.json"
            records = json.loads(path.read_text())
            self.assertEqual([r["frame_id"] for r in records], [10, 20])
            self.assertTrue(all("visual_description" not in r and r["vqa_answers"] == {} for r in records))
            self.assertEqual(records[0]["caption_generation"]["model"], FAST_CAPTION_CONFIG["model"])
            self.assertEqual(records[0]["caption_generation"]["compute_dtype"], "bf16")
            edited = dict(records[0], caption="Updated caption.", retrieval_text="stale")
            self.assertEqual(normalize_caption_record(edited, "L19_V001", {1: 10})["retrieval_text"], "Updated caption")

            builder = VideoDatasetBuilder(root)
            old = caption_generation_config("qwen", True, "fp32")
            builder._save_state("L19_V001", {"keyframes": {"unchanged": True}, "caption_generation": old})
            before = file_sha256(path)
            new = light_caption_generation_config("fp32")
            with self.assertRaisesRegex(ValueError, "model/prompt changed"):
                restart_caption_runtime_if_changed(builder, "L19_V001", new)
            restart_caption_runtime_if_changed(builder, "L19_V001", new, force=True)
            state = builder._state("L19_V001")
            backup = root / state["caption_generation_history"][0]["backup"] / path.name
            self.assertEqual(file_sha256(backup), before)
            self.assertEqual(state["keyframes"], {"unchanged": True})

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "requires artifact audit dependencies")
    def test_dense_schema_passes_artifact_audit(self):
        from kis.artifact_io import atomic_write_json
        from tools.audit_artifact_formats import audit_video_formats
        from kis.caption_schema import file_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data" / "L19" / "L19_V001" / "001.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image)
            (root / "mapping" / "L19").mkdir(parents=True)
            (root / "mapping" / "L19" / "L19_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0,30,0\n", encoding="utf-8")
            record = {
                "video_id": "L19_V001", "frame_id": 0, "keyframe_n": 1, "image_path": "data/L19/L19_V001/001.jpg",
                "caption": "A cook pours eggs into soup.", "details": "", "retrieval_text": "A cook pours eggs into soup.",
                "vqa_answers": {}, "caption_generation": light_caption_generation_config("fp32"),
                "image_sha256": file_sha256(image), "generation_seconds": 1.0, "generation_batch_size": 16,
            }
            atomic_write_json(root / "caption_generator" / "L19" / "L19_V001.json", [record])
            report = next(item for item in audit_video_formats(root, "L19_V001") if item["folder"] == "caption_generator")
            self.assertEqual(report["status"], "OK", report)


if __name__ == "__main__":
    unittest.main()
