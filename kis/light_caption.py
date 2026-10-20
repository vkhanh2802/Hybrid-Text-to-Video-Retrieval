"""InternVL3.5-1B-HF captioning with short retrieval-focused output."""

import hashlib
import re

from kis.config import FAST_CAPTION_CONFIG


INTERNVL_SYSTEM = "Describe only visible image content for visual search. Give only a concise English caption, no reasoning. Never transcribe television graphics, tickers, subtitles, channel names or timestamps."
INTERNVL_PROMPT = """Describe the visible action, objects, people's positions and clothing colors/patterns in 2-3 sentences, at most 50 words. Keep each person's attributes together. A displayed city skyline can be a studio backdrop. Read short clearly legible teaching-board text only, not TV overlays. Name ingredients only if certain; otherwise describe color/shape. Do not infer past or future events. /no_think"""


def light_caption_generation_config(compute_dtype="fp32", use_4bit=False):
    if compute_dtype not in {"bf16", "fp32"}:
        raise ValueError("InternVL compute dtype must be bf16 or fp32")
    return {
        "backend": "internvl_dense_v3", "schema_version": 3,
        **FAST_CAPTION_CONFIG,
        "prompt_sha256": hashlib.sha256((INTERNVL_SYSTEM + "\n" + INTERNVL_PROMPT).encode()).hexdigest(),
        "precision": "nf4" if use_4bit else compute_dtype, "compute_dtype": compute_dtype,
        "crop_to_patches": True, "thinking": False, "output_format": "dense_text",
        "render_version": 1, "inference_revision": 1,
    }


def validate_dense_caption(text):
    if not isinstance(text, str):
        raise ValueError("Dense caption must be a string")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) > 2400 or not any(character.isalpha() for character in text):
        raise ValueError("Empty, repetitive or oversized dense caption")
    if len(text) >= 32 and len(set(text)) <= 2:
        raise ValueError("Degenerate dense caption")
    if any(token in text for token in ("<pad>", "<s>", "</s>", "<think>", "</think>", "<|im_end|>")):
        raise ValueError("Special-token leakage in dense caption")
    return text


class InternVLCaptioner:
    def __init__(self, device="cuda", use_4bit=None, compute_dtype=None):
        import numpy
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, InternVLForConditionalGeneration
        from tools.structured_caption_generator import resolve_caption_backend_runtime

        self.device = torch.device(device)
        runtime = resolve_caption_backend_runtime("internvl", device, use_4bit, compute_dtype)
        compute_dtype, use_4bit = runtime["compute_dtype"], runtime["use_4bit"]
        self.config = light_caption_generation_config(compute_dtype, use_4bit)
        self.dtype = torch.bfloat16 if compute_dtype == "bf16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"], revision=self.config["revision"], trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "left"
        kwargs = {"revision": self.config["revision"], "torch_dtype": self.dtype,
                  "low_cpu_mem_usage": True, "trust_remote_code": False, "attn_implementation": "sdpa"}
        if use_4bit:
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            ), device_map={"": str(self.device)})
        print(f"[InternVL caption] model={self.config['model']}; compute={compute_dtype}; weights={self.config['precision']}", flush=True)
        self.model = InternVLForConditionalGeneration.from_pretrained(self.config["model"], **kwargs)
        if not use_4bit:
            self.model.to(self.device)
        self.model.eval()

    def describe(self, image):
        return self.describe_batch([image])[0]

    def describe_batch(self, images):
        if not images:
            return []
        import torch
        from transformers import LogitsProcessorList
        from transformers import StoppingCriteriaList
        from tools.structured_caption_generator import FiniteCaptionLogits, RepeatedCaptionTokenStop

        texts = []
        for image in images:
            messages = [
                {"role": "system", "content": INTERNVL_SYSTEM},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": INTERNVL_PROMPT}]},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True,
                                crop_to_patches=True, max_patches=self.config["max_patches"])
        inputs = {key: (value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        output = None
        print(f"[InternVL batch] size={len(images)}; max_patches={self.config['max_patches']}", flush=True)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=self.config["max_new_tokens"], num_beams=1,
                    do_sample=False, use_cache=True, remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteCaptionLogits()]),
                    stopping_criteria=StoppingCriteriaList([RepeatedCaptionTokenStop(inputs["input_ids"].shape[1])]),
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            generated = output[:, inputs["input_ids"].shape[1]:]
            think_id = self.processor.tokenizer.convert_tokens_to_ids("<think>")
            if isinstance(think_id, int) and think_id != self.processor.tokenizer.unk_token_id and (generated == think_id).any():
                raise ValueError("InternVL returned reasoning instead of a direct caption")
            texts = self.processor.batch_decode(generated, skip_special_tokens=True)
            if len(texts) != len(images):
                raise ValueError("InternVL batch output count does not match image count")
            return [validate_dense_caption(text) for text in texts]
        finally:
            del inputs, output
