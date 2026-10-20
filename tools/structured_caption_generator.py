"""Versioned caption generation with batching and atomic per-image checkpoints."""

import gc
import json
from pathlib import Path
from time import perf_counter

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.caption_schema import file_sha256
from kis.caption_schema import load_frame_mapping, normalize_caption_record, rebuild_retrieval_text, validate_unique_caption_records
from kis.paths import CAPTION_DIR, DATA_DIR, KIS_ROOT, MAPPING_DIR, canonical_video_artifact_path, discover_video_image_dirs, resolve_video_artifact
from kis.structured_caption import CAPTION_PROMPT, caption_generation_config, parse_description, render_description


def resolve_caption_runtime(device="cuda", use_4bit=None, compute_dtype=None):
    import numpy  # Initialize Windows OpenMP before torch.
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen structured captioning requires CUDA; use the internvl backend for CPU")
    with torch.cuda.device(device):
        # T4/P100 must not use emulated BF16 or FP16 for this BF16 checkpoint.
        native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = compute_dtype or ("bf16" if native_bf16 else "fp32")
        if dtype not in {"bf16", "fp32"} or (dtype == "bf16" and not native_bf16):
            raise ValueError(f"Unsupported Qwen compute dtype {dtype} on this GPU")
        if use_4bit is None:
            total_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
            use_4bit = total_gib < (10 if dtype == "bf16" else 20)
    return {"use_4bit": bool(use_4bit), "compute_dtype": dtype}


def resolve_caption_backend_runtime(backend, device="cuda", use_4bit=None, compute_dtype=None):
    if backend == "qwen":
        return resolve_caption_runtime(device, use_4bit, compute_dtype)
    if backend == "internvl":
        import numpy
        import torch

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for InternVL captioning")
        native_bf16 = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = compute_dtype or ("bf16" if native_bf16 else "fp32")
        if dtype not in {"bf16", "fp32"} or (dtype == "bf16" and not native_bf16):
            raise ValueError("InternVL requires native BF16 or FP32 compute")
        if use_4bit and device.type != "cuda":
            raise ValueError("4-bit InternVL captioning requires CUDA")
        return {"use_4bit": bool(use_4bit), "compute_dtype": dtype}
    raise ValueError("Unknown caption runtime backend")


class FiniteCaptionLogits:
    def __call__(self, input_ids, scores):
        import torch

        # Negative infinity can be a legitimate token mask. NaN/+Inf and rows
        # without any finite choice cannot produce a meaningful next token.
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(dim=-1).all():
            raise FloatingPointError(
                "Caption model produced non-finite logits. Caption was not saved. "
                "Check the logged model/compute dtype/GPU."
            )
        return scores


class RepeatedCaptionTokenStop:
    def __init__(self, prompt_length, window=32):
        self.prompt_length, self.window = prompt_length, window

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[1] - self.prompt_length < self.window:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        tail = input_ids[:, -self.window:]
        return (tail == tail[:, :1]).all(dim=-1)


def parse_caption_response(response):
    compact = "".join(response.split())
    if not compact or (len(compact) >= 32 and len(set(compact)) <= 2):
        raise ValueError("Degenerate repetitive Qwen output (possible numerical instability)")
    return parse_description(response)


class QwenCaptioner:
    def __init__(self, device="cuda", use_4bit=None, compute_dtype=None):
        import numpy  # Initialize Windows OpenMP before torch.
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.device = torch.device(device)
        runtime = resolve_caption_runtime(device, use_4bit, compute_dtype)
        use_4bit = runtime["use_4bit"]
        dtype = torch.bfloat16 if runtime["compute_dtype"] == "bf16" else torch.float32
        self.config = caption_generation_config("qwen", use_4bit, runtime["compute_dtype"])
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"], revision=self.config["revision"],
            min_pixels=256 * 256, max_pixels=self.config["max_pixels"],
        )
        self.processor.tokenizer.padding_side = "left"
        kwargs = {"revision": self.config["revision"], "attn_implementation": "sdpa", "low_cpu_mem_usage": True,
                  "torch_dtype": dtype}
        if use_4bit:
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ), device_map={"": str(self.device)})
        print(f"[Caption V2] GPU={torch.cuda.get_device_name(self.device)}; "
              f"weights={self.config['precision']}; compute={self.config['compute_dtype']}; torch={torch.__version__}", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config["model"], **kwargs)
        if not use_4bit:
            self.model.to(self.device)
        self.model.eval()

    def describe(self, image):
        last_error = None
        for attempt in range(self.config["format_attempts"]):
            prompt = CAPTION_PROMPT
            if attempt:
                prompt += "\nFORMAT CORRECTION: " + str(last_error) + ". Use the exact field names and string/list types above."
            response = self._generate(image, prompt)
            try:
                return parse_caption_response(response)
            except (ValueError, TypeError) as exc:
                last_error = exc
        raise ValueError(f"Invalid Qwen caption after format retry: {last_error}. "
                         f"Compute={self.config.get('compute_dtype', 'unknown')}. Raw response (first 240 chars): {response[:240]}")

    def _generate(self, image, prompt):
        return self._generate_batch([image], [prompt])[0]

    def describe_batch(self, images):
        if not images:
            return []
        if len(images) == 1:
            return [self.describe(images[0])]
        responses = self._generate_batch(images, [CAPTION_PROMPT] * len(images))
        if len(responses) != len(images):
            raise ValueError("Qwen batch output count does not match image count")
        descriptions = []
        for image, response in zip(images, responses):
            try:
                descriptions.append(parse_caption_response(response))
            except (ValueError, TypeError):
                # Retry only the malformed item, preserving other batch outputs.
                descriptions.append(self.describe(image))
        return descriptions

    def _generate_batch(self, images, prompts):
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList

        texts = []
        for image, prompt in zip(images, prompts):
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        print(f"[Qwen batch] size={len(images)}", flush=True)
        output = None
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False, num_beams=1, use_cache=True,
                    remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteCaptionLogits()]),
                    stopping_criteria=StoppingCriteriaList([RepeatedCaptionTokenStop(inputs["input_ids"].shape[1])]),
                    pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                )
            return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            del inputs, output


def describe_image_batches(captioner, images):
    """Yield successful GPU sub-batches in input order, splitting only on OOM."""
    import numpy  # Initialize Windows OpenMP before torch.
    import torch

    started = perf_counter()
    retry_smaller = False
    try:
        if hasattr(captioner, "describe_batch"):
            descriptions = captioner.describe_batch(images)
        else:
            descriptions = [captioner.describe(image) for image in images]
    except torch.cuda.OutOfMemoryError:
        if len(images) == 1:
            raise
        retry_smaller = True
    # Exit the exception handler before retrying so the failed CUDA traceback
    # releases its tensors. Never convert numerical/JSON failures into OOMs.
    if retry_smaller:
        smaller = max(1, len(images) // 2)
        print(f"[Caption batch] CUDA OOM at size={len(images)}; retrying smaller batches <= {len(images) - smaller}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        yield from describe_image_batches(captioner, images[:smaller])
        yield from describe_image_batches(captioner, images[smaller:])
        return
    if len(descriptions) != len(images):
        raise ValueError("Caption batch output count does not match input count")
    yield descriptions, perf_counter() - started, len(images)


def generate_structured_captions_with_mapping(
    image_root=DATA_DIR, csv_root=MAPPING_DIR, output_dir=CAPTION_DIR,
    video_ids=None, batch_size=1, num_workers=0, device="cuda", path_root=KIS_ROOT,
    use_4bit=None, keyframes_by_video=None, captioner=None, compute_dtype=None, backend="qwen",
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if captioner is None:
        runtime = resolve_caption_backend_runtime(backend, device, use_4bit, compute_dtype)
        use_4bit, compute_dtype = runtime["use_4bit"], runtime["compute_dtype"]
    else:
        provided = getattr(captioner, "config", None)
        if isinstance(provided, dict) and "backend" in provided:
            actual_compute = provided.get("compute_dtype")
            actual_quantized = provided.get("precision") == "nf4"
            if (compute_dtype is not None and compute_dtype != actual_compute) or (use_4bit is not None and bool(use_4bit) != actual_quantized):
                raise ValueError("Requested caption runtime differs from the provided model")
            expected = caption_generation_config(backend, actual_quantized, actual_compute)
            if provided != expected:
                raise ValueError("Provided caption model/config does not match the selected backend")
            use_4bit, compute_dtype = actual_quantized, actual_compute
    config = caption_generation_config(backend, use_4bit, compute_dtype or "fp32")
    discovered = discover_video_image_dirs(image_root)
    requested = set(video_ids) if video_ids is not None else set(discovered)
    if requested - set(discovered):
        raise FileNotFoundError(f"Missing image directories: {sorted(requested - set(discovered))}")
    saved, generated = 0, 0
    effective_batch = batch_size
    for video_id in sorted(requested):
        mapping = load_frame_mapping(video_id, csv_root)
        destination = canonical_video_artifact_path(Path(output_dir), video_id, ".json")
        existing = resolve_video_artifact(Path(output_dir), video_id, ".json")
        records = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else []
        if existing.is_file() and existing.resolve() != destination.resolve():
            raise ValueError(f"Migrate legacy caption layout first: {existing}")
        if not isinstance(records, list):
            raise ValueError("Caption checkpoint must be an array")
        records = [normalize_caption_record(record, video_id, mapping) for record in records]
        validate_unique_caption_records(records, video_id)
        selected = set(keyframes_by_video[video_id]) if keyframes_by_video is not None else set(mapping)
        if selected - set(mapping):
            raise ValueError(f"Unknown selected keyframes for {video_id}")
        images = {
            int(path.stem): path for path in discovered[video_id].iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit()
        }
        if selected - set(images):
            raise FileNotFoundError(f"Missing selected images for {video_id}: {sorted(selected - set(images))}")
        by_n = {record["keyframe_n"]: record for record in records}
        if set(by_n) - set(images):
            raise FileNotFoundError(f"Checkpoint source images are missing for {video_id}")
        for record in records:
            n = record["keyframe_n"]
            if record.get("caption_generation") != config:
                raise ValueError(f"Caption generation changed or V1 captions found: {destination}. Use a separate output root.")
            if record.get("image_sha256") != file_sha256(images[n]):
                raise ValueError(f"Caption source image changed: {video_id}/{n}")
            if (Path(path_root) / record["image_path"]).resolve() != images[n].resolve():
                raise ValueError(f"Caption source image identity mismatch: {video_id}/{n}")
            if not record.get("retrieval_text") or (backend == "qwen" and "visual_description" not in record):
                raise ValueError(f"Incomplete structured checkpoint: {video_id}/{n}")
        pending = sorted(selected - set(by_n))
        if pending:
            print(f"[Caption] {video_id}: backend={backend}; pending={len(pending)}; requested batch={batch_size}", flush=True)
        offset = 0
        while offset < len(pending):
            if captioner is None:
                if backend == "internvl":
                    from kis.light_caption import InternVLCaptioner

                    captioner = InternVLCaptioner(device=device, use_4bit=use_4bit, compute_dtype=compute_dtype)
                else:
                    captioner = QwenCaptioner(device=device, use_4bit=use_4bit, compute_dtype=compute_dtype)
            batch_ns = pending[offset:offset + effective_batch]
            source_hashes = {n: file_sha256(images[n]) for n in batch_ns}
            loaded = []
            processed = 0
            completed_groups, largest_successful = 0, 0
            current_n = batch_ns[0]
            try:
                for n in batch_ns:
                    with Image.open(images[n]) as source:
                        loaded.append(source.convert("RGB"))
                try:
                    for descriptions, seconds, actual_batch in describe_image_batches(captioner, loaded):
                        group_ns = batch_ns[processed:processed + actual_batch]
                        for n, description in zip(group_ns, descriptions):
                            current_n = n
                            if backend == "internvl":
                                from kis.light_caption import validate_dense_caption

                                caption, details = validate_dense_caption(description), ""
                            else:
                                caption, details = render_description(description)
                            if file_sha256(images[n]) != source_hashes[n]:
                                raise ValueError(f"Caption source image changed during inference: {video_id}/{n}")
                            record = {
                                "video_id": video_id, "frame_id": mapping[n], "keyframe_n": n,
                                "image_path": images[n].resolve().relative_to(Path(path_root).resolve()).as_posix(),
                                "caption": caption, "details": details, "vqa_answers": {},
                                "caption_generation": config,
                                "image_sha256": source_hashes[n], "generation_seconds": seconds / actual_batch,
                                "generation_batch_size": actual_batch,
                            }
                            if backend == "qwen":
                                record["visual_description"] = description
                            record["retrieval_text"] = rebuild_retrieval_text(record)
                            by_n[n] = normalize_caption_record(record, video_id, mapping)
                            atomic_write_json(destination, [by_n[key] for key in sorted(by_n)])
                            generated += 1
                            print(f"[Caption] {video_id}/{n:03d}: {caption}", flush=True)
                        processed += actual_batch
                        completed_groups += 1
                        largest_successful = max(largest_successful, actual_batch)
                        if processed < len(batch_ns):
                            current_n = batch_ns[processed]
                        print(f"[Caption] {video_id}: completed batch={actual_batch} in {seconds:.1f}s", flush=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Caption failed for {video_id}/{current_n:03d} (compute={config['compute_dtype']}): {exc}"
                    ) from exc
            finally:
                for image in loaded:
                    image.close()
            if completed_groups > 1:
                effective_batch = min(effective_batch, largest_successful)
            offset += len(batch_ns)
        saved += len(by_n)
    return {"generated": generated, "records": saved, "backend": config["backend"]}
