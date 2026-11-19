"""Frozen 20-image Qwen batch-size experiment, with strict (no split) OOM tests.

No DRES/Groq calls. Selects a fixed timing packet from a prior PUBLIC query's
top-100 output, using image availability, never score labels, before inference.
"""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SIZES = (1, 4, 8, 12, 20)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources():
    names = ["kis/vlm_reranker.py", "kis/ranking.py", "kis/solve_kis.py", "kis/web_app.py",
             "config/models.json", "tools/benchmark_qwen_batch_sizes.py"]
    return {name: sha(ROOT / name) for name in names}


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(output, response_path, worker_config):
    from PIL import Image
    from io import BytesIO
    from kis.remote_worker import RemoteWorker
    from kis.config import load_runtime_config, MODEL_MANIFEST, VLM_CONFIG
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    previous = json.loads(Path(response_path).read_text(encoding="utf-8"))
    response = previous["response"]
    config = load_runtime_config()
    settings = json.loads(Path(worker_config).read_text(encoding="utf-8"))
    client = RemoteWorker(settings["url"], settings["token"], config.video_groups)
    client.connect(config, MODEL_MANIFEST)
    rows, missing, seen = [], [], set()
    for candidate in response["results"]:
        vid, frame, number = candidate["video_id"], candidate["frame_id"], candidate["image_number"]
        identity = (vid, frame, number)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            payload, mime = client.read(vid, number)
        except FileNotFoundError:
            missing.append({"video_id": vid, "image_number": number})
            continue
        with Image.open(BytesIO(payload)) as image:
            image.load()
            width, height = image.size
        path = Path("images") / f"{vid}_{number:03d}{'.png' if mime == 'image/png' else '.jpg'}"
        with (output / path).open("xb") as stream:
            stream.write(payload)
        rows.append({"video_id": vid, "frame_id": frame, "image_number": number,
                     "path": path.as_posix(), "sha256": sha(output / path),
                     "caption": candidate["caption"], "width": width, "height": height})
        if len(rows) == 20:
            break
    if len(rows) != 20:
        raise RuntimeError("Could not freeze 20 distinct real images; no padding")
    packet = {"schema": "qwen_batch_timing_v1", "query": response["raw_query"], "images": rows,
              "model": VLM_CONFIG, "batch_sizes": SIZES, "timed_repetitions": 3,
              "model_load_cpu_threads": 1, "inference_cpu_threads": 4,
              "source_sha256": sources(), "selection_source_sha256": sha(response_path),
              "selection": "first 20 available distinct images from saved public-query top100; no score-label use",
              "unavailable_before_packet_complete": missing,
              "generation": {"min_pixels": 256 * 256, "max_pixels": 512 * 512, "max_new_tokens": 8,
                             "do_sample": False, "num_beams": 1, "quantization": "NF4 double quant", "padding_side": "left"},
              "notes": ["timing-only packet, no DRES content or submission",
                        "not a change to production top20 no-backfill policy",
                        "strict OOM, no splitting in batch-size performance measurements",
                        "same 20 original image bytes, prompts and captions for every size",
                        "batch-dependent numeric/output changes are reported, not called relevance gains"]}
    save(output / "PACKET.json", packet)
    print(json.dumps({"phase": "frozen", "images": 20, "batch_sizes": SIZES}), flush=True)


def run(output):
    output = Path(output)
    packet = json.loads((output / "PACKET.json").read_text(encoding="utf-8"))
    if (output / "REPORT.json").exists() or (output / "runs.jsonl").exists():
        raise FileExistsError("Existing experiment retained; choose a new output")
    if packet["source_sha256"] != sources():
        raise RuntimeError("Code changed after packet freeze")
    for image in packet["images"]:
        if sha(output / image["path"]) != image["sha256"]:
            raise RuntimeError("Frozen image bytes changed")
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy
    import torch
    from kis.vlm_reranker import VLMReranker
    from kis.config import VLM_CONFIG
    if packet["model"] != VLM_CONFIG or not torch.cuda.is_available():
        raise RuntimeError("Pinned model / CUDA prerequisite failed")
    # Isolate a Windows native loader crash: serialize CPU weight preparation.
    # Restore the original four inference threads BEFORE any warm-up/timing.
    torch.set_num_threads(packet.get("model_load_cpu_threads", 1))
    torch.set_num_interop_threads(1)
    initial_free, total = torch.cuda.mem_get_info()
    import psutil
    load_context = {"gpu_initial_free_bytes": initial_free, "gpu_total_bytes": total,
                    "ram_available_bytes": psutil.virtual_memory().available,
                    "model_load_cpu_threads": torch.get_num_threads(),
                    "inference_cpu_threads": packet.get("inference_cpu_threads", 4)}
    save(output / "LOAD_CONTEXT.json", load_context)
    start = perf_counter()
    scorer = VLMReranker()
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    torch.set_num_threads(packet.get("inference_cpu_threads", 4))
    images = [output / image["path"] for image in packet["images"]]
    captions = [image["caption"] for image in packet["images"]]
    runs, warmups, unsupported = [], [], set()

    def measure(size, phase, repetition=0):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        values, error = None, None
        try:
            values = scorer.score_batch(images, packet["query"], captions, batch_size=size, oom_fallback=False)
            torch.cuda.synchronize()
            status = "ok" if len(values) == 20 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 10 for v in values) else "invalid_scores"
        except torch.cuda.OutOfMemoryError:
            status, error = "oom", "CUDA out of memory; no hidden split"
        except Exception as exc:
            status, error = "error", type(exc).__name__ + ": " + str(exc)[:300]
        elapsed = perf_counter() - start
        row = {"phase": phase, "repetition": repetition, "batch_size": size, "status": status,
               "seconds": elapsed, "scores": values, "error": error,
               "valid_scores": sum(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 10 for v in values) if values else 0,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "batch_diagnostics": deepcopy(scorer.last_batch_stats)}
        with (output / "runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({k: row[k] for k in ("phase", "repetition", "batch_size", "status", "seconds", "valid_scores", "peak_allocated_bytes")}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return row

    # Every size gets its own unmeasured warm-up on the SAME 20 images.
    for size in SIZES:
        row = measure(size, "warmup")
        warmups.append(row)
        if row["status"] in {"oom", "error"}:
            unsupported.add(size)
    orders = [SIZES, tuple(reversed(SIZES)), SIZES[2:] + SIZES[:2]]
    for repetition, order in enumerate(orders, 1):
        for size in order:
            if size in unsupported:
                continue
            row = measure(size, "timed", repetition)
            runs.append(row)
            if row["status"] in {"oom", "error"}:
                unsupported.add(size)
    reference = next((r["scores"] for r in runs if r["batch_size"] == 1 and r["status"] == "ok"), None)
    summaries = []
    for size in SIZES:
        subset = [r for r in runs if r["batch_size"] == size]
        complete = [r for r in subset if r["status"] in {"ok", "invalid_scores"}]
        valid = len(subset) == 3 and all(r["status"] == "ok" for r in subset)
        agreement = [sum(a == b for a, b in zip(reference, r["scores"])) / 20 for r in complete] if reference else []
        summaries.append({"batch_size": size, "eligible_all_3_runs_valid": valid,
                          "timed_runs": len(subset), "successful_valid_runs": sum(r["status"] == "ok" for r in subset),
                          "unsupported": size in unsupported,
                          "median_seconds": statistics.median(r["seconds"] for r in complete) if complete else None,
                          "seconds": [r["seconds"] for r in subset],
                          "valid_scores": [r["valid_scores"] for r in subset],
                          "peak_allocated_bytes": max((r["peak_allocated_bytes"] for r in subset), default=None),
                          "peak_reserved_bytes": max((r["peak_reserved_bytes"] for r in subset), default=None),
                          "mean_exact_score_agreement_with_scalar": statistics.mean(agreement) if agreement else None})
    eligible = [r for r in summaries if r["eligible_all_3_runs_valid"]]
    best = min(eligible, key=lambda r: r["median_seconds"]) if eligible else None
    report = {"status": "completed" if best else "no_fully_valid_configuration", "model_load_seconds": load_seconds,
              "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": total, "gpu_initial_free_bytes": initial_free,
               "torch": torch.__version__, "images": 20, "warmups": warmups, "runs": runs,
              "load_context": load_context,
              "summaries": summaries, "fastest_valid_batch_size": best["batch_size"] if best else None,
              "source_unchanged": packet["source_sha256"] == sources(),
              "notes": packet["notes"] + ["timing excludes image network download, retrieval and Groq",
                                           "score equality is not ground-truth accuracy", "only one query and 20 images, no universal optimum claim"]}
    save(output / "REPORT.json", report)
    print(json.dumps({"status": report["status"], "summaries": summaries, "fastest_valid_batch_size": report["fastest_valid_batch_size"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--response", type=Path)
    parser.add_argument("--worker-config", type=Path, default=ROOT / "config/school-runtime.local.json")
    args = parser.parse_args()
    if args.prepare_only:
        if not args.response:
            parser.error("prepare requires a saved PUBLIC-query response")
        prepare(args.output, args.response, args.worker_config)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
