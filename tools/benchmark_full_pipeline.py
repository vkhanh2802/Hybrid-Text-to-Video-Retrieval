"""Real production text-search timing through a private loopback HTTP server.

No replayed planner, fake scorer, changed dataset scope or DRES operation. Timing
wrappers are process-local and delegate to the original production methods.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch


QUERIES = (
    "Hai người ngồi trên thuyền ở một con sông, người đàn ông mặc áo xanh dương đội mũ trắng và người phụ nữ mặc áo hồng đội mũ xanh lá cây.",
    "Một người đang nấu ăn trong nhà bếp, đứng trước bếp và dùng dụng cụ để đảo thức ăn trong nồi.",
    "Một người đội mũ bảo hiểm đang lái xe máy trên đường phố có nhiều phương tiện qua lại.",
)


def result_checks(response, score_rows):
    frames = response.get("results", [])
    distinct = len({(r["video_id"], r["frame_id"]) for r in frames})
    calls = sum(row.get("batch_size", 1) for row in score_rows)
    valid = sum(row.get("valid_scores", int(row.get("valid_score", False))) for row in score_rows)
    return {"frames": len(frames), "distinct_frames": distinct,
            "qwen_calls": calls, "qwen_valid_scores": valid,
            "all_qwen_calls_valid": bool(score_rows) and valid == calls,
            "result_vlm_status_counts": {status: sum(r.get("vlm_status") == status for r in frames)
                                         for status in sorted({r.get("vlm_status", "unknown") for r in frames})},
            "complete_100": len(frames) == distinct == 100,
            "qwen_executed": valid > 0,
            "qwen_budget_ok": calls <= 20}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    setup_started = perf_counter()
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    # Preserve production VLM enablement; refuse an inherited disabled benchmark.
    if os.environ.get("KIS_DISABLE_VLM", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("VLM disabled in environment; no full-pipeline timing claim")
    import numpy  # Existing Windows environment requires NumPy before Torch.
    import torch
    import uvicorn
    import psutil
    from kis import web_app
    from kis.vlm_reranker import VLMReranker
    from kis.llm import load_query_api_config

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen benchmark requires the configured CUDA GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats()
    service = web_app.service
    api = load_query_api_config()
    api.require_ready()
    if api.provider != "groq":
        raise RuntimeError("Expected the existing Groq production planner")
    root = Path(__file__).resolve().parents[1]
    files = [root / "config/models.json", root / "config/pipeline.json",
             root / "config/object_labels.json", *sorted((root / "kis").glob("*.py"))]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol = {"entry": "POST /api/search over private loopback HTTP", "queries": QUERIES,
                "schedule": [0, 0, 1, 2],
                "first_run": ("cold laptop/Qwen, resident school worker already ready" if service.remote_worker is not None
                              else "cold process/model, existing OS cache"),
                "later_runs": "same process/models; metadata may be warm, no cached image bytes",
                "query_api": api.public(), "video_groups": service.video_groups,
                "image_source": (getattr(service.image_source, "source_name", "drive_api") if service.image_source else "local_files"),
                "retrieval_backend": "school_worker" if service.remote_worker is not None else "local_milvus",
                "vlm_budget": 20, "expected_results": 100,
                "temporal": service.runtime_config.temporal.public(),
                "ocr": service.runtime_config.ocr.public(),
                "text_reranker": service.runtime_config.text_reranker.public(),
                "source_sha256": hashes,
                "gpu": torch.cuda.get_device_name(0),
                "ram_available_before": psutil.virtual_memory().available,
                "notes": ["real planner/model/image calls; no replay or mock outputs",
                          "no DRES connection, hint fetch, confirmation or submission",
                          "text search only; no supplied video task",
                          "optional OCR/cross-encoder follow production off settings",
                          "response timing excludes browser image-grid rendering",
                          "backend and image source explicitly recorded; no silent fallback"]}
    (output / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    measurements = {"qwen_load": [], "qwen_score": [], "image_read": []}
    original_init, original_batch = VLMReranker.__init__, VLMReranker.score_batch

    def timed_init(instance, *args, **kwargs):
        start = perf_counter()
        success = False
        try:
            result = original_init(instance, *args, **kwargs)
            torch.cuda.synchronize()
            success = True
            return result
        finally:
            measurements["qwen_load"].append({"seconds": perf_counter() - start, "success": success})

    def timed_batch(instance, image_paths, *args, **kwargs):
        torch.cuda.synchronize()
        start = perf_counter()
        value = None
        success = False
        try:
            value = original_batch(instance, image_paths, *args, **kwargs)
            torch.cuda.synchronize()
            success = True
            return value
        finally:
            measurements["qwen_score"].append({"seconds": perf_counter() - start,
                                               "returned": success,
                                               "batch_size": len(image_paths),
                                               "valid_scores": sum(isinstance(v, (float, int)) and 0 <= v <= 10 for v in value) if isinstance(value, list) else 0,
                                               "batch_diagnostics": instance.last_batch_stats})

    original_read = service.image_source.read if service.image_source else None

    def timed_read(video_id, number):
        start = perf_counter()
        success = False
        byte_count = 0
        try:
            value = original_read(video_id, number)
            byte_count = len(value[0])
            success = True
            return value
        finally:
            measurements["image_read"].append({"seconds": perf_counter() - start, "success": success,
                                               "bytes": byte_count, "video_id": video_id, "image_number": number})

    ready = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            try:
                await super().startup(sockets=sockets)
            finally:
                ready.set()

    config = uvicorn.Config(web_app.app, host="127.0.0.1", port=0, log_level="warning", access_log=False)
    sock = config.bind_socket()
    port = sock.getsockname()[1]
    server = Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    rows = []
    try:
        with patch.object(VLMReranker, "__init__", timed_init), patch.object(VLMReranker, "score_batch", timed_batch):
            if service.image_source:
                service.image_source.read = timed_read
            thread.start()
            if not ready.wait(timeout=60) or not server.started:
                raise RuntimeError("Private benchmark HTTP server did not start")
            setup_seconds = perf_counter() - setup_started
            for run_index, query_index in enumerate(protocol["schedule"]):
                for values in measurements.values():
                    values.clear()
                before_cpu = psutil.Process().cpu_times()
                request = Request(f"http://127.0.0.1:{port}/api/search",
                                  data=json.dumps({"query": QUERIES[query_index]}).encode("utf-8"),
                                  headers={"Content-Type": "application/json"}, method="POST")
                start = perf_counter()
                try:
                    with urlopen(request, timeout=1200) as response:
                        code = response.status
                        payload = json.loads(response.read())
                except HTTPError as error:
                    code = error.code
                    payload = json.loads(error.read())
                elapsed = perf_counter() - start
                after_cpu = psutil.Process().cpu_times()
                checks = result_checks(payload, measurements["qwen_score"]) if code == 200 else {}
                row = {"run": run_index + 1, "kind": "cold" if run_index == 0 else "warm",
                       "query_index": query_index, "http_status": code, "client_seconds": elapsed,
                       "backend_seconds": payload.get("elapsed_seconds"),
                       "stage_seconds": payload.get("stage_seconds"), "checks": checks,
                       "qwen_load": list(measurements["qwen_load"]),
                       "qwen_scores": list(measurements["qwen_score"]),
                       "image_reads": list(measurements["image_read"]),
                       "process_cpu_seconds": (after_cpu.user + after_cpu.system) - (before_cpu.user + before_cpu.system),
                       "process_rss_bytes": psutil.Process().memory_info().rss,
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "response": payload}
                rows.append(row)
                (output / f"run_{run_index + 1:02d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({k: row[k] for k in ("run", "kind", "http_status", "client_seconds", "checks", "stage_seconds")}), flush=True)
                if code != 200 or not all(checks.get(k) for k in ("complete_100", "qwen_executed", "qwen_budget_ok")):
                    print("Stopping: no successful full-pipeline timing claim for a degraded run.", flush=True)
                    break
    finally:
        if service.image_source:
            service.image_source.read = original_read
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=30)
        sock.close()
        unchanged = all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        successful = len(rows) == len(protocol["schedule"]) and all(
            row["http_status"] == 200 and all(row["checks"].get(k) for k in
                ("complete_100", "qwen_executed", "qwen_budget_ok")) for row in rows)
        degraded = any(not row["checks"].get("all_qwen_calls_valid", False)
                       or any(not image["success"] for image in row["image_reads"]) for row in rows)
        status = "partial_or_failed" if not successful or not unchanged else "completed_with_degradation" if degraded else "passed"
        report = {"status": status,
                  "setup_seconds": locals().get("setup_seconds"), "source_unchanged": unchanged,
                  "private_server_stopped": not thread.is_alive(),
                  "runs": [{k: v for k, v in row.items() if k != "response"} for row in rows],
                  "limitations": protocol["notes"], "production_state": service.state,
                  "worker": service.remote_worker.info if service.remote_worker is not None else None,
                  "qwen_model_loaded": bool(service.vlm_reranker and service.vlm_reranker.model)}
        warm = [r["client_seconds"] for r in rows if r["kind"] == "warm"]
        if warm:
            report["warm_seconds"] = {"count": len(warm), "median": statistics.median(warm),
                                      "min": min(warm), "max": max(warm), "mean": statistics.mean(warm)}
        (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "warm_seconds": report.get("warm_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
