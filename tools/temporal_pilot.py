"""Read-only temporal/baseline pilot with fixed queries, CPU and cached models.

No live planner, image fetch, VLM, index publication, or model downloads. The
default fixtures are UNLABELED: output supports manual review, not accuracy claims.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import CAPTION_CONFIG, VISUAL_CONFIG, load_runtime_config
from kis.paths import KIS_ROOT
from kis.query_plan import build_event_plan
from kis.temporal import align_events


def load_queries(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("queries"), list) or not data["queries"]:
        raise ValueError("Expected schema_version=1 and a nonempty query list")
    result, ids = [], set()
    for item in data["queries"]:
        query_id = item["query_id"]
        if (not isinstance(query_id, str) or not query_id or len(query_id) > 64
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in query_id)
                or query_id in ids):
            raise ValueError("Invalid or duplicate pilot query_id")
        ids.add(query_id)
        for field in ("raw_query", "english", "keywords"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Missing pilot field: {field}")
        actions = item.get("events")
        if not isinstance(actions, list) or any(not isinstance(action, str) or not action.strip() for action in actions):
            raise ValueError("Expected English action sentences in events")
        query = {"query_id": query_id, "raw_query": item["raw_query"],
                 "query_variants": [item["raw_query"], item["english"], item["keywords"]],
                 "stream_queries": {stream: [item["english"], item["keywords"]]
                                    for stream in ("visual", "bm25", "semantic")},
                 "entities": [], "events": [{"action": action} for action in actions],
                 "temporal_constraints": [{"type": "sequence", "order": actions}]}
        if not build_event_plan(query).active:
            raise ValueError(f"Pilot query is not a supported complete sequence: {query_id}")
        result.append(query)
    return result


def alignment_controls(frames, scores, supported, config):
    """Identical local evidence/pool: unordered, zero penalty, original, reverse."""
    result = {}
    for mode, matrix, mask, settings in (
        ("ordered", scores, supported, config),
        ("ordered_zero_penalty", scores, supported, replace(config, gap_penalty=0)),
        ("reversed_order", scores[::-1], supported[::-1], config),
    ):
        aligned = align_events(frames, matrix, mask, settings)
        result[mode] = None if aligned is None else {
            **{key: value for key, value in aligned.items() if key != "path"},
            "frames": [{"event_index": event, "frame_id": frames[index]["frame_id"],
                        "pts_time": frames[index]["pts_time"]} for event, index in aligned["path"]],
        }
    unordered = []
    for event in range(len(scores)):
        eligible = np.flatnonzero(supported[event])
        if eligible.size:
            index = int(eligible[np.argmax(scores[event, eligible])])
            unordered.append({"event_index": event, "frame_id": frames[index]["frame_id"],
                              "pts_time": frames[index]["pts_time"], "score": float(scores[event, index])})
    result["unordered"] = {"coverage": len(unordered) / len(scores), "frames": unordered}
    return result


class RecordingScorer:
    def __init__(self, scorer, config):
        self.scorer, self.config, self.controls = scorer, config, {}

    def score(self, video_id, actions, vectors, seeds):
        data = self.scorer.score(video_id, actions, vectors, seeds)
        self.controls[video_id] = alignment_controls(data[0], data[1], data[2], self.config)
        return data


def require_model_cache():
    from huggingface_hub import snapshot_download

    for repo_id, revision, weight_name in (
        (VISUAL_CONFIG["hf_repo"], VISUAL_CONFIG["hf_revision"], "open_clip_model.safetensors"),
        (CAPTION_CONFIG["model"], CAPTION_CONFIG["revision"], "model.safetensors"),
    ):
        try:
            path = Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
        except Exception as exc:
            raise RuntimeError(f"STOP: pinned model not cached: {repo_id}@{revision}; no download attempted") from exc
        if not (path / weight_name).is_file():
            raise RuntimeError(f"STOP: missing cached weights: {repo_id}/{weight_name}; no download attempted")


def run(query_path, output):
    # Set before importing transformers/open_clip/hub runtime. No GPU use.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    queries = load_queries(query_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier run, even if it was interrupted.
    with (output / "run.lock").open("x", encoding="utf-8") as file:
        file.write(datetime.now(timezone.utc).isoformat())
    try:
        require_model_cache()
        import faiss
        import torch
        from kis.config import load_caption_runtime
        from kis.event_retrieval import LocalEventScorer
        from kis.objects import ObjectMetadataLookup
        from kis.ranking import get_search_profile
        from kis.solve_kis import solve_kis
        from kis.text import BM25Retriever
        from kis.visual import SiglipEncoder, SiglipRetriever

        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        config = load_runtime_config()
        temporal = replace(config.temporal, enabled=True)
        profile = get_search_profile(config.retrieval_profile)
        with (output / "retrieval.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            # Check Milvus before paying the cost of loading cached CPU encoders.
            visual = SiglipRetriever(video_groups=config.video_groups, nprobe=profile.nprobe)
            try:
                collection = visual.client.describe_collection(collection_name=visual.collection_name, timeout=15)
                stats = visual.client.get_collection_stats(collection_name=visual.collection_name, timeout=15)
                started = perf_counter()
                captions, _ = load_caption_runtime(config)
                encoder = SiglipEncoder(device="cpu")
                bm25 = BM25Retriever(captions.caption_data, tokenizer_mode=profile.tokenizer_mode)
                lookup = ObjectMetadataLookup()
                context = {"mode": "unlabeled_read_only_cpu_pilot", "accuracy_metrics": None,
                           "planner": "fixed English events, no live API", "vlm": False, "model_downloads": False,
                           "video_groups": list(config.video_groups), "caption_vectors": int(captions.index.ntotal),
                           "collection_id": collection.get("collection_id"), "collection_rows": stats.get("row_count"),
                           "collection_metadata": collection.get("description"), "temporal_config": temporal.public(),
                           "retrieval_profile": profile.name, "model_load_seconds": perf_counter() - started,
                           "query_sha256": hashlib.sha256(Path(query_path).read_bytes()).hexdigest(),
                           "source_sha256": {name: hashlib.sha256((KIS_ROOT / name).read_bytes()).hexdigest() for name in (
                               "kis/solve_kis.py", "kis/temporal.py", "kis/event_retrieval.py", "kis/text.py",
                               "kis/ranking.py", "config/models.json", "config/pipeline.json", "tools/temporal_pilot.py")},
                           "budget_note": "Baseline: 100/stream/variant; temporal: event_top_k/stream/event + local rescore. Not equal cost or an accuracy experiment."}
                atomic_write_json(output / "context.json", context)
                records = []
                for query in queries:
                    record = {"query_id": query["query_id"], "structured_query": query, "modes": {}}
                    # Alternate mode order to disclose rather than systematically favor warm-cache temporal.
                    modes = (False, True) if len(records) % 2 == 0 else (True, False)
                    for enabled in modes:
                        diagnostics = {}
                        scorer = RecordingScorer(LocalEventScorer(encoder, captions, temporal), temporal) if enabled else None
                        started = perf_counter()
                        results = solve_kis(query, visual, encoder, bm25, captions, lookup,
                                            vlm_reranker=None, rerank_top_k=0, result_top_k=20,
                                            policy=profile.policy, temporal_config=replace(temporal, enabled=enabled),
                                            event_scorer=scorer, diagnostics=diagnostics)
                        record["modes"]["temporal" if enabled else "baseline"] = {
                            "elapsed_seconds": perf_counter() - started, "results": results,
                            "diagnostics": diagnostics, "same_pool_controls": scorer.controls if scorer else None}
                    atomic_write_json(output / "results" / f"{query['query_id']}.json", record)
                    records.append({"query_id": query["query_id"], **{
                        mode: {"seconds": values["elapsed_seconds"], "status": values["diagnostics"]["temporal"]["status"],
                               "top_video": values["results"][0]["video_id"] if values["results"] else None,
                               "result_count": len(values["results"]),
                               "top_coverage": values["results"][0].get("temporal", {}).get("coverage") if values["results"] else None,
                               "artifact_failures": values["diagnostics"]["temporal"].get("failures", {})}
                        for mode, values in record["modes"].items()}})
                    atomic_write_json(output / "summary.json", {"accuracy_metrics": None, "queries": records})
                    print(f"Completed {query['query_id']}", flush=True)
            finally:
                visual.client.close()
                LocalEventScorer._video_data.cache_clear()
        print(json.dumps({"output": str(output.resolve()), "queries": records}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=KIS_ROOT / "benchmarks/temporal_pilot.json")
    parser.add_argument("--output-dir", type=Path, default=KIS_ROOT / "benchmark_results" / ("temporal_pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    run(args.queries, args.output_dir)
