"""Development-only, same-seed comparison of temporal video pooling/budgets.

Replay uses saved event hits, not fresh searches or any holdout outcomes.
Optional local rescoring uses cached CPU encoders and shared per-video results;
reported scoring costs are work estimates, not full-request latency benchmarks.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from kis.artifact_io import atomic_write_json
from kis.candidate_pool import select_candidate_videos
from kis.config import load_runtime_config
from kis.paths import KIS_ROOT
from kis.temporal import TemporalConfig, align_events
from tools.groundtruth_evaluation import known_path_metrics, load_labels


def load_dev_records(labels_path, source_run):
    labels = load_labels(labels_path)
    context = json.loads((Path(source_run) / "context.json").read_text(encoding="utf-8"))
    if context["label_sha256"] != hashlib.sha256(Path(labels_path).read_bytes()).hexdigest():
        raise ValueError("Saved seed run and frozen label hashes differ")
    records = []
    for query in labels["queries"]:
        if query["split"] != "dev" or query.get("task") != "sequence":
            continue
        path = Path(source_run) / f"{query['query_id']}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["split"] != "dev" or data["query"] != query:
            raise ValueError("Development record/label mismatch")
        records.append(data)
    if not records:
        raise ValueError("No development sequence queries")
    return records, context


def compact_alignment(video_id, frames, scores, supported, config):
    aligned = align_events(frames, scores, supported, config)
    if aligned is None:
        return None
    selected = dict(aligned["path"])
    events = []
    for event in range(len(scores)):
        index = selected.get(event)
        events.append({"status": "unknown", "keyframe_n": None, "frame_id": None, "pts_time": None} if index is None
                      else dict(frames[index], status="retrieved"))
    first = frames[aligned["path"][0][1]]
    return {"video_id": video_id, "frame_id": first["frame_id"], "keyframe_n": first["keyframe_n"],
            "score": aligned["score"], "temporal": {"events": events, "coverage": aligned["coverage"],
                                                       "status": aligned["status"], "verified": False}}


def run(labels_path, source_run, output, rescore=False):
    records, source_context = load_dev_records(labels_path, source_run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime_config()
    seed_config = TemporalConfig(**source_context["temporal_config"])
    cases = [(strategy, budget) for strategy in ("round_robin", "event_rrf", "balanced_rrf") for budget in (12, 24, 48)]
    context = {"split": "dev_only", "holdout_used_for_selection": False, "source_run": str(source_run),
               "source_context_sha256": hashlib.sha256((Path(source_run) / "context.json").read_bytes()).hexdigest(),
               "label_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
               "seed_depth": seed_config.event_top_k, "same_seed_lists": True,
               "thresholds_changed": False, "runtime_config_modified": False,
               "rescore": rescore, "cases": cases,
               "source_sha256": {name: hashlib.sha256((KIS_ROOT / name).read_bytes()).hexdigest() for name in (
                   "kis/candidate_pool.py", "kis/event_retrieval.py", "kis/temporal.py", "tools/candidate_pool_ablation.py")},
               "limitation": "Four small AI-labeled dev queries; no independent new holdout. Scoring reused across cases; time sums are estimates, not request latency."}
    atomic_write_json(output / "context.json", context)
    rows = []
    try:
        with (output / "ablation.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            captions = encoder = None
            if rescore:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from tools.temporal_pilot import require_model_cache
                require_model_cache()
                import faiss
                import torch
                from kis.config import load_caption_runtime
                from kis.event_retrieval import LocalEventScorer
                from kis.visual import SiglipEncoder
                torch.set_num_threads(4)
                faiss.omp_set_num_threads(4)
                captions, _ = load_caption_runtime(runtime)
                encoder = SiglipEncoder(device="cpu")
            for record in records:
                query = record["query"]
                hits = record["modes"]["temporal"]["diagnostics"]["temporal"]["events"]
                selections, seeds = {}, None
                for strategy, budget in cases:
                    started = perf_counter()
                    videos, seeds, diagnostics = select_candidate_videos(hits, budget, strategy, seed_config.candidate_rrf_k)
                    target_rank = next((item["global_rank"] for item in diagnostics["ranked_videos"] if item["video_id"] == query["video_id"]), None)
                    selections[f"{strategy}_{budget}"] = {"selected_videos": videos, "candidate_pool": diagnostics,
                        "target_in_pool": query["video_id"] in videos, "target_global_evidence_rank": target_rank,
                        "selection_seconds": perf_counter() - started}
                cache, failures = {}, {}
                if rescore:
                    actions = tuple(event["action"] for event in hits)
                    vectors = [encoder.encode_text(action) for action in actions]
                    scorer = LocalEventScorer(encoder, captions, seed_config)
                    union = list(dict.fromkeys(video for selection in selections.values() for video in selection["selected_videos"]))
                    for video_id in union:
                        started = perf_counter()
                        try:
                            frames, scores, supported, _, _, stats = scorer.score(video_id, actions, vectors, seeds[video_id])
                            aligned = compact_alignment(video_id, frames, scores, supported, seed_config)
                            cache[video_id] = {"result": aligned, "stats": stats, "seconds": perf_counter() - started}
                        except (OSError, ValueError, RuntimeError) as exc:
                            failures[video_id] = str(exc)
                    LocalEventScorer._video_data.cache_clear()
                for name, selection in selections.items():
                    row = {"query_id": query["query_id"], "case": name, "selected_count": len(selection["selected_videos"]),
                           "target_in_pool": selection["target_in_pool"], "target_global_evidence_rank": selection["target_global_evidence_rank"]}
                    if rescore:
                        videos = selection["selected_videos"]
                        results = sorted((cache[vid]["result"] for vid in videos if vid in cache and cache[vid]["result"] is not None),
                                         key=lambda result: (-result["score"], result["video_id"]))
                        selection.update(results=results, known_path_metrics=known_path_metrics(results, query))
                        row.update(top_target_video=bool(results and results[0]["video_id"] == query["video_id"]),
                                   known_path_hit_at_20=selection["known_path_metrics"]["known_path_hit_at_20"],
                                   scoring_seconds_sum=sum(cache[vid]["seconds"] for vid in videos if vid in cache),
                                   scored_frames=sum(cache[vid]["stats"]["scored_frames"] for vid in videos if vid in cache),
                                   artifact_failures=sum(vid in failures for vid in videos))
                    rows.append(row)
                atomic_write_json(output / f"{query['query_id']}.json", {"query_id": query["query_id"], "split": "dev",
                                  "selections": selections, "failures": failures})
            summary = []
            for strategy, budget in cases:
                name = f"{strategy}_{budget}"
                subset = [row for row in rows if row["case"] == name]
                summary.append({"case": name, "queries": len(subset), "target_in_pool": sum(row["target_in_pool"] for row in subset),
                                "mean_selected_videos": sum(row["selected_count"] for row in subset) / len(subset),
                                **({"top_target_video": sum(row["top_target_video"] for row in subset),
                                    "known_path_hits_at_20": sum(row["known_path_hit_at_20"] for row in subset),
                                    "scored_frames": sum(row["scored_frames"] for row in subset),
                                    "scoring_seconds_sum": sum(row["scoring_seconds_sum"] for row in subset),
                                    "artifact_failures": sum(row["artifact_failures"] for row in subset)} if rescore else {})})
            atomic_write_json(output / "summary.json", {"split": "dev", "cases": summary, "rows": rows})
        print(json.dumps({"output": str(output.resolve()), "cases": summary}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=KIS_ROOT / "benchmark_results" / ("candidate_pool_dev_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    run(args.labels, args.source_run, args.output_dir, args.rescore)
