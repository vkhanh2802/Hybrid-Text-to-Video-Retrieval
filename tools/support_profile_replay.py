"""Replay frozen support thresholds against old dev sequences with fixed pools.

Regression diagnostic only: the old four dev queries are not a new holdout.
Gate calibration is kept separate from candidate selection and scoring weights.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from kis.artifact_io import atomic_write_json
from kis.config import load_caption_runtime, load_runtime_config
from kis.event_retrieval import LocalEventScorer
from kis.temporal import TemporalConfig, evidence_scores
from tools.candidate_pool_ablation import compact_alignment, load_dev_records
from tools.groundtruth_evaluation import known_path_metrics


def run(labels_path, source_run, calibration_path, output):
    records, source_context = load_dev_records(labels_path, source_run)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration.get("split") != "dev" or not calibration.get("selected"):
        raise ValueError("Expected a frozen development-selected support profile")
    baseline = TemporalConfig(**source_context["temporal_config"])
    selected = calibration["selected"]
    experiment = replace(baseline, min_visual_similarity=selected["visual_threshold"],
                         min_semantic_similarity=selected["semantic_threshold"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from tools.temporal_pilot import require_model_cache
        require_model_cache()
        import faiss
        import torch
        from kis.visual import SiglipEncoder
        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        context = {"mode": "old_dev_regression_not_independent_holdout", "same_saved_pools_and_seed_frames": True,
                   "candidate_pooling_modified": False, "runtime_thresholds_modified": False,
                   "calibration_sha256": hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
                   "baseline": baseline.public(), "experiment": experiment.public(),
                   "source_run": str(source_run), "label_sha256": source_context["label_sha256"]}
        atomic_write_json(output / "context.json", context)
        rows = []
        with (output / "replay.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            captions, _ = load_caption_runtime(load_runtime_config())
            encoder = SiglipEncoder(device="cpu")
            for record in records:
                query = record["query"]
                old = record["modes"]["temporal"]
                diagnostic = old["diagnostics"]["temporal"]
                actions = [event["action"] for event in diagnostic["events"]]
                vectors = [encoder.encode_text(action) for action in actions]
                seeds = {}
                for event in diagnostic["events"]:
                    for hits in event["streams"].values():
                        for hit in hits:
                            seeds.setdefault(hit["video_id"], set()).add(int(hit["frame_id"]))
                scorer = LocalEventScorer(encoder, captions, baseline)
                results = {"baseline": [], "experimental": []}
                failures, frame_count = {}, 0
                started = perf_counter()
                for video_id in diagnostic["candidate_videos"]:
                    try:
                        frames, _, _, visual, semantic, stats = scorer.score(video_id, actions, vectors, seeds[video_id])
                        frame_count += stats["scored_frames"]
                        for name, config in (("baseline", baseline), ("experimental", experiment)):
                            scores, supported = evidence_scores(visual, semantic, (len(actions), len(frames)), config)
                            result = compact_alignment(video_id, frames, scores, supported, config)
                            if result is not None:
                                results[name].append(result)
                    except (OSError, ValueError, RuntimeError) as exc:
                        failures[video_id] = str(exc)
                for values in results.values():
                    values.sort(key=lambda item: (-item["score"], item["video_id"]))
                row = {"query_id": query["query_id"], "scored_frames": frame_count,
                       "seconds_shared_scoring": perf_counter() - started, "failures": failures, "modes": {}}
                for name, values in results.items():
                    row["modes"][name] = {"top_video": values[0]["video_id"] if values else None,
                        "top_target_video": bool(values and values[0]["video_id"] == query["video_id"]),
                        "full_paths_in_pool": sum(item["temporal"]["coverage"] == 1 for item in values),
                        "known_path_metrics": known_path_metrics(values, query), "results": values}
                row["baseline_matches_source_top"] = row["modes"]["baseline"]["top_video"] == (old["results"][0]["video_id"] if old["results"] else None)
                rows.append(row)
                atomic_write_json(output / f"{query['query_id']}.json", row)
                LocalEventScorer._video_data.cache_clear()
        summary = {"queries": len(rows), "baseline_matches_source_top": sum(row["baseline_matches_source_top"] for row in rows),
                   "modes": {name: {"top_target_video": sum(row["modes"][name]["top_target_video"] for row in rows),
                                     "full_paths_in_pool": sum(row["modes"][name]["full_paths_in_pool"] for row in rows),
                                     "known_path_hits_at_20": sum(row["modes"][name]["known_path_metrics"]["known_path_hit_at_20"] for row in rows)}
                             for name in ("baseline", "experimental")}}
        atomic_write_json(output / "summary.json", summary)
        print(json.dumps({"output": str(output.resolve()), **summary}, indent=2))
    finally:
        LocalEventScorer._video_data.cache_clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.labels, args.source_run, args.calibration, args.output_dir)
