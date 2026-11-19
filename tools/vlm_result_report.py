"""Describe returned fixed-protocol VLM replay; no search, model or tuning."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from kis.artifact_io import atomic_write_json
from tools.human_holdout import sha
from tools.pipeline_ablation import ARMS


def transitions(rows, metric):
    counts = Counter()
    examples = {name: [] for name in ("improved", "regressed", "unchanged")}
    for row in rows:
        delta = row["vlm_metrics"][metric] - row["baseline_metrics"][metric]
        key = "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "unchanged"
        counts[key] += 1
        examples[key].append(row["case_id"])
    return {"counts": {name: counts[name] for name in examples}, "case_ids": examples}


def timing_stats(values):
    values = sorted(values)
    if not values:
        return {"tasks": 0, "sum_seconds": 0, "median_seconds": None, "p95_seconds": None}
    return {"tasks": len(values), "sum_seconds": sum(values), "median_seconds": statistics.median(values),
            "p95_seconds": values[min(len(values)-1, int((len(values)-1)*0.95))]}


def write_report(source_run, packet_root, result_dir, replay_dir, output):
    source_run, packet_root, result_dir, replay_dir, output = map(Path, (source_run, packet_root, result_dir, replay_dir, output))
    if output.exists():
        raise FileExistsError("Report is versioned; choose a new path")
    context = json.loads((result_dir / "run_context.json").read_text(encoding="utf-8"))
    packet = json.loads((packet_root / "vlm_packet/manifest.json").read_text(encoding="utf-8"))
    bindings = json.loads((packet_root / "bindings_private.json").read_text(encoding="utf-8"))
    if sha(packet_root / "vlm_packet/manifest.json") != context["manifest_sha256"]:
        raise ValueError("Replay/result packet provenance mismatch")
    scores = {row["task_id"]: row for row in (
        json.loads(line) for line in (result_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines())}
    tasks = {task["task_id"]: task for task in packet["tasks"]}
    if set(scores) != set(tasks):
        raise ValueError("Full task set required for reporting")
    summary = json.loads((replay_dir / "summary.json").read_text(encoding="utf-8"))
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((source_run / "cases").glob("*.json"))]
    rows = []
    for case in cases:
        for mode in ("with_caption", "image_only"):
            row = json.loads((replay_dir / f"{case['case_id']}__{mode}.json").read_text(encoding="utf-8"))
            if row["baseline_metrics"] != case["metrics"]:
                raise ValueError("Replay baseline differs from frozen CPU case")
            rows.append(row)
    comparisons = {}
    for role in ("development", "observed_regression"):
        comparisons[role] = {}
        for task in ("image", "sequence"):
            comparisons[role][task] = {}
            metric = "known_frame_hit_at_1" if task == "image" else "known_path_hit_at_1"
            metric20 = "known_frame_hit_at_20" if task == "image" else "known_path_hit_at_20"
            for arm in ARMS:
                for mode in ("with_caption", "image_only"):
                    group = [row for row in rows if row["role"] == role and row["task"] == task
                             and row["arm"] == arm and row["caption_mode"] == mode]
                    count = len(group)
                    fields = (metric, "known_frame_hit_at_5" if task == "image" else "known_path_hit_at_5", metric20,
                              "mrr_at_20" if task == "image" else "top_target_video")
                    comparisons[role][task][arm + "__" + mode] = {
                        "queries": count, "baseline": {f: sum(row["baseline_metrics"][f] for row in group)/count if count else None for f in fields},
                        "with_vlm": {f: sum(row["vlm_metrics"][f] for row in group)/count if count else None for f in fields},
                        "paired_top1": transitions(group, metric), "paired_top20": transitions(group, metric20),
                        "queries_with_any_valid_vlm_score": sum(row["vlm_applied"] for row in group),
                        "queries_with_all_requested_scores_valid": sum(row["tasks_expected"] > 0 and row["tasks_scored"] == row["tasks_expected"] for row in group),
                        "requested_score_references": sum(row["tasks_expected"] for row in group),
                        "valid_score_references": sum(row["tasks_scored"] for row in group)}
    paired_rows = {}
    for role in ("development", "observed_regression"):
        paired_rows[role] = []
        for pair in bindings["pairs"]:
            if pair["role"] != role:
                continue
            modes = {mode: scores[task_id] for mode, task_id in pair["modes"].items()}
            valid = all(row["status"] == "scored" for row in modes.values())
            paired_rows[role].append({"query_id": pair["query_id"], "video_id": pair["video_id"],
                "keyframe_n": pair["keyframe_n"], "judgment": pair["judgment"], "paired_valid": valid,
                "scores": {mode: row["score"] for mode, row in modes.items()},
                "status": {mode: row["status"] for mode, row in modes.items()}})
    timing = {}
    for kind in ("image", "path"):
        timing[kind] = {mode: timing_stats([row["seconds"] for task_id, row in scores.items()
            if tasks[task_id]["kind"] == kind and tasks[task_id]["caption_mode"] == mode and row["status"] == "scored"])
                       for mode in ("with_caption", "image_only")}
    distribution = {kind: {mode: dict(sorted(Counter(str(row["score"]) for task_id, row in scores.items()
        if tasks[task_id]["kind"] == kind and tasks[task_id]["caption_mode"] == mode and row["status"] == "scored").items()))
                          for mode in ("with_caption", "image_only")} for kind in ("image", "path")}
    result = {"schema": "kis_vlm_return_analysis_v1", "posthoc_descriptive_only": True,
        "source_sha256": {str(path): sha(path) for path in (source_run / "protocol.json", packet_root / "bindings_private.json",
            result_dir / "scores.jsonl", result_dir / "run_context.json", replay_dir / "summary.json")},
        "report_tool_sha256": sha(Path(__file__)),
        "returned_status_counts": dict(Counter(row["status"] for row in scores.values())),
        "comparisons": comparisons, "pair_diagnostics": summary["pair_diagnostics"], "pair_rows": paired_rows,
        "timing_successful_inferences_only": timing, "score_distribution": distribution,
        "reported_model_load_seconds": json.loads((result_dir / "hardware.json").read_text(encoding="utf-8")).get("model_load_seconds"),
        "total_recorded_task_seconds": sum(row["seconds"] for row in scores.values()),
        "automatically_promoted": False, "models_rerun": False, "new_independent_holdout": False,
        "limits": ["Image/path scores are ordinal model judgments, not calibrated probabilities",
            "Incomplete relevance labels: outside-qrel results are not automatically false positives",
            "Human v3 is already observed regression, not untouched holdout",
            "Timing excludes startup/download and does not constitute end-to-end latency",
            "Invalid score can include parse failure or OOM returning None; raw model responses are not present in this ZIP"]}
    atomic_write_json(output, result)
    print(json.dumps({"report": str(output.resolve()), "status": result["returned_status_counts"],
                      "total_recorded_task_seconds": result["total_recorded_task_seconds"],
                      "pair_diagnostics": result["pair_diagnostics"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "packet-root", "result-dir", "replay-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    write_report(args.source_run, args.packet_root, args.result_dir, args.replay_dir, args.output)
