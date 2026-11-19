"""Frozen, cache-only multipath replay. No model, retrieval or network calls."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.paths import KIS_ROOT
from kis.temporal import TemporalConfig, align_events, evidence_scores
from tools.candidate_pool_ablation import load_dev_records
from tools.groundtruth_evaluation import known_path_metrics
from tools.human_holdout import sha
from tools.human_holdout_evaluation import load_frozen_labels, sequence_queries
from tools.temporal_multipath import align_multiple, candidate_mask, compact_path


SETTINGS = {"event_frame_budget": 32, "search_paths": 32, "max_paths": 4}
ARMS = ("single_12", "top4_equal12", "diverse4_equal12", "top4_expanded48", "diverse4_expanded48")
SOURCES = ("tools/temporal_multipath.py", "tools/temporal_multipath_experiment.py",
           "kis/temporal.py", "tools/candidate_pool_ablation.py", "tools/groundtruth_evaluation.py",
           "tools/human_holdout_evaluation.py", "tools/human_holdout.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hashes(hashes):
    for path, fingerprint in hashes.items():
        if sha(path) != fingerprint:
            raise ValueError(f"Frozen input changed: {path}")


def prepare(role, output, development_run=None):
    output = Path(output)
    hashes, cases = {}, []

    def freeze(path):
        path = Path(path).resolve()
        hashes[str(path)] = sha(path)
        return str(path)

    if role == "development":
        source = KIS_ROOT / "benchmark_results/groundtruth_eval_20261102_002150"
        labels = KIS_ROOT / "benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json"
        records, context = load_dev_records(labels, source)
        evidence = KIS_ROOT / "benchmark_results/condition_evidence_v2_dev_20261102"
        report = read_json(evidence / "report.json")
        if (report["sequence_label_sha256"] != sha(labels)
                or report["sequence_source_context_sha256"] != sha(source / "context.json")):
            raise ValueError("Development evidence provenance differs")
        freeze(labels)
        freeze(source / "context.json")
        freeze(evidence / "report.json")
        for record in records:
            query = record["query"]
            diag = record["modes"]["temporal"]["diagnostics"]["temporal"]
            actions = [event["action"] for event in diag["events"]]
            cases.append({"query": query, "actions": actions, "candidate_videos": diag["candidate_videos"],
                "record": freeze(source / (query["query_id"] + ".json")),
                "baseline_record": freeze(evidence / (query["query_id"] + ".json")),
                "evidence": {vid: freeze(evidence / f"{query['query_id']}_{vid}_evidence.json")
                             for vid in diag["candidate_videos"]}})
    elif role == "observed_regression":
        if development_run is None:
            raise ValueError("Regression requires a completed development replay")
        development_run = Path(development_run)
        dev = read_json(development_run / "protocol.json")
        summary = read_json(development_run / "summary.json")
        if (dev["role"] != "development" or dev["settings"] != SETTINGS
                or summary["protocol_sha256"] != sha(development_run / "protocol.json")
                or summary["decision"] != "keep_production_unchanged"):
            raise ValueError("Development protocol/decision mismatch")
        verify_hashes(dev["input_sha256"])
        verify_hashes(dev["source_sha256"])
        freeze(development_run / "protocol.json")
        freeze(development_run / "summary.json")
        packet = KIS_ROOT / "benchmark_results/human_holdout_v3_seed2026110203_v2"
        labels_path = packet / "frozen_human_labels_v1.json"
        labels, _ = load_frozen_labels(packet, labels_path)
        source = KIS_ROOT / "benchmark_results/human_holdout_v3_evaluation_20261102"
        context = read_json(source / "context.json")
        if context["label_sha256"] != sha(labels_path):
            raise ValueError("Human evidence and frozen labels differ")
        freeze(labels_path)
        freeze(packet / "human_annotations.json")
        freeze(source / "context.json")
        for query in sequence_queries(labels):
            record_path = source / (query["query_id"] + ".json")
            record = read_json(record_path)
            if record["query"] != query:
                raise ValueError("Regression query differs from frozen human labels")
            cases.append({"query": query, "actions": query["actions"],
                "candidate_videos": record["candidate_videos"], "record": freeze(record_path),
                "baseline_record": freeze(record_path),
                "evidence": {vid: freeze(source / "raw_evidence" / f"{query['query_id']}_{vid}.json")
                             for vid in record["candidate_videos"] if vid not in record["failures"]}})
    else:
        raise ValueError("Unknown experiment role")
    config = TemporalConfig(**context["temporal_config"])
    if (config.condition_evidence != "off" or config.max_videos != 12 or config.max_frames_per_video != 1500
            or config.min_visual_similarity != .25 or config.min_semantic_similarity != .45):
        raise ValueError("Expected unchanged production gates/budgets")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema": "kis_temporal_multipath_v1", "role": role, "settings": SETTINGS,
        "temporal_config": config.public(), "arms": ARMS, "cases": cases,
        "input_sha256": hashes, "source_sha256": {str(KIS_ROOT / name): sha(KIS_ROOT / name) for name in SOURCES},
        "model_runs": False, "new_retrieval": False, "llm_planner_measured": False,
        "production_modified": False, "decision_rule": "keep_production_unchanged_even_if_small_sample_improves",
        "selection": "label-blind top32 supported frames/event + pinned baseline, top32 DAG paths; top4 or greedy <=50% overlap then score fill",
        "ranking": "unchanged alignment score, video_id, pinned path index; no VLM score reuse",
        "budget_control": "same cached frame matrices and <=12 videos; equal12 limits total output paths, expanded48 increases path budget",
        "limitations": ["Bounded DAG not all paths; diversity only within fixed search set",
            "Sparse image qrels; unjudged predictions are not negatives",
            "Development labels are AI-observed; human-v3 is observed regression, not independent validation",
            "Fixed event plans; no planner or full end-to-end latency measured"]}
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "source_snapshot.json", {name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in SOURCES})
    print(json.dumps({"prepared": str(output), "role": role, "queries": len(cases),
                      "protocol_sha256": sha(output / "protocol.json")}, indent=2))


def ordered(results):
    return sorted(results, key=lambda row: (-row["score"], row["video_id"], row["path_index"]))


def signature(row):
    return row["video_id"], tuple((e["status"], e.get("keyframe_n")) for e in row["temporal"]["events"])


def path_feasibility(frames, scores, support, query, config, baseline):
    """Diagnostic only AFTER label-blind paths have been generated."""
    known = np.array([[frame["keyframe_n"] in allowed for frame in frames]
                      for allowed in query["event_relevant_keyframes"]], dtype=bool)
    presence = known.any(axis=1).tolist()
    supported = (known & support).any(axis=1).tolist()
    selected = candidate_mask(scores, support, baseline, SETTINGS["event_frame_budget"])

    def complete(mask):
        result = align_events(frames, scores, mask, config)
        return result is not None and result["coverage"] == 1

    return {"known_frame_present_by_event": presence, "known_frame_supported_by_event": supported,
        "known_ordered_path_in_cached_frames": complete(known),
        "known_ordered_supported_path": complete(known & support),
        "known_ordered_supported_path_in_bounded_dag": complete(known & selected)}


def replay_case(case, config, role):
    query, actions = case["query"], case["actions"]
    paths, stats, target = {"single": [], "top4": [], "diverse4": []}, {}, None
    for vid in case["candidate_videos"]:
        if vid not in case["evidence"]:
            continue  # Source scoring failure, never silently regenerated.
        evidence = read_json(case["evidence"][vid])
        frames = evidence["frames"]
        if (evidence["split"] != ("dev" if role == "development" else "holdout")
                or evidence["actions"] != actions or any(f["video_id"] != vid for f in frames)):
            raise ValueError("Cached evidence identity mismatch")
        visual, semantic = (None if evidence[name] is None else np.asarray(evidence[name], dtype=float)
                            for name in ("visual", "semantic"))
        scores, support = evidence_scores(visual, semantic, (len(actions), len(frames)), config)
        baseline = align_events(frames, scores, support, config)
        stats[vid] = {"scored_frames": len(frames), "supported_nodes": int(support.sum())}
        for name, diverse in (("top4", False), ("diverse4", True)):
            started = perf_counter()
            aligned, work = align_multiple(frames, scores, support, config, diverse=diverse, **SETTINGS)
            stats[vid][name] = {**work, "cpu_replay_seconds": perf_counter() - started}
            if aligned[:1] != ([baseline] if baseline else []):
                raise ValueError("Pinned path differs from production")
            for index, result in enumerate(aligned):
                row = {**compact_path(vid, frames, result, len(actions)), "path_index": index}
                paths[name].append(row)
                if name == "top4" and index == 0:
                    paths["single"].append(row)
        if vid == query["video_id"]:
            target = path_feasibility(frames, scores, support, query, config, baseline)
    baseline_saved = read_json(case["baseline_record"])["modes"]["baseline"]["results"]
    single = ordered(paths["single"])
    if ([signature(r) for r in single] != [signature(r) for r in baseline_saved]
            or any(abs(a["score"]-b["score"]) > 1e-12 for a, b in zip(single, baseline_saved))):
        raise ValueError("CPU baseline differs from saved original baseline")
    arms = {}
    for arm in ARMS:
        method = "single" if arm == "single_12" else ("diverse4" if arm.startswith("diverse") else "top4")
        budget = 48 if arm.endswith("48") else 12
        values = ordered(paths[method])[:budget]
        eligible = [r for r in values if r["temporal"]["coverage"] == 1][:5]
        arms[arm] = {"metrics": {**known_path_metrics(values, query),
            "known_path_candidate_coverage": bool(known_path_metrics(values, query)["first_known_path_rank"]),
            "known_path_in_first5_complete_for_future_vlm": bool(known_path_metrics(eligible, query)["first_known_path_rank"]),
            "top_target_video": bool(values and values[0]["video_id"] == query["video_id"])},
            "path_budget": budget, "path_count": len(values), "unique_videos": len({r["video_id"] for r in values}),
            "complete_paths": sum(r["temporal"]["coverage"] == 1 for r in values), "results": values}
    if query["video_id"] not in case["candidate_videos"]:
        attribution = "target_outside_pool"
    elif target is None:
        attribution = "target_scoring_failure"
    elif not target["known_ordered_path_in_cached_frames"]:
        attribution = "known_path_absent_or_temporally_infeasible_in_cached_frames"
    elif not target["known_ordered_supported_path"]:
        attribution = "known_path_blocked_by_gate"
    elif arms["single_12"]["metrics"]["known_path_candidate_coverage"]:
        attribution = "known_path_already_retained"
    else:
        attribution = "known_supported_path_lost_by_single_alignment"
    return {"query_id": query["query_id"], "role": role, "query": query,
        "candidate_videos": case["candidate_videos"], "source_failure_videos": sorted(set(case["candidate_videos"])-set(case["evidence"])),
        "baseline_matches_saved": True, "attribution": attribution, "target_feasibility": target,
        "work": stats, "arms": arms}


def run(output):
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    if protocol["settings"] != SETTINGS or protocol["arms"] != list(ARMS):
        raise ValueError("Experimental settings changed after freeze")
    verify_hashes(protocol["input_sha256"])
    verify_hashes(protocol["source_sha256"])
    (output / "cases").mkdir(exist_ok=False)
    rows = []
    for case in protocol["cases"]:
        row = replay_case(case, TemporalConfig(**protocol["temporal_config"]), protocol["role"])
        atomic_write_json(output / "cases" / (row["query_id"] + ".json"), row)
        rows.append(row)
    verify_hashes(protocol["input_sha256"])
    verify_hashes(protocol["source_sha256"])
    summary = {"role": protocol["role"], "protocol_sha256": sha(output / "protocol.json"), "queries": len(rows),
        "attribution": dict(Counter(row["attribution"] for row in rows)),
        "source_failure_videos": sum(len(row["source_failure_videos"]) for row in rows),
        "decision": "keep_production_unchanged", "new_vlm_scores": False, "arms": {}}
    for arm in ARMS:
        summary["arms"][arm] = {"queries": len(rows),
            "metric_counts": {field: sum(row["arms"][arm]["metrics"][field] for row in rows)
                              for field in rows[0]["arms"][arm]["metrics"] if field != "first_known_path_rank"},
            "total_paths": sum(row["arms"][arm]["path_count"] for row in rows),
            "total_complete_paths": sum(row["arms"][arm]["complete_paths"] for row in rows),
            "mean_unique_videos": sum(row["arms"][arm]["unique_videos"] for row in rows)/len(rows),
            "paired_hit20": dict(Counter("improved" if row["arms"][arm]["metrics"]["known_path_hit_at_20"] > row["arms"]["single_12"]["metrics"]["known_path_hit_at_20"] else
                "regressed" if row["arms"][arm]["metrics"]["known_path_hit_at_20"] < row["arms"]["single_12"]["metrics"]["known_path_hit_at_20"] else "unchanged" for row in rows))}
    summary["cpu_replay_seconds_by_method"] = {method: sum(work[method]["cpu_replay_seconds"] for row in rows for work in row["work"].values())
                                               for method in ("top4", "diverse4")}
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "observed_regression"), default="development")
    parser.add_argument("--development-run", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.role, args.output_dir, args.development_run)
    else:
        run(args.output_dir)
