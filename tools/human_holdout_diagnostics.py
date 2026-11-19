"""Posthoc stage attribution from saved holdout evidence; no tuning or models."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.temporal import TemporalConfig, align_events, evidence_scores
from tools.human_holdout import sha


def classify(row, evidence, config):
    """Known-positive restricted paths are diagnostics, never ranked answers."""
    sequence = row["task"] == "sequence"
    metric = "known_path_hit_at_20" if sequence else "known_frame_hit_at_20"
    if row["modes"]["baseline"]["metrics"][metric]:
        return "known_positive_hit_at_20"
    if not row["target_video_in_pool"]:
        return "target_video_missing_from_pool"
    if row["query"]["video_id"] in row["failures"] or evidence is None:
        return "target_video_scoring_failure"
    frames = evidence["frames"]
    visual, semantic = (None if evidence[name] is None else np.asarray(evidence[name], dtype=float)
                        for name in ("visual", "semantic"))
    scores, support = evidence_scores(visual, semantic, (len(evidence["actions"]), len(frames)), config)
    positives = (row["query"]["event_relevant_keyframes"] if sequence else
                 [[j["keyframe_n"] for j in row["query"]["judgments"]
                   if j["judgment"] == "positive" and j["video_id"] == row["query"]["video_id"]]])
    known = np.array([[frame["keyframe_n"] in allowed for frame in frames] for allowed in positives], dtype=bool)
    if not all(known.any(axis=1)):
        return "known_positive_missing_from_local_frames"
    if not sequence:
        return "known_positive_support_failure" if not (known & support).any() else "known_positive_rank_below_20"
    known_path = align_events(frames, scores, known, config)
    if known_path is None or known_path["coverage"] != 1:
        return "known_positive_temporal_gap_failure"
    supported_path = align_events(frames, scores, known & support, config)
    if supported_path is None or supported_path["coverage"] != 1:
        return "known_positive_support_failure"
    return "alignment_chose_other_frames_than_known_positives"


def run(source_run, labels_path):
    source_run = Path(source_run)
    output = source_run / "posthoc_diagnostics.json"
    if output.exists():
        raise FileExistsError("Posthoc diagnostic is versioned; do not overwrite")
    context = json.loads((source_run / "context.json").read_text(encoding="utf-8"))
    if sha(labels_path) != context["label_sha256"]:
        raise ValueError("Diagnostic labels/source run mismatch")
    config = TemporalConfig(**context["temporal_config"])
    rows = []
    for path in sorted(source_run.glob("drive_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        raw_path = source_run / "raw_evidence" / f"{row['query_id']}_{row['query']['video_id']}.json"
        evidence = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else None
        rows.append({"query_id": row["query_id"], "task": row["task"], "stage": classify(row, evidence, config),
                     "target_video": row["query"]["video_id"],
                     "baseline_top_video": row["modes"]["baseline"]["results"][0]["video_id"]
                         if row["modes"]["baseline"]["results"] else None,
                     "top_result_changed": row["modes"]["baseline"]["results"][:1] != row["modes"]["semantic_guard"]["results"][:1],
                     "raw_evidence_sha256": sha(raw_path) if evidence is not None else None})
    pairs = json.loads((source_run / "pair_predictions.json").read_text(encoding="utf-8"))
    report = {"mode": "posthoc_description_not_threshold_selection", "models_loaded": False,
              "source_label_sha256": context["label_sha256"], "source_tool_sha256": sha(Path(__file__)),
              "event_frame_stages": dict(Counter(row["stage"] for row in rows if row["task"] == "event_frame")),
              "sequence_stages": dict(Counter(row["stage"] for row in rows if row["task"] == "sequence")),
              "unknown_condition_reasons": dict(Counter(row["condition_evidence"]["reason"] for row in pairs
                                                        if row["condition_evidence"]["status"] == "unknown")),
              "rows": rows,
              "limitations": "A path differing from sparse known positives is not automatically visually wrong. Oracle restriction is diagnostic only, never injected into retrieval."}
    atomic_write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run, args.labels)
