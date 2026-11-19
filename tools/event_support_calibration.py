"""Freeze image judgments, train thresholds on dev, evaluate once on new holdout.

Use existing embeddings and cached CPU text encoders only. Never read holdout
scores during training or overwrite calibration artifacts/runtime thresholds.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import load_caption_runtime, load_runtime_config
from kis.event_retrieval import LocalEventScorer, _unit_vectors
from kis.paths import KIS_ROOT
from kis.support_calibration import choose_support_thresholds, support_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_labels(manifest_path, observations_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Frozen labels must not be overwritten")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    video_splits = {}
    for sample in samples.values():
        if sample["split"] not in {"dev", "holdout"}:
            raise ValueError("Unknown sample split")
        prior = video_splits.setdefault(sample["video_id"], sample["split"])
        if prior != sample["split"]:
            raise ValueError("Same video cannot appear in dev and holdout")
    queries, seen = [], set()
    for query in observations["queries"]:
        if query["query_id"] in seen or not isinstance(query.get("event_text"), str) or not query["event_text"].strip():
            raise ValueError("Duplicate ID or missing event text")
        seen.add(query["query_id"])
        anchor = samples[query["anchor_sample_id"]]
        judgments, identities = [], set()
        for judgment in query["judgments"]:
            sample = samples[judgment["sample_id"]]
            if sample["split"] != anchor["split"]:
                raise ValueError("Positive/negative frames must stay within their split")
            frame = next((frame for frame in sample["frames"] if frame["keyframe_n"] == judgment["keyframe_n"]), None)
            if frame is None or frame["status"] != "downloaded" or not frame.get("image_sha256"):
                raise ValueError("Cannot label unavailable images")
            identity = (frame["video_id"], frame["keyframe_n"])
            if identity in identities or judgment["judgment"] not in {"positive", "negative", "unknown"}:
                raise ValueError("Duplicate frame or invalid judgment")
            identities.add(identity)
            judgments.append({**judgment, **{key: frame[key] for key in ("video_id", "frame_id", "keyframe_n", "pts_time", "image_sha256")}})
        classes = {row["judgment"] for row in judgments}
        if not {"positive", "negative"}.issubset(classes):
            raise ValueError("Each event needs visually judged positive and negative frames")
        queries.append({**query, "split": anchor["split"], "judgments": judgments})
    labels = {"schema_version": 1, "annotation_source": "ai_visual_observation", "human_verified": False,
              "caption_used_to_author_labels": False, "manifest_sha256": sha(manifest_path),
              "observations_sha256": sha(observations_path), "queries": queries,
              "excluded_samples": observations.get("excluded_samples", [])}
    atomic_write_json(output, labels)
    print(json.dumps({"output": str(output.resolve()), "label_sha256": sha(output),
                      "dev_queries": sum(q["split"] == "dev" for q in queries),
                      "holdout_queries": sum(q["split"] == "holdout" for q in queries)}, indent=2))


def score_split(labels, split):
    # Filter before any encoder calls; training cannot score holdout queries.
    queries = [query for query in labels["queries"] if query["split"] == split]
    if not queries:
        raise ValueError(f"No {split} queries")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tools.temporal_pilot import require_model_cache
    require_model_cache()
    import faiss
    import torch
    from kis.visual import SiglipEncoder
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    runtime = load_runtime_config()
    captions, _ = load_caption_runtime(runtime)
    encoder = SiglipEncoder(device="cpu")
    rows = []
    try:
        for query in queries:
            vector = _unit_vectors([encoder.encode_text(query["event_text"])])[0]
            frames = query["judgments"]
            semantic = captions.score_frames(query["event_text"], frames)
            for index, frame in enumerate(frames):
                mapping, features = LocalEventScorer._video_data(frame["video_id"])
                positions = [i for i, row in enumerate(mapping) if row["frame_id"] == frame["frame_id"] and row["keyframe_n"] == frame["keyframe_n"]]
                if len(positions) != 1:
                    raise ValueError("Frozen judgment/mapping identity mismatch")
                visual = float((_unit_vectors(features[positions]) @ vector)[0]) if features is not None else None
                rows.append({"query_id": query["query_id"], "split": split, "event_text": query["event_text"],
                             **frame, "visual": visual, "semantic": float(semantic[index]) if np.isfinite(semantic[index]) else None})
    finally:
        LocalEventScorer._video_data.cache_clear()
    return rows


def run(labels_path, output, phase="train", calibration_path=None):
    if phase not in {"train", "holdout"}:
        raise ValueError("Unknown calibration phase")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if labels.get("schema_version") != 1 or labels.get("annotation_source") != "ai_visual_observation":
        raise ValueError("Expected frozen image-authored event judgments")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calibration = None
    if phase == "holdout":
        if calibration_path is None:
            raise ValueError("Freeze a development calibration artifact before holdout")
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if calibration["label_sha256"] != sha(labels_path) or calibration["split"] != "dev":
            raise ValueError("Calibration does not match frozen labels/development split")
    try:
        with (output / "scoring.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            rows = score_split(labels, "dev" if phase == "train" else "holdout")
            atomic_write_json(output / "scores.json", rows)
            report = {"split": "dev" if phase == "train" else "holdout", "label_sha256": sha(labels_path),
                      "annotation_source": "AI visual judgments; not human verified", "runtime_thresholds_modified": False,
                      "models": "cached CPU SigLIP/MiniLM; no VLM or new artifact generation",
                      "source_sha256": {name: sha(KIS_ROOT / name) for name in (
                          "kis/support_calibration.py", "tools/event_support_calibration.py", "kis/temporal.py", "kis/text.py", "config/models.json")}}
            if phase == "train":
                report.update(choose_support_thresholds(rows))
            else:
                report.update(calibration_sha256=sha(calibration_path), baseline_metrics=support_metrics(rows, 0.25, 0.45))
                selected = calibration.get("selected")
                report["selected_thresholds"] = None if selected is None else {key: selected[key] for key in ("visual_threshold", "semantic_threshold")}
                report["selected_metrics"] = None if selected is None else support_metrics(rows, selected["visual_threshold"], selected["semantic_threshold"])
                report["selection_rule"] = "Frozen development selection; no holdout tuning"
            atomic_write_json(output / "calibration.json", report)
        print(json.dumps({key: value for key, value in report.items() if key not in {"grid", "source_sha256"}}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--observations", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for phase in ("train", "holdout"):
        cmd = sub.add_parser(phase)
        cmd.add_argument("--labels", type=Path, required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
        if phase == "holdout":
            cmd.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_labels(args.manifest, args.observations, args.output)
    else:
        run(args.labels, args.output_dir, args.command, getattr(args, "calibration", None))
