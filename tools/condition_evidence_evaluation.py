"""Development-only condition diagnostic; optional fixed-pool CPU replay.

No threshold selection, holdout scoring, image/model downloads or index writes.
Pair evaluation needs only saved scores and original caption JSON artifacts.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.condition_evidence import (apply_condition_evidence, caption_for_frame,
                                    check_conditions, guarded_support)
from kis.config import load_runtime_config
from kis.paths import KIS_ROOT, resolve_video_artifact
from kis.support_calibration import gate, support_metrics
from kis.temporal import TemporalConfig, evidence_scores
from tools.candidate_pool_ablation import compact_alignment, load_dev_records
from tools.groundtruth_evaluation import known_path_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArtifactCaptions:
    """Read original caption JSON without loading FAISS or an encoder."""

    def __init__(self, root):
        self.root = Path(root) / "caption_generator"
        self.hashes = {}

    @lru_cache(maxsize=16)
    def _video(self, video_id):
        path = resolve_video_artifact(self.root, video_id, ".json")
        if not path.is_file():
            self.hashes[video_id] = None
            return {}
        records = json.loads(path.read_text(encoding="utf-8"))
        self.hashes[video_id] = sha(path)
        by_frame = {}
        for row in records:
            frame = int(row["frame_id"])
            if row["video_id"] != video_id or frame in by_frame:
                raise ValueError("Invalid/duplicate original caption identity")
            by_frame[frame] = row
        return by_frame

    def get_caption(self, video_id, frame_id):
        return self._video(video_id).get(int(frame_id))


def load_dev_scores(scores_path, labels_path):
    rows = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    if not rows or any(row["split"] != "dev" for row in rows):
        raise ValueError("Condition evaluation accepts development-only score rows")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    expected = {}
    for query in labels["queries"]:
        if query["split"] == "dev":
            for item in query["judgments"]:
                key = (query["query_id"], item["video_id"], item["frame_id"], item["keyframe_n"])
                expected[key] = {**item, "event_text": query["event_text"]}
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in ("query_id", "video_id", "frame_id", "keyframe_n"))
        target = expected.get(key)
        if key in seen or target is None or any(row[field] != target[field] for field in (
                "event_text", "judgment", "image_sha256")):
            raise ValueError("Scored development rows do not match frozen labels")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Incomplete scored development judgments")
    return rows


def evaluate_pairs(rows, captions, config):
    evaluated = []
    for row in rows:
        evidence = check_conditions(row["event_text"], caption_for_frame(captions, row))
        baseline = gate(row.get("visual"), row.get("semantic"),
                        config.min_visual_similarity, config.min_semantic_similarity)
        visual = None if row.get("visual") is None else [[row["visual"]]]
        guarded = bool(guarded_support([[baseline]], visual, [[evidence["status"] == "contradicted"]],
                                      config.min_visual_similarity)[0, 0])
        evaluated.append({**row, "condition_evidence": evidence, "baseline_support": baseline,
                          "guarded_support": guarded})
    def metrics(field):
        predictions = [{**row, "visual": float(row[field]), "semantic": None} for row in evaluated]
        return support_metrics(predictions, 0.5, 0.5)
    return evaluated, {"baseline": metrics("baseline_support"), "semantic_guard": metrics("guarded_support"),
                       "condition_status_counts": {status: sum(row["condition_evidence"]["status"] == status for row in evaluated)
                                                   for status in ("consistent", "contradicted", "unknown")},
                       "support_removed": sum(row["baseline_support"] and not row["guarded_support"] for row in evaluated)}


def replay(records, source_context, captions, output, evidence_run=None):
    # The only model work is optional and cache-only, over the old four dev pools.
    if evidence_run is None:
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
        caption_vectors, _ = load_caption_runtime(load_runtime_config())
        encoder = SiglipEncoder(device="cpu")
    config = replace(TemporalConfig(**source_context["temporal_config"]), condition_evidence="off")
    guarded_config = replace(config, condition_evidence="semantic_guard")
    rows = []
    try:
        for record in records:
            query = record["query"]
            source = record["modes"]["temporal"]
            diagnostic = source["diagnostics"]["temporal"]
            actions = [event["action"] for event in diagnostic["events"]]
            vectors = [encoder.encode_text(action) for action in actions] if evidence_run is None else None
            seeds = {}
            for event in diagnostic["events"]:
                for hits in event["streams"].values():
                    for hit in hits:
                        seeds.setdefault(hit["video_id"], set()).add(int(hit["frame_id"]))
            scorer = LocalEventScorer(encoder, caption_vectors, config) if evidence_run is None else None
            results = {name: [] for name in ("baseline", "semantic_guard")}
            failures, condition_stats = {}, {}
            for video_id in diagnostic["candidate_videos"]:
                try:
                    if evidence_run is None:
                        frames, scores, support, visual, semantic, stats = scorer.score(video_id, actions, vectors, seeds[video_id])
                    else:
                        evidence = json.loads((Path(evidence_run) / f"{query['query_id']}_{video_id}_evidence.json").read_text(encoding="utf-8"))
                        if (evidence["split"] != "dev" or evidence["actions"] != actions
                                or any(frame["video_id"] != video_id for frame in evidence["frames"])):
                            raise ValueError("Cached sequence evidence identity mismatch")
                        frames, stats = evidence["frames"], evidence["stats"]
                        visual, semantic = (None if evidence[name] is None else np.asarray(evidence[name], dtype=float)
                                            for name in ("visual", "semantic"))
                        scores, support = evidence_scores(visual, semantic, (len(actions), len(frames)), config)
                    guarded, condition_stats[video_id] = apply_condition_evidence(
                        actions, frames, captions, support, visual, guarded_config)
                    # Save shared raw evidence so future dev-only replays need no models.
                    atomic_write_json(output / f"{query['query_id']}_{video_id}_evidence.json", {
                        "split": "dev", "actions": actions, "frames": frames, "stats": stats,
                        "visual": None if visual is None else np.where(np.isfinite(visual), visual, None).tolist(),
                        "semantic": None if semantic is None else np.where(np.isfinite(semantic), semantic, None).tolist()})
                    for name, mask in (("baseline", support), ("semantic_guard", guarded)):
                        result = compact_alignment(video_id, frames, scores, mask, config)
                        if result is not None:
                            results[name].append(result)
                except (OSError, ValueError, RuntimeError) as exc:
                    failures[video_id] = str(exc)
            row = {"query_id": query["query_id"], "failures": failures, "condition_stats": condition_stats, "modes": {}}
            for name, values in results.items():
                values.sort(key=lambda item: (-item["score"], item["video_id"]))
                row["modes"][name] = {"top_video": values[0]["video_id"] if values else None,
                    "top_target_video": bool(values and values[0]["video_id"] == query["video_id"]),
                    "full_paths_in_pool": sum(item["temporal"]["coverage"] == 1 for item in values),
                    "known_path_metrics": known_path_metrics(values, query), "results": values}
            row["baseline_matches_source_top"] = row["modes"]["baseline"]["top_video"] == (
                source["results"][0]["video_id"] if source["results"] else None)
            atomic_write_json(output / f"{query['query_id']}.json", row)
            rows.append(row)
            if evidence_run is None:
                LocalEventScorer._video_data.cache_clear()
    finally:
        if evidence_run is None:
            LocalEventScorer._video_data.cache_clear()
    summary = {"queries": len(rows), "baseline_matches_source_top": sum(row["baseline_matches_source_top"] for row in rows),
               "failures": sum(len(row["failures"]) for row in rows),
               "support_removed": sum(stats["support_removed"] for row in rows for stats in row["condition_stats"].values()),
               "modes": {name: {"top_target_video": sum(row["modes"][name]["top_target_video"] for row in rows),
                                 "full_paths_in_pool": sum(row["modes"][name]["full_paths_in_pool"] for row in rows),
                                 "known_path_hits_at_20": sum(row["modes"][name]["known_path_metrics"]["known_path_hit_at_20"] for row in rows)}
                         for name in ("baseline", "semantic_guard")}}
    atomic_write_json(output / "sequence_summary.json", summary)
    return summary


def run(scores_path, labels_path, output, sequence_labels=None, source_run=None, evidence_run=None):
    rows = load_dev_scores(scores_path, labels_path)
    if bool(sequence_labels) != bool(source_run):
        raise ValueError("Sequence labels and saved source run must be provided together")
    records, context = load_dev_records(sequence_labels, source_run) if source_run else (None, None)
    if evidence_run is not None:
        if source_run is None:
            raise ValueError("Cached evidence requires the original saved sequence run")
        cached = json.loads((Path(evidence_run) / "report.json").read_text(encoding="utf-8"))
        if (cached["sequence_source_context_sha256"] != sha(Path(source_run) / "context.json")
                or cached["sequence_label_sha256"] != sha(sequence_labels)):
            raise ValueError("Cached evidence/source labels mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    captions = ArtifactCaptions(load_runtime_config().caption_root)
    try:
        evaluated, summary = evaluate_pairs(rows, captions, TemporalConfig())
        atomic_write_json(output / "dev_pairs.json", evaluated)
        report = {"split": "dev_only", "holdout_used": False, "runtime_config_modified": False,
                  "labels_sha256": sha(labels_path), "scores_sha256": sha(scores_path),
                  "thresholds": {"visual": 0.25, "semantic": 0.45}, "pair_metrics": summary,
                  "source_sha256": {name: sha(KIS_ROOT / name) for name in (
                      "kis/condition_evidence.py", "kis/event_retrieval.py", "kis/temporal.py",
                      "tools/condition_evidence_evaluation.py")}}
        atomic_write_json(output / "source_snapshot.json", {
            name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in report["source_sha256"]})
        if records is not None:
            report["sequence_source_context_sha256"] = sha(Path(source_run) / "context.json")
            report["sequence_label_sha256"] = sha(sequence_labels)
            report["sequence_evidence_mode"] = "saved_raw_scores_no_models" if evidence_run else "cached_cpu_encoders"
            if evidence_run:
                report["evidence_source_sha256"] = {path.name: sha(path) for path in sorted(Path(evidence_run).glob("*_evidence.json"))}
            with (output / "replay.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                report["sequence_regression"] = replay(records, context, captions, output, evidence_run)
        report["caption_sha256_by_video"] = captions.hashes
        atomic_write_json(output / "report.json", report)
        print(json.dumps({"output": str(output.resolve()), **report}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-labels", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--evidence-run", type=Path)
    args = parser.parse_args()
    run(args.dev_scores, args.labels, args.output_dir, args.sequence_labels, args.source_run, args.evidence_run)
