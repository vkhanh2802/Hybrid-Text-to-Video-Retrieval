"""One frozen human holdout comparison, cache-only CPU and read-only retrieval.

Independent of development-only runners: never relabel holdout as development,
never select thresholds, never inject a known target into a candidate pool.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.candidate_pool import select_candidate_videos
from kis.condition_evidence import apply_condition_evidence
from kis.config import load_runtime_config
from kis.event_retrieval import LocalEventScorer, _unit_vectors
from kis.paths import KIS_ROOT, video_csv_path, video_clip_path
from tools.candidate_pool_ablation import compact_alignment
from tools.condition_evidence_evaluation import evaluate_pairs
from tools.groundtruth_evaluation import known_path_metrics
from tools.human_holdout import sha, validate_annotations, verify_protocol


MODES = ("baseline", "semantic_guard")


def load_frozen_labels(packet, labels_path):
    protocol = verify_protocol(packet)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if (labels.get("schema") != "kis_human_holdout_labels_v1"
            or labels.get("annotation_source") != "human_visual_observation"
            or labels.get("human_verified") is not True or labels.get("caption_used_for_annotation") is not False):
        raise ValueError("Expected frozen human image-only holdout labels")
    validated = validate_annotations(packet, labels["human_annotations"], require_english=True)
    for key in ("queries", "skipped_samples", "translations_pending", "reviewed_windows", "eligible_pair_queries"):
        if labels.get(key) != validated[key]:
            raise ValueError("Frozen qrels differ from approved human annotations")
    if (labels["protocol_sha256"] != sha(Path(packet) / "protocol.json")
            or labels["manifest_sha256"] != labels["human_annotations"]["manifest_sha256"]
            or any(query["split"] != "holdout" for query in labels["queries"])):
        raise ValueError("Frozen label protocol/split mismatch")
    return labels, protocol


def frame_metrics(results, query):
    positives = {(j["video_id"], j["keyframe_n"]) for j in query["judgments"] if j["judgment"] == "positive"}
    first = next((rank for rank, frame in enumerate(results, 1)
                  if (frame["video_id"], frame["keyframe_n"]) in positives), None)
    return {"first_known_frame_rank": first, **{f"known_frame_hit_at_{k}": float(first is not None and first <= k)
                                               for k in (1, 5, 20)},
            "mrr_at_20": 1 / first if first is not None and first <= 20 else 0.0,
            "top_target_video": bool(results and results[0]["video_id"] == query["video_id"])}


def sequence_queries(labels):
    by_id = {query["query_id"]: query for query in labels["queries"]}
    sequences = []
    for sample in labels["human_annotations"]["samples"]:
        if sample["status"] != "reviewed" or len(sample["events"]) < 2:
            continue
        events = [by_id[f"{sample['sample_id']}_event_{i+1}"] for i in range(len(sample["events"]))]
        sequences.append({"query_id": sample["sample_id"] + "_sequence", "video_id": sample["video_id"],
                          "event_query_ids": [event["query_id"] for event in events],
                          "actions": [event["event_text"] for event in events],
                          "event_relevant_keyframes": [[j["keyframe_n"] for j in event["judgments"]
                                                        if j["judgment"] == "positive" and j["video_id"] == sample["video_id"]]
                                                       for event in events]})
    return sequences


def score_labeled_pairs(queries, encoder, captions):
    rows, vectors = [], {}
    for query in queries:
        action = query["event_text"]
        if action not in vectors:
            vectors[action] = encoder.encode_text(action)
        query_vector = _unit_vectors([vectors[action]])[0]
        frames = query["judgments"]
        semantic = captions.score_frames(action, frames)
        for index, frame in enumerate(frames):
            mapping, features = LocalEventScorer._video_data(frame["video_id"])
            positions = [i for i, row in enumerate(mapping) if row["frame_id"] == frame["frame_id"]]
            if len(positions) != 1 or mapping[positions[0]]["keyframe_n"] != frame["keyframe_n"]:
                raise ValueError("Human judgment/local mapping is ambiguous or mismatched")
            visual = float((_unit_vectors(features[positions]) @ query_vector)[0]) if features is not None else None
            metadata = captions.get_caption(frame["video_id"], frame["frame_id"])
            caption_hash = metadata.get("image_sha256") if metadata else None
            rows.append({**frame, "query_id": query["query_id"], "split": "holdout", "event_text": action,
                         "visual": visual, "semantic": float(semantic[index]) if np.isfinite(semantic[index]) else None,
                         "caption_image_sha256": caption_hash,
                         "caption_image_hash_matches": caption_hash == frame["image_sha256"] if caption_hash else None})
    return rows, vectors


def shared_retrieval(actions, vectors, hits_by_action, visual, bm25, captions, config):
    for action in actions:
        if action not in hits_by_action:
            hits_by_action[action] = {"action": action, "streams": {
                "visual": visual.search(vectors[action], top_k=config.event_top_k),
                "bm25": bm25.search_bm25(action, top_k=config.event_top_k),
                "semantic": captions.search(action, top_k=config.event_top_k)}}
    hits = [hits_by_action[action] for action in actions]
    videos, seeds, diagnostics = select_candidate_videos(
        hits, config.max_videos, strategy=config.candidate_pooling, rrf_k=config.candidate_rrf_k)
    return hits, videos, seeds, diagnostics


def save_evidence(output, unit_id, video_id, actions, frames, visual, semantic, stats):
    path = Path(output) / "raw_evidence" / f"{unit_id}_{video_id}.json"
    atomic_write_json(path, {"split": "holdout", "unit_id": unit_id, "video_id": video_id,
        "actions": actions, "frames": frames, "stats": stats,
        "visual": None if visual is None else np.where(np.isfinite(visual), visual, None).tolist(),
        "semantic": None if semantic is None else np.where(np.isfinite(semantic), semantic, None).tolist()})


def compare_unit(query, actions, vectors, hits_by_action, scorer, visual, bm25, captions, config, output, sequence=False):
    hits, video_ids, seeds, pool = shared_retrieval(actions, vectors, hits_by_action, visual, bm25, captions, config)
    results = {name: [] for name in MODES}
    failures, stats_by_video = {}, {}
    guarded_config = replace(config, condition_evidence="semantic_guard")
    started = perf_counter()
    for video_id in video_ids:
        try:
            frames, scores, support, vis, sem, stats = scorer.score(
                video_id, actions, [vectors[action] for action in actions], seeds[video_id])
            guarded, condition_stats = apply_condition_evidence(actions, frames, captions, support, vis, guarded_config)
            stats_by_video[video_id] = {**stats, "guard_condition_evidence": condition_stats}
            save_evidence(output, query["query_id"], video_id, actions, frames, vis, sem, stats)
            for name, mask in (("baseline", support), ("semantic_guard", guarded)):
                if sequence:
                    result = compact_alignment(video_id, frames, scores, mask, config)
                    if result is not None:
                        results[name].append(result)
                else:
                    results[name].extend(dict(frames[index], score=float(scores[0, index]))
                                         for index in np.flatnonzero(mask[0]))
        except (OSError, ValueError, RuntimeError) as exc:
            failures[video_id] = {"type": type(exc).__name__, "message": str(exc)}
    row = {"query_id": query["query_id"], "task": "sequence" if sequence else "event_frame",
           "split": "holdout", "query": query, "events": hits, "candidate_videos": video_ids,
           "target_video_in_pool": query["video_id"] in video_ids, "candidate_pool": pool,
           "failures": failures, "video_stats": stats_by_video, "same_pool_and_raw_scores": True,
           "seconds_shared_scoring": perf_counter() - started, "modes": {}}
    for name, values in results.items():
        values.sort(key=lambda item: (-item["score"], item["video_id"], item["frame_id"]))
        metrics = ({**known_path_metrics(values, query),
                    "top_target_video": bool(values and values[0]["video_id"] == query["video_id"])}
                   if sequence else frame_metrics(values, query))
        row["modes"][name] = {"metrics": metrics, "candidate_count": len(values),
                             "full_paths_in_pool": sum(item["temporal"]["coverage"] == 1 for item in values) if sequence else None,
                             "results": values[:20]}
    atomic_write_json(Path(output) / f"{query['query_id']}.json", row)
    return row


def ranking_summary(rows, sequence=False):
    fields = ("known_path_hit_at_1", "known_path_hit_at_5", "known_path_hit_at_20", "top_target_video") if sequence else (
        "known_frame_hit_at_1", "known_frame_hit_at_5", "known_frame_hit_at_20", "mrr_at_20", "top_target_video")
    count = len(rows)
    return {"queries": count, "target_video_in_pool": sum(row["target_video_in_pool"] for row in rows),
            "video_scoring_failures": sum(len(row["failures"]) for row in rows),
            "support_removed": sum(stats["guard_condition_evidence"]["support_removed"] for row in rows
                                   for stats in row["video_stats"].values()),
            "modes": {name: {field: sum(row["modes"][name]["metrics"][field] for row in rows) / count if count else None
                             for field in fields} for name in MODES}}


def adoption_gate(pair_metrics, event_summary, sequence_summary):
    baseline, guard = pair_metrics["baseline"], pair_metrics["semantic_guard"]
    checks = {
        "enough_event_queries": baseline["eligible_queries"] >= 8,
        "enough_sequences": sequence_summary["queries"] >= 4,
        "macro_fpr_strictly_lower": (baseline["macro_fpr"] is not None and guard["macro_fpr"] < baseline["macro_fpr"]),
        "macro_tpr_not_lower": (baseline["macro_tpr"] is not None and guard["macro_tpr"] >= baseline["macro_tpr"]),
        "frame_hit_at_20_not_lower": bool(event_summary["queries"] and
            event_summary["modes"]["semantic_guard"]["known_frame_hit_at_20"] >= event_summary["modes"]["baseline"]["known_frame_hit_at_20"]),
        "path_hit_at_20_not_lower": bool(sequence_summary["queries"] and
            sequence_summary["modes"]["semantic_guard"]["known_path_hit_at_20"] >= sequence_summary["modes"]["baseline"]["known_path_hit_at_20"]),
        "no_scoring_failure": not (event_summary["video_scoring_failures"] or sequence_summary["video_scoring_failures"])}
    return {"checks": checks, "all_checks_passed": all(checks.values()), "automatically_promoted": False,
            "decision": "review_required_not_promoted" if all(checks.values()) else "insufficient_evidence_keep_off"}


def run(packet, labels_path, output):
    labels, protocol = load_frozen_labels(packet, labels_path)
    runtime = load_runtime_config()
    config = replace(runtime.temporal, condition_evidence="off")
    if (config.min_visual_similarity != protocol["thresholds"]["visual"]
            or config.min_semantic_similarity != protocol["thresholds"]["semantic"]):
        raise ValueError("Runtime thresholds differ from frozen protocol")
    pipeline = json.loads((KIS_ROOT / "config/pipeline.json").read_text(encoding="utf-8"))
    if list(runtime.video_groups) != pipeline["video_groups"]:
        raise ValueError("Runtime video scope differs from frozen pipeline")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source_files = ("tools/human_holdout_evaluation.py", "tools/human_holdout.py",
                    "tools/condition_evidence_evaluation.py", "tools/candidate_pool_ablation.py",
                    "tools/groundtruth_evaluation.py", "kis/text.py", "kis/visual.py", "kis/ranking.py")
    context = {"mode": "frozen_human_holdout", "label_sha256": sha(labels_path),
               "protocol_sha256": sha(Path(packet) / "protocol.json"), "protocol": protocol,
               "temporal_config": config.public(), "annotation_source": "human_visual_observation",
               "human_verified": True, "vlm": False, "device": "cpu", "threshold_selection": False,
               "runtime_config_modified": False, "same_raw_scores_and_pools": True,
               "event_ranking_scope": "bounded temporal per-event local ranking, not production single-image solve_kis",
               "sequence_scope": "fixed approved event plan, same retrieval/scorer/pooling/DP; no LLM planner",
               "source_sha256": {name: sha(KIS_ROOT / name) for name in source_files}}
    atomic_write_json(output / "context.json", context)
    atomic_write_json(output / "source_snapshot.json", {
        name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in source_files})
    visual = None
    try:
        with (output / "evaluation.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            from tools.temporal_pilot import require_model_cache
            require_model_cache()
            import torch
            import faiss
            from kis.config import load_caption_runtime
            from kis.ranking import get_search_profile
            from kis.text import BM25Retriever
            from kis.visual import SiglipEncoder, SiglipRetriever
            torch.set_num_threads(4)
            faiss.omp_set_num_threads(4)
            profile = get_search_profile(runtime.retrieval_profile)
            visual = SiglipRetriever(video_groups=runtime.video_groups, nprobe=profile.nprobe)
            captions, _ = load_caption_runtime(runtime)
            encoder = SiglipEncoder(device="cpu")
            bm25 = BM25Retriever(captions.caption_data, tokenizer_mode=profile.tokenizer_mode)
            context.update(collection_id=visual.client.describe_collection(collection_name=visual.collection_name).get("collection_id"),
                           collection_stats=visual.client.get_collection_stats(collection_name=visual.collection_name),
                           caption_vectors=int(captions.index.ntotal), nprobe=profile.nprobe,
                           tokenizer_mode=profile.tokenizer_mode, video_groups=runtime.video_groups)
            context["caption_artifact_sha256"] = {str(Path(path).relative_to(runtime.caption_root)): sha(path)
                                                  for path in captions.json_path}
            atomic_write_json(output / "context.json", context)
            raw_pairs, vectors = score_labeled_pairs(labels["queries"], encoder, captions)
            atomic_write_json(output / "pair_raw_scores.json", raw_pairs)
            evaluated, pair_summary = evaluate_pairs(raw_pairs, captions, config)
            atomic_write_json(output / "pair_predictions.json", evaluated)
            atomic_write_json(output / "pair_summary.json", pair_summary)
            scorer, hits_by_action = LocalEventScorer(encoder, captions, config), {}
            event_rows, sequence_rows = [], []
            fingerprints = {}
            for query in labels["queries"]:
                row = compare_unit(query, [query["event_text"]], vectors, hits_by_action,
                                   scorer, visual, bm25, captions, config, output)
                event_rows.append(row)
                for video_id in row["candidate_videos"]:
                    if video_id not in fingerprints:
                        fingerprints[video_id] = {"mapping_sha256": sha(video_csv_path(video_id)),
                            "features_sha256": sha(video_clip_path(video_id)) if video_clip_path(video_id).is_file() else None}
                LocalEventScorer._video_data.cache_clear()
                atomic_write_json(output / "event_summary.json", ranking_summary(event_rows))
            for query in sequence_queries(labels):
                row = compare_unit(query, query["actions"], vectors, hits_by_action,
                                   scorer, visual, bm25, captions, config, output, sequence=True)
                sequence_rows.append(row)
                for video_id in row["candidate_videos"]:
                    if video_id not in fingerprints:
                        fingerprints[video_id] = {"mapping_sha256": sha(video_csv_path(video_id)),
                            "features_sha256": sha(video_clip_path(video_id)) if video_clip_path(video_id).is_file() else None}
                LocalEventScorer._video_data.cache_clear()
                atomic_write_json(output / "sequence_summary.json", ranking_summary(sequence_rows, sequence=True))
            event_summary, sequence_summary = ranking_summary(event_rows), ranking_summary(sequence_rows, sequence=True)
            atomic_write_json(output / "local_artifact_sha256.json", fingerprints)
            verify_protocol(packet)
            if sha(labels_path) != context["label_sha256"]:
                raise ValueError("Frozen labels changed during evaluation")
            summary = {"split": "holdout", "label_sha256": context["label_sha256"],
                       "annotation_source": "human_visual_observation", "human_verified": True,
                       "pair_metrics": pair_summary, "event_retrieval": event_summary,
                       "sequence_retrieval": sequence_summary,
                       "adoption": adoption_gate(pair_summary, event_summary, sequence_summary),
                       "caption_image_hash_mismatches": sum(row["caption_image_hash_matches"] is False for row in raw_pairs),
                       "labels_unchanged": True, "runtime_config_modified": False,
                       "limitations": ["Sparse known-positive qrels, not exhaustive corpus relevance",
                           "Human queries/labels; English translations explicitly approved",
                           "Small random windows and repeated neighboring frames; no independence/significance claim",
                           "Condition evidence is generated-caption evidence, never pixel verification",
                           "Single-event frame ranking is a bounded diagnostic, not production single-image retrieval"]}
            atomic_write_json(output / "summary.json", summary)
        print(json.dumps({"output": str(output.resolve()), "pair_metrics": {name: {
            key: pair_summary[name][key] for key in ("eligible_queries", "macro_tpr", "macro_fpr", "macro_balanced_accuracy", "totals")}
            for name in MODES}, "condition_status_counts": pair_summary["condition_status_counts"],
            "event_retrieval": event_summary, "sequence_retrieval": sequence_summary,
            "adoption": summary["adoption"]}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if visual is not None:
            visual.client.close()
        LocalEventScorer._video_data.cache_clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.packet, args.labels, args.output_dir)
