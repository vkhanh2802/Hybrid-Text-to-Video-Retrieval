"""CPU/cache-only retrieval evaluation on image-authored known targets.

No runtime threshold tuning: split is reported separately, and unknown frames
are NOT labeled false positives merely because they are not the sampled anchor.
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

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import load_runtime_config
from kis.paths import KIS_ROOT
from kis.query_plan import build_event_plan
from tools.evaluation import known_target_metrics
from tools.temporal_pilot import require_model_cache


def load_labels(path):
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    if labels.get("schema_version") != 1 or not labels.get("queries"):
        raise ValueError("Expected schema_version=1 and image-authored queries")
    seen, splits = set(), {}
    for query in labels["queries"]:
        key = query["query_id"]
        if not isinstance(key, str) or not key or len(key) > 64 or any(not c.isascii() or not (c.isalnum() or c in "_-") for c in key) or key in seen:
            raise ValueError("Invalid/duplicate query_id")
        seen.add(key)
        if query.get("split") not in {"dev", "holdout"}:
            raise ValueError("Set dev/holdout before evaluation")
        old = splits.setdefault(query["video_id"], query["split"])
        if old != query["split"]:
            raise ValueError("Video leakage between dev and holdout")
        if query.get("annotation_source") != "ai_visual_observation" or not query.get("image_sha256"):
            raise ValueError("Image-only annotation provenance and image checksum required")
        for field in ("raw_query", "english", "keywords"):
            if not isinstance(query.get(field), str) or not query[field].strip():
                raise ValueError(f"Missing query text: {field}")
        if not query.get("relevant_keyframes") or query.get("anchor_keyframe_n") not in query["relevant_keyframes"]:
            raise ValueError("Anchor must be explicitly judged relevant")
        events = query.get("events", [])
        if events:
            positives = query.get("event_relevant_keyframes", [])
            if len(events) != len(positives) or any(not rows for rows in positives):
                raise ValueError("Sequence events need visually reviewed positive keyframes")
            if not build_event_plan(structured(query)).active:
                raise ValueError("Unsupported labeled event sequence")
    return labels


def structured(query):
    variants = [query["english"], query["keywords"]]
    actions = query.get("events", [])
    return {"query_id": query["query_id"], "raw_query": query["raw_query"],
            "query_variants": [query["raw_query"], *variants],
            "stream_queries": {stream: variants for stream in ("visual", "bm25", "semantic")},
            "entities": [], "events": [{"action": action} for action in actions],
            "temporal_constraints": [{"type": "sequence", "order": actions}] if len(actions) > 1 else []}


def summarize(records):
    result = {}
    for split in ("dev", "holdout", "all"):
        subset = [record for record in records if split == "all" or record["split"] == split]
        result[split] = {}
        for mode in ("baseline", "temporal"):
            values = [record["modes"][mode] for record in subset if mode in record["modes"]]
            result[split][mode] = {"queries": len(values), **{
                field: sum(v["metrics"][field] for v in values) / len(values) if values else None
                for field in ("hit_at_1", "hit_at_5", "hit_at_20", "mrr_at_20", "candidate_target_hit")}}
            result[split][mode]["by_task"] = {}
            for task in ("image", "sequence"):
                selected = [record["modes"][mode] for record in subset
                            if record.get("task", "image") == task and mode in record["modes"]]
                result[split][mode]["by_task"][task] = {"queries": len(selected), **{
                    field: sum(v["metrics"][field] for v in selected) / len(selected) if selected else None
                    for field in ("hit_at_1", "hit_at_5", "hit_at_20", "mrr_at_20")}}
                if task == "sequence" and mode == "temporal":
                    result[split][mode]["by_task"][task].update({
                        field: sum(v["known_path_metrics"][field] for v in selected) / len(selected) if selected else None
                        for field in ("known_path_hit_at_1", "known_path_hit_at_5", "known_path_hit_at_20")})
    return result


def known_path_metrics(results, query):
    positives = query["event_relevant_keyframes"]
    ranks = []
    for rank, result in enumerate(results, 1):
        if result["video_id"] != query["video_id"]:
            continue
        events = result.get("temporal", {}).get("events", [])
        if len(events) == len(positives) and all(
            event.get("status") == "retrieved" and event.get("keyframe_n") in allowed
            for event, allowed in zip(events, positives)
        ):
            ranks.append(rank)
    first = min(ranks) if ranks else None
    return {"first_known_path_rank": first, **{
        f"known_path_hit_at_{cutoff}": float(first is not None and first <= cutoff) for cutoff in (1, 5, 20)}}


def run(labels_path, output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    labels = load_labels(labels_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        require_model_cache()
        import torch
        import faiss
        from kis.config import load_caption_runtime
        from kis.event_retrieval import LocalEventScorer, _unit_vectors
        from kis.objects import ObjectMetadataLookup
        from kis.ranking import get_search_profile
        from kis.solve_kis import solve_kis
        from kis.text import BM25Retriever
        from kis.visual import SiglipEncoder, SiglipRetriever

        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        config = load_runtime_config()
        profile = get_search_profile(config.retrieval_profile)
        with (output / "retrieval.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            visual = SiglipRetriever(video_groups=config.video_groups, nprobe=profile.nprobe)
            try:
                captions, _ = load_caption_runtime(config)
                encoder = SiglipEncoder(device="cpu")
                bm25 = BM25Retriever(captions.caption_data, tokenizer_mode=profile.tokenizer_mode)
                lookup = ObjectMetadataLookup()
                collection = visual.client.describe_collection(collection_name=visual.collection_name, timeout=15)
                context = {"label_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
                           "annotation_source": "AI visual observation; not human adjudicated", "vlm": False,
                           "planner": "fixed image-authored queries; no caption used to author labels", "device": "cpu",
                           "temporal_config": config.temporal.public(), "video_groups": config.video_groups,
                           "collection_id": collection.get("collection_id"),
                           "collection_stats": visual.client.get_collection_stats(collection_name=visual.collection_name, timeout=15),
                           "caption_vectors": int(captions.index.ntotal),
                           "source_sha256": {name: hashlib.sha256((KIS_ROOT / name).read_bytes()).hexdigest() for name in (
                               "tools/groundtruth_evaluation.py", "kis/solve_kis.py", "kis/temporal.py", "kis/text.py", "config/models.json")},
                           "limitations": ["Known-anchor metrics, not exhaustive relevance", "No threshold tuning or accuracy guarantee",
                                           "Random image queries do not validate temporal order", "Tiny video-disjoint sample; content duplicates not excluded"]}
                atomic_write_json(output / "context.json", context)
                records = []
                for query in labels["queries"]:
                    plan = structured(query)
                    frames, vectors = LocalEventScorer._video_data(query["video_id"])
                    matched = [i for i, frame in enumerate(frames) if frame["frame_id"] == query["frame_id"]
                               and frame["keyframe_n"] == query["anchor_keyframe_n"]]
                    if len(matched) != 1:
                        raise ValueError("Ground truth identity no longer matches mapping")
                    anchor = frames[matched[0]]
                    visual_similarity = None
                    if vectors is not None:
                        visual_similarity = float((_unit_vectors([encoder.encode_text(query["english"])]) @ _unit_vectors(vectors[matched]).T)[0, 0])
                    semantic = float(captions.score_frames(query["english"], [anchor])[0])
                    record = {"query_id": query["query_id"], "split": query["split"], "task": query.get("task", "image"), "query": query,
                              "anchor_visual_similarity": visual_similarity,
                              "anchor_caption_similarity": semantic if np.isfinite(semantic) else None, "modes": {}}
                    for enabled in ((False, True) if build_event_plan(plan).active else (False,)):
                        diagnostics = {}
                        started = perf_counter()
                        results = solve_kis(plan, visual, encoder, bm25, captions, lookup, vlm_reranker=None,
                                            rerank_top_k=0, result_top_k=300, policy=profile.policy,
                                            temporal_config=replace(config.temporal, enabled=enabled), diagnostics=diagnostics)
                        record["modes"]["temporal" if enabled else "baseline"] = {
                            "seconds": perf_counter() - started, "metrics": known_target_metrics(results, query),
                            "results": results, "diagnostics": diagnostics}
                        if enabled:
                            record["modes"]["temporal"]["known_path_metrics"] = known_path_metrics(results, query)
                            record["modes"]["temporal"]["target_video_in_local_pool"] = query["video_id"] in diagnostics["temporal"].get("candidate_videos", [])
                    records.append(record)
                    atomic_write_json(output / f"{query['query_id']}.json", record)
                    atomic_write_json(output / "summary.json", {"known_anchor_metrics": summarize(records)})
            finally:
                visual.client.close()
                LocalEventScorer._video_data.cache_clear()
        print(json.dumps({"output": str(output.resolve()), "known_anchor_metrics": summarize(records)}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=KIS_ROOT / "benchmark_results" / ("groundtruth_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    run(args.labels, args.output_dir)
