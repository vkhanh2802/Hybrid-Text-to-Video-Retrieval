"""Evaluate the active caption runtime with fixed, image-authored query variants."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.caption_schema import file_sha256
from kis.paths import KIS_ROOT


def load_benchmark():
    data = json.loads((KIS_ROOT / "benchmarks/l19_l20.json").read_text(encoding="utf-8"))
    queries = data["queries"]
    if len({item["query_id"] for item in queries}) != len(queries):
        raise ValueError("Duplicate benchmark query IDs")
    dev = {item["video_id"] for item in queries if item["split"] == "dev"}
    test = {item["video_id"] for item in queries if item["split"] == "test"}
    if dev & test:
        raise ValueError("Development and test videos overlap")
    return data


def known_target_metrics(results, query):
    positives = set(query["relevant_keyframes"])
    positions = [rank for rank, item in enumerate(results, 1) if item["video_id"] == query["video_id"] and item.get("keyframe_n") in positives]
    anchors = [rank for rank, item in enumerate(results, 1) if item["video_id"] == query["video_id"] and item["frame_id"] == query["frame_id"]]
    first = min(positions) if positions else None
    return {
        "first_known_target_rank": first, "anchor_rank": min(anchors) if anchors else None,
        **{f"hit_at_{cutoff}": float(first is not None and first <= cutoff) for cutoff in (1, 5, 20, 100, 300)},
        "candidate_target_hit": float(first is not None),
        "mrr_at_20": 1 / first if first is not None and first <= 20 else 0.0,
        "known_qrel_recall_at_20": sum(rank <= 20 for rank in positions) / len(positives),
    }


def structured_query(query):
    variants = [query["english"], query["keywords"]]
    return {"query_id": query["query_id"], "raw_query": query["raw_query"],
            "query_variants": [query["raw_query"], *variants],
            "stream_queries": {name: variants for name in ("visual", "bm25", "semantic")},
            "entities": [{"type": "object", "value": value} for value in query["objects"]]}


def run(output):
    import faiss
    import torch
    from kis.text import BM25Retriever
    from kis.objects import ObjectMetadataLookup
    from kis.ranking import get_search_profile
    from kis.config import load_caption_runtime, load_runtime_config
    from kis.visual import SiglipRetriever, SiglipEncoder
    from kis.solve_kis import solve_kis

    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    config = load_runtime_config()
    queries = load_benchmark()["queries"] + json.loads((KIS_ROOT / "benchmarks/additional_queries.json").read_text(encoding="utf-8"))["queries"]
    context = {"caption_root": str(config.caption_root), "profile": config.retrieval_profile,
               "queries_sha256": hashlib.sha256(json.dumps(queries, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
               "index_manifests": {path.relative_to(config.caption_root).as_posix(): file_sha256(path) for path in sorted((config.caption_root / "index").rglob("*.meta.json"))},
               "source_sha256": {name: file_sha256(KIS_ROOT / name) for name in ("kis/solve_kis.py", "kis/ranking.py", "config/models.json", "kis/text.py", "tools/evaluation.py")}}
    context_path = output / "context.json"
    if context_path.exists() and json.loads(context_path.read_text(encoding="utf-8")) != context:
        raise ValueError("Inputs changed; choose a new --output-dir")
    atomic_write_json(context_path, context)
    records = []
    with (output / "retrieval.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        caption, ids = load_caption_runtime(config)
        if set(query["video_id"] for query in queries) - set(ids):
            raise ValueError("Benchmark targets are not covered by the active runtime")
        visual = SiglipRetriever(video_groups=config.video_groups, nprobe=128)
        encoder = SiglipEncoder(device="cpu")
        lexical = BM25Retriever(caption.caption_data, tokenizer_mode="unicode")
        lookup = ObjectMetadataLookup()
        for query in queries:
            started = perf_counter()
            results = solve_kis(structured_query(query), visual, encoder, lexical, caption, lookup,
                                policy=get_search_profile(config.retrieval_profile).policy, result_top_k=3000)
            record = {"query_id": query["query_id"], "split": query["split"], "elapsed_seconds": perf_counter() - started,
                      "metrics": known_target_metrics(results, query), "results": results}
            records.append(record)
            atomic_write_json(output / "results" / f"{query['query_id']}.json", record)
            print(f"{query['query_id']}: rank={record['metrics']['first_known_target_rank']}", file=sys.__stdout__, flush=True)
    summary = {}
    for split in ("dev", "test", "fresh_test", "all"):
        selected = [record for record in records if split == "all" or record["split"] == split]
        summary[split] = {"queries": len(selected), **{key: float(np.mean([record["metrics"][key] for record in selected]))
                          for key in ("hit_at_1", "hit_at_5", "hit_at_20", "mrr_at_20", "candidate_target_hit")}}
    atomic_write_json(output / "summary.json", {"vlm": False, "planner": "fixed variants, no live Gemini", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=KIS_ROOT / "benchmark_results/current")
    run(parser.parse_args().output_dir.resolve())
