"""Versioned CPU smoke and fixed-candidate text reranker evaluation.

Prepare freezes inputs before inference; run never searches, plans or loads VLM.
Development and observed human regression are separate, never new holdout claims.
"""

import argparse
from collections import Counter
import copy
from dataclasses import replace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter
import traceback

from kis.artifact_io import atomic_write_json
from kis.config import load_runtime_config
from kis.paths import KIS_ROOT, resolve_video_artifact
from kis.ranking import finalize_vlm_ranking
from kis.text_reranker import LazyCrossEncoder, TextRerankerConfig, rerank_text_candidates
from tools.condition_evidence_evaluation import ArtifactCaptions
from tools.human_holdout import sha
from tools.pipeline_ablation import ARMS, metrics


SOURCES = ("tools/text_reranker_benchmark.py", "kis/text_reranker.py", "kis/ranking.py",
           "tools/pipeline_ablation.py", "tools/groundtruth_evaluation.py", "tools/human_holdout_evaluation.py")
CONFIG = replace(TextRerankerConfig(), mode="rerank")
LABELS = {"dev_pairs": "benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json",
          "dev_sequences": "benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json",
          "human_regression": "benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(hashes):
    for name, fingerprint in hashes.items():
        path = Path(name)
        actual = sha(path) if path.is_file() else None
        if actual != fingerprint:
            raise ValueError(f"Frozen input changed: {name}")


def source_hashes():
    return {str(KIS_ROOT / name): sha(KIS_ROOT / name) for name in SOURCES}


def model_files(config):
    from huggingface_hub import try_to_load_from_cache

    result = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt",
                 "special_tokens_map.json", "model.safetensors"):
        path = try_to_load_from_cache(config.model, name, revision=config.revision)
        if not isinstance(path, str) or not Path(path).is_file():
            raise FileNotFoundError(f"Pinned checkpoint cache incomplete: {name}")
        result[str(Path(path).resolve())] = sha(path)
    return result


def cpu_runtime():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return torch


def runtime_info(torch):
    return {"python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "cuda_initialized": torch.cuda.is_initialized()}


def smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"schema": "kis_text_smoke_v1", "config": CONFIG.public(),
        "source_sha256": source_hashes(), "model_sha256": model_files(CONFIG),
        "synthetic_pairs": [
            ["A person in a red shirt rides a bicycle.", "A person wearing a red shirt rides a bicycle on a road."],
            ["A person in a red shirt rides a bicycle.", "A forest contains tall trees and a river."],
            ["A cook pours beaten eggs into a pot.", "A cook pours a bowl of beaten eggs into a pot of soup."]],
        "checks": "finite raw logits, CPU only, repeated scores, valid batch16, missing-text/failure fallback",
        "query_labels_used": False, "new_retrieval": False, "vlm": False}
    atomic_write_json(output / "protocol.json", frozen)
    try:
        torch = cpu_runtime()
        scorer = LazyCrossEncoder(CONFIG)
        started = perf_counter()
        first = scorer.score_pairs(frozen["synthetic_pairs"])
        cold_seconds = perf_counter() - started
        started = perf_counter()
        second = scorer.score_pairs(frozen["synthetic_pairs"])
        warm_seconds = perf_counter() - started
        started = perf_counter()
        batch16 = scorer.score_pairs([frozen["synthetic_pairs"][i % 3] for i in range(16)])
        batch_seconds = perf_counter() - started
        if (not all(math.isfinite(v) for v in first + second + batch16)
                or max(abs(a-b) for a, b in zip(first, second)) > 1e-6):
            raise ValueError("CPU smoke logits invalid or not repeatable")
        candidates = [{"video_id": "synthetic", "frame_id": i, "score": score, "retrieval_text": text}
                      for i, score, text in ((1, .9, frozen["synthetic_pairs"][1][1]), (2, .8, None),
                                             (3, .7, frozen["synthetic_pairs"][0][1]))]
        ranked, diag = rerank_text_candidates(candidates, {"raw_query": frozen["synthetic_pairs"][0][0]},
                                              None, CONFIG, scorer)
        if ranked[1]["frame_id"] != 2 or ranked[1]["score"] != .8 or diag["candidates_scored"] != 2:
            raise ValueError("Missing-caption fallback changed")
        class BrokenScorer:
            def score_pairs(self, pairs):
                raise RuntimeError("intentional synthetic failure, not a real model error")
        fallback, error_diag = rerank_text_candidates(candidates, {"raw_query": "test query"},
                                                      None, CONFIG, BrokenScorer())
        if ([r["frame_id"] for r in fallback] != [1, 2, 3]
                or error_diag["candidates_scored"] != 0):
            raise ValueError("Synthetic error fallback failed")
        if torch.cuda.is_initialized() or any(p.device.type != "cpu" for p in scorer.model.parameters()):
            raise ValueError("Smoke must be CPU-only")
        verify(frozen["source_sha256"])
        verify(frozen["model_sha256"])
        result = {"success": True, "protocol_sha256": sha(output / "protocol.json"),
            "raw_logits": first, "repeated_logits": second, "batch16_all_finite": True,
            "cold_load_and_first_batch_seconds": cold_seconds, "warm_batch3_seconds": warm_seconds,
            "warm_batch16_seconds": batch_seconds,
            "parameters": sum(p.numel() for p in scorer.model.parameters()),
            "parameter_bytes": sum(p.numel()*p.element_size() for p in scorer.model.parameters()),
            "runtime": runtime_info(torch), "missing_caption_fallback_passed": True,
            "injected_error_fallback_passed": True, "synthetic_ranked_ids": [r["frame_id"] for r in ranked],
            "actual_checkpoint_error": False, "synthetic_scores_are_not_quality_metrics": True}
        atomic_write_json(output / "summary.json", result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


class PreparedCaptions:
    def __init__(self, rows):
        self.rows = rows

    def get_caption(self, video_id, frame_id):
        return self.rows.get(f"{video_id}:{int(frame_id)}")


def validate_case(case):
    if case.get("planner_failure") or case.get("plan") is None:
        raise ValueError("This cache-only protocol requires an existing successful fixed plan")
    if metrics(case["results"], case["unit"]) != case["metrics"]:
        raise ValueError("Saved baseline/qrels metrics differ")
    if any("text_ranking_score" in r or "text_logit" in r for r in case["results"]):
        raise ValueError("Expected text-reranker-free CPU baseline")


def prepare(source, smoke_dir, output, role="development", development_run=None):
    source, smoke_dir, output = map(Path, (source, smoke_dir, output))
    if output.exists():
        raise FileExistsError("Never overwrite a prepared benchmark")
    smoke_protocol, smoke_result = read_json(smoke_dir / "protocol.json"), read_json(smoke_dir / "summary.json")
    if (not smoke_result["success"] or smoke_result["protocol_sha256"] != sha(smoke_dir / "protocol.json")
            or smoke_protocol["config"] != CONFIG.public()):
        raise ValueError("A matching successful smoke is required")
    verify(smoke_protocol["source_sha256"])
    verify(smoke_protocol["model_sha256"])
    frozen_inputs = {}
    def freeze(path):
        path = Path(path).resolve()
        frozen_inputs[str(path)] = sha(path) if path.is_file() else None
        return str(path)
    for name in ("protocol.json", "summary.json"):
        freeze(smoke_dir / name)
    original = read_json(source / "protocol.json")
    if original["planner"] != "fixed" or (source / "failure.json").exists():
        raise ValueError("Expected completed fixed-plan CPU source")
    for name in ("protocol.json", "case_index.json", "source_snapshot.json"):
        freeze(source / name)
    # Old code is intentionally not required to match the newly added reranker.
    # Check the immutable original CPU files against their published bindings.
    binding_path = KIS_ROOT / "benchmark_results/full_pipeline_vlm_packet_20261102/bindings_private.json"
    bindings = read_json(binding_path)
    if bindings["source_protocol_sha256"] != sha(source / "protocol.json"):
        raise ValueError("Original CPU protocol differs from published bindings")
    freeze(binding_path)
    for key in (("dev_pairs", "dev_sequences") if role == "development" else ("human_regression",)):
        path = KIS_ROOT / LABELS[key]
        if sha(path) != original["label_sha256"][key]:
            raise ValueError("Frozen qrels changed")
        freeze(path)
    if role == "observed_regression":
        if development_run is None:
            raise ValueError("Observed regression requires completed development")
        development_run = Path(development_run)
        dev_protocol, dev = read_json(development_run / "protocol.json"), read_json(development_run / "summary.json")
        if (dev["role"] != "development" or dev_protocol["config"] != CONFIG.public()
                or dev["protocol_sha256"] != sha(development_run / "protocol.json")):
            raise ValueError("Development configuration/provenance differs")
        verify(dev_protocol["source_sha256"])
        freeze(development_run / "protocol.json")
        freeze(development_run / "summary.json")
    elif role != "development":
        raise ValueError("Unknown experiment role")
    runtime = load_runtime_config(environ={})
    captions = ArtifactCaptions(runtime.caption_root)
    old_caption_context = KIS_ROOT / "benchmark_results/human_holdout_v3_evaluation_20261102/context.json"
    old_caption_hashes = read_json(old_caption_context)["caption_artifact_sha256"]
    # Read only historical artifact fingerprints, NOT regression outcomes/qrels.
    freeze(old_caption_context)
    cases, caption_rows = [], {}
    index = read_json(source / "case_index.json")
    for item in index:
        if item["role"] != role:
            continue
        path = source / "cases" / (item["case_id"] + ".json")
        case = read_json(path)
        validate_case(case)
        if (sha(path) != bindings["cases"][case["case_id"]]["case_sha256"]
                or case["unit"]["role"] != role or case["arm"] != item["arm"]):
            raise ValueError("Original CPU case changed")
        freeze(path)
        for result in case["results"][:CONFIG.pool_size]:
            rows = result.get("temporal", {}).get("events", []) if case["unit"]["task"] == "sequence" else [result]
            for row in rows:
                if row.get("status") == "unknown":
                    continue
                video_id, frame_id = row["video_id"], row["frame_id"]
                meta = captions.get_caption(video_id, frame_id)
                caption_path = resolve_video_artifact(runtime.caption_root / "caption_generator", video_id, ".json")
                freeze(caption_path)
                if meta is not None:
                    relative = str(caption_path.relative_to(runtime.caption_root))
                    if old_caption_hashes.get(relative) != sha(caption_path):
                        raise ValueError("Caption corpus differs from historical fingerprint")
                    if row.get("keyframe_n") is not None and row["keyframe_n"] != meta.get("keyframe_n"):
                        raise ValueError("Caption/keyframe identity mismatch")
                    for field in ("retrieval_text", "caption"):
                        if isinstance(row.get(field), str) and row[field].strip() and row[field] != meta.get(field):
                            raise ValueError("Inline evidence differs from original caption")
                caption_rows[f"{video_id}:{int(frame_id)}"] = meta
        cases.append(case)
    expected = 45 if role == "development" else 108
    if len(cases) != expected:
        raise ValueError(f"Expected all {expected} cases; never filter query denominators")
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared_inputs.json"
    atomic_write_json(prepared, {"cases": cases, "captions": caption_rows})
    freeze(prepared)
    protocol = {"schema": "kis_cached_text_reranker_v1", "role": role, "config": CONFIG.public(),
        "cases": len(cases), "query_units": len({case["unit"]["query_id"] for case in cases}),
        "input_sha256": frozen_inputs, "source_sha256": source_hashes(),
        "model_sha256": model_files(CONFIG), "prepared_inputs": str(prepared.resolve()),
        "source_cpu_run": str(source.resolve()), "smoke_dir": str(smoke_dir.resolve()),
        "arms": list(ARMS), "new_retrieval": False, "llm_calls": False, "vlm": False,
        "threshold_selection": False, "independent_holdout": False, "production_modified": False,
        "current_code_may_differ_from_old_cpu_source": True,
        "metrics": "All frozen queries retained; baseline metrics recomputed and verified. Unknown predictions not negatives.",
        "timing": "No score memoization. CPU rerank timings exclude caption preparation, retrieval and planner; first case includes model load.",
        "adoption_rule": "No automatic promotion. Development descriptive only; human v3 observed regression only. No parameter retuning after outcomes."}
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "source_snapshot.json", {name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in SOURCES})
    verify(frozen_inputs)
    print(json.dumps({"prepared": str(output), "role": role, "cases": len(cases),
                      "query_units": protocol["query_units"], "protocol_sha256": sha(output / "protocol.json")}, indent=2))


def paired(rows, field):
    result = {key: [] for key in ("improved", "regressed", "unchanged")}
    for row in rows:
        delta = row["text_metrics"][field] - row["baseline_metrics"][field]
        name = "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "unchanged"
        result[name].append(row["case_id"])
    return {"counts": {k: len(v) for k, v in result.items()}, "case_ids": result}


def run_case(case, captions, scorer, config=CONFIG):
    validate_case(case)
    baseline = case["results"]
    working = copy.deepcopy(baseline)
    ranked, diagnostic = rerank_text_candidates(working, case["plan"], captions, config, scorer)
    ranked, applied = finalize_vlm_ranking(ranked, top_k=0, base_score_field="text_ranking_score")
    if applied or sorted(r["score"] for r in ranked) != sorted(r["score"] for r in baseline):
        raise ValueError("Retrieval scores or VLM state changed")
    if any(r.get("temporal", {}).get("verified", False) for r in ranked):
        raise ValueError("Text score cannot verify a temporal path")
    return {"case_id": case["case_id"], "role": case["unit"]["role"], "task": case["unit"]["task"],
        "arm": case["arm"], "baseline_metrics": case["metrics"], "text_metrics": metrics(ranked, case["unit"]),
        "diagnostics": diagnostic, "results": ranked}


def summarize(rows):
    result = {}
    for task in ("image", "sequence"):
        result[task] = {}
        for arm in ARMS:
            group = [row for row in rows if row["task"] == task and row["arm"] == arm]
            fields = (["known_frame_hit_at_1", "known_frame_hit_at_5", "known_frame_hit_at_20", "mrr_at_20"] if task == "image"
                      else ["known_path_hit_at_1", "known_path_hit_at_5", "known_path_hit_at_20", "top_target_video"])
            metric1, metric5 = fields[0], fields[1]
            times = sorted(row["diagnostics"]["seconds"] for row in group)
            statuses = Counter()
            for row in group:
                statuses.update(row["diagnostics"]["candidate_status_counts"])
            result[task][arm] = {"queries": len(group),
                "baseline": {field: sum(row["baseline_metrics"][field] for row in group)/len(group) for field in fields},
                "text": {field: sum(row["text_metrics"][field] for row in group)/len(group) for field in fields},
                "paired_top1": paired(group, metric1), "paired_top5": paired(group, metric5),
                "paired_mrr" if task == "image" else "paired_top20": paired(group, "mrr_at_20" if task == "image" else fields[2]),
                "candidate_status_counts": dict(statuses),
                "pairs_requested": sum(row["diagnostics"]["pairs_requested"] for row in group),
                "pairs_valid": sum(row["diagnostics"]["pairs_valid"] for row in group),
                "batch_errors": sum(row["diagnostics"]["batch_errors"] for row in group),
                "rerank_seconds_sum": sum(times), "case_seconds_median": statistics.median(times),
                "case_seconds_p95_nearest_rank": times[math.ceil(.95*len(times))-1]}
    return result


def run(output):
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    if protocol["config"] != CONFIG.public():
        raise ValueError("Settings changed after preparation")
    for key in ("source_sha256", "input_sha256", "model_sha256"):
        verify(protocol[key])
    (output / "cases").mkdir(exist_ok=False)
    try:
        torch = cpu_runtime()
        prepared = read_json(protocol["prepared_inputs"])
        scorer, captions = LazyCrossEncoder(CONFIG), PreparedCaptions(prepared["captions"])
        rows = []
        for case in prepared["cases"]:
            row = run_case(case, captions, scorer)
            atomic_write_json(output / "cases" / (case["case_id"] + ".json"), row)
            rows.append(row)
        if torch.cuda.is_initialized() or scorer.load_error:
            raise RuntimeError("Unexpected CUDA/model load state during CPU benchmark")
        for key in ("source_sha256", "input_sha256", "model_sha256"):
            verify(protocol[key])
        summary = {"schema": "kis_cached_text_reranker_results_v1", "role": protocol["role"],
            "protocol_sha256": sha(output / "protocol.json"), "cases": len(rows),
            "query_units": protocol["query_units"], "runtime": runtime_info(torch),
            "comparisons": summarize(rows), "automatically_promoted": False,
            "production_mode": load_runtime_config(environ={}).text_reranker.mode,
            "limits": ["Fixed cached retrieval and event plans, not live LLM/planner or end-to-end latency",
                "Development AI-observed qrels; human v3 already observed regression, not untouched holdout",
                "Three retrieval arms share query units; do not count as independent replications",
                "Text cannot recover outside-pool evidence or change a missing temporal path"]}
        atomic_write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "prepare", "run"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-run", type=Path, default=KIS_ROOT / "benchmark_results/full_pipeline_ablation_20261102")
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--role", choices=("development", "observed_regression"), default="development")
    parser.add_argument("--development-run", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.output_dir)
    elif args.command == "prepare":
        if args.smoke_dir is None:
            parser.error("prepare requires --smoke-dir")
        prepare(args.source_run, args.smoke_dir, args.output_dir, args.role, args.development_run)
    else:
        run(args.output_dir)
