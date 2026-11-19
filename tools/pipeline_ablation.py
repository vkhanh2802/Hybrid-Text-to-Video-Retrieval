"""CPU production-path retrieval ablation; heavy VLM is a separate Kaggle job.

No changes to production modules. Development and observed human regression are
reported separately. LLM planning is optional and never silently substituted.
"""

import argparse
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import load_runtime_config
from kis.event_retrieval import LocalEventScorer
from kis.objects import LABEL_MAP
from kis.paths import KIS_ROOT
from kis.ranking import fuse_query_variants, get_search_profile
from tools.groundtruth_evaluation import known_path_metrics
from tools.human_holdout import sha
from tools.human_holdout_evaluation import frame_metrics, load_frozen_labels, sequence_queries


ARMS = ("full", "full_deep", "full_compact")
STOPWORDS = frozenset("a an the is are was were be been being in on at of to and with from by for as "
                     "his her their its him she he it this that there has have wearing wears dressed "
                     "behind front background backdrop showing shows stands standing".split())
PROTOCOL = {"schema": "kis_full_pipeline_ablation_v1", "arms": list(ARMS),
    "static_depth": {"full": 100, "full_deep": 200, "full_compact": "100 per variant, 2 variants"},
    "temporal_depth": {"full": 50, "full_deep": 100, "full_compact": "50 per variant, 2 variants"},
    "compact_rule": "first 20 non-stopword alphanumeric tokens in original order; full query retained alongside",
    "vlm_top_k_images": 20, "vlm_top_k_paths": 5, "vlm_weight": 0.5,
    "vlm_caption_modes": ["with_caption", "image_only"], "pair_vlm_positive_threshold": 8,
    "sequence_vlm": "joint ordered evidence images for complete selected paths only; not continuous-video verification",
    "selection": "No automated selection/promotion. Compare methods on dev; already-observed human v3 is regression only.",
    "limits": "Reranking cannot recover candidates outside retrieval or generate alternate temporal paths.",
    "frozen_sources_unchanged": True}


def compact_query(text):
    tokens = re.findall(r"[^\W_]+(?:-[^\W_]+)*", text, flags=re.UNICODE)
    kept = [token for token in tokens if token.casefold() not in STOPWORDS]
    return " ".join(kept[:20]) or text


def literal_entities(text):
    # A fixed-plan control, NOT the production LLM entity extractor.
    return [{"type": "object", "value": name, "attributes": {}}
            for name in LABEL_MAP if re.search(r"\b" + re.escape(name) + r"\b", text, re.I)]


def fixed_plan(unit):
    actions = unit.get("actions", [])
    english = unit.get("event_text") or " Then ".join(actions)
    return {"query_id": unit["query_id"], "raw_query": unit["raw_query"],
            "query_variants": [english], "stream_queries": {name: [english] for name in ("visual", "bm25", "semantic")},
            "entities": literal_entities(english), "events": [{"action": action} for action in actions],
            "temporal_constraints": [{"type": "sequence", "order": actions}] if len(actions) > 1 else []}


def arm_plan(plan, arm):
    result = copy.deepcopy(plan)
    if arm == "full_compact":
        result["stream_queries"] = {name: list(dict.fromkeys(
            [query for text in values for query in (text, compact_query(text))]))
            for name, values in result["stream_queries"].items()}
    return result


class CachedEncoder:
    def __init__(self, encoder):
        self.encoder, self.cache = encoder, {}

    def encode_text(self, text):
        if text not in self.cache:
            self.cache[text] = self.encoder.encode_text(text)
        return self.cache[text]


class TemporalVariantVisual:
    def __init__(self, visual, encoder, actions, arm, policy, event_top_k=50):
        self.visual, self.encoder, self.arm, self.policy = visual, encoder, arm, policy
        self.event_top_k = event_top_k
        self.by_vector = {}
        for action in actions:
            key = np.asarray(encoder.encode_text(action), dtype=np.float32).tobytes()
            if key in self.by_vector and self.by_vector[key] != action:
                raise ValueError("Ambiguous text/vector identity in query variant wrapper")
            self.by_vector[key] = action

    def search(self, vector, top_k=50):
        if top_k != self.event_top_k:
            return self.visual.search(vector, top_k=top_k)  # Static fallback already expands its own variants.
        if self.arm == "full_deep":
            return self.visual.search(vector, top_k=top_k * 2)
        action = self.by_vector[np.asarray(vector, dtype=np.float32).tobytes()]
        texts = list(dict.fromkeys([action, compact_query(action)]))
        return fuse_query_variants([self.visual.search(self.encoder.encode_text(text), top_k=top_k) for text in texts], self.policy)


class TemporalVariantText:
    def __init__(self, base, arm, policy, event_top_k=50):
        self.base, self.arm, self.policy = base, arm, policy
        self.event_top_k = event_top_k

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _search(self, text, top_k, method):
        if top_k != self.event_top_k:
            return method(text, top_k=top_k)
        if self.arm == "full_deep":
            return method(text, top_k=top_k * 2)
        return fuse_query_variants([method(query, top_k=top_k)
                                   for query in dict.fromkeys([text, compact_query(text)])], self.policy)

    def search(self, text, top_k=50):
        return self._search(text, top_k, self.base.search)

    def search_bm25(self, text, top_k=50):
        return self._search(text, top_k, self.base.search_bm25)


def load_units(dev_labels, dev_sequence_labels, human_packet, human_labels):
    data = json.loads(Path(dev_labels).read_text(encoding="utf-8"))
    if data.get("annotation_source") != "ai_visual_observation":
        raise ValueError("Expected frozen development pair labels")
    units = []
    for query in data["queries"]:
        if query["split"] != "dev":
            continue
        primary = next(j["video_id"] for j in query["judgments"]
                       if j["judgment"] == "positive" and j["sample_id"] == query["anchor_sample_id"])
        units.append({**query, "query_id": "dev_" + query["query_id"], "task": "image", "role": "development",
                      "video_id": primary, "raw_query": query["event_text"], "annotation_source": "ai_visual_observation"})
    old = json.loads(Path(dev_sequence_labels).read_text(encoding="utf-8"))
    for query in old["queries"]:
        if query["split"] != "dev" or query.get("task") != "sequence":
            continue
        units.append({**query, "query_id": "dev_" + query["query_id"], "role": "development",
                      "actions": query["events"], "annotation_source": "ai_visual_observation"})
    human, _ = load_frozen_labels(human_packet, human_labels)
    for query in human["queries"]:
        units.append({**query, "query_id": "human_" + query["query_id"], "task": "image", "role": "observed_regression",
                      "raw_query": query["text_vi"]})
    by_sample = {sample["sample_id"]: sample for sample in human["human_annotations"]["samples"]}
    for query in sequence_queries(human):
        sample = by_sample[query["query_id"].removesuffix("_sequence")]
        raw = " Sau đó: ".join(event["text_vi"].strip() for event in sample["events"])
        units.append({**query, "query_id": "human_" + query["query_id"], "task": "sequence", "split": "holdout",
                      "role": "observed_regression", "raw_query": "Đầu tiên: " + raw,
                      "annotation_source": "human_visual_observation"})
    if len({unit["query_id"] for unit in units}) != len(units):
        raise ValueError("Duplicate benchmark query IDs")
    return units


def metrics(results, unit):
    if unit["task"] == "sequence":
        return {**known_path_metrics(results, unit), "top_target_video": bool(results and results[0]["video_id"] == unit["video_id"])}
    return frame_metrics(results, unit)


def summarize(cases):
    output = {}
    for role in ("development", "observed_regression"):
        output[role] = {}
        for task in ("image", "sequence"):
            output[role][task] = {}
            for arm in ARMS:
                subset = [case for case in cases if case["unit"]["role"] == role and case["unit"]["task"] == task and case["arm"] == arm]
                fields = (["known_frame_hit_at_1", "known_frame_hit_at_5", "known_frame_hit_at_20", "mrr_at_20"]
                          if task == "image" else ["known_path_hit_at_1", "known_path_hit_at_5", "known_path_hit_at_20", "top_target_video"])
                output[role][task][arm] = {"queries": len(subset), **{
                    field: sum(case["metrics"][field] for case in subset) / len(subset) if subset else None for field in fields},
                    "seconds_total": sum(case["seconds"] for case in subset),
                    "planner_failures": sum(bool(case.get("planner_failure")) for case in subset)}
    return output


def prepare(dev_labels, dev_sequence_labels, human_packet, human_labels, output, planner="fixed"):
    units = load_units(dev_labels, dev_sequence_labels, human_packet, human_labels)
    if planner not in {"fixed", "api"}:
        raise ValueError("Unknown planner mode")
    generator = None
    if planner == "api":
        from kis.structure_query import create_query_structurer
        generator = create_query_structurer()  # Fail before scoring if not configured.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_files = ("tools/pipeline_ablation.py", "kis/solve_kis.py", "kis/ranking.py", "kis/objects.py",
                    "kis/text.py", "kis/visual.py", "kis/event_retrieval.py", "kis/temporal.py", "kis/candidate_pool.py",
                    "kis/vlm_reranker.py", "kis/structure_query.py", "config/models.json", "config/pipeline.json")
    protocol = {**PROTOCOL, "planner": planner,
                "planner_limit": "fixed approved events and literal detector labels; LLM planning not measured" if planner == "fixed" else "live production planner; failures included, no manual repair",
                "source_sha256": {name: sha(KIS_ROOT / name) for name in source_files},
                "label_sha256": {name: sha(path) for name, path in (("dev_pairs", dev_labels), ("dev_sequences", dev_sequence_labels), ("human_regression", human_labels))}}
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "source_snapshot.json", {name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in source_files})
    atomic_write_json(output / "units_private.json", units)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    visual = None
    try:
        from tools.temporal_pilot import require_model_cache
        require_model_cache()
        with (output / "retrieval.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            import torch
            import faiss
            from kis.config import load_caption_runtime
            from kis.objects import ObjectMetadataLookup
            from kis.solve_kis import solve_kis
            from kis.text import BM25Retriever
            from kis.visual import SiglipEncoder, SiglipRetriever
            torch.set_num_threads(4)
            faiss.omp_set_num_threads(4)
            runtime = load_runtime_config()
            profile = get_search_profile(runtime.retrieval_profile)
            visual = SiglipRetriever(video_groups=runtime.video_groups, nprobe=profile.nprobe)
            captions, _ = load_caption_runtime(runtime)
            encoder = CachedEncoder(SiglipEncoder(device="cpu"))
            bm25 = BM25Retriever(captions.caption_data, tokenizer_mode=profile.tokenizer_mode)
            objects = ObjectMetadataLookup()
            temporal = replace(runtime.temporal, condition_evidence="off")
            atomic_write_json(output / "runtime.json", {"temporal": temporal.public(), "profile": runtime.retrieval_profile,
                "nprobe": profile.nprobe, "tokenizer_mode": profile.tokenizer_mode, "planner": planner,
                "collection_id": visual.client.describe_collection(collection_name=visual.collection_name).get("collection_id"),
                "caption_vectors": int(captions.index.ntotal), "vlm_loaded": False})
            cases = []
            for unit in units:
                planner_failure = None
                try:
                    if planner == "fixed":
                        plan = fixed_plan(unit)
                    else:
                        from kis.structure_query import create_structured_query
                        plan = create_structured_query(unit["raw_query"], query_id=unit["query_id"], output_path=None, generator=generator)
                except Exception as exc:
                    # No API response text or credentials in benchmark artifacts.
                    plan, planner_failure = None, type(exc).__name__
                for arm in ARMS:
                    case_id = unit["query_id"] + "__" + arm
                    diagnostic, results = {}, []
                    started = perf_counter()
                    if plan is not None:
                        arm_query = arm_plan(plan, arm)
                        v, b, c = visual, bm25, captions
                        if unit["task"] == "sequence" and arm != "full":
                            actions = [event["action"] for event in plan.get("events", [])]
                            v = TemporalVariantVisual(visual, encoder, actions, arm, profile.policy, temporal.event_top_k)
                            b = TemporalVariantText(bm25, arm, profile.policy, temporal.event_top_k)
                            c = TemporalVariantText(captions, arm, profile.policy, temporal.event_top_k)
                        results = solve_kis(arm_query, v, encoder, b, c, objects, vlm_reranker=None,
                            retrieval_top_k=200 if arm == "full_deep" else 100, rerank_top_k=20,
                            result_top_k=300, policy=profile.policy, diagnostics=diagnostic, temporal_config=temporal)
                    # Hydrate identity for visual-only static hits without guessing duplicate frame IDs.
                    for result in results:
                        if result.get("keyframe_n") is None:
                            rows, _ = LocalEventScorer._video_data(result["video_id"])
                            matches = [row for row in rows if row["frame_id"] == result["frame_id"]]
                            if len(matches) == 1:
                                result.update({key: matches[0][key] for key in ("keyframe_n", "pts_time")})
                    case = {"case_id": case_id, "unit": unit, "arm": arm, "plan": plan,
                            "planner_failure": planner_failure, "seconds": perf_counter() - started,
                            "metrics": metrics(results, unit), "results": results, "diagnostics": diagnostic}
                    atomic_write_json(output / "cases" / (case_id + ".json"), case)
                    cases.append(case)
                    atomic_write_json(output / "cpu_summary.json", summarize(cases))
                    LocalEventScorer._video_data.cache_clear()
            atomic_write_json(output / "case_index.json", [{"case_id": case["case_id"], "task": case["unit"]["task"],
                "role": case["unit"]["role"], "arm": case["arm"]} for case in cases])
            for name, fingerprint in protocol["source_sha256"].items():
                if sha(KIS_ROOT / name) != fingerprint:
                    raise ValueError("A frozen source changed during ablation")
        print(json.dumps({"output": str(output.resolve()), "cases": len(cases), "planner": planner,
                          "vlm_loaded": False, "cpu_summary": summarize(cases)}, indent=2))
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__})
        raise
    finally:
        if visual is not None:
            visual.client.close()
        LocalEventScorer._video_data.cache_clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-labels", type=Path, required=True)
    parser.add_argument("--dev-sequence-labels", type=Path, required=True)
    parser.add_argument("--human-packet", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planner", choices=("fixed", "api"), default="fixed")
    args = parser.parse_args()
    prepare(args.dev_labels, args.dev_sequence_labels, args.human_packet, args.human_labels, args.output_dir, args.planner)
