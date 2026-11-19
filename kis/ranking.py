"""Fusion settings, reciprocal rank fusion and ranking helpers."""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import numpy as np
import pandas as pd

from kis.paths import video_clip_path, video_csv_path

MAX_QUERY_CHARS = 2000
MAX_CAPTION_CHARS = 3000


def bounded_prompt_text(value, max_chars):
    return str(value or "").strip().replace("<", "[").replace(">", "]")[:max_chars]


def parse_vlm_score(response):
    match = re.fullmatch(r"(?:10|[0-9])", str(response).strip())
    return float(match.group()) if match else None


def weighted_rrf(streams, weights=None, k=60):
    weights = weights or {"visual": 0.5, "bm25": 0.2, "semantic": 0.3}
    if k < 1 or set(streams) != set(weights):
        raise ValueError("RRF needs matching channel weights and a positive k")
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("RRF weights must be finite, non-negative and sum to one")
    candidates = {}
    for channel, results in streams.items():
        seen = set()
        for rank, item in enumerate(results, 1):
            if not all(math.isfinite(float(item.get(field, item["score"]))) for field in ("score", "norm_score", "raw_score")):
                raise ValueError("RRF evidence contains non-finite scores")
            key = (item["video_id"], item["frame_id"])
            if key in seen:
                continue
            seen.add(key)
            candidate = candidates.setdefault(key, {"video_id": key[0], "frame_id": key[1], "score": 0.0,
                "fusion_score": 0.0, "evidence": {}, **{f"{name}_score": 0.0 for name in weights}})
            for field, value in item.items():
                if field not in {"score", "norm_score", "evidence"} and candidate.get(field) in (None, ""):
                    candidate[field] = value
            contribution = weights[channel] * (k + 1) / (k + rank)
            candidate["fusion_score"] += contribution
            candidate[f"{channel}_score"] = float(item.get("norm_score", item["score"]))
            candidate["evidence"][channel] = {"rank": rank, "raw_score": float(item.get("raw_score", item["score"])),
                "normalized_score": float(item.get("norm_score", item["score"])), "rrf_contribution": contribution}
    for candidate in candidates.values():
        candidate["score"] = candidate["fusion_score"]
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)


@dataclass(frozen=True)
class RetrievalPolicy:
    route_queries: bool = False
    variant_fusion: str = "raw_max"
    flat_score: float = 1.0
    verified_first: bool = False
    cross_fusion: str = "weighted_sum"
    rrf_k: int = 60

    def __post_init__(self):
        if self.variant_fusion not in {"raw_max", "normalized_max", "rank_max"}:
            raise ValueError("Unknown variant fusion policy")
        if not 0.0 <= self.flat_score <= 1.0:
            raise ValueError("flat_score must be between 0 and 1")
        if self.cross_fusion not in {"weighted_sum", "rrf"} or self.rrf_k < 1:
            raise ValueError("Invalid cross-channel fusion configuration")


def max_score_fusion(multi_results):
    best = {}
    for results in multi_results:
        for item in results:
            key = (item["video_id"], item["frame_id"])
            if key not in best or float(item["score"]) > float(best[key]["score"]):
                best[key] = dict(item, score=float(item["score"]))
    return sorted(best.values(), key=lambda item: item["score"], reverse=True)


def normalize_scores(results, flat_score=1.0):
    if not results:
        return []
    scores = [float(item["score"]) for item in results]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Retrieval results contain non-finite scores")
    low, high = min(scores), max(scores)
    for item, score in zip(results, scores):
        item["norm_score"] = flat_score if high == low else (score - low) / (high - low)
    return results


def fuse_query_variants(multi_results, policy):
    if policy.variant_fusion == "raw_max":
        return normalize_scores(max_score_fusion(multi_results), flat_score=policy.flat_score)
    calibrated = []
    for results in multi_results:
        copied = [dict(item) for item in results]
        if policy.variant_fusion == "normalized_max":
            normalize_scores(copied, flat_score=policy.flat_score)
        for rank, item in enumerate(copied, 1):
            item["raw_score"] = item["score"]
            item["score"] = item["norm_score"] if policy.variant_fusion == "normalized_max" else 61.0 / (60.0 + rank)
        calibrated.append(copied)
    fused = max_score_fusion(calibrated)
    for item in fused:
        item["norm_score"] = item["score"]
    return fused


def static_weight_fusion_3_way(visual_results, bm25_results, semantic_results,
                              w_visual=0.5, w_bm25=0.2, w_semantic=0.3):
    weights = {"visual": w_visual, "bm25": w_bm25, "semantic": w_semantic}
    if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights.values()):
        raise ValueError("Fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Fusion weights must sum to 1")
    candidates = {}
    for channel, results in (("visual", visual_results), ("bm25", bm25_results), ("semantic", semantic_results)):
        for rank, item in enumerate(results, 1):
            key = (item["video_id"], item["frame_id"])
            candidate = candidates.setdefault(key, {"visual_score": 0.0, "bm25_score": 0.0, "semantic_score": 0.0, "evidence": {}})
            for field, value in item.items():
                if field not in {"score", "norm_score", "evidence"} and candidate.get(field) in (None, ""):
                    candidate[field] = value
            candidate[f"{channel}_score"] = item.get("norm_score", 0.0)
            candidate["evidence"][channel] = {"rank": rank, "raw_score": float(item.get("raw_score", item["score"])),
                                              "normalized_score": float(item.get("norm_score", 0.0))}
    for item in candidates.values():
        item["score"] = item["fusion_score"] = sum(item[f"{channel}_score"] * weight for channel, weight in weights.items())
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)


def finalize_vlm_ranking(candidates, top_k=20, vlm_weight=0.5, base_score_field="score"):
    if not 0.0 <= vlm_weight <= 1.0:
        raise ValueError("vlm_weight must be between 0 and 1")
    for item in candidates:
        retrieval = float(item.get(base_score_field, item["score"]))
        if not math.isfinite(retrieval):
            raise ValueError("Candidate contains a non-finite retrieval score")
        item["final_score"] = retrieval
    applied = False
    for item in candidates[:top_k]:
        raw_score = item.get("vlm_raw_score")
        if not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
            continue
        applied = True
        retrieval = max(0.0, min(1.0, float(item.get(base_score_field, item["score"]))))
        delta = max(0.0, min(10.0, float(raw_score))) / 10.0 - 0.5
        item["final_score"] = retrieval + 2.0 * vlm_weight * delta * (1.0 - retrieval if delta >= 0.0 else retrieval)
    return sorted(candidates, key=lambda item: item["final_score"], reverse=True), applied


@dataclass(frozen=True)
class SearchProfile:
    name: str
    tokenizer_mode: str
    nprobe: int
    policy: RetrievalPolicy
    rerank_top_k: int = 20


def get_search_profile(name):
    if name == "rrf":
        return SearchProfile(name, "unicode", 128, RetrievalPolicy(
            route_queries=True, variant_fusion="normalized_max", flat_score=0.0,
            cross_fusion="rrf", rrf_k=60,
        ))
    if name == "legacy":
        return SearchProfile(name, "legacy", 10, RetrievalPolicy())
    if name == "upgraded":
        return SearchProfile(name, "unicode", 128, RetrievalPolicy(
            route_queries=True, variant_fusion="normalized_max", flat_score=0.0,
        ))
    if name == "verified40":
        return SearchProfile(name, "unicode", 128, RetrievalPolicy(
            route_queries=True, variant_fusion="normalized_max", flat_score=0.0,
            verified_first=True,
        ), rerank_top_k=40)
    raise ValueError("Unknown KIS_RETRIEVAL_PROFILE; use rrf, upgraded, legacy or verified40")


def queries_for_stream(structured_query, stream, policy):
    fallback = structured_query.get("query_variants") or [structured_query.get("raw_query", "")]
    if not policy.route_queries:
        return fallback
    routed = structured_query.get("stream_queries", {}).get(stream)
    if routed is None:
        return fallback
    if not isinstance(routed, list) or not routed or any(
        not isinstance(value, str) or not value.strip() for value in routed
    ):
        raise ValueError(f"stream_queries.{stream} must be a non-empty string list")
    return list(dict.fromkeys(value.strip() for value in routed))


class FrameDiversitySelector:
    """Suppress similar nearby frames; this is a proxy, not a shot detector.

    Compare only frames from the same video within the time window. Keep the
    best retrieval-ranked representative and attach suppressed neighbors.
    Missing features never cause an otherwise valid candidate to disappear.
    """

    def __init__(self, similarity_threshold=0.96, window_seconds=12.0):
        if not 0.0 < similarity_threshold <= 1.0 or window_seconds <= 0:
            raise ValueError("Invalid diversity thresholds")
        self.similarity_threshold = similarity_threshold
        self.window_seconds = window_seconds

    @staticmethod
    @lru_cache(maxsize=64)
    def _video_data(video_id):
        csv_path, feature_path = video_csv_path(video_id), video_clip_path(video_id)
        if not csv_path.is_file() or not feature_path.is_file():
            return None
        rows = pd.read_csv(csv_path)
        vectors = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if len(rows) != len(vectors):
            raise ValueError(f"Feature/mapping count mismatch: {video_id}")
        return {
            int(row.frame_idx): (position, float(row.pts_time), int(row.n))
            for position, row in enumerate(rows.itertuples())
        }, vectors

    def describe(self, candidate):
        data = self._video_data(candidate["video_id"])
        if data is None:
            return None
        mapping, vectors = data
        row = mapping.get(int(candidate["frame_id"]))
        if row is None:
            return None
        position, timestamp, n = row
        candidate.setdefault("keyframe_n", n)
        candidate.setdefault("pts_time", timestamp)
        vector = np.array(vectors[position], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            return None
        return timestamp, vector / norm

    def select(self, candidates):
        kept = []
        by_video = {}
        for candidate in candidates:
            item = candidate.copy()
            description = self.describe(item)
            duplicate = None
            if description is not None:
                timestamp, vector = description
                for previous, previous_time, previous_vector in by_video.get(item["video_id"], []):
                    if abs(timestamp - previous_time) <= self.window_seconds and float(
                        np.dot(vector, previous_vector)
                    ) >= self.similarity_threshold:
                        duplicate = previous
                        break
            if duplicate is not None:
                duplicate.setdefault("suppressed_neighbors", []).append({
                    "video_id": item["video_id"], "frame_id": int(item["frame_id"]),
                    "keyframe_n": item.get("keyframe_n"), "pts_time": item.get("pts_time"),
                })
                continue
            kept.append(item)
            if description is not None:
                timestamp, vector = description
                by_video.setdefault(item["video_id"], []).append((item, timestamp, vector))
        return kept
