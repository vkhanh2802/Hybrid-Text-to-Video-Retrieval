"""Bounded per-event retrieval and local rescoring; no new model is loaded."""

import csv
from collections import Counter
from functools import lru_cache
from time import perf_counter

import numpy as np

from kis.paths import video_csv_path, video_clip_path
from kis.candidate_pool import select_candidate_videos
from kis.condition_evidence import apply_condition_evidence, caption_for_frame, check_conditions
from kis.temporal import align_events, evidence_scores


def _unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("Invalid local embedding")
    return values / norms


class LocalEventScorer:
    def __init__(self, encoder, captions, config):
        self.encoder, self.captions, self.config = encoder, captions, config
        self._semantic_queries = {}

    @staticmethod
    @lru_cache(maxsize=2)
    def _video_data(video_id):
        with video_csv_path(video_id).open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if not {"frame_idx", "n", "pts_time"}.issubset(reader.fieldnames or []):
                raise ValueError("Temporal search requires frame_idx, n, pts_time mapping")
            frames = [{"video_id": video_id, "frame_id": int(row["frame_idx"]),
                       "keyframe_n": int(row["n"]), "pts_time": float(row["pts_time"])} for row in reader]
        if (len({f["keyframe_n"] for f in frames}) != len(frames)
                or any(not np.isfinite(f["pts_time"]) or f["pts_time"] < 0 for f in frames)):
            raise ValueError("Invalid temporal mapping identity/timestamp")
        path = video_clip_path(video_id)
        vectors = np.load(path, mmap_mode="r", allow_pickle=False) if path.is_file() else None
        if vectors is not None and (vectors.ndim != 2 or len(vectors) != len(frames)):
            raise ValueError("Local visual feature/mapping count mismatch")
        return frames, vectors

    def score(self, video_id, actions, query_vectors, seed_frames):
        all_frames, vectors = self._video_data(video_id)
        limit = self.config.max_frames_per_video
        counts = Counter(frame["frame_id"] for frame in all_frames)
        # Legacy sampling may repeat frame_idx with distinct n/pts_time. The
        # retrievers identify by frame_idx, so neither row is safe to join.
        eligible = [i for i, frame in enumerate(all_frames) if counts[frame["frame_id"]] == 1]
        # Seed rows survive subsampling; remaining capacity covers the whole video.
        seed_positions = [i for i in eligible if all_frames[i]["frame_id"] in seed_frames][:limit]
        positions = set(seed_positions)
        remaining = [i for i in eligible if i not in positions]
        count = min(limit - len(positions), len(remaining))
        if count:
            positions.update(remaining[i] for i in np.linspace(0, len(remaining) - 1, count, dtype=int))
        positions = sorted(positions, key=lambda i: (all_frames[i]["pts_time"], all_frames[i]["frame_id"]))
        frames = [dict(all_frames[i]) for i in positions]
        visual = None
        if vectors is not None and frames:
            visual = _unit_vectors(query_vectors) @ _unit_vectors(vectors[positions]).T
        semantic = None
        score_frames = getattr(self.captions, "score_frames", None)
        if score_frames is not None and frames:
            for action in actions:
                if action not in self._semantic_queries:
                    self._semantic_queries[action] = self.captions.encode_query(action)[0]
            semantic = np.asarray([score_frames(action, frames, query_vector=self._semantic_queries[action]) for action in actions])
        shape = (len(actions), len(frames))
        scores, supported = evidence_scores(visual, semantic, shape, self.config)
        supported, condition_stats = apply_condition_evidence(
            actions, frames, self.captions, supported, visual, self.config)
        return frames, scores, supported, visual, semantic, {
            "total_frames": len(all_frames), "scored_frames": len(frames),
            "sampled": len(frames) < len(all_frames),
            "ambiguous_frame_rows_excluded": len(all_frames) - len(eligible),
            "seed_frames_total": len(seed_frames),
            "seed_frames_scored": sum(frame["frame_id"] in seed_frames for frame in frames),
            "visual_available": visual is not None,
            "semantic_available": semantic is not None and bool(np.isfinite(semantic).any()),
            "condition_evidence": condition_stats,
        }


def retrieve_event_segments(plan, visual, encoder, bm25, captions, config, diagnostics, scorer=None,
                            ocr_hits=None, ocr_weight=0.5):
    started = perf_counter()
    actions = plan.actions
    vectors = [encoder.encode_text(action) for action in actions]
    event_hits = []
    for event_index, (action, vector) in enumerate(zip(actions, vectors)):
        hits = {"visual": visual.search(vector, top_k=config.event_top_k),
                "bm25": bm25.search_bm25(action, top_k=config.event_top_k),
                 "semantic": captions.search(action, top_k=config.event_top_k)}
        if ocr_hits and ocr_hits.get(event_index):
            hits["ocr"] = ocr_hits[event_index]
        event_hits.append({"action": action, "streams": hits})
    video_ids, seeds, pool_diagnostics = select_candidate_videos(
        event_hits, config.max_videos, strategy=config.candidate_pooling, rrf_k=config.candidate_rrf_k,
        ocr_weight=ocr_weight,
    )
    scorer = scorer or LocalEventScorer(encoder, captions, config)
    segments, failures, video_stats = [], {}, {}
    for video_id in video_ids:
        try:
            frames, scores, supported, vis, sem, stats = scorer.score(video_id, actions, vectors, seeds[video_id])
            video_stats[video_id] = stats
            alignment = align_events(frames, scores, supported, config)
        except (OSError, ValueError, RuntimeError) as exc:
            failures[video_id] = str(exc)
            continue
        if alignment is None:
            continue
        by_event = dict(alignment["path"])
        evidence = []
        for index, action in enumerate(actions):
            frame_index = by_event.get(index)
            item = {"action": action, "status": "unknown", "frame_id": None, "pts_time": None,
                    "keyframe_n": None, "score": None, "visual_similarity": None, "semantic_similarity": None}
            if frame_index is not None:
                item.update(frames[frame_index], status="retrieved", score=float(scores[index, frame_index]))
                for name, matrix in (("visual_similarity", vis), ("semantic_similarity", sem)):
                    value = float(matrix[index, frame_index]) if matrix is not None else float("nan")
                    item[name] = value if np.isfinite(value) else None
                if config.condition_evidence != "off":
                    item["condition_evidence"] = check_conditions(
                         action, caption_for_frame(captions, frames[frame_index]))
                if ocr_hits:
                    match = next((hit for hit in ocr_hits.get(index, []) if hit["video_id"] == video_id
                                  and hit["frame_id"] == item["frame_id"]), None)
                    if match is not None:
                        item["ocr"] = match["ocr"]
            evidence.append(item)
        first, last = alignment["path"][0][1], alignment["path"][-1][1]
        representative = dict(frames[first])
        get_caption = getattr(captions, "get_caption", None)
        metadata = get_caption(video_id, representative["frame_id"]) if get_caption else None
        if metadata:
            for field in ("retrieval_text", "caption", "image_path", "image_sha256"):
                if field in metadata:
                    representative[field] = metadata[field]
        representative.update(score=alignment["score"], final_score=alignment["score"], vlm_status="skipped_temporal",
                              temporal={"status": alignment["status"], "verified": False,
                                        "support_basis": "heuristic_cosine_thresholds", "coverage": alignment["coverage"],
                                        "quality": alignment["quality"], "score": alignment["score"],
                                        "start_seconds": frames[first]["pts_time"], "end_seconds": frames[last]["pts_time"],
                                        "boundary_type": "evidence_span", "events": evidence, **stats})
        segments.append(representative)
    diagnostics.update(status="applied" if segments else "fallback", events=event_hits,
                       candidate_pool=pool_diagnostics,
                       candidate_videos=video_ids, candidate_videos_seen=len(seeds),
                       failures=failures, video_stats=video_stats,
                       seconds=perf_counter() - started, config=config.public())
    return sorted(segments, key=lambda item: (-item["score"], item["video_id"]))
