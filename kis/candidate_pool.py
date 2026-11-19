"""Video-level candidate selection, separate from event verification/alignment.

Rank evidence here is only a way to allocate local rescoring work. It does not
verify actions. Caption BM25/semantic remain a correlated family of evidence.
"""

from itertools import zip_longest
import math


STREAM_WEIGHTS = {"visual": 0.5, "bm25": 0.2, "semantic": 0.3}


def select_candidate_videos(event_hits, max_videos, strategy="round_robin", rrf_k=60, ocr_weight=0.5):
    if strategy not in {"round_robin", "event_rrf", "balanced_rrf"}:
        raise ValueError("Unknown temporal candidate pooling strategy")
    if max_videos < 1 or rrf_k < 1 or not event_hits:
        raise ValueError("Candidate pooling needs events and positive budgets")
    if not math.isfinite(ocr_weight) or not 0 <= ocr_weight <= 1:
        raise ValueError("Invalid OCR pool weight")
    streams, seeds, evidence = [], {}, {}
    for event_index, event in enumerate(event_hits):
        if set(event["streams"]) not in (set(STREAM_WEIGHTS), set(STREAM_WEIGHTS) | {"ocr"}):
            raise ValueError("Candidate pooling expects visual, bm25, semantic and optional OCR streams")
        weights = STREAM_WEIGHTS
        if event["streams"].get("ocr"):
            weights = {**{name: weight * (1 - ocr_weight) for name, weight in STREAM_WEIGHTS.items()}, "ocr": ocr_weight}
        for stream, weight in weights.items():
            hits = event["streams"][stream]
            streams.append(hits)
            seen = set()
            for frame_rank, hit in enumerate(hits, 1):
                video_id = hit["video_id"]
                seeds.setdefault(video_id, set()).add(int(hit["frame_id"]))
                if video_id in seen:
                    continue
                seen.add(video_id)
                # Rank distinct videos, not every repeated frame of long videos.
                video_rank = len(seen)
                record = evidence.setdefault(video_id, {"video_id": video_id, "event_scores": [0.0] * len(event_hits),
                                                        "streams": [], "reserved_for_events": []})
                contribution = weight * (rrf_k + 1) / (rrf_k + video_rank)
                record["event_scores"][event_index] += contribution
                record["streams"].append({"event_index": event_index, "stream": stream,
                                          "frame_rank": frame_rank, "video_rank": video_rank,
                                          "contribution": contribution})
    for record in evidence.values():
        scores = record["event_scores"]
        record.update(score=0.7 * sum(scores) / len(scores) + 0.3 * max(scores),
                      events_with_rank_evidence=sum(score > 0 for score in scores))
    ranked = sorted(evidence, key=lambda video_id: (-evidence[video_id]["score"], video_id))
    if strategy == "round_robin":
        selected = []
        for row in zip_longest(*streams):
            for hit in row:
                if hit is not None and hit["video_id"] not in selected and len(selected) < max_videos:
                    selected.append(hit["video_id"])
    elif strategy == "event_rrf":
        selected = ranked[:max_videos]
    else:
        # Reserve the winner of each event. If winners exceed budget, global
        # score breaks the tie, not event order. Never require intersection.
        reserved = set()
        for event_index in range(len(event_hits)):
            eligible = [video_id for video_id in ranked if evidence[video_id]["event_scores"][event_index] > 0]
            if eligible:
                best = min(eligible, key=lambda video_id: (-evidence[video_id]["event_scores"][event_index], video_id))
                evidence[best]["reserved_for_events"].append(event_index)
                reserved.add(best)
        selected = [video_id for video_id in ranked if video_id in reserved][:max_videos]
        selected.extend(video_id for video_id in ranked if video_id not in reserved)
        selected = selected[:max_videos]
    selected_set = set(selected)
    diagnostics = {"strategy": strategy, "rrf_k": rrf_k, "max_videos": max_videos,
                   "video_count_seen": len(seeds), "rank_evidence_is_verification": False,
                   "ranked_videos": [dict(evidence[video_id], selected=video_id in selected_set,
                                          global_rank=rank) for rank, video_id in enumerate(ranked, 1)]}
    return selected, seeds, diagnostics
