"""CPU ordered-event alignment over timestamped, locally scored keyframes.

Support thresholds are engineering heuristics, NOT calibrated probabilities or
verification that an action occurred. Unsupported events may be skipped.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool = True
    max_events: int = 6
    event_top_k: int = 50
    max_videos: int = 12
    candidate_pooling: str = "round_robin"
    candidate_rrf_k: int = 60
    max_frames_per_video: int = 1500
    max_gap_seconds: float = 60.0
    gap_penalty: float = 0.001
    min_visual_similarity: float = 0.25
    min_semantic_similarity: float = 0.45
    condition_evidence: str = "off"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Temporal enabled must be a boolean")
        if self.candidate_pooling not in {"round_robin", "event_rrf", "balanced_rrf"}:
            raise ValueError("Unknown temporal candidate pooling strategy")
        if self.condition_evidence not in {"off", "diagnostics", "semantic_guard"}:
            raise ValueError("Unknown temporal condition evidence mode")
        for field in ("max_events", "event_top_k", "max_videos", "max_frames_per_video", "candidate_rrf_k"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("max_gap_seconds", "gap_penalty", "min_visual_similarity", "min_semantic_similarity"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.max_gap_seconds <= 0 or self.gap_penalty < 0:
            raise ValueError("Invalid temporal gap configuration")
        if any(not -1 <= value <= 1 for value in (self.min_visual_similarity, self.min_semantic_similarity)):
            raise ValueError("Similarity thresholds must be in [-1, 1]")

    def public(self):
        return asdict(self)


def evidence_scores(visual, semantic, shape, config):
    """Combine available raw cosine similarities without per-query min/max."""
    total, weight = np.zeros(shape), np.zeros(shape)
    supported = np.zeros(shape, dtype=bool)
    for values, channel_weight, threshold in (
        (visual, 0.6, config.min_visual_similarity),
        (semantic, 0.4, config.min_semantic_similarity),
    ):
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != shape:
            raise ValueError("Event evidence dimensions do not match the frame table")
        valid = np.isfinite(values)
        total += np.where(valid, np.clip(values, 0, 1), 0) * channel_weight
        weight += valid * channel_weight
        supported |= valid & (values >= threshold)
    scores = np.divide(total, weight, out=np.zeros(shape), where=weight > 0)
    return scores, supported


def align_events(frames, scores, supported, config):
    """O(events * frames) DP, lexicographic support-count then penalized score.

    Each state ends at a particular timestamp; skipping an event retains it.
    Transitions require strictly later time and a bounded gap. Equal timestamps
    cannot satisfy two ordered events. Ties keep the earlier deterministic path.
    """
    times = [float(frame["pts_time"]) for frame in frames]
    if any(not math.isfinite(t) or t < 0 for t in times) or times != sorted(times):
        raise ValueError("Temporal frames must have sorted finite, non-negative timestamps")
    scores, supported = np.asarray(scores, dtype=float), np.asarray(supported, dtype=bool)
    if scores.ndim != 2 or scores.shape[1] != len(frames) or supported.shape != scores.shape:
        raise ValueError("Invalid event score matrix")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Event scores must be finite and in [0, 1]")
    # state = (support count, score minus gap penalties, ((event, frame), ...))
    previous = [None] * len(frames)
    for event in range(scores.shape[0]):
        current = list(previous)  # Explicit abstention for this event.
        queue, cursor = deque(), 0
        for frame, timestamp in enumerate(times):
            while cursor < frame and times[cursor] < timestamp:
                state = previous[cursor]
                if state is not None:
                    key = (state[0], state[1] + config.gap_penalty * times[cursor])
                    while queue and queue[-1][1] < key:
                        queue.pop()
                    queue.append((cursor, key))
                cursor += 1
            while queue and times[queue[0][0]] < timestamp - config.max_gap_seconds:
                queue.popleft()
            if not supported[event, frame]:
                continue
            candidate = (1, float(scores[event, frame]), ((event, frame),))
            if queue:
                last = queue[0][0]
                state = previous[last]
                candidate = (state[0] + 1, state[1] + float(scores[event, frame])
                             - config.gap_penalty * (timestamp - times[last]),
                             state[2] + ((event, frame),))
            if current[frame] is None or candidate[:2] > current[frame][:2]:
                current[frame] = candidate
        previous = current
    valid = [state for state in previous if state is not None]
    if not valid:
        return None
    best = max(valid, key=lambda state: state[:2])
    coverage = best[0] / scores.shape[0]
    quality = max(0.0, min(1.0, best[1] / best[0]))
    return {"path": best[2], "coverage": coverage, "quality": quality,
            # Support count dominates quality across videos too.
            "score": (best[0] + 0.5 * quality) / (scores.shape[0] + 0.5),
            "status": "retrieved_complete" if best[0] == scores.shape[0] else "partial"}
