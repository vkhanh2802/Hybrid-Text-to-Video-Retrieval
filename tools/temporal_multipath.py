"""Experimental bounded k-best temporal DAG. Not imported by production.

Exact k-best within the selected event/frame DAG, not over all corpus frames.
Always pin the unchanged production path first, including its tie behaviour.
No labels, captions, encoders or VLM are used to select alternative paths.
"""

import numpy as np

from kis.temporal import align_events


def _bound(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def candidate_mask(scores, supported, baseline, event_frame_budget):
    """Top scores per event, plus at most one pinned baseline frame per event."""
    mask = np.zeros_like(supported, dtype=bool)
    for event in range(len(scores)):
        indices = sorted(np.flatnonzero(supported[event]), key=lambda i: (-scores[event, i], int(i)))
        mask[event, indices[:event_frame_budget]] = True
    if baseline is not None:
        for event, frame in baseline["path"]:
            mask[event, frame] = True
    return mask


def _alignment(state, event_count):
    count, objective, path = state
    quality = max(0.0, min(1.0, objective / count))
    return {"path": path, "coverage": count / event_count, "quality": quality,
            "score": (count + 0.5 * quality) / (event_count + 0.5),
            "status": "retrieved_complete" if count == event_count else "partial"}


def _key(state):
    # Deterministic alternatives: count first, un-clipped objective second,
    # then event/frame tuple. Production ties are preserved by the pinned path.
    return -state[0], -state[1], state[2]


def align_multiple(frames, scores, supported, config, *, event_frame_budget=32,
                   search_paths=32, max_paths=4, diverse=False):
    """Return pinned baseline + bounded alternatives, and explicit work counts.

    Nodes are (event, frame). Edges advance event AND timestamp, can skip events,
    and obey the production max gap between supported frames. Keeping the best
    K prefixes at each node is exact for the top K DAG paths: all suffixes from
    a shared ending node have identical future feasibility and additive costs.
    Complexity O((events * (frame_budget+1))**2 * search_paths), bounded to a
    small experimental DAG. Diversity is a greedy output selector over that
    fixed search set; it does not claim globally optimal diverse paths.
    """
    _bound(event_frame_budget, "event_frame_budget", 128)
    _bound(search_paths, "search_paths", 128)
    _bound(max_paths, "max_paths", search_paths)
    if not isinstance(diverse, bool):
        raise ValueError("diverse must be boolean")
    # Reuse all production validation, score semantics and top1 tie rules.
    baseline = align_events(frames, scores, supported, config)
    scores, supported = np.asarray(scores, dtype=float), np.asarray(supported, dtype=bool)
    if len(frames) > config.max_frames_per_video or len(scores) > config.max_events:
        raise ValueError("Experimental input exceeds configured event/frame bounds")
    mask = candidate_mask(scores, supported, baseline, event_frame_budget)
    nodes, all_states, transitions = [], [], 0
    times = [float(frame["pts_time"]) for frame in frames]
    for event in range(len(scores)):
        current = []
        for frame in np.flatnonzero(mask[event]):
            frame = int(frame)
            candidates = [(1, float(scores[event, frame]), ((event, frame),))]
            for last_event, last_frame, prefixes in nodes:
                gap = times[frame] - times[last_frame]
                if last_event >= event or not 0 < gap <= config.max_gap_seconds:
                    continue
                for count, objective, path in prefixes:
                    candidates.append((count + 1, objective + float(scores[event, frame])
                                       - config.gap_penalty * gap, path + ((event, frame),)))
                    transitions += 1
            states = sorted(candidates, key=_key)[:search_paths]
            current.append((event, frame, states))
            all_states.extend(states)
        nodes.extend(current)
    ranked = sorted(all_states, key=_key)[:search_paths]
    alternatives = [_alignment(state, len(scores)) for state in ranked]
    # Baseline might lie outside the top K after deterministic alternative ties.
    pool, seen = [], set()
    for aligned in ([baseline] if baseline is not None else []) + alternatives:
        if aligned["path"] not in seen:
            seen.add(aligned["path"])
            pool.append(aligned)
    selected = pool[:1]
    if diverse and max_paths > 1:
        # Prefer <=50% identical event/frame pairs against every selected path.
        for aligned in pool[1:]:
            if aligned["coverage"] != baseline["coverage"]:
                continue
            path = set(aligned["path"])
            if all(len(path & set(other["path"])) / max(len(path), len(other["path"])) <= 0.5
                   for other in selected):
                selected.append(aligned)
                if len(selected) == max_paths:
                    break
    for aligned in pool[1:]:
        if len(selected) >= max_paths:
            break
        if aligned not in selected:
            selected.append(aligned)
    # For max_paths=1, diversity must not append beyond the pinned baseline.
    selected = selected[:max_paths]
    return selected, {"event_frame_budget": event_frame_budget, "search_paths": search_paths,
        "max_paths": max_paths, "diverse": diverse, "selected_nodes": int(mask.sum()),
        "selected_nodes_by_event": mask.sum(axis=1).tolist(), "prefix_states": len(all_states),
        "transitions": transitions, "search_pool_paths": len(pool), "returned_paths": len(selected)}


def compact_path(video_id, frames, aligned, event_count):
    selected = dict(aligned["path"])
    first = frames[aligned["path"][0][1]]
    events = [dict(frames[selected[event]], status="retrieved") if event in selected else
              {"status": "unknown", "keyframe_n": None, "frame_id": None, "pts_time": None}
              for event in range(event_count)]
    return {"video_id": video_id, "frame_id": first["frame_id"], "keyframe_n": first["keyframe_n"],
            "score": aligned["score"], "temporal": {"events": events, "coverage": aligned["coverage"],
            "quality": aligned["quality"], "status": aligned["status"], "verified": False}}
