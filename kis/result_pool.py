"""A full frame list is a contract, never padding or duplicated identities."""

import math


MAX_EXPANDED_TOP_K = 6400
WEB_RESULT_LIMIT = 200


class IncompleteResultPool(RuntimeError):
    def __init__(self, target, actual):
        self.target, self.actual = target, actual
        super().__init__(
            f"Chưa lấy đủ {target} frame khác nhau (hiện có {actual}). "
            "Kiểm tra scope/index/dữ liệu; hệ thống không nhân bản frame để bù số lượng.")


def unique_frames(results):
    values, seen = [], set()
    for item in results:
        key = (item["video_id"], int(item["frame_id"]))
        if key not in seen:
            if not math.isfinite(float(item["score"])):
                raise ValueError("Frame pool contains a non-finite score")
            values.append(item)
            seen.add(key)
    return values


def complete_frame_pool(initial, fetch, *, target=100, initial_k=100,
                        max_k=MAX_EXPANDED_TOP_K):
    """Expand actual retrieval depth only on shortfall; fail explicitly at budget.

    fetch(k) must return a fresh, ranked pool for the same query and dataset scope.
    It may contain more than target items so later reranking retains its pool.
    """
    if target < 1 or initial_k < 1 or max_k < initial_k:
        raise ValueError("Invalid frame pool budget")
    values, depth, attempts = unique_frames(initial), initial_k, 0
    while len(values) < target and depth < max_k:
        depth = min(max_k, max(depth * 2, target))
        values = unique_frames(fetch(depth))
        attempts += 1
    if len(values) < target:
        raise IncompleteResultPool(target, len(values))
    return values, {"status": "complete", "target": target, "unique_candidates": len(values),
                    "retrieval_depth": depth, "expansions": attempts}


def require_frame_count(results, target=100):
    """Final guard: a successful response contains exactly target unique frames."""
    count = len({(row["video_id"], int(row["frame_id"])) for row in results})
    if len(results) != target or count != target:
        raise IncompleteResultPool(target, count)
