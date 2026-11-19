"""Pure metrics/selection for provisional event-to-frame support judgments.

Select on development only. Unknown judgments are never negatives. The selected
thresholds remain an experimental profile, not a calibrated probability.
"""

import math


VISUAL_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3)
SEMANTIC_GRID = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


def gate(visual, semantic, visual_threshold, semantic_threshold):
    return any(value is not None and math.isfinite(value) and value >= threshold
               for value, threshold in ((visual, visual_threshold), (semantic, semantic_threshold)))


def support_metrics(rows, visual_threshold, semantic_threshold):
    by_query, skipped = {}, 0
    for row in rows:
        if row["judgment"] == "unknown":
            skipped += 1
            continue
        if row["judgment"] not in {"positive", "negative"}:
            raise ValueError("Unknown judgment type")
        group = by_query.setdefault(row["query_id"], {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        prediction = gate(row.get("visual"), row.get("semantic"), visual_threshold, semantic_threshold)
        positive = row["judgment"] == "positive"
        group["tp" if positive and prediction else "fn" if positive else "fp" if prediction else "tn"] += 1
    eligible = []
    for key, group in by_query.items():
        positives, negatives = group["tp"] + group["fn"], group["fp"] + group["tn"]
        if positives and negatives:
            tpr, fpr = group["tp"] / positives, group["fp"] / negatives
            eligible.append({"query_id": key, **group, "tpr": tpr, "fpr": fpr,
                             "balanced_accuracy": (tpr + 1 - fpr) / 2})
    count = len(eligible)
    return {"eligible_queries": count, "queries_without_both_classes": len(by_query) - count,
            "unknown_rows_skipped": skipped,
            "macro_tpr": sum(group["tpr"] for group in eligible) / count if count else None,
            "macro_fpr": sum(group["fpr"] for group in eligible) / count if count else None,
            "macro_balanced_accuracy": sum(group["balanced_accuracy"] for group in eligible) / count if count else None,
            "totals": {field: sum(group[field] for group in by_query.values()) for field in ("tp", "tn", "fp", "fn")},
            "by_query": eligible}


def choose_support_thresholds(rows, baseline=(0.25, 0.45), max_macro_fpr=0.1, min_macro_tpr=0.5):
    if not rows or any(row["split"] != "dev" for row in rows):
        raise ValueError("Threshold selection requires development-only scored rows")
    candidates = []
    for visual in VISUAL_GRID:
        for semantic in SEMANTIC_GRID:
            metrics = support_metrics(rows, visual, semantic)
            if (metrics["eligible_queries"] and metrics["macro_fpr"] <= max_macro_fpr
                    and metrics["macro_tpr"] >= min_macro_tpr):
                candidates.append({"visual_threshold": visual, "semantic_threshold": semantic, "metrics": metrics})
    baseline_metrics = support_metrics(rows, *baseline)
    if not candidates:
        return {"status": "no_safe_candidate", "selected": None, "baseline_metrics": baseline_metrics,
                "selection_rule": {"max_macro_fpr": max_macro_fpr, "min_macro_tpr": min_macro_tpr}}
    selected = max(candidates, key=lambda candidate: (
        candidate["metrics"]["macro_balanced_accuracy"], -candidate["metrics"]["macro_fpr"],
        -(abs(candidate["visual_threshold"] - baseline[0]) + abs(candidate["semantic_threshold"] - baseline[1])),
        candidate["visual_threshold"], candidate["semantic_threshold"]))
    return {"status": "experimental_profile_selected", "selected": selected, "baseline_metrics": baseline_metrics,
            "selection_rule": {"max_macro_fpr": max_macro_fpr, "min_macro_tpr": min_macro_tpr},
            "grid": {"visual": VISUAL_GRID, "semantic": SEMANTIC_GRID}, "qualifying_candidates": len(candidates)}
