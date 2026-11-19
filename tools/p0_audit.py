"""Read-only, deterministic audit of one saved retrieval evidence run.

The audit never loads an encoder, retriever, VLM or live Milvus client. It reads
saved rows, raw per-video evidence, frozen labels and local artifact files only.
Missing or malformed inputs are reported explicitly; they are never silently
treated as a zero recall denominator.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from kis.paths import KIS_ROOT, VIDEO_ID_PATTERN, video_group
from kis.temporal import TemporalConfig, align_events, evidence_scores


REPORT_SCHEMA = "kis_p0_audit_v1"
CONTEXT_MODE = "frozen_human_holdout"
LABEL_SCHEMA = "kis_human_holdout_labels_v1"
STREAMS = ("visual", "bm25", "semantic")
MODES = ("baseline", "semantic_guard")
STAGE_ORDER = (
    "visual_hits",
    "bm25_hits",
    "semantic_hits",
    "union_hits",
    "candidate_pool",
    "local_frames",
    "support",
    "temporal_feasibility",
    "saved_top_1",
    "saved_top_5",
    "saved_top_20",
)
STAGE_UNITS = {
    "visual_hits": "positive_frames",
    "bm25_hits": "positive_frames",
    "semantic_hits": "positive_frames",
    "union_hits": "positive_frames",
    "candidate_pool": "positive_videos",
    "local_frames": "positive_frames",
    "support": "positive_frames",
    "temporal_feasibility": "queries",
    "saved_top_1": "queries",
    "saved_top_5": "queries",
    "saved_top_20": "queries",
}
SEQUENCE_COMPLETE_STATUS = "retrieved_complete"
MANIFEST_HASH_FIELDS = ("caption_sha256", "mapping_sha256", "index_sha256")
TEMPORAL_CONFIG_FIELDS = (
    "enabled",
    "max_events",
    "event_top_k",
    "max_videos",
    "candidate_pooling",
    "candidate_rrf_k",
    "max_frames_per_video",
    "max_gap_seconds",
    "gap_penalty",
    "min_visual_similarity",
    "min_semantic_similarity",
    "condition_evidence",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MODELS_CONFIG_RELATIVE = "config/models.json"
LIMITATIONS = (
    "Read-only saved-artifact audit: no encoder, retriever, caption model, VLM, planner or live Milvus client is loaded or rerun.",
    "Only explicit human positive judgments form stage denominators. Negative, unknown, unjudged, missing and out-of-scope values are never counted as zero recall.",
    "Raw evidence stores only visual/semantic captured scores and frames; it does not store the semantic_guard per-frame removal mask, so guard support is reported as unknown.",
    "Saved result lists are capped at 20 entries per mode; rankings or hits beyond the saved rows cannot be audited.",
    "Historical model/embedding expectations are used only when the saved context pins a models-config hash that matches the local config file; the current checkout configuration is reported separately and never treated as historical.",
    "The historical models-config hash proves file identity only when the local file matches; it does not prove that live models, Milvus or the planner behaved as intended.",
    "Artifact checks cover local filesystem copies; they do not prove that the original run read exactly these copies, nor that live Milvus, the planner or a VLM behaved as intended.",
)


class AuditError(ValueError):
    """Fatal source/label error: no audit report can be produced."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_pairs(pairs):
    digest = hashlib.sha256()
    for name, value in sorted(pairs):
        digest.update(f"{name}\0{value}\n".encode("utf-8"))
    return digest.hexdigest()


def _json_load(path, code):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"{code}: missing file {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{code}: cannot read {path}: {exc}") from exc


def _ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def _integer(value, field):
    """Parse integer JSON/CSV values, rejecting fractions, bools and non-finite values."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field} must be an integer")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(f"{field} must be an integer")
        return int(text)
    raise ValueError(f"{field} must be an integer")


def _number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _video_id(value, field="video_id"):
    if not isinstance(value, str) or not VIDEO_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a canonical video id: {value!r}")
    return value


def _issue(code, detail):
    return {"code": code, "detail": detail}


def _sorted_issues(issues):
    return sorted(issues, key=lambda item: (item["code"], item["detail"]))


def _merge_issues(*groups):
    seen = set()
    merged = []
    for group in groups:
        for item in group:
            key = (item["code"], item["detail"])
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return _sorted_issues(merged)


def _relative(path, root):
    path, root = Path(path), Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return _normalize_path(path)


def _normalize_path(value):
    return str(value).replace("\\", "/")


def _normalized_path_map(mapping):
    if not isinstance(mapping, dict):
        return {}
    return {_normalize_path(key): mapping[key] for key in sorted(mapping)}


def _in_groups(video_id, groups):
    return video_id.split("_", 1)[0] in set(groups)


def _index_root(root):
    index = {}
    root = Path(root)
    if not root.is_dir():
        return index
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path.resolve())
    return {name: sorted(paths, key=str) for name, paths in index.items()}


def _resolve(index, video_id, suffix):
    return list(index.get(f"{video_id}{suffix}", []))


def _inventory(index, suffix):
    paths = [path for name, items in index.items() if name.endswith(suffix) for path in items]
    return len(paths)


def _count_json_files(root, suffix=".json"):
    root = Path(root)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob(f"*{suffix}") if path.is_file())


def _current_config():
    result = {
        "status": "ok",
        "source": "config/pipeline.json",
        "scope": "current_checkout_snapshot_not_historical_run",
    }
    try:
        from kis.config import load_runtime_config

        runtime = load_runtime_config(environ={})
        pipeline_path = KIS_ROOT / "config" / "pipeline.json"
        models_path = KIS_ROOT / "config" / "models.json"
        models = json.loads(models_path.read_text(encoding="utf-8"))
        result.update(
            pipeline_path=_relative(pipeline_path, KIS_ROOT),
            pipeline_sha256=_sha256_file(pipeline_path),
            models_path=_relative(models_path, KIS_ROOT),
            models_sha256=_sha256_file(models_path),
            temporal_config=runtime.temporal.public(),
            video_groups=list(runtime.video_groups),
            retrieval_profile=runtime.retrieval_profile,
            ocr_mode=getattr(runtime.ocr, "mode", None),
            text_reranker_mode=getattr(runtime.text_reranker, "mode", None),
            visual_embedding=models.get("visual_embedding"),
            caption_embedding=models.get("caption_embedding"),
            fast_caption_generation=models.get("fast_caption_generation"),
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result.update(status="unknown", reason=f"{type(exc).__name__}: {exc}")
    return result


def _captured_model_config(context):
    """Return historical embedding config only when its saved hash verifies.

    The saved run pins ``config/models.json`` in its context/protocol source
    hashes.  A local file that does not match that hash cannot stand in for the
    historical configuration, so the caller must treat embeddings, dimensions
    and revisions as unknown rather than using the current checkout snapshot.
    """
    result = {
        "status": "unknown",
        "source": None,
        "path": None,
        "sha256": None,
        "expected_sha256": None,
        "reason": None,
        "visual_embedding": None,
        "caption_embedding": None,
        "fast_caption_generation": None,
    }
    if not isinstance(context, dict):
        result["reason"] = "context is not an object"
        return result
    expected = None
    direct = context.get("models_sha256")
    if isinstance(direct, str) and direct:
        expected = direct
        result["source"] = "context.models_sha256"
    if expected is None:
        sources = [("context.source_sha256", context.get("source_sha256"))]
        protocol = context.get("protocol")
        if isinstance(protocol, dict):
            sources.append(("context.protocol.source_sha256",
                            protocol.get("source_sha256")))
        for source_name, source in sources:
            if not isinstance(source, dict):
                continue
            for key in (MODELS_CONFIG_RELATIVE, MODELS_CONFIG_RELATIVE.replace("/", "\\")):
                value = source.get(key)
                if isinstance(value, str) and value:
                    expected = value
                    result["source"] = f"{source_name}[{MODELS_CONFIG_RELATIVE}]"
                    break
            if expected is not None:
                break
    if expected is None:
        result["reason"] = "captured models-config hash is absent"
        return result
    result["expected_sha256"] = expected
    if not SHA256_PATTERN.fullmatch(expected):
        result["status"] = "invalid"
        result["reason"] = "captured models-config hash is malformed"
        return result
    path = Path(KIS_ROOT) / MODELS_CONFIG_RELATIVE
    result["path"] = _relative(path, KIS_ROOT)
    if not path.is_file():
        result["reason"] = "local models config is missing"
        return result
    try:
        actual = _sha256_file(path)
        result["sha256"] = actual
        if actual != expected:
            result["status"] = "unverified"
            result["reason"] = "local models config hash differs from captured hash"
            return result
        models = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(models, dict):
            raise ValueError("models config root must be an object")
        for field in ("visual_embedding", "caption_embedding", "fast_caption_generation"):
            if models.get(field) is not None and not isinstance(models[field], dict):
                raise ValueError(f"{field} must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["status"] = "invalid"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        status="verified",
        visual_embedding=models.get("visual_embedding"),
        caption_embedding=models.get("caption_embedding"),
        fast_caption_generation=models.get("fast_caption_generation"),
    )
    return result


def _expected_sequences(labels, event_by_id, errors):
    sequences = []
    annotations = labels.get("human_annotations")
    samples = annotations.get("samples") if isinstance(annotations, dict) else None
    if not isinstance(samples, list):
        errors.append(_issue("label_annotations_malformed", "human_annotations.samples must be an array"))
        return sequences
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            errors.append(_issue("label_sample_malformed", "sample must be an object"))
            continue
        sample_id = sample.get("sample_id")
        if sample.get("status") != "reviewed" or not isinstance(sample.get("events"), list) or len(sample["events"]) < 2:
            continue
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            errors.append(_issue("label_sample_id_invalid", f"invalid/duplicate sample_id {sample_id!r}"))
            continue
        seen.add(sample_id)
        try:
            video_id = _video_id(sample.get("video_id"), "sample video_id")
        except ValueError as exc:
            errors.append(_issue("label_sample_video_invalid", str(exc)))
            continue
        event_ids, actions, positives, event_texts = [], [], [], []
        valid = True
        for offset, event in enumerate(sample["events"]):
            event_id = f"{sample_id}_event_{offset + 1}"
            if not isinstance(event, dict) or event_id not in event_by_id:
                errors.append(_issue("sequence_event_missing", event_id))
                valid = False
                continue
            label_query = event_by_id[event_id]
            text = label_query.get("event_text")
            if not isinstance(text, str) or not text.strip():
                errors.append(_issue("label_event_text_missing", event_id))
                valid = False
                continue
            if "text_en" in event and event["text_en"] != text:
                errors.append(_issue("sequence_event_text_mismatch", event_id))
                valid = False
                continue
            allowed = sorted({int(item["keyframe_n"]) for item in label_query["_positives"]
                               if item["video_id"] == video_id})
            if not allowed:
                errors.append(_issue("sequence_event_without_target_positive", event_id))
                valid = False
            event_ids.append(event_id)
            actions.append(text)
            positives.append(allowed)
            event_texts.append(event.get("text_en", text))
        if valid:
            sequences.append({
                "query_id": f"{sample_id}_sequence",
                "video_id": video_id,
                "event_query_ids": event_ids,
                "actions": actions,
                "event_relevant_keyframes": positives,
                "sample_id": sample_id,
            })
    return sequences


def _load_labels(path, errors):
    labels = _json_load(path, "labels_unreadable")
    if not isinstance(labels, dict):
        raise AuditError("labels_malformed: labels root must be an object")
    if labels.get("schema") != LABEL_SCHEMA:
        raise AuditError(f"label_schema_unsupported: expected {LABEL_SCHEMA}")
    if (labels.get("annotation_source") != "human_visual_observation"
            or labels.get("human_verified") is not True
            or labels.get("caption_used_for_annotation") is not False):
        raise AuditError("label_provenance_invalid: expected frozen human image-only labels")
    queries = labels.get("queries")
    if not isinstance(queries, list) or not queries:
        raise AuditError("label_queries_missing: expected a non-empty queries array")

    event_by_id, event_queries = {}, []
    for query in queries:
        if not isinstance(query, dict):
            errors.append(_issue("label_query_malformed", "query must be an object"))
            continue
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in event_by_id:
            errors.append(_issue("label_query_id_invalid", repr(query_id)))
            continue
        try:
            video_id = _video_id(query.get("video_id"))
            text = query.get("event_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("event_text is empty")
        except ValueError as exc:
            errors.append(_issue("label_query_invalid", f"{query_id}: {exc}"))
            continue
        judgments = query.get("judgments")
        if not isinstance(judgments, list) or not judgments:
            errors.append(_issue("label_judgments_missing", query_id))
            judgments = []
        positives, negatives, unknown, frame_keys, keyframe_keys = [], 0, 0, set(), set()
        normalized = []
        for judgment in judgments:
            if not isinstance(judgment, dict):
                errors.append(_issue("label_judgment_malformed", f"{query_id}: judgment must be an object"))
                continue
            try:
                item_video = _video_id(judgment.get("video_id"), "judgment video_id")
                frame_id = _integer(judgment.get("frame_id"), "judgment frame_id")
                keyframe_n = _integer(judgment.get("keyframe_n"), "judgment keyframe_n")
                pts_time = _number(judgment.get("pts_time"), "judgment pts_time")
                if frame_id < 0 or keyframe_n < 1 or pts_time < 0:
                    raise ValueError("judgment identity/timestamp is outside its valid range")
                status = judgment.get("judgment")
                if status not in {"positive", "negative", "unknown"}:
                    raise ValueError(f"unsupported judgment {status!r}")
            except ValueError as exc:
                errors.append(_issue("label_judgment_invalid", f"{query_id}: {exc}"))
                continue
            frame_key, keyframe_key = (item_video, frame_id), (item_video, keyframe_n)
            if frame_key in frame_keys:
                errors.append(_issue("duplicate_label_frame_identity", f"{query_id}: {frame_key}"))
            if keyframe_key in keyframe_keys and status == "positive":
                errors.append(_issue("duplicate_label_keyframe_identity", f"{query_id}: {keyframe_key}"))
            frame_keys.add(frame_key)
            if status == "positive":
                keyframe_keys.add(keyframe_key)
            normalized.append({
                "video_id": item_video,
                "frame_id": frame_id,
                "keyframe_n": keyframe_n,
                "pts_time": pts_time,
                "image_sha256": judgment.get("image_sha256"),
                "judgment": status,
                "sample_id": judgment.get("sample_id"),
            })
            if status == "positive":
                positives.append(normalized[-1])
            elif status == "negative":
                negatives += 1
            else:
                unknown += 1
        query = dict(query)
        query["_positives"] = sorted(positives, key=lambda item: (item["video_id"], item["frame_id"], item["keyframe_n"]))
        query["_judgment_counts"] = {"positive": len(positives), "negative": negatives, "unknown": unknown}
        event_by_id[query_id] = query
        event_queries.append(query)

    sequences = _expected_sequences(labels, event_by_id, errors)
    if not event_queries:
        raise AuditError("label_queries_missing: no valid event queries")
    return labels, event_by_id, event_queries, sequences


def _load_context(path, labels_sha):
    context = _json_load(path, "context_unreadable")
    if not isinstance(context, dict):
        raise AuditError("context_malformed: context root must be an object")
    if context.get("mode") != CONTEXT_MODE:
        raise AuditError(f"context_mode_unsupported: expected {CONTEXT_MODE}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(context.get("label_sha256", ""))):
        raise AuditError("context_label_hash_invalid: context.label_sha256 is missing or malformed")
    if context["label_sha256"] != labels_sha:
        raise AuditError("label_hash_mismatch: frozen labels and saved context differ")
    temporal = context.get("temporal_config")
    if not isinstance(temporal, dict):
        raise AuditError("context_temporal_config_missing: temporal_config is required")
    missing = [field for field in TEMPORAL_CONFIG_FIELDS if field not in temporal]
    if missing:
        raise AuditError("context_temporal_config_missing_fields: "
                         + ", ".join(sorted(missing)))
    try:
        config = TemporalConfig(**temporal)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"context_temporal_config_invalid: {exc}") from exc
    groups = context.get("video_groups")
    if not isinstance(groups, list) or not groups or any(not isinstance(group, str) for group in groups):
        raise AuditError("context_video_groups_invalid: captured video_groups are required")
    context["_temporal"] = config
    context["_groups"] = tuple(groups)
    return context


def _read_mapping(path):
    detail = {"status": "ok", "rows": 0, "order_valid": None, "issues": []}
    frames = []
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"n", "pts_time", "frame_idx"}.issubset(fields):
                raise ValueError("mapping requires n, pts_time and frame_idx columns")
            for line_number, row in enumerate(reader, 2):
                try:
                    keyframe_n = _integer(row["n"], "n")
                    frame_id = _integer(row["frame_idx"], "frame_idx")
                    pts_time = _number(row["pts_time"], "pts_time")
                    if keyframe_n < 1 or frame_id < 0 or pts_time < 0:
                        raise ValueError("identity/timestamp outside valid range")
                    if "fps" in fields and row.get("fps") not in (None, ""):
                        fps = _number(row["fps"], "fps")
                        if fps <= 0:
                            raise ValueError("fps must be positive")
                    else:
                        fps = None
                except ValueError as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
                frames.append({"frame_id": frame_id, "keyframe_n": keyframe_n, "pts_time": pts_time, "fps": fps})
    except (OSError, ValueError, csv.Error) as exc:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("mapping_invalid", str(exc)))
        return detail, frames

    detail["rows"] = len(frames)
    times = [frame["pts_time"] for frame in frames]
    detail["order_valid"] = bool(frames) and all(a <= b for a, b in zip(times, times[1:]))
    if not frames:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("mapping_empty", "mapping has no rows"))
    frame_counts = Counter(frame["frame_id"] for frame in frames)
    keyframe_counts = Counter(frame["keyframe_n"] for frame in frames)
    detail["duplicate_frame_ids"] = sorted(frame_id for frame_id, count in frame_counts.items() if count > 1)
    detail["duplicate_keyframe_ns"] = sorted(keyframe_n for keyframe_n, count in keyframe_counts.items() if count > 1)
    if detail["duplicate_frame_ids"]:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("duplicate_mapping_frame_ids", ",".join(map(str, detail["duplicate_frame_ids"]))))
    if detail["duplicate_keyframe_ns"]:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("duplicate_mapping_keyframes", ",".join(map(str, detail["duplicate_keyframe_ns"]))))
    if not detail["order_valid"]:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("mapping_timestamp_order", "pts_time is not non-decreasing"))
    return detail, frames


def _read_caption_records(path, expected_video, kind):
    detail = {"status": "ok", "rows": 0, "issues": [], "image_hashes": {}, "generation": []}
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("caption artifact must be a JSON array")
        identities = []
        frame_counts, keyframe_counts = Counter(), Counter()
        for offset, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"record {offset} is not an object")
            video_id = _video_id(record.get("video_id"), "caption video_id")
            if video_id != expected_video:
                raise ValueError(f"record {offset} video_id {video_id} != {expected_video}")
            frame_id = _integer(record.get("frame_id"), "caption frame_id")
            keyframe_n = _integer(record.get("keyframe_n"), "caption keyframe_n")
            if frame_id < 0 or keyframe_n < 1:
                raise ValueError(f"record {offset} identity outside valid range")
            identities.append((frame_id, keyframe_n))
            frame_counts[frame_id] += 1
            keyframe_counts[keyframe_n] += 1
            if kind == "caption_generator":
                if not (record.get("retrieval_text") or record.get("caption")):
                    detail["issues"].append(_issue("caption_text_missing", f"frame_id={frame_id}"))
                image_hash = record.get("image_sha256")
                if isinstance(image_hash, str):
                    detail["image_hashes"][frame_id] = image_hash
                generation = record.get("caption_generation")
                if isinstance(generation, dict):
                    detail["generation"].append({
                        "backend": generation.get("backend"),
                        "model": generation.get("model"),
                        "revision": generation.get("revision"),
                    })
        detail["rows"] = len(records)
        detail["identities"] = identities
        detail["duplicate_frame_ids"] = sorted(frame_id for frame_id, count in frame_counts.items() if count > 1)
        detail["duplicate_keyframe_ns"] = sorted(keyframe_n for keyframe_n, count in keyframe_counts.items() if count > 1)
        if detail["duplicate_frame_ids"]:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("duplicate_caption_frame_ids", ",".join(map(str, detail["duplicate_frame_ids"]))))
        if detail["duplicate_keyframe_ns"]:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("duplicate_caption_keyframes", ",".join(map(str, detail["duplicate_keyframe_ns"]))))
    except (OSError, ValueError, TypeError) as exc:
        detail["status"] = "invalid"
        detail["issues"].append(_issue(f"{kind}_invalid", str(exc)))
        detail["identities"] = []
    return detail


def _compare_identity_order(mapping_frames, caption_detail, label):
    issues = []
    mapping_pairs = [(frame["frame_id"], frame["keyframe_n"]) for frame in mapping_frames]
    caption_pairs = caption_detail.get("identities") or []
    order_matches = None
    if mapping_frames and caption_detail.get("status") == "ok":
        if len(mapping_pairs) != len(caption_pairs):
            issues.append(_issue(f"{label}_row_count_mismatch", f"mapping={len(mapping_pairs)} caption={len(caption_pairs)}"))
            order_matches = False
        elif mapping_pairs != caption_pairs:
            order_matches = False
            if sorted(mapping_pairs) == sorted(caption_pairs):
                issues.append(_issue(f"{label}_row_order_mismatch", "same identities in a different row order"))
            else:
                issues.append(_issue(f"{label}_identity_sequence_mismatch", "frame_id/keyframe_n identities differ"))
        else:
            order_matches = True
    return order_matches, issues


def _read_features(path, expected_rows, expected_dimension):
    detail = {"status": "ok", "issues": [], "rows": None, "dimension": None, "dtype": None}
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        detail.update(rows=int(array.shape[0]) if array.ndim >= 1 else None,
                      dimension=int(array.shape[1]) if array.ndim == 2 else None,
                      dtype=str(array.dtype))
        if array.ndim != 2:
            raise ValueError(f"feature array must be 2-D, got {array.ndim}-D")
        if expected_rows is not None and len(array) != expected_rows:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("feature_row_count_mismatch", f"mapping={expected_rows} features={len(array)}"))
        if expected_dimension is not None and array.shape[1] != expected_dimension:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("feature_dimension_mismatch", f"expected={expected_dimension} actual={array.shape[1]}"))
        if detail["status"] == "ok":
            for start in range(0, len(array), 256):
                block = np.asarray(array[start:start + 256], dtype=np.float32)
                if not np.isfinite(block).all():
                    detail["status"] = "invalid"
                    detail["issues"].append(_issue("non_finite_features", f"rows {start}:{start + len(block)}"))
                    break
                if np.any(np.linalg.norm(block, axis=1) <= np.finfo(np.float32).eps):
                    detail["status"] = "invalid"
                    detail["issues"].append(_issue("zero_norm_features", f"rows {start}:{start + len(block)}"))
                    break
    except (OSError, ValueError, TypeError) as exc:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("feature_invalid", str(exc)))
    return detail


def _faiss_status():
    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        return {"status": "unknown", "reason": f"faiss unavailable: {exc}"}
    except Exception as exc:  # pragma: no cover - host-specific import failure
        return {"status": "unknown", "reason": f"faiss import failed: {type(exc).__name__}: {exc}"}
    return {"status": "available", "reason": None}


def _read_index(path, expected_count, expected_dimension, faiss_state):
    detail = {"status": "unknown", "issues": [], "rows": None, "dimension": None}
    if faiss_state["status"] != "available":
        detail["reason"] = faiss_state["reason"]
        return detail
    try:  # pragma: no cover - not exercised when faiss is unavailable
        import faiss

        index = faiss.read_index(str(path))
        detail.update(status="ok", rows=int(index.ntotal), dimension=int(index.d),
                      metric=str(index.metric_type))
        if expected_count is not None and index.ntotal != expected_count:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("index_row_count_mismatch", f"expected={expected_count} actual={index.ntotal}"))
        if expected_dimension is not None and index.d != expected_dimension:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("index_dimension_mismatch", f"expected={expected_dimension} actual={index.d}"))
    except Exception as exc:  # pragma: no cover - host-specific
        detail.update(status="invalid", issues=[_issue("index_read_failed", f"{type(exc).__name__}: {exc}")])
    return detail


def _read_manifest(path, video_id, actual_hashes, caption_count, embedding_config):
    detail = {"status": "ok", "issues": []}
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("manifest schema_version must be 1")
        if manifest.get("video_id") != video_id:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("manifest_video_mismatch", f"{manifest.get('video_id')} != {video_id}"))
        if manifest.get("count") != caption_count:
            detail["status"] = "invalid"
            detail["issues"].append(_issue("manifest_count_mismatch", f"manifest={manifest.get('count')} caption={caption_count}"))
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("manifest files must be an object")
        for name in MANIFEST_HASH_FIELDS:
            if name not in files:
                detail["status"] = "invalid"
                detail["issues"].append(_issue("manifest_hash_missing", name))
        # The caption manifest's mapping hash is the caption-mapping JSON, not
        # the frame mapping CSV used by retrieval.
        actual_names = {"mapping_sha256": "caption_mapping_sha256"}
        for name, expected in sorted(files.items()):
            if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
                detail["status"] = "invalid"
                detail["issues"].append(_issue("manifest_hash_invalid", name))
                continue
            actual = actual_hashes.get(actual_names.get(name, name))
            if actual is None:
                detail["status"] = "invalid"
                detail["issues"].append(_issue("manifest_hash_unverifiable", name))
            elif actual != expected:
                detail["status"] = "invalid"
                detail["issues"].append(_issue("manifest_hash_mismatch", name))
        embedding = manifest.get("embedding")
        if not isinstance(embedding, dict):
            raise ValueError("manifest embedding must be an object")
        detail["embedding"] = embedding
        for field in ("model", "revision", "dimension", "normalized", "metric"):
            if embedding_config is not None and embedding.get(field) != embedding_config.get(field):
                detail["status"] = "invalid"
                detail["issues"].append(_issue("manifest_embedding_mismatch", field))
    except (OSError, ValueError, TypeError) as exc:
        detail["status"] = "invalid"
        detail["issues"].append(_issue("manifest_invalid", str(exc)))
    return detail


def _artifact_status(statuses, issues):
    """Aggregate artifact and cross-artifact issues before reporting overall status."""
    if issues:
        return "issues"
    if any(status in {"missing", "ambiguous", "invalid"} for status in statuses):
        return "issues"
    if statuses and all(status == "ok" for status in statuses):
        return "ok"
    return "unknown"


def _artifact_detail(
    video_id,
    roots,
    indexes,
    groups,
    faiss_state,
    video_model_config,
    caption_model_config,
    fast_caption_config,
    label_judgments,
):
    public = {
        "video_id": video_id,
        "captured_scope": video_id.split("_", 1)[0] in set(groups),
        "status": "ok",
        "issues": [],
        "artifacts": {},
        "identity": {},
    }
    identity = {"mapping": {}, "keyframe_n": {}, "caption_image_hashes": {}, "caption_generation": []}
    actual_hashes = {}

    mapping_paths = _resolve(indexes["mapping"], video_id, ".csv")
    mapping_detail, mapping_frames = {"status": "missing", "issues": []}, []
    if len(mapping_paths) > 1:
        mapping_detail = {"status": "ambiguous", "issues": [_issue("duplicate_mapping_artifact",
                                                                   ", ".join(str(path) for path in mapping_paths))]}
    elif mapping_paths:
        mapping_path = mapping_paths[0]
        mapping_detail, mapping_frames = _read_mapping(mapping_path)
        mapping_detail.update(path=_relative(mapping_path, roots["mapping"]),
                              sha256=_sha256_file(mapping_path))
        actual_hashes["mapping_csv_sha256"] = mapping_detail["sha256"]
    public["artifacts"]["mapping"] = mapping_detail
    if mapping_frames:
        identity["mapping"] = {
            frame["frame_id"]: {"keyframe_n": frame["keyframe_n"], "pts_time": frame["pts_time"]}
            for frame in mapping_frames
        }
        keyframe_map = {}
        for frame in mapping_frames:
            keyframe_map.setdefault(frame["keyframe_n"], frame["frame_id"])
        identity["keyframe_n"] = keyframe_map
    public["identity"]["mapping_rows"] = len(mapping_frames)
    public["identity"]["duplicate_frame_ids"] = mapping_detail.get("duplicate_frame_ids", [])
    public["identity"]["duplicate_keyframe_ns"] = mapping_detail.get("duplicate_keyframe_ns", [])

    feature_paths = _resolve(indexes["features"], video_id, ".npy")
    feature_detail = {"status": "missing", "issues": []}
    if len(feature_paths) > 1:
        feature_detail = {"status": "ambiguous", "issues": [_issue("duplicate_feature_artifact",
                                                                   ", ".join(str(path) for path in feature_paths))]}
    elif feature_paths:
        feature_path = feature_paths[0]
        feature_detail = _read_features(feature_path, len(mapping_frames) or None,
                                        (video_model_config or {}).get("dimension"))
        feature_detail.update(path=_relative(feature_path, roots["features"]),
                              sha256=_sha256_file(feature_path))
        actual_hashes["features_sha256"] = feature_detail["sha256"]
    public["artifacts"]["visual_features"] = feature_detail

    caption_paths = _resolve(indexes["captions"], video_id, ".json")
    caption_detail = {"status": "missing", "issues": [], "image_hashes": {}, "generation": []}
    if len(caption_paths) > 1:
        caption_detail = {"status": "ambiguous", "issues": [_issue("duplicate_caption_artifact",
                                                                   ", ".join(str(path) for path in caption_paths))]}
    elif caption_paths:
        caption_path = caption_paths[0]
        caption_detail = _read_caption_records(caption_path, video_id, "caption_generator")
        caption_detail.update(path=_relative(caption_path, roots["captions"]),
                              sha256=_sha256_file(caption_path))
        actual_hashes["caption_sha256"] = caption_detail["sha256"]
        identity["caption_image_hashes"] = caption_detail.get("image_hashes", {})
        identity["caption_generation"] = caption_detail.get("generation", [])
    public["artifacts"]["caption_generator"] = caption_detail

    caption_mapping_paths = _resolve(indexes["caption_mapping"], video_id, ".json")
    caption_mapping_detail = {"status": "missing", "issues": []}
    if len(caption_mapping_paths) > 1:
        caption_mapping_detail = {"status": "ambiguous", "issues": [_issue("duplicate_caption_mapping_artifact",
                                                                            ", ".join(str(path) for path in caption_mapping_paths))]}
    elif caption_mapping_paths:
        caption_mapping_path = caption_mapping_paths[0]
        caption_mapping_detail = _read_caption_records(caption_mapping_path, video_id, "caption_mapping")
        caption_mapping_detail.update(path=_relative(caption_mapping_path, roots["caption_mapping"]),
                                      sha256=_sha256_file(caption_mapping_path))
        actual_hashes["caption_mapping_sha256"] = caption_mapping_detail["sha256"]
    public["artifacts"]["caption_mapping"] = caption_mapping_detail

    index_paths = _resolve(indexes["index"], video_id, ".index")
    index_detail = {"status": "missing", "issues": []}
    if len(index_paths) > 1:
        index_detail = {"status": "ambiguous", "issues": [_issue("duplicate_index_artifact",
                                                                 ", ".join(str(path) for path in index_paths))]}
    elif index_paths:
        index_path = index_paths[0]
        index_detail = _read_index(index_path, len(mapping_frames) or None,
                                   (caption_model_config or {}).get("dimension"), faiss_state)
        index_detail.update(path=_relative(index_path, roots["index"]),
                            sha256=_sha256_file(index_path))
        actual_hashes["index_sha256"] = index_detail["sha256"]
    public["artifacts"]["index"] = index_detail

    manifest_paths = _resolve(indexes["index"], video_id, ".meta.json")
    manifest_detail = {"status": "missing", "issues": []}
    if len(manifest_paths) > 1:
        manifest_detail = {"status": "ambiguous", "issues": [_issue("duplicate_index_manifest",
                                                                    ", ".join(str(path) for path in manifest_paths))]}
    elif manifest_paths:
        manifest_path = manifest_paths[0]
        caption_count = len(caption_detail.get("identities") or []) or len(mapping_frames)
        manifest_detail = _read_manifest(manifest_path, video_id, actual_hashes,
                                         caption_count, caption_model_config)
        manifest_detail.update(path=_relative(manifest_path, roots["index"]),
                               sha256=_sha256_file(manifest_path))
    public["artifacts"]["index_manifest"] = manifest_detail

    identity_checks = {
        "mapping_vs_caption_generator": _compare_identity_order(mapping_frames, caption_detail, "caption_generator"),
        "mapping_vs_caption_mapping": _compare_identity_order(mapping_frames, caption_mapping_detail, "caption_mapping"),
    }
    for name, (matches, issues) in identity_checks.items():
        public["identity"][name] = matches
        for issue in issues:
            public["issues"].append(issue)

    generation = {tuple(sorted(item.items())) for item in identity["caption_generation"]}
    if len(generation) > 1:
        public["issues"].append(_issue("caption_generation_model_mixed", "multiple caption model/revision records"))
    elif generation and fast_caption_config:
        backend = dict(next(iter(generation)))
        if backend.get("model") is not None and backend.get("model") != fast_caption_config.get("model"):
            public["issues"].append(_issue("caption_generation_model_mismatch", str(backend.get("model"))))
        if backend.get("revision") is not None and backend.get("revision") != fast_caption_config.get("revision"):
            public["issues"].append(_issue("caption_generation_revision_mismatch", str(backend.get("revision"))))

    for judgment in label_judgments:
        frame_id = judgment["frame_id"]
        expected_hash = judgment.get("image_sha256")
        actual_hash = identity["caption_image_hashes"].get(frame_id)
        if expected_hash and actual_hash and expected_hash != actual_hash:
            public["issues"].append(_issue("label_caption_image_hash_mismatch", f"frame_id={frame_id}"))
        if expected_hash and not actual_hash:
            public["issues"].append(_issue("label_caption_image_hash_unavailable", f"frame_id={frame_id}"))
        mapped = identity["mapping"].get(frame_id)
        if not mapped:
            public["issues"].append(_issue("label_frame_missing_from_mapping", f"frame_id={frame_id}"))
        elif mapped["keyframe_n"] != judgment["keyframe_n"]:
            public["issues"].append(_issue("label_mapping_keyframe_mismatch",
                                           f"frame_id={frame_id} mapping={mapped['keyframe_n']} label={judgment['keyframe_n']}"))
        elif abs(mapped["pts_time"] - judgment["pts_time"]) > 1e-6:
            public["issues"].append(_issue("label_mapping_timestamp_mismatch", f"frame_id={frame_id}"))

    for artifact_name, artifact in sorted(public["artifacts"].items()):
        if artifact.get("status") == "missing":
            public["issues"].append(_issue(f"{artifact_name}_missing", video_id))
        for issue in artifact.get("issues", []):
            public["issues"].append({"code": f"{artifact_name}_{issue['code']}", "detail": issue["detail"]})
    public["issues"] = _merge_issues(public["issues"])
    statuses = [artifact.get("status") for artifact in public["artifacts"].values()]
    public["status"] = _artifact_status(statuses, public["issues"])
    return public, identity


def _load_saved_rows(source_run, issues=None):
    """Load saved drive_*.json rows, keeping unreadable rows as explicit issues."""
    issues = [] if issues is None else issues
    root = Path(source_run)
    if not root.is_dir():
        raise AuditError(f"source_run_missing: {root}")
    paths = sorted(root.glob("drive_*.json"), key=lambda path: path.name)
    if not paths:
        raise AuditError("saved_rows_missing: no drive_*.json rows in source run")
    rows, files, seen = [], [], set()
    for path in paths:
        try:
            file_hash = _sha256_file(path)
        except OSError as exc:
            issues.append(_issue("saved_row_unreadable", f"{path.name}: {exc}"))
            continue
        files.append({"path": _relative(path, root), "sha256": file_hash})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue("saved_row_unreadable", f"{path.name}: {type(exc).__name__}"))
            continue
        if not isinstance(data, dict):
            issues.append(_issue("saved_row_invalid", f"{path.name}: root must be an object"))
            continue
        query_id = data.get("query_id")
        if not isinstance(query_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", query_id):
            issues.append(_issue("saved_row_invalid", f"{path.name}: invalid query_id {query_id!r}"))
            continue
        if path.stem != query_id:
            issues.append(_issue("saved_row_filename_mismatch", f"{path.name}: query_id={query_id}"))
        if query_id in seen:
            issues.append(_issue("duplicate_saved_row", query_id))
            continue
        seen.add(query_id)
        if data.get("task") not in {"event_frame", "sequence"}:
            issues.append(_issue("saved_row_invalid", f"{query_id}: unsupported task {data.get('task')!r}"))
            continue
        if data.get("split") != "holdout":
            issues.append(_issue("saved_row_split_mismatch", f"{query_id}: {data.get('split')!r}"))
            continue
        query = data.get("query")
        if not isinstance(query, dict) or query.get("query_id") != query_id:
            issues.append(_issue("saved_row_invalid", f"{query_id}: query object is missing or mismatched"))
            continue
        try:
            _video_id(query.get("video_id"), "query video_id")
        except ValueError as exc:
            issues.append(_issue("saved_row_invalid", f"{query_id}: {exc}"))
            continue
        if data["task"] == "event_frame" and not isinstance(query.get("judgments"), list):
            issues.append(_issue("saved_row_invalid", f"{query_id}: event judgments are missing"))
            continue
        if data["task"] == "sequence" and not isinstance(query.get("event_relevant_keyframes"), list):
            issues.append(_issue("saved_row_invalid", f"{query_id}: sequence positives are missing"))
            continue
        if data.get("same_pool_and_raw_scores") is False:
            issues.append(_issue("saved_row_pool_not_shared", query_id))
        if not isinstance(data.get("failures", {}), dict):
            issues.append(_issue("saved_row_invalid", f"{query_id}: failures must be an object"))
        rows.append(data)
    if not rows:
        raise AuditError("saved_rows_missing: no valid saved row could be loaded")
    return rows, files, issues


def _validate_raw_evidence(data, filename, record):
    try:
        if not isinstance(data, dict):
            raise ValueError("root must be an object")
        unit_id = data.get("unit_id")
        video_id = _video_id(data.get("video_id"), "raw evidence video_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("unit_id is missing")
        if filename != f"{unit_id}_{video_id}.json":
            raise ValueError("filename does not match unit_id/video_id")
        actions = data.get("actions")
        if not isinstance(actions, list) or not actions or any(not isinstance(action, str) or not action.strip()
                                                             for action in actions):
            raise ValueError("actions must be a non-empty array of non-empty strings")
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("frames must be a non-empty array")
        normalized, frame_index, keyframe_index = [], {}, {}
        previous_time = None
        for offset, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"frame {offset} is not an object")
            frame_video = _video_id(frame.get("video_id"), f"frame {offset} video_id")
            if frame_video != video_id:
                raise ValueError(f"frame {offset} video_id {frame_video} != {video_id}")
            frame_id = _integer(frame.get("frame_id"), f"frame {offset} frame_id")
            keyframe_n = _integer(frame.get("keyframe_n"), f"frame {offset} keyframe_n")
            pts_time = _number(frame.get("pts_time"), f"frame {offset} pts_time")
            if frame_id < 0 or keyframe_n < 1 or pts_time < 0:
                raise ValueError(f"frame {offset} identity/timestamp outside valid range")
            if previous_time is not None and pts_time < previous_time:
                raise ValueError("frame timestamps are not non-decreasing")
            previous_time = pts_time
            if frame_id in frame_index:
                raise ValueError(f"duplicate local frame_id {frame_id}")
            if keyframe_n in keyframe_index:
                raise ValueError(f"duplicate local keyframe_n {keyframe_n}")
            frame_index[frame_id] = offset
            keyframe_index[keyframe_n] = offset
            normalized.append({"video_id": frame_video, "frame_id": frame_id,
                               "keyframe_n": keyframe_n, "pts_time": pts_time})
        arrays = {}
        shape = (len(actions), len(frames))
        for name in ("visual", "semantic"):
            values = data.get(name)
            if values is None:
                arrays[name] = None
                continue
            if not isinstance(values, list) or len(values) != len(actions):
                raise ValueError(f"{name} must have one row per action")
            matrix = []
            for event_index, event_values in enumerate(values):
                if not isinstance(event_values, list) or len(event_values) != len(frames):
                    raise ValueError(f"{name} event {event_index} must have one value per frame")
                row = []
                for value in event_values:
                    row.append(math.nan if value is None else _number(value, f"{name}[{event_index}]"))
                matrix.append(row)
            array = np.asarray(matrix, dtype=float)
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} != {shape}")
            arrays[name] = array
        stats = data.get("stats")
        if stats is not None and not isinstance(stats, dict):
            raise ValueError("stats must be an object")
        record.update(status="ok", data=data, unit_id=unit_id, video_id=video_id,
                      actions=list(actions), frames=normalized, frame_index=frame_index,
                      keyframe_index=keyframe_index, visual=arrays["visual"],
                      semantic=arrays["semantic"], stats=stats or {})
    except (TypeError, ValueError) as exc:
        record["status"] = "invalid"
        record["issues"].append(_issue("raw_evidence_invalid", f"{filename}: {exc}"))
    return record


def _load_raw_evidence(source_run, issues=None):
    """Load raw per-unit/per-video evidence; invalid or missing evidence stays unresolved."""
    issues = [] if issues is None else issues
    root = Path(source_run) / "raw_evidence"
    records, files = {}, []
    if not root.is_dir():
        issues.append(_issue("raw_evidence_missing", "raw_evidence directory is missing"))
        return records, files, issues
    paths = sorted(root.glob("*.json"), key=lambda path: path.name)
    if not paths:
        issues.append(_issue("raw_evidence_missing", "raw_evidence directory has no JSON files"))
        return records, files, issues
    for path in paths:
        try:
            file_hash = _sha256_file(path)
        except OSError as exc:
            issues.append(_issue("raw_evidence_unreadable", f"{path.name}: {exc}"))
            continue
        files.append({"path": _relative(path, Path(source_run)), "sha256": file_hash})
        record = {"status": "invalid", "issues": [], "path": _relative(path, Path(source_run)),
                  "sha256": file_hash}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["issues"].append(_issue("raw_evidence_unreadable",
                                           f"{path.name}: {type(exc).__name__}"))
            issues.extend(record["issues"])
            continue
        record = _validate_raw_evidence(data, path.name, record)
        if record["status"] == "ok":
            key = (record["unit_id"], record["video_id"])
            if key in records:
                record["status"] = "invalid"
                record["issues"].append(_issue("duplicate_raw_evidence",
                                               f"{key[0]}/{key[1]}: {path.name}"))
        issues.extend(record["issues"])
        key = (record.get("unit_id"), record.get("video_id"))
        if key[0] is not None and key[1] is not None:
            records.setdefault(key, record)
    return records, files, issues


def _audit_provenance(labels_path, context_path, context, row_files, raw_files,
                      artifact_hashes, labels=None, captured_model_config=None):
    """Hash every saved input class and keep current config explicitly non-historical."""
    labels_path, context_path = Path(labels_path), Path(context_path)
    labels = _json_load(labels_path, "labels_unreadable") if labels is None else labels
    labels_sha = _sha256_file(labels_path)
    context_sha = _sha256_file(context_path)
    tool_path = Path(__file__)
    tool_sha = _sha256_file(tool_path)
    rows_sha = _sha256_pairs((item["path"], item["sha256"]) for item in row_files)
    raw_sha = _sha256_pairs((item["path"], item["sha256"]) for item in raw_files)
    artifact_pairs = [(name, digest) for name, digest in artifact_hashes]
    artifacts_sha = _sha256_pairs(artifact_pairs)
    caption_hashes = _normalized_path_map(context.get("caption_artifact_sha256"))
    source_hashes = _normalized_path_map(context.get("source_sha256"))
    caption_hashes_sha = _sha256_pairs(caption_hashes.items())

    snapshot_path = context_path.parent / "source_snapshot.json"
    local_artifacts_path = context_path.parent / "local_artifact_sha256.json"
    snapshot_sha = _sha256_file(snapshot_path) if snapshot_path.is_file() else None
    local_artifacts_sha = _sha256_file(local_artifacts_path) if local_artifacts_path.is_file() else None

    return {
        "tool": {
            "path": _relative(tool_path, KIS_ROOT),
            "sha256": tool_sha,
            "models_loaded": False,
            "network_used": False,
        },
        "labels": {
            "path": _relative(labels_path, KIS_ROOT),
            "sha256": labels_sha,
            "schema": labels.get("schema"),
            "annotation_source": labels.get("annotation_source"),
            "human_verified": labels.get("human_verified"),
            "caption_used_for_annotation": labels.get("caption_used_for_annotation"),
            "protocol_sha256": labels.get("protocol_sha256"),
            "manifest_sha256": labels.get("manifest_sha256"),
        },
        "context": {
            "path": _relative(context_path, KIS_ROOT),
            "sha256": context_sha,
            "mode": context.get("mode"),
            "label_sha256": context.get("label_sha256"),
            "protocol_sha256": context.get("protocol_sha256"),
            "annotation_source": context.get("annotation_source"),
            "human_verified": context.get("human_verified"),
            "collection_id": context.get("collection_id"),
            "collection_stats": context.get("collection_stats"),
            "nprobe": context.get("nprobe"),
            "device": context.get("device"),
            "caption_vectors": context.get("caption_vectors"),
            "temporal_config": context.get("temporal_config"),
            "video_groups": context.get("video_groups"),
            "caption_artifact_sha256": caption_hashes,
            "caption_artifact_sha256_digest": caption_hashes_sha,
            "source_sha256": source_hashes,
            "source_snapshot_sha256": snapshot_sha,
        },
        "rows": {
            "count": len(row_files),
            "sha256": rows_sha,
            "files": sorted(row_files, key=lambda item: item["path"]),
        },
        "raw_evidence": {
            "count": len(raw_files),
            "sha256": raw_sha,
            "files": sorted(raw_files, key=lambda item: item["path"]),
        },
        "artifacts": {
            "count": len(artifact_pairs),
            "sha256": artifacts_sha,
            "local_artifact_sha256_file": _relative(local_artifacts_path, KIS_ROOT)
            if local_artifacts_path.is_file() else None,
            "local_artifact_sha256": local_artifacts_sha,
        },
        "hashes": {
            "labels_sha256": labels_sha,
            "context_sha256": context_sha,
            "rows_sha256": rows_sha,
            "raw_evidence_sha256": raw_sha,
            "artifacts_sha256": artifacts_sha,
            "tool_sha256": tool_sha,
        },
        "captured_model_config": (captured_model_config
                                  if captured_model_config is not None
                                  else _captured_model_config(context)),
        "current_config_snapshot": {
            **_current_config(),
            "historical": False,
            "note": "Current checkout snapshot only; it is not the configuration captured by the saved run.",
        },
        "unknown_state": {
            "live_milvus": {"status": "unknown",
                            "reason": "read-only audit; no live Milvus connection or query provenance is saved"},
            "planner": {"status": "unknown",
                        "reason": "no planner trace is saved; the sequence plan is audited as frozen input only"},
            "vlm": {"status": "unknown",
                    "reason": "no VLM trace is saved; current config is not historical run state"},
        },
    }


def _judgment_signature(judgments):
    if not isinstance(judgments, list):
        return None
    try:
        return tuple(sorted(
            (_video_id(item.get("video_id")), _integer(item.get("frame_id"), "judgment frame_id"),
             _integer(item.get("keyframe_n"), "judgment keyframe_n"), item.get("judgment"))
            for item in judgments if isinstance(item, dict)
        ))
    except (TypeError, ValueError):
        return None


def _event_query_mismatch(row, label):
    query = row.get("query")
    if not isinstance(query, dict):
        return ["query object is missing"]
    problems = []
    for field in ("query_id", "video_id", "event_text"):
        if query.get(field) != label.get(field):
            problems.append(field)
    if _judgment_signature(query.get("judgments")) != _judgment_signature(label.get("judgments")):
        problems.append("judgments")
    events = row.get("events")
    if not isinstance(events, list) or len(events) != 1:
        problems.append("events")
    elif not isinstance(events[0], dict) or events[0].get("action") != label.get("event_text"):
        problems.append("event_action")
    return problems


def _sequence_query_mismatch(row, sequence):
    query = row.get("query")
    if not isinstance(query, dict):
        return ["query object is missing"]
    problems = []
    for field in ("video_id", "event_query_ids", "actions", "event_relevant_keyframes"):
        if query.get(field) != sequence.get(field):
            problems.append(field)
    events = row.get("events")
    if not isinstance(events, list) or len(events) != len(sequence["actions"]):
        problems.append("events")
    else:
        for offset, event in enumerate(events):
            if not isinstance(event, dict) or event.get("action") != sequence["actions"][offset]:
                problems.append(f"event_order_{offset}")
                break
    return problems


def _stream_hits(row, event_count):
    issues = []
    events = row.get("events")
    if not isinstance(events, list) or len(events) != event_count:
        issues.append(_issue("saved_events_invalid",
                             f"{row.get('query_id')}: expected {event_count} events"))
        return None, issues
    result = []
    for event_index, event in enumerate(events):
        streams = event.get("streams") if isinstance(event, dict) else None
        hit_map = {}
        if not isinstance(streams, dict):
            issues.append(_issue("saved_streams_invalid", f"event {event_index}: streams missing"))
            result.append({name: None for name in STREAMS})
            continue
        for stream in STREAMS:
            items = streams.get(stream)
            if not isinstance(items, list):
                hit_map[stream] = None
                issues.append(_issue("saved_stream_missing", f"event {event_index}: {stream}"))
                continue
            identities, valid = set(), True
            for item in items:
                if not isinstance(item, dict):
                    valid = False
                    break
                try:
                    identities.add((_video_id(item.get("video_id")),
                                    _integer(item.get("frame_id"), "stream frame_id")))
                except ValueError:
                    valid = False
                    break
            hit_map[stream] = identities if valid else None
            if not valid:
                issues.append(_issue("saved_stream_invalid", f"event {event_index}: {stream}"))
        result.append(hit_map)
    return result, issues


def _pool_for_row(row, config, target_video):
    issues = []
    query_id = row.get("query_id")
    pool = row.get("candidate_pool")
    ranked_selected = set()
    if isinstance(pool, dict):
        ranked = pool.get("ranked_videos")
        if isinstance(ranked, list):
            for item in ranked:
                if isinstance(item, dict) and item.get("selected") is True:
                    try:
                        ranked_selected.add(_video_id(item.get("video_id")))
                    except ValueError:
                        issues.append(_issue("saved_pool_invalid", f"{query_id}: ranked video id"))
        else:
            issues.append(_issue("saved_pool_invalid", f"{query_id}: ranked_videos missing"))
        for field, key in (("max_videos", "max_videos"), ("rrf_k", "candidate_rrf_k"),
                           ("strategy", "candidate_pooling")):
            if field in pool and pool.get(field) != getattr(config, key):
                issues.append(_issue("saved_pool_config_mismatch",
                                     f"{query_id}: {field}={pool.get(field)!r} expected {getattr(config, key)!r}"))
    else:
        issues.append(_issue("saved_pool_invalid", f"{query_id}: candidate_pool missing"))

    candidate_videos = row.get("candidate_videos")
    if not isinstance(candidate_videos, list):
        issues.append(_issue("saved_pool_invalid", f"{query_id}: candidate_videos missing"))
        pool_set = None
    else:
        pool_set = set()
        valid = True
        for video in candidate_videos:
            try:
                pool_set.add(_video_id(video))
            except ValueError:
                valid = False
                break
        if not valid:
            issues.append(_issue("saved_pool_invalid", f"{query_id}: candidate video id"))
            pool_set = None
        elif ranked_selected and ranked_selected != pool_set:
            issues.append(_issue("saved_pool_selection_mismatch",
                                 f"{query_id}: candidate_videos and selected ranked_videos differ"))
    if pool_set is not None and bool(row.get("target_video_in_pool")) != (target_video in pool_set):
        issues.append(_issue("saved_target_pool_flag_mismatch", query_id))
    return pool_set, issues


def _support_matrices(record, config):
    shape = (len(record["actions"]), len(record["frames"]))
    visual, semantic = record.get("visual"), record.get("semantic")
    scores, supported = evidence_scores(visual, semantic, shape, config)
    valid = np.zeros(shape, dtype=bool)
    if visual is not None:
        valid |= np.isfinite(visual)
    if semantic is not None:
        valid |= np.isfinite(semantic)
    return scores, supported, valid


def _stage_detail(outcomes, reasons, unit):
    covered = sum(1 for outcome in outcomes if outcome is True)
    resolved = sum(1 for outcome in outcomes if outcome is not None)
    total = len(outcomes)
    unresolved = total - resolved
    reason_counts = Counter(reason for outcome, reason in zip(outcomes, reasons)
                            if outcome is None and reason)
    if total == 0:
        status = "not_applicable"
    elif resolved == 0:
        status = "unresolved"
    elif unresolved:
        status = "partial"
    else:
        status = "known"
    detail = {"status": status, "unit": unit, "covered": covered, "resolved": resolved,
              "total": total, "unresolved": unresolved}
    if reason_counts:
        detail["reasons"] = dict(sorted(reason_counts.items()))
    return detail


def _video_stage_detail(videos, pool_set):
    outcomes = [None if pool_set is None else video in pool_set for video in videos]
    reasons = ["pool_unavailable" if pool_set is None else None for _ in videos]
    return _stage_detail(outcomes, reasons, "positive_videos")


def _saved_detail(hit, valid, reason="saved_results_unavailable"):
    if not valid:
        return {"status": "unresolved", "unit": "queries", "covered": 0, "resolved": 0,
                "total": 1, "unresolved": 1, "reasons": {reason: 1}}
    return {"status": "known", "unit": "queries", "covered": int(bool(hit)), "resolved": 1,
            "total": 1, "unresolved": 0}


def _temporal_detail(feasible, reason, event_frame):
    if event_frame:
        return {"status": "not_applicable", "unit": "queries", "covered": 0, "resolved": 0,
                "total": 0, "unresolved": 0}
    if reason is not None:
        return {"status": "unresolved", "unit": "queries", "covered": 0, "resolved": 0,
                "total": 1, "unresolved": 1, "reasons": {reason: 1}}
    return {"status": "known", "unit": "queries", "covered": int(bool(feasible)), "resolved": 1,
            "total": 1, "unresolved": 0, "feasible": bool(feasible)}


def _missing_row_stages(positives, task, reason="saved_row_missing"):
    """Keep every stage denominator visible when a saved row cannot be audited."""
    frame_count = len(positives)
    video_count = len({positive["video_id"] for positive in positives})
    frame_detail = _stage_detail([None] * frame_count,
                                 [reason] * frame_count, "positive_frames")
    video_detail = _stage_detail([None] * video_count,
                                 [reason] * video_count, "positive_videos")
    query_detail = _stage_detail([None], [reason], "queries")
    temporal_detail = (_temporal_detail(None, None, True) if task == "event_frame"
                       else _temporal_detail(None, reason, False))
    stages = {
        "visual_hits": frame_detail,
        "bm25_hits": frame_detail,
        "semantic_hits": frame_detail,
        "union_hits": frame_detail,
        "candidate_pool": video_detail,
        "local_frames": frame_detail,
        "support": frame_detail,
        "temporal_feasibility": temporal_detail,
        "saved_top_1": query_detail,
        "saved_top_5": query_detail,
        "saved_top_20": query_detail,
    }
    return {mode: {"stages": stages, "saved_metrics_match": None} for mode in MODES}


def _local_outcome(record, positive, query_id, issues):
    if record is None:
        return None, "raw_evidence_missing"
    if record.get("status") != "ok":
        return None, "raw_evidence_invalid"
    index = record["frame_index"].get(positive["frame_id"])
    if index is None:
        return False, None
    frame = record["frames"][index]
    if (frame["keyframe_n"] != positive["keyframe_n"]
            or abs(frame["pts_time"] - positive["pts_time"]) > 1e-6):
        issues.append(_issue("positive_local_identity_mismatch",
                             f"{query_id}: frame_id={positive['frame_id']}"))
        return None, "local_identity_mismatch"
    return True, None


def _support_outcome(record, positive, matrices):
    if matrices is None:
        return None, "raw_evidence_support_unavailable"
    scores, supported, valid = matrices
    event_index = positive["event_index"]
    if event_index >= supported.shape[0]:
        return None, "event_index_out_of_range"
    index = record["frame_index"].get(positive["frame_id"]) if record else None
    if index is None:
        return None, "local_frame_unavailable"
    if not valid[event_index, index]:
        return None, "no_captured_channel"
    return bool(supported[event_index, index]), None


def _temporal_feasibility(record, sequence, positives, config, query_id, issues):
    if record is None:
        return None, "raw_evidence_missing", []
    if record.get("status") != "ok":
        return None, "raw_evidence_invalid", []
    if list(record["actions"]) != list(sequence["actions"]):
        issues.append(_issue("raw_evidence_actions_mismatch", query_id))
        return None, "raw_evidence_actions_mismatch", []
    known = np.zeros((len(sequence["actions"]), len(record["frames"])), dtype=bool)
    missing = []
    for positive in positives:
        index = record["frame_index"].get(positive["frame_id"])
        reason = None
        if index is None:
            reason = "positive_local_frame_missing"
        else:
            frame = record["frames"][index]
            if (frame["keyframe_n"] != positive["keyframe_n"]
                    or abs(frame["pts_time"] - positive["pts_time"]) > 1e-6):
                reason = "local_identity_mismatch"
        if reason is not None:
            code = ("positive_local_identity_mismatch" if reason == "local_identity_mismatch"
                    else "positive_local_frame_missing")
            issues.append(_issue(code, f"{query_id}: frame_id={positive['frame_id']}"))
            missing.append({
                "event_index": positive["event_index"],
                "video_id": positive["video_id"],
                "frame_id": positive["frame_id"],
                "keyframe_n": positive["keyframe_n"],
                "reason": reason,
            })
            continue
        known[positive["event_index"], index] = True
    if any(not known[event].any() for event in range(known.shape[0])):
        return None, "sequence_event_without_local_positive", missing
    try:
        scores, _, _ = _support_matrices(record, config)
    except ValueError:
        return None, "raw_evidence_support_unavailable", missing
    aligned = align_events(record["frames"], scores, known, config)
    if aligned is not None and aligned["coverage"] == 1:
        return True, None, missing
    if missing:
        # A missing alternative positive could change the path; a partial result
        # over the remaining positives is not enough to call feasibility false.
        return None, "sequence_alternative_positive_unavailable", missing
    return False, None, missing


def _mode_results(row, mode, task, query_id, issues):
    modes = row.get("modes")
    if not isinstance(modes, dict) or not isinstance(modes.get(mode), dict):
        issues.append(_issue("saved_mode_missing", f"{query_id}: {mode}"))
        return None
    results = modes[mode].get("results")
    if not isinstance(results, list):
        issues.append(_issue("saved_results_invalid", f"{query_id}: {mode}"))
        return None
    for offset, item in enumerate(results):
        if not isinstance(item, dict):
            issues.append(_issue("saved_result_invalid", f"{query_id}: {mode}[{offset}]"))
            return None
        try:
            _video_id(item.get("video_id"))
            _integer(item.get("frame_id"), "result frame_id")
        except ValueError:
            issues.append(_issue("saved_result_invalid", f"{query_id}: {mode}[{offset}]"))
            return None
        if task == "sequence":
            temporal = item.get("temporal")
            events = temporal.get("events") if isinstance(temporal, dict) else None
            if not isinstance(events, list):
                issues.append(_issue("saved_sequence_result_invalid", f"{query_id}: {mode}[{offset}]"))
                return None
            for event in events:
                if not isinstance(event, dict):
                    issues.append(_issue("saved_sequence_result_invalid", f"{query_id}: {mode}[{offset}]"))
                    return None
                if event.get("status") == "retrieved":
                    try:
                        _integer(event.get("keyframe_n"), "temporal keyframe_n")
                    except ValueError:
                        issues.append(_issue("saved_sequence_result_invalid",
                                             f"{query_id}: {mode}[{offset}]"))
                        return None
    return results


def _saved_event_hits(results, positives, cutoffs=(1, 5, 20)):
    identities = {(positive["video_id"], positive["frame_id"]) for positive in positives}
    return {cutoff: any((item["video_id"], item["frame_id"]) in identities
                        for item in results[:cutoff]) for cutoff in cutoffs}


def _sequence_path_problems(result, sequence, record, config, positives):
    """Validate one saved sequence result path.

    ``None`` means the result does not claim a complete path (for example a
    fallback/partial candidate).  Otherwise return ``(problems, positive_match)``
    where structural problems are reportable issues and ``positive_match`` only
    says whether every event matches one explicit frozen positive identity and
    valid local raw evidence.  Canonical positives are the identity source of
    truth; the caller must not call this without a valid raw record.
    """
    temporal = result.get("temporal")
    if not isinstance(temporal, dict) or temporal.get("status") != SEQUENCE_COMPLETE_STATUS:
        return None
    problems = []
    try:
        coverage = _number(temporal.get("coverage"), "temporal coverage")
    except ValueError:
        coverage = None
    if coverage != 1.0:
        problems.append("temporal coverage is not 1")
    expected = sequence["event_relevant_keyframes"]
    positives_by_event = {}
    for positive in positives:
        positives_by_event.setdefault(positive["event_index"], []).append(positive)
    events = temporal.get("events")
    if not isinstance(events, list) or len(events) != len(expected):
        problems.append("temporal event count does not match the sequence")
        return problems, False
    times, frame_ids, keyframes = [], set(), set()
    record_ok = record is not None and record.get("status") == "ok"
    positive_match = True
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            problems.append(f"event {index + 1} is not an object")
            positive_match = False
            continue
        if event.get("status") != "retrieved":
            problems.append(f"event {index + 1} status is {event.get('status')!r}")
            positive_match = False
            continue
        try:
            video_id = _video_id(event.get("video_id"), f"event {index + 1} video_id")
            frame_id = _integer(event.get("frame_id"), f"event {index + 1} frame_id")
            keyframe_n = _integer(event.get("keyframe_n"), f"event {index + 1} keyframe_n")
            pts_time = _number(event.get("pts_time"), f"event {index + 1} pts_time")
            if video_id != sequence["video_id"]:
                raise ValueError(f"event {index + 1} video_id {video_id} != {sequence['video_id']}")
            if frame_id < 0 or keyframe_n < 1 or pts_time < 0:
                raise ValueError(f"event {index + 1} identity/timestamp outside valid range")
        except ValueError as exc:
            problems.append(str(exc))
            positive_match = False
            continue
        event_positives = positives_by_event.get(index, [])
        canonical = any(
            positive["video_id"] == video_id
            and positive["frame_id"] == frame_id
            and positive["keyframe_n"] == keyframe_n
            and abs(positive["pts_time"] - pts_time) <= 1e-6
            for positive in event_positives
        )
        if not canonical:
            positive_match = False
        if frame_id in frame_ids:
            problems.append(f"event {index + 1} repeats frame_id {frame_id}")
        if keyframe_n in keyframes:
            problems.append(f"event {index + 1} repeats keyframe_n {keyframe_n}")
        frame_ids.add(frame_id)
        keyframes.add(keyframe_n)
        times.append((index, pts_time))
        if record_ok:
            local_index = record["frame_index"].get(frame_id)
            if local_index is None:
                problems.append(f"event {index + 1} frame_id {frame_id} is absent from local evidence")
            else:
                local = record["frames"][local_index]
                if (local["keyframe_n"] != keyframe_n
                        or abs(local["pts_time"] - pts_time) > 1e-6):
                    problems.append(f"event {index + 1} identity differs from local evidence")
                elif not canonical:
                    # Raw and saved agree, but both disagree with the frozen
                    # explicit positive identity: never credit this path.
                    problems.append(f"event {index + 1} identity does not match an explicit positive")
    if len(times) == len(expected):
        for (left_index, left), (right_index, right) in zip(times, times[1:]):
            if not left < right:
                problems.append("event timestamps are not strictly increasing at "
                                f"{left_index + 1}->{right_index + 1}")
            elif right - left > config.max_gap_seconds:
                problems.append(f"event gap {right - left:.6f}s exceeds captured "
                                f"max_gap_seconds={config.max_gap_seconds} at "
                                f"{left_index + 1}->{right_index + 1}")
    return problems, positive_match


def _saved_sequence_hits(results, sequence, record, config, query_id, mode, issues,
                         positives, cutoffs=(1, 5, 20)):
    """Return cutoff hits and an unresolved reason for the saved sequence path.

    A sequence success requires both canonical frozen positive identities and
    valid local raw evidence with matching ordered actions.  Missing, invalid
    or action-mismatched raw evidence leaves every cutoff unresolved (``None``)
    with the corresponding reason; it is never credited as a hit and never
    silently treated as a miss.
    """
    if record is None:
        return {cutoff: None for cutoff in cutoffs}, "raw_evidence_missing"
    if record.get("status") != "ok":
        return {cutoff: None for cutoff in cutoffs}, "raw_evidence_invalid"
    if list(record.get("actions") or []) != list(sequence["actions"]):
        return {cutoff: None for cutoff in cutoffs}, "raw_evidence_actions_mismatch"
    evaluated = {}
    for rank, result in enumerate(results[:max(cutoffs)], 1):
        if result.get("video_id") != sequence["video_id"]:
            continue
        path = _sequence_path_problems(result, sequence, record, config, positives)
        if path is None:
            evaluated[rank] = False
            continue
        problems, positive_match = path
        if problems:
            issues.append(_issue("saved_sequence_path_invalid",
                                 f"{query_id}: {mode} result {rank}: {'; '.join(problems)}"))
        evaluated[rank] = not problems and positive_match
    return {cutoff: any(evaluated.get(rank, False)
                        for rank in range(1, min(cutoff, len(results)) + 1))
            for cutoff in cutoffs}, None


def _saved_metric_issues(row, mode, task, hits, results, target_video, query_id):
    issues = []
    metrics = (row.get("modes") or {}).get(mode, {}).get("metrics")
    if not isinstance(metrics, dict):
        issues.append(_issue("saved_metrics_missing", f"{query_id}: {mode}"))
        return issues
    prefix = "known_path_hit_at_" if task == "sequence" else "known_frame_hit_at_"
    for cutoff in (1, 5, 20):
        if hits[cutoff] is None:
            # Unresolved saved outcomes cannot confirm or refute the saved metric.
            continue
        actual = metrics.get(f"{prefix}{cutoff}")
        if actual is None:
            issues.append(_issue("saved_metric_missing", f"{query_id}: {mode} {prefix}{cutoff}"))
        elif bool(actual) != bool(hits[cutoff]):
            issues.append(_issue("saved_metric_mismatch",
                                 f"{query_id}: {mode} {prefix}{cutoff} saved={actual} recomputed={int(hits[cutoff])}"))
    expected_top = bool(results and results[0].get("video_id") == target_video)
    actual_top = metrics.get("top_target_video")
    if actual_top is None:
        issues.append(_issue("saved_metric_missing", f"{query_id}: {mode} top_target_video"))
    elif bool(actual_top) != expected_top:
        issues.append(_issue("saved_metric_mismatch",
                             f"{query_id}: {mode} top_target_video saved={actual_top} recomputed={int(expected_top)}"))
    return issues


def _coverage_report(query_id, task, label, sequence, row, event_by_id, raw_evidence,
                     config, groups, issues, support_cache):
    target_video = sequence["video_id"] if task == "sequence" else label["video_id"]
    local_issues = []
    report = {
        "query_id": query_id,
        "task": task,
        "target_video": target_video,
        "status": "ok",
        "explicit_positive_frames": 0,
        "judgment_counts": {"positive": 0, "negative": 0, "unknown": 0},
        "explicit_positives": [],
        "modes": {},
    }
    positives = []
    if task == "event_frame":
        counts = dict(label["_judgment_counts"])
        for positive in label["_positives"]:
            positives.append({**{key: positive[key] for key in
                                ("video_id", "frame_id", "keyframe_n", "pts_time")},
                              "event_index": 0})
    else:
        counts = {"positive": 0, "negative": 0, "unknown": 0}
        missing_event = False
        for event_index, event_query_id in enumerate(sequence["event_query_ids"]):
            event_query = event_by_id.get(event_query_id)
            if event_query is None:
                local_issues.append(_issue("sequence_event_missing", event_query_id))
                missing_event = True
                continue
            for key in counts:
                counts[key] += event_query["_judgment_counts"][key]
            event_positives = [positive for positive in event_query["_positives"]
                               if positive["video_id"] == sequence["video_id"]]
            if not event_positives:
                missing_event = True
            for positive in event_positives:
                positives.append({**{key: positive[key] for key in
                                    ("video_id", "frame_id", "keyframe_n", "pts_time")},
                                  "event_index": event_index})
        if missing_event:
            report.update(status="unresolved", reason="sequence_event_without_explicit_positive",
                          judgment_counts=counts)
            report["issues"] = _merge_issues(local_issues)
            issues.extend(local_issues)
            return report

    report["judgment_counts"] = counts
    if not _in_groups(target_video, groups):
        local_issues.append(_issue("query_out_of_scope", f"{query_id}: {target_video}"))
        report.update(status="out_of_scope", explicit_positive_frames=0)
        report["issues"] = _merge_issues(local_issues)
        issues.extend(local_issues)
        return report

    scoped = []
    for positive in positives:
        if _in_groups(positive["video_id"], groups):
            scoped.append(positive)
        else:
            local_issues.append(_issue("positive_frame_out_of_scope",
                                       f"{query_id}: {positive['video_id']}/{positive['frame_id']}"))
    positives = scoped
    report["explicit_positives"] = sorted(positives, key=lambda item: (
        item["event_index"], item["video_id"], item["frame_id"]))
    report["explicit_positive_frames"] = len(positives)
    if not positives:
        report["status"] = "no_explicit_positive"
        report["issues"] = _merge_issues(local_issues)
        issues.extend(local_issues)
        return report

    if row is None:
        code = "sequence_saved_row_missing" if task == "sequence" else "saved_row_missing"
        local_issues.append(_issue(code, query_id))
        report["status"] = "missing_row"
        report["modes"] = _missing_row_stages(positives, task)
        report["issues"] = _merge_issues(local_issues)
        issues.extend(local_issues)
        return report
    if row.get("task") != task:
        local_issues.append(_issue("saved_row_task_mismatch",
                                   f"{query_id}: {row.get('task')!r} != {task!r}"))
        report["status"] = "label_mismatch"
        report["modes"] = _missing_row_stages(positives, task, "saved_row_task_mismatch")
        report["issues"] = _merge_issues(local_issues)
        issues.extend(local_issues)
        return report

    problems = _event_query_mismatch(row, label) if task == "event_frame" else _sequence_query_mismatch(row, sequence)
    if problems:
        code = "saved_row_query_mismatch" if task == "event_frame" else "sequence_event_order_mismatch"
        local_issues.append(_issue(code, f"{query_id}: {','.join(problems)}"))
        report["status"] = "label_mismatch"
        report["modes"] = _missing_row_stages(positives, task, code)
        report["issues"] = _merge_issues(local_issues)
        issues.extend(local_issues)
        return report

    expected_actions = ([label["event_text"]] if task == "event_frame"
                        else list(sequence["actions"]))
    action_mismatch_videos = set()
    for video_id in sorted({positive["video_id"] for positive in positives}):
        record = raw_evidence.get((query_id, video_id))
        if record is None or record.get("status") != "ok":
            continue
        if list(record.get("actions") or []) != expected_actions:
            action_mismatch_videos.add(video_id)
            local_issues.append(_issue("raw_evidence_actions_mismatch",
                                       f"{query_id}/{video_id}"))

    event_count = 1 if task == "event_frame" else len(sequence["actions"])
    pool_set, pool_issues = _pool_for_row(row, config, target_video)
    local_issues.extend(pool_issues)
    stream_hits, stream_issues = _stream_hits(row, event_count)
    local_issues.extend(stream_issues)

    stream_outcomes = {name: [None] * len(positives) for name in STREAMS}
    stream_reasons = {name: [None] * len(positives) for name in STREAMS}
    for index, positive in enumerate(positives):
        event_index = positive["event_index"]
        for stream in STREAMS:
            if stream_hits is None or event_index >= len(stream_hits):
                stream_reasons[stream][index] = "saved_stream_unavailable"
                continue
            identities = stream_hits[event_index].get(stream)
            if identities is None:
                stream_reasons[stream][index] = "saved_stream_unavailable"
                continue
            stream_outcomes[stream][index] = (positive["video_id"], positive["frame_id"]) in identities
    union_outcomes, union_reasons = [], []
    for index in range(len(positives)):
        values = [stream_outcomes[stream][index] for stream in STREAMS]
        if any(value is True for value in values):
            union_outcomes.append(True)
            union_reasons.append(None)
        elif all(value is not None for value in values):
            union_outcomes.append(False)
            union_reasons.append(None)
        else:
            union_outcomes.append(None)
            union_reasons.append("saved_stream_unavailable")

    local_outcomes, local_reasons = [], []
    for positive in positives:
        if pool_set is not None and positive["video_id"] not in pool_set:
            local_outcomes.append(None)
            local_reasons.append("not_reached_pool")
            continue
        record = raw_evidence.get((query_id, positive["video_id"]))
        if record is None and pool_set is not None and positive["video_id"] in pool_set:
            local_issues.append(_issue("raw_evidence_missing",
                                       f"{query_id}/{positive['video_id']}"))
        outcome, reason = _local_outcome(record, positive, query_id, local_issues)
        local_outcomes.append(outcome)
        local_reasons.append(reason)

    if task == "sequence":
        sequence_record = raw_evidence.get((query_id, sequence["video_id"]))
        if sequence["video_id"] in action_mismatch_videos:
            feasible, temporal_reason, temporal_missing = (
                None, "raw_evidence_actions_mismatch", [])
        else:
            feasible, temporal_reason, temporal_missing = _temporal_feasibility(
                sequence_record, sequence, positives, config, query_id, local_issues)
        temporal_detail = _temporal_detail(feasible, temporal_reason, False)
        if temporal_missing:
            report["temporal_missing_positives"] = temporal_missing
    else:
        temporal_detail = _temporal_detail(None, None, True)

    def matrices_for(record):
        if record is None or record.get("status") != "ok":
            return None
        key = record.get("path", query_id)
        if key not in support_cache:
            try:
                support_cache[key] = _support_matrices(record, config)
            except (TypeError, ValueError):
                support_cache[key] = None
        return support_cache[key]

    baseline_outcomes, baseline_reasons = [], []
    for positive, local_outcome, local_reason in zip(positives, local_outcomes, local_reasons):
        if local_outcome is not True:
            baseline_outcomes.append(None)
            baseline_reasons.append(local_reason or "local_frame_unavailable")
            continue
        if positive["video_id"] in action_mismatch_videos:
            baseline_outcomes.append(None)
            baseline_reasons.append("raw_evidence_actions_mismatch")
            continue
        record = raw_evidence.get((query_id, positive["video_id"]))
        outcome, reason = _support_outcome(record, positive, matrices_for(record))
        baseline_outcomes.append(outcome)
        baseline_reasons.append(reason)
    guard_outcomes = [None] * len(positives)
    guard_reasons = ["guard_mask_absent"] * len(positives)
    videos = sorted({positive["video_id"] for positive in positives})

    for mode in MODES:
        results = _mode_results(row, mode, task, query_id, local_issues)
        if results is None:
            hits = {cutoff: False for cutoff in (1, 5, 20)}
            metric_match = None
            top_details = {cutoff: _saved_detail(False, False) for cutoff in (1, 5, 20)}
        else:
            unresolved_reason = None
            if task == "sequence":
                hits, unresolved_reason = _saved_sequence_hits(
                    results, sequence,
                    raw_evidence.get((query_id, sequence["video_id"])),
                    config, query_id, mode, local_issues, positives)
            else:
                hits = _saved_event_hits(results, positives)
            metric_issues = _saved_metric_issues(row, mode, task, hits, results, target_video, query_id)
            local_issues.extend(metric_issues)
            # Unresolved outcomes leave the metric comparison unknown instead of
            # reporting a spurious mismatch against an unverifiable recomputation.
            metric_match = (None if any(hit is None for hit in hits.values())
                            else not bool(metric_issues))
            top_details = {
                cutoff: _saved_detail(hits[cutoff], unresolved_reason is None,
                                      unresolved_reason or "saved_results_unavailable")
                for cutoff in (1, 5, 20)
            }
        stages = {
            "visual_hits": _stage_detail(stream_outcomes["visual"], stream_reasons["visual"], "positive_frames"),
            "bm25_hits": _stage_detail(stream_outcomes["bm25"], stream_reasons["bm25"], "positive_frames"),
            "semantic_hits": _stage_detail(stream_outcomes["semantic"], stream_reasons["semantic"], "positive_frames"),
            "union_hits": _stage_detail(union_outcomes, union_reasons, "positive_frames"),
            "candidate_pool": _video_stage_detail(videos, pool_set),
            "local_frames": _stage_detail(local_outcomes, local_reasons, "positive_frames"),
            "support": _stage_detail(baseline_outcomes if mode == "baseline" else guard_outcomes,
                                     baseline_reasons if mode == "baseline" else guard_reasons,
                                     "positive_frames"),
            "temporal_feasibility": temporal_detail,
            "saved_top_1": top_details[1],
            "saved_top_5": top_details[5],
            "saved_top_20": top_details[20],
        }
        report["modes"][mode] = {"stages": stages, "saved_metrics_match": metric_match}

    if action_mismatch_videos:
        report["status"] = "unresolved"
        report["reason"] = "raw_evidence_actions_mismatch"
    report["issues"] = _merge_issues(local_issues)
    issues.extend(local_issues)
    return report


def _aggregate_stages(reports):
    summary = {}
    for task in ("event_frame", "sequence"):
        summary[task] = {}
        for mode in MODES:
            summary[task][mode] = {}
            for stage in STAGE_ORDER:
                summary[task][mode][stage] = {
                    "status": "not_applicable",
                    "unit": STAGE_UNITS[stage],
                    "covered": 0,
                    "resolved": 0,
                    "total": 0,
                    "unresolved": 0,
                    "queries": 0,
                    "queries_with_unresolved": 0,
                    "not_applicable_queries": 0,
                    "coverage": None,
                    "reasons": {},
                }
    for report in reports:
        if report.get("task") not in summary:
            continue
        for mode in MODES:
            stages = report.get("modes", {}).get(mode, {}).get("stages", {})
            for stage in STAGE_ORDER:
                detail = stages.get(stage)
                if not isinstance(detail, dict):
                    continue
                accumulator = summary[report["task"]][mode][stage]
                if detail.get("status") == "not_applicable":
                    accumulator["not_applicable_queries"] += 1
                    continue
                accumulator["queries"] += 1
                accumulator["covered"] += detail.get("covered", 0)
                accumulator["resolved"] += detail.get("resolved", 0)
                accumulator["total"] += detail.get("total", 0)
                accumulator["unresolved"] += detail.get("unresolved", 0)
                if detail.get("unresolved"):
                    accumulator["queries_with_unresolved"] += 1
                for reason, count in detail.get("reasons", {}).items():
                    accumulator["reasons"][reason] = accumulator["reasons"].get(reason, 0) + count
    for task in summary:
        for mode in summary[task]:
            for stage, accumulator in summary[task][mode].items():
                accumulator["coverage"] = _ratio(accumulator["covered"], accumulator["resolved"])
                accumulator["reasons"] = dict(sorted(accumulator["reasons"].items()))
                if accumulator["queries"] == 0:
                    accumulator["status"] = "not_applicable"
                elif accumulator["resolved"] == 0 and accumulator["total"] > 0:
                    accumulator["status"] = "unknown"
                elif accumulator["unresolved"]:
                    accumulator["status"] = "partial"
                else:
                    accumulator["status"] = "known"
    return summary


def _stage_coverage(rows, event_by_id, sequences, raw_evidence, context, issues):
    config = context["_temporal"]
    groups = set(context["_groups"])
    row_by_id = {row["query_id"]: row for row in rows}
    sequence_by_id = {sequence["query_id"]: sequence for sequence in sequences}
    reports = []
    support_cache = {}
    for query_id in sorted(event_by_id):
        reports.append(_coverage_report(query_id, "event_frame", event_by_id[query_id], None,
                                        row_by_id.get(query_id), event_by_id, raw_evidence,
                                        config, groups, issues, support_cache))
    for query_id in sorted(sequence_by_id):
        reports.append(_coverage_report(query_id, "sequence", None, sequence_by_id[query_id],
                                        row_by_id.get(query_id), event_by_id, raw_evidence,
                                        config, groups, issues, support_cache))
    extra = sorted(set(row_by_id) - set(event_by_id) - set(sequence_by_id))
    for query_id in extra:
        row = row_by_id[query_id]
        query = row.get("query") if isinstance(row.get("query"), dict) else {}
        issues.append(_issue("saved_row_out_of_scope", query_id))
        reports.append({
            "query_id": query_id,
            "task": row.get("task"),
            "target_video": query.get("video_id"),
            "status": "out_of_scope",
            "explicit_positive_frames": 0,
            "judgment_counts": {"positive": 0, "negative": 0, "unknown": 0},
            "explicit_positives": [],
            "modes": {},
            "issues": [_issue("saved_row_out_of_scope", query_id)],
        })
    reports.sort(key=lambda item: (item["query_id"], item["task"]))
    return reports, _aggregate_stages(reports)


def _artifact_summary(details, faiss_state):
    status_counts = Counter(detail.get("status") for detail in details.values())
    artifact_counts = Counter()
    for detail in details.values():
        for artifact in detail.get("artifacts", {}).values():
            artifact_counts[artifact.get("status")] += 1
    return {
        "audited_videos": len(details),
        "status_counts": dict(sorted(status_counts.items())),
        "artifact_status_counts": dict(sorted(artifact_counts.items())),
        "faiss": faiss_state,
    }


def _write_report(path, report):
    path = Path(path)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run(source_run, labels_path, mapping_root, caption_root, output):
    """Build and write one deterministic report; no model, network or live-service access."""
    source_run = Path(source_run)
    labels_path = Path(labels_path)
    mapping_root = Path(mapping_root)
    caption_root = Path(caption_root)
    output = Path(output)
    if output.exists():
        raise AuditError(f"output_exists: refusing to overwrite {output}")
    if not source_run.is_dir():
        raise AuditError(f"source_run_missing: {source_run}")
    if not labels_path.is_file():
        raise AuditError(f"labels_missing: {labels_path}")
    if not mapping_root.is_dir():
        raise AuditError(f"mapping_root_missing: {mapping_root}")
    if not caption_root.is_dir():
        raise AuditError(f"caption_root_missing: {caption_root}")
    context_path = source_run / "context.json"
    if not context_path.is_file():
        raise AuditError(f"context_missing: {context_path}")
    if output.resolve() in {labels_path.resolve(), context_path.resolve()}:
        raise AuditError("output_conflicts_with_input: output path is an input file")

    label_issues = []
    labels, event_by_id, event_queries, sequences = _load_labels(labels_path, label_issues)
    labels_sha = _sha256_file(labels_path)
    context = _load_context(context_path, labels_sha)
    if (labels.get("protocol_sha256") and context.get("protocol_sha256")
            and labels["protocol_sha256"] != context["protocol_sha256"]):
        raise AuditError("label_protocol_mismatch: labels and saved context differ")
    if (context.get("annotation_source") != labels.get("annotation_source")
            or context.get("human_verified") is not True):
        raise AuditError("label_provenance_mismatch: labels and saved context differ")

    issues = list(label_issues)
    rows, row_files, row_issues = _load_saved_rows(source_run)
    issues.extend(row_issues)
    raw_evidence, raw_files, raw_issues = _load_raw_evidence(source_run)
    issues.extend(raw_issues)
    protected = {labels_path.resolve(), context_path.resolve()}
    protected.update((source_run / item["path"]).resolve() for item in row_files)
    protected.update((source_run / item["path"]).resolve() for item in raw_files)
    if output.resolve() in protected:
        raise AuditError("output_conflicts_with_input: output path is a saved input")

    summary_path = source_run / "summary.json"
    if summary_path.is_file():
        try:
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue("run_summary_unreadable", type(exc).__name__))
        else:
            if (isinstance(run_summary, dict) and run_summary.get("label_sha256")
                    and run_summary["label_sha256"] != labels_sha):
                issues.append(_issue("summary_label_hash_mismatch", "summary.json labels differ"))

    captured_model_config = _captured_model_config(context)
    if captured_model_config.get("status") == "invalid":
        issues.append(_issue("captured_model_config_invalid",
                             str(captured_model_config.get("reason"))))
    # Only a hash-verified captured config may drive historical artifact
    # expectations.  An unavailable current checkout config is never substituted.
    captured_verified = captured_model_config.get("status") == "verified"
    video_model_config = (captured_model_config.get("visual_embedding")
                          if captured_verified else None)
    caption_model_config = (captured_model_config.get("caption_embedding")
                            if captured_verified else None)
    fast_caption_config = (captured_model_config.get("fast_caption_generation")
                           if captured_verified else None)

    roots = {
        "mapping": mapping_root,
        "features": caption_root / "clip-features-32",
        "captions": caption_root / "caption_generator",
        "caption_mapping": caption_root / "caption_mapping",
        "index": caption_root / "index",
    }
    indexes = {name: _index_root(root) for name, root in roots.items()}
    faiss_state = _faiss_status()

    videos = set()
    for query in event_queries:
        videos.add(query["video_id"])
    for record in raw_evidence.values():
        if record.get("video_id"):
            videos.add(record["video_id"])
    for row in rows:
        candidate_videos = row.get("candidate_videos")
        if isinstance(candidate_videos, list):
            videos.update(video for video in candidate_videos if isinstance(video, str))
    videos = {video for video in videos if VIDEO_ID_PATTERN.fullmatch(video)}

    label_judgments = {}
    seen_judgments = set()
    for query in event_queries:
        for positive in query["_positives"]:
            key = (positive["video_id"], positive["frame_id"], positive["keyframe_n"],
                   positive["pts_time"], positive.get("image_sha256"))
            if key in seen_judgments:
                continue
            seen_judgments.add(key)
            label_judgments.setdefault(positive["video_id"], []).append(positive)

    artifact_details = {}
    artifact_hashes = []
    artifact_issues = []
    for video_id in sorted(videos):
        try:
            public, _ = _artifact_detail(
                video_id, roots, indexes, context["_groups"], faiss_state,
                video_model_config, caption_model_config, fast_caption_config,
                label_judgments.get(video_id, []),
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            public = {
                "video_id": video_id,
                "captured_scope": _in_groups(video_id, context["_groups"]),
                "status": "issues",
                "issues": [_issue("artifact_audit_failed", f"{type(exc).__name__}: {exc}")],
                "artifacts": {},
                "identity": {},
            }
        artifact_details[video_id] = public
        for name, artifact in public.get("artifacts", {}).items():
            if artifact.get("sha256"):
                artifact_hashes.append((f"{video_id}/{name}", artifact["sha256"]))
        for issue in public.get("issues", []):
            artifact_issues.append({"code": issue["code"],
                                    "detail": f"{video_id}: {issue['detail']}"})
    issues.extend(artifact_issues)

    caption_hashes = _normalized_path_map(context.get("caption_artifact_sha256"))
    for video_id in sorted(artifact_details):
        caption = artifact_details[video_id].get("artifacts", {}).get("caption_generator", {})
        if not caption.get("sha256"):
            continue
        key = _normalize_path(f"caption_generator/{caption.get('path', '')}")
        expected = caption_hashes.get(key)
        if expected is None:
            issues.append(_issue("caption_artifact_hash_unregistered", video_id))
        elif expected != caption["sha256"]:
            issues.append(_issue("caption_artifact_hash_mismatch", video_id))

    local_artifacts_path = source_run / "local_artifact_sha256.json"
    if local_artifacts_path.is_file():
        try:
            local_artifacts = json.loads(local_artifacts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue("local_artifact_hashes_unreadable", type(exc).__name__))
            local_artifacts = {}
        if not isinstance(local_artifacts, dict):
            issues.append(_issue("local_artifact_hashes_invalid", "root must be an object"))
            local_artifacts = {}
        for video_id, expected in sorted(local_artifacts.items()):
            detail = artifact_details.get(video_id)
            if not isinstance(expected, dict):
                issues.append(_issue("local_artifact_hashes_invalid", video_id))
                continue
            for key, artifact_name in (("mapping_sha256", "mapping"),
                                       ("features_sha256", "visual_features")):
                expected_hash = expected.get(key)
                if not expected_hash:
                    continue
                actual = detail.get("artifacts", {}).get(artifact_name, {}).get("sha256") if detail else None
                if not actual:
                    issues.append(_issue("local_artifact_hash_unverifiable", f"{video_id}:{key}"))
                elif actual != expected_hash:
                    issues.append(_issue("local_artifact_hash_mismatch", f"{video_id}:{key}"))

    query_reports, stages = _stage_coverage(rows, event_by_id, sequences, raw_evidence, context, issues)
    # Deduplicate once all producers have contributed, so the top-level issues,
    # histogram, total and overall status all use the same deterministic list.
    issues = _merge_issues(issues)
    artifacts = _artifact_summary(artifact_details, faiss_state)
    provenance = _audit_provenance(labels_path, context_path, context, row_files, raw_files,
                                   artifact_hashes, labels=labels,
                                   captured_model_config=captured_model_config)

    status_counts = Counter(report["status"] for report in query_reports)
    issue_counts = Counter(issue["code"] for issue in issues)
    unresolved_stage_units = sum(
        accumulator["unresolved"]
        for task in stages.values() for mode in task.values() for accumulator in mode.values()
    )
    if issues:
        overall_status = "issues"
    elif unresolved_stage_units or artifacts["status_counts"].get("unknown"):
        overall_status = "unknown"
    else:
        overall_status = "ok"

    label_counts = Counter()
    for query in event_queries:
        for key, value in query["_judgment_counts"].items():
            label_counts[key] += value
    inventory = {
        "source_run": _relative(source_run, KIS_ROOT),
        "labels": {
            "path": _relative(labels_path, KIS_ROOT),
            "queries": len(event_queries),
            "sequences": len(sequences),
            "judgments": dict(sorted(label_counts.items())),
            "skipped_samples": len(labels.get("skipped_samples") or []),
        },
        "context": {
            "path": _relative(context_path, KIS_ROOT),
            "mode": context.get("mode"),
            "video_groups": list(context["_groups"]),
            "caption_vectors": context.get("caption_vectors"),
            "collection_id": context.get("collection_id"),
            "nprobe": context.get("nprobe"),
            "device": context.get("device"),
        },
        "saved_rows": {
            "files": len(row_files),
            "loaded": len(rows),
            "event_frame": sum(row.get("task") == "event_frame" for row in rows),
            "sequence": sum(row.get("task") == "sequence" for row in rows),
        },
        "raw_evidence": {
            "files": len(raw_files),
            "units": len({record.get("unit_id") for record in raw_evidence.values()}),
            "videos": len({record.get("video_id") for record in raw_evidence.values()}),
        },
        "artifacts": artifacts,
        "roots": {name: _relative(root, KIS_ROOT) for name, root in roots.items()},
    }
    summary = {
        "status": overall_status,
        "counts": {
            "label_event_queries": len(event_by_id),
            "label_sequences": len(sequences),
            "saved_rows": len(rows),
            "report_queries": len(query_reports),
            "explicit_positive_frames": sum(report["explicit_positive_frames"]
                                             for report in query_reports),
            "unknown_judgments": sum(report["judgment_counts"].get("unknown", 0)
                                     for report in query_reports),
            "audited_events": status_counts.get("ok", 0),
            "missing_rows": status_counts.get("missing_row", 0),
            "out_of_scope_queries": status_counts.get("out_of_scope", 0),
            "label_mismatches": status_counts.get("label_mismatch", 0),
            "no_explicit_positive_queries": status_counts.get("no_explicit_positive", 0),
            "unresolved_queries": status_counts.get("unresolved", 0),
            "artifact_videos": artifacts["audited_videos"],
        },
        "query_status": dict(sorted(status_counts.items())),
        "artifact_status": artifacts["status_counts"],
        "unresolved_stage_units": unresolved_stage_units,
        "issues": {"total": len(issues), "by_code": dict(sorted(issue_counts.items()))},
        "stages": stages,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "provenance": provenance,
        "inventory": inventory,
        "artifacts": artifact_details,
        "queries": query_reports,
        "stages": stages,
        "summary": summary,
        "issues": issues,
        "limitations": list(LIMITATIONS),
    }
    _write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--caption-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.source_run, args.labels, args.mapping_root, args.caption_root, args.output)
    except AuditError as exc:
        print(f"p0_audit: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"p0_audit: fatal {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
