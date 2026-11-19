"""Explicit L04/L05 visual-only local Milvus staging; never publish or clean up.

Each mode exclusively creates its report: preflight.json, report.json (ingest),
or verify.json. Existing collections are rejected by preflight/ingest. Verify is
read-only apart from loading the explicitly named, schema-marked staging index.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from tools import l04_l05_staging as offline


local = offline.local
VIDEO_IDS = offline.VIDEO_IDS
EXPECTED = offline.EXPECTED
DIMENSION = 768
TAG = "kis_l04_l05_local_milvus_staging_v1"
SCORE_ATOL = 1e-4
TIE_ATOL = 1e-6
MODES = ("preflight", "ingest", "verify")
READINESS_TIMEOUT = 300
READINESS_POLL_INTERVAL = 1
SCHEMA = {
    "auto_id": True,
    "enable_dynamic_field": False,
    "fields": [
        {"name": "id", "datatype": "INT64", "is_primary": True},
        {"name": "video_id", "datatype": "VARCHAR", "max_length": 256},
        {"name": "frame_id", "datatype": "INT64"},
        {"name": "embedding", "datatype": "FLOAT_VECTOR", "dim": DIMENSION},
    ],
    "index": {"field_name": "embedding", "index_type": "IVF_FLAT",
              "metric_type": "IP", "params": {"nlist": 128}},
}


def validate_target(uri, collection):
    parsed = urlsplit(uri)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.port is None or not 1 <= parsed.port <= 65535):
        raise ValueError("Only explicit credential-free loopback HTTP host:port permitted")
    if not re.fullmatch(r"kis_l04_l05_stg_[0-9]{8}_[0-9]{2,}", collection):
        raise ValueError("Only isolated kis_l04_l05_stg_DATE_NUMBER collection permitted")
    return f"http://localhost:{parsed.port}"


def read_video(dataset_root, features_root, video_id):
    offline.require_video(video_id)  # Before any path validation/lstat/read.
    mapping = offline.artifact_path(dataset_root, "mapping", video_id)
    features = local.safe_path(features_root, video_id[:3], video_id + ".npy")
    frames = local.read_mapping(mapping)
    pairs = [(video_id, frame["frame_id"]) for frame in frames]
    offline.validate_identities(pairs)
    if [frame["keyframe_n"] for frame in frames] != list(range(1, len(frames) + 1)):
        raise ValueError("Mapping keyframe rows must be contiguous from 1")
    raw = np.load(features, allow_pickle=False)
    offline.validate_vectors(raw, len(frames))
    # Exact builder convention; neither raw scores nor caption FAISS is used.
    vectors = raw.astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    offline.validate_vectors(vectors, len(frames))
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Float32 L2 normalization failed")
    return vectors, pairs, {
        "video_id": video_id, "records": len(pairs),
        "mapping_sha256": local.sha256(mapping), "features_sha256": local.sha256(features),
        "identity_sha256": offline.digest_json(pairs),
    }


def preflight(dataset_root, features_root):
    arrays, identities, videos, query_rows = [], [], [], []
    for video_id in VIDEO_IDS:
        vectors, pairs, detail = read_video(dataset_root, features_root, video_id)
        query_rows.append(len(identities) + len(pairs) // 2)
        arrays.append(vectors)
        identities.extend(pairs)
        videos.append(detail)
    offline.validate_identities(identities)
    groups = {group: {"videos": sum(v["video_id"].startswith(group + "_") for v in videos),
                      "records": sum(v["records"] for v in videos
                                     if v["video_id"].startswith(group + "_"))}
              for group in EXPECTED}
    if groups != EXPECTED:
        raise ValueError("Explicit L04/L05 group counts differ from fixed expected counts")
    return np.concatenate(arrays), identities, query_rows, videos, groups


def create_collection(client, collection, owner_token):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False,
                                  description=f"{TAG};owner={owner_token}")
    for field in SCHEMA["fields"]:
        options = {key: value for key, value in field.items() if key not in {"name", "datatype"}}
        schema.add_field(field_name=field["name"], datatype=DataType[field["datatype"]], **options)
    index = client.prepare_index_params()
    index.add_index(**SCHEMA["index"])
    client.create_collection(collection_name=collection, schema=schema, index_params=index,
                             consistency_level="Strong", timeout=120)


def owner_from_description(description):
    match = re.fullmatch(re.escape(TAG) + r";owner=([0-9a-f]{32})",
                         description.get("description", ""))
    if not match:
        raise ValueError("Missing or invalid staging ownership marker")
    return match[1]


def verify_ownership(client, collection, report):
    ownership = report["ownership"]
    ownership["status"] = "unverified"
    description = client.describe_collection(collection_name=collection, timeout=30)
    observed = owner_from_description(description)
    ownership["observed_owner_token"] = observed
    if report["mode"] == "ingest" and observed != ownership["owner_token"]:
        ownership["status"] = "mismatch"
        raise ValueError("Collection belongs to another run; insert forbidden")
    ownership["status"] = "owned_by_this_run" if report["mode"] == "ingest" else "marker_verified_read_only"


def verify_schema(client, collection):
    from pymilvus import DataType

    description = client.describe_collection(collection_name=collection, timeout=30)
    owner_from_description(description)
    if (description.get("auto_id") is not True
            or description.get("enable_dynamic_field") is not False):
        raise ValueError("Staging marker/auto_id/dynamic schema mismatch")
    fields = description.get("fields", [])
    if len(fields) != len(SCHEMA["fields"]):
        raise ValueError("Unexpected collection fields")
    by_name = {field["name"]: field for field in fields}
    if len(by_name) != len(fields):
        raise ValueError("Duplicate schema fields")
    for expected in SCHEMA["fields"]:
        actual = by_name[expected["name"]]
        if int(actual["type"]) != int(DataType[expected["datatype"]]):
            raise ValueError("Field datatype mismatch")
        if bool(actual.get("is_primary", False)) != expected.get("is_primary", False):
            raise ValueError("Primary field mismatch")
        for key in ("max_length", "dim"):
            if key in expected and int(actual.get("params", {}).get(key, -1)) != expected[key]:
                raise ValueError("Field dimension/length mismatch")
    index = client.describe_index(collection_name=collection, index_name="embedding", timeout=30)
    for key in ("field_name", "index_type", "metric_type"):
        if index.get(key) != SCHEMA["index"][key]:
            raise ValueError("Index schema mismatch")
    if int(index.get("nlist", index.get("params", {}).get("nlist", -1))) != 128:
        raise ValueError("Index nlist mismatch")
    return {"status": "ok", **SCHEMA}


class ReadinessError(ValueError):
    pass


def wait_for_readiness(client, collection, expected_rows, report):
    """Require observed full index coverage, never infer it from count/Finished."""
    started = time.monotonic()
    deadline = started + READINESS_TIMEOUT
    gate = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": READINESS_TIMEOUT, "poll_interval_seconds": READINESS_POLL_INTERVAL,
        "started_monotonic": started, "deadline_monotonic": deadline,
        "expected_rows": expected_rows, "snapshots": [], "terminal_reason": None,
    }
    report["readiness"] = gate
    snapshot = {}

    def rpc(method, *, budget_fraction=1, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReadinessError("timeout")
        result = method(collection_name=collection,
                        timeout=min(30, remaining * budget_fraction), **kwargs)
        if time.monotonic() >= deadline:
            raise ReadinessError("timeout")
        return result

    def row_count(value):
        if type(value) not in (int, str) or not re.fullmatch(r"[0-9]+", str(value)):
            raise ReadinessError("missing_or_invalid_count_field")
        return int(value)

    def record():
        now = time.monotonic()
        snapshot.update(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=now - started, remaining_seconds=max(0, deadline - now))
        gate["snapshots"].append(dict(snapshot))
        gate["final_snapshot"] = dict(snapshot)
        gate["elapsed_seconds"] = now - started

    try:
        snapshot = {"operation": "load_collection"}
        rpc(client.load_collection)
        while True:
            snapshot = {"operation": "poll", "collection_entity_count": None,
                        "strong_query_count": None, "total_rows": None, "indexed_rows": None,
                        "pending_index_rows": None, "index_state": None,
                        "load_state": None, "load_progress": None}
            stats = rpc(client.get_collection_stats)
            snapshot["collection_entity_count"] = row_count(stats.get("row_count"))
            report["counts"]["collection_entity_count"] = snapshot["collection_entity_count"]
            count = rpc(client.query, output_fields=["count(*)"], consistency_level="Strong")
            snapshot["strong_query_count"] = row_count(count[0].get("count(*)"))
            report["counts"]["strong_query_count"] = snapshot["strong_query_count"]
            index = rpc(client.describe_index, index_name="embedding")
            for field in ("total_rows", "indexed_rows", "pending_index_rows"):
                snapshot[field] = row_count(index.get(field))
            snapshot["index_state"] = index.get("state")
            if snapshot["index_state"] == "Failed":
                raise ReadinessError("index_failed")
            if snapshot["index_state"] not in {"Unissued", "InProgress", "Finished", "Retry"}:
                raise ReadinessError("missing_or_invalid_index_state")
            # SDK get_load_state can make two RPCs (state then Loading progress).
            load = rpc(client.get_load_state, budget_fraction=0.5)
            state = load.get("state")
            snapshot["load_state"] = str(state) if state is not None else None
            snapshot["load_progress"] = load.get("progress")  # Absent is unknown, not 100%.
            if snapshot["load_state"] not in {"NotLoad", "Loading", "Loaded"}:
                raise ReadinessError("missing_or_invalid_load_state")
            if (snapshot["collection_entity_count"] != expected_rows
                    or snapshot["strong_query_count"] != expected_rows
                    or snapshot["total_rows"] > expected_rows
                    or snapshot["indexed_rows"] > expected_rows):
                raise ReadinessError("count_mismatch")
            ready = (snapshot["total_rows"] == expected_rows
                     and snapshot["indexed_rows"] == expected_rows
                     and snapshot["pending_index_rows"] == 0
                     and snapshot["index_state"] == "Finished"
                     and snapshot["load_state"] == "Loaded")
            record()
            if ready:
                gate.update(status="ok", terminal_reason="full_index_coverage_and_loaded")
                return
            time.sleep(min(READINESS_POLL_INTERVAL, max(0, deadline - time.monotonic())))
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, ReadinessError) else "rpc_or_api_error"
        gate.update(status="failed", terminal_reason=reason, error_type=type(exc).__name__)
        record()
        raise


def verify_collection(client, collection, vectors, identities, query_rows, report):
    verify_ownership(client, collection, report)
    report["stage"] = "readiness"
    wait_for_readiness(client, collection, len(identities), report)
    report["verified_schema"] = verify_schema(client, collection)
    report["stage"] = "verify"
    primary_ids = set()
    primary_pairs = {}
    expected_pairs = set(identities)
    actual_counts = {}
    for video_id in VIDEO_IDS:
        offline.require_video(video_id)
        rows = client.query(collection_name=collection, filter=f'video_id == "{video_id}"',
                            output_fields=["id", "video_id", "frame_id"], limit=16384,
                            consistency_level="Strong", timeout=30)
        pairs = [(row["video_id"], row["frame_id"]) for row in rows]
        for pair in pairs:
            offline.require_video(pair[0])
        wanted = {pair for pair in expected_pairs if pair[0] == video_id}
        if len(pairs) != len(set(pairs)) or set(pairs) != wanted:
            raise ValueError("Per-video count/identity/duplicate mismatch")
        for row in rows:
            if type(row["id"]) is not int or row["id"] in primary_ids:
                raise ValueError("Invalid or duplicate primary ID")
            primary_ids.add(row["id"])
            primary_pairs[row["id"]] = (row["video_id"], row["frame_id"])
        actual_counts[video_id] = len(rows)
        report["counts"]["per_video"] = dict(actual_counts)
    if len(primary_ids) != len(identities):
        raise ValueError("Scope/primary count mismatch")
    report["counts"]["groups"] = {
        group: sum(n for video, n in actual_counts.items() if video.startswith(group + "_"))
        for group in EXPECTED}
    report["counts"]["identities_verified"] = True

    baseline = offline.exact_ip(vectors, identities, query_rows, k=5)
    baseline.update(normalized=True, normalization="float32_l2_builder_convention")
    report["normalized_offline_baseline"] = baseline
    lookup = {pair: i for i, pair in enumerate(identities)}
    pair_primary = {pair: primary for primary, pair in primary_pairs.items()}
    matrix = np.asarray(vectors, dtype=np.float64)
    comparison = {
        "status": "running", "k": 5, "query_count": 0, "matched_queries": 0,
        "mismatched_queries": 0, "strict_id_rank_matches": 0, "tie_adjusted_matches": 0,
        "score_atol": SCORE_ATOL, "tie_atol": TIE_ATOL, "rtol": 0,
        "baseline_arithmetic": "float32_L2_vectors_then_float64_exhaustive_IP_fixed_reduction",
        "tie_policy": ("Require every ID scoring above cutoff + tie_atol; boundary alternatives "
                       "must be within tie_atol of cutoff. At each rank, exact IP must be within "
                       "tie_atol of baseline rank score; otherwise IDs/ranks must match exactly. "
                       "Distinct IDs only; returned scores must match their exact IP within score_atol."),
        "semantic_quality_evaluated": False,
    }
    report["top_k_comparison"] = comparison
    queries = []
    for row, baseline_query in zip(query_rows, baseline["queries"]):
        exact_scores = np.einsum("ij,j->i", matrix, matrix[row], optimize=False)
        baseline_rows = [result["row_id"] for result in baseline_query["results"]]
        cutoff = exact_scores[baseline_rows[-1]]
        required = set(np.flatnonzero(exact_scores > cutoff + TIE_ATOL).tolist())
        boundary = set(np.flatnonzero(np.abs(exact_scores - cutoff) <= TIE_ATOL).tolist())
        hits = client.search(collection_name=collection, data=[vectors[row].tolist()],
                             anns_field="embedding", limit=5,
                             output_fields=["video_id", "frame_id"],
                             search_params={"metric_type": "IP", "params": {"nprobe": 128}},
                             consistency_level="Strong", timeout=30)[0]
        if len(hits) != min(5, len(identities)):
            raise ValueError("Structural result count mismatch")
        results, seen, seen_ids, returned_rows, issues = [], set(), set(), [], []
        previous = float("inf")
        for hit in hits:
            entity = hit["entity"]
            pair = (entity["video_id"], entity["frame_id"])
            offline.require_video(pair[0])
            score = float(hit["distance"])
            if (pair not in lookup or pair in seen or hit["id"] in seen_ids
                    or primary_pairs.get(hit["id"]) != pair
                    or not np.isfinite(score) or score > previous + SCORE_ATOL):
                raise ValueError("Invalid structural result identity/scope/score/order")
            label = lookup[pair]
            expected_score = float(exact_scores[label])
            if not np.isclose(score, expected_score, atol=SCORE_ATOL, rtol=0):
                issues.append({"reason": "score_mismatch", "rank": len(results) + 1})
            if abs(expected_score - exact_scores[baseline_rows[len(results)]]) > TIE_ATOL:
                issues.append({"reason": "rank_or_id_mismatch", "rank": len(results) + 1})
            returned_rows.append(label)
            seen.add(pair)
            seen_ids.add(hit["id"])
            previous = score
            results.append({"id": hit["id"], "video_id": pair[0], "frame_id": pair[1],
                            "score": score, "normalized_offline_ip": expected_score})
        missing = sorted(required - set(returned_rows))
        unexpected = sorted(set(returned_rows) - required - boundary)
        if missing:
            issues.append({"reason": "missing_high_score_ids", "row_ids": missing})
        if unexpected:
            issues.append({"reason": "outside_top_k_or_boundary_ties", "row_ids": unexpected})
        strict = returned_rows == baseline_rows
        matched = not issues
        comparison["query_count"] += 1
        comparison["matched_queries"] += int(matched)
        comparison["mismatched_queries"] += int(not matched)
        comparison["strict_id_rank_matches"] += int(matched and strict)
        comparison["tie_adjusted_matches"] += int(matched and not strict)
        queries.append({"query_row": row, "video_id": identities[row][0],
                        "frame_id": identities[row][1], "results": results,
                        "baseline_primary_ids": [pair_primary[identities[i]] for i in baseline_rows],
                        "top_k_status": "ok" if matched else "issues", "issues": issues,
                        "strict_id_rank_match": strict,
                        "boundary_tied_candidates": len(boundary), "cutoff_score": float(cutoff)})
        report["queries"] = list(queries)
    report["query_count"] = len(queries)
    comparison["status"] = "issues" if comparison["mismatched_queries"] else "ok"
    if comparison["mismatched_queries"]:
        raise ValueError("Milvus top-k differs from exhaustive normalized IP baseline")


def run(mode, dataset_root, features_root, output_dir, uri, collection, *, batch_size=256,
        client_factory=None):
    endpoint = validate_target(uri, collection)  # No filesystem or client before validation.
    if mode not in MODES or type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise ValueError("Invalid staging mode/batch size")
    owner_token = uuid4().hex if mode == "ingest" else None
    # Validate both root names before the first lstat (including excluded-group roots).
    for root in (dataset_root, features_root, output_dir):
        if ".." in Path(root).parts or str(root).startswith(("\\\\", "//")):
            raise local.UnsafePath("Traversal or network root rejected before I/O")
        for part in Path(root).parts:
            if (re.match(r"L06(?:_|$)", part, re.IGNORECASE)
                    or (re.fullmatch(r"L\d+(?:_V.*)?", part, re.IGNORECASE)
                        and part not in {"L04", "L05", *VIDEO_IDS})):
                raise local.UnsafePath("Excluded group/video root rejected before I/O")
    dataset_root = local.safe_path(dataset_root)
    features_root = local.safe_path(features_root)
    if features_root != dataset_root / "clip-features-32":
        raise local.UnsafePath("Only explicit dataset/clip-features-32 root permitted")
    output_dir = local.safe_path(output_dir)
    if output_dir.is_relative_to(dataset_root) or dataset_root.is_relative_to(output_dir):
        raise local.UnsafePath("Report output must be outside and not contain dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = {"preflight": "preflight.json", "ingest": "report.json", "verify": "verify.json"}[mode]
    report_path = output_dir / filename
    # Reserve the report before any input payload read, connection or mutation.
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        report = {
            "schema": TAG, "mode": mode, "status": "failed", "stage": "preflight",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint, "collection_name": collection, "requested_schema": SCHEMA,
            "normalized": True, "normalization": "float32_l2_builder_convention", "metric": "IP",
            "scope": {"video_ids": list(VIDEO_IDS), "expected": EXPECTED},
            "production_published": False, "l06_included": False,
            "accuracy": {"status": "not_evaluated", "Hit@K": "not_evaluated", "MRR": "not_evaluated"},
            "counts": {"acknowledged_inserted_this_run": 0, "actual_inserted_this_run": 0,
                       "collection_entity_count": None, "per_video": {}, "identities_verified": False},
            "create_attempted": False, "collection_created": False, "collection_retained": False,
            "ownership": {"owner_token": owner_token, "observed_owner_token": None,
                          "status": "not_checked"},
            "query_count": 0, "queries": [], "batch_size": batch_size,
            "completed_batches": [],
            "versions": {"python": sys.version, "numpy": np.__version__},
            "consistency_level": "Strong", "search_params": {"metric_type": "IP", "nprobe": 128},
            "safety": {"dataset_written": False, "production_config_accessed": False,
                       "models_loaded": False, "discovery": "none_explicit_allowlist_only",
                       "cleanup_performed": False, "alias_operations": False},
            "limitations": ["Structural visual queries only, not semantic quality or production approval",
                            "Inputs must remain stable; path guards are not an adversarial OS sandbox",
                            "On an unacknowledged batch, actual inserted rows are unknown; no retry/cleanup"],
            "source_sha256": local.sha256(__file__),
            "helper_sha256": local.sha256(offline.__file__),
        }
        client = None
        try:
            vectors, identities, query_rows, videos, groups = preflight(dataset_root, features_root)
            report["preflight"] = {"status": "ok", "records": len(identities),
                                   "groups": groups, "videos": videos}
            report["stage"] = "connect_and_check_target"
            if client_factory is None:
                import pymilvus
                from pymilvus import MilvusClient

                client_factory = MilvusClient
                report["versions"]["pymilvus"] = pymilvus.__version__
            client = client_factory(uri=uri, timeout=30)
            exists = client.has_collection(collection_name=collection, timeout=30)
            if mode != "verify" and exists:
                raise ValueError("Collection already exists; never reuse or replace")
            if mode == "verify" and not exists:
                raise ValueError("Explicit staging collection does not exist")
            if mode == "ingest":
                report["stage"] = "create_collection"
                report["create_attempted"] = True
                report["collection_retained"] = True
                create_collection(client, collection, owner_token)
                report["stage"] = "verify_ownership_before_insert"
                verify_ownership(client, collection, report)
                report["collection_created"] = True
                report["stage"] = "insert"
                for start in range(0, len(identities), batch_size):
                    end = min(start + batch_size, len(identities))
                    report["pending_batch"] = {"start": start, "rows": end - start}
                    rows = [{"video_id": identities[i][0], "frame_id": identities[i][1],
                             "embedding": vectors[i].tolist()} for i in range(start, end)]
                    report["counts"]["actual_inserted_this_run"] = None
                    result = client.insert(collection_name=collection, data=rows, timeout=60)
                    inserted = int(result["insert_count"])
                    if not 0 <= inserted <= len(rows):
                        raise ValueError("Invalid batch acknowledgement count")
                    report["counts"]["acknowledged_inserted_this_run"] += inserted
                    report["counts"]["actual_inserted_this_run"] = report["counts"]["acknowledged_inserted_this_run"]
                    if inserted != len(rows):
                        raise ValueError("Partial batch insert; stopping without retry/cleanup")
                    report["completed_batches"].append({"start": start, "insert_count": inserted})
                    del report["pending_batch"]
                report["stage"] = "flush"
                client.flush(collection_name=collection, timeout=120)
            if mode in {"ingest", "verify"}:
                report["collection_retained"] = True
                report["stage"] = "verify"
                verify_collection(client, collection, vectors, identities, query_rows, report)
            report.update(status="ok", stage="complete")
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "partial" if report["create_attempted"] else "failed"
            if (report["stage"] == "verify_ownership_before_insert"
                    or report.get("readiness", {}).get("status") == "failed"
                    or report.get("top_k_comparison", {}).get("status") == "issues"):
                report["status"] = "failed"
            # Do not serialize arbitrary server errors, which might contain secrets.
            report["error"] = {"type": type(exc).__name__,
                               "reason": "Stage failed; stopped without retry, publish or cleanup"}
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    report["client_close_status"] = "failed"
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = run(args.mode, args.dataset_root, args.features_root, args.output_dir,
                     args.uri, args.collection, batch_size=args.batch_size)
    except (OSError, ValueError):
        parser.exit(2, "Invalid/unsafe target, paths or existing report; no ingest started\n")
    print(json.dumps({key: report[key] for key in ("status", "stage", "collection_name", "counts", "query_count")},
                     sort_keys=True, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
