"""Offline-only L04/L05 preflight and visual self-retrieval structural smoke.

No service client, model, production config or whole-root discovery is used.
Run as ``python -B -m tools.l04_l05_staging`` with explicit local input roots.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from tools import l04_l05_local_smoke as local


EXPECTED = {"L04": {"videos": 30, "records": 7471}, "L05": {"videos": 31, "records": 7447}}
VIDEO_IDS = tuple(f"L04_V{n:03d}" for n in range(1, 31)) + tuple(
    f"L05_V{n:03d}" for n in range(1, 32))
DIMENSION = 768
MODES = ("preflight", "structural-smoke")
REPO_ROOT = Path(__file__).resolve().parents[1]


def require_video(video_id):
    # Must precede even lstat, including for caller-supplied roots.
    if video_id not in VIDEO_IDS:
        raise local.UnsafePath("Video outside fixed L04/L05 allowlist")


def artifact_path(dataset_root, kind, video_id):
    require_video(video_id)
    if kind not in {"mapping", "visual_features", "caption_generator", "caption_mapping"}:
        raise ValueError("Only offline mapping/caption/visual artifacts permitted")
    return local.canonical_path(dataset_root, kind, video_id)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_identities(identities):
    seen = set()
    previous = None
    for video_id, frame_id in identities:
        require_video(video_id)
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("Invalid frame identity")
        identity = (video_id, frame_id)
        if identity in seen:
            raise ValueError("Duplicate (video_id, frame_id)")
        if previous is not None and identity <= previous:
            raise ValueError("Mapping video/frame order invalid")
        seen.add(identity)
        previous = identity


def validate_vectors(vectors, count):
    if vectors.shape != (count, DIMENSION) or vectors.dtype.kind != "f" or count == 0:
        raise ValueError("Visual dimension/count/dtype mismatch (requires 768)")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite visual vectors")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("Zero or invalid visual vector norm")
    return {"status": "ok", "dimension": DIMENSION, "rows": count,
            "finite": True, "nonzero": True,
            "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def exact_ip(vectors, identities, query_rows, k=5):
    """Exhaustive float64 IP, fixed reduction order, ties by ascending global row.

    Raw vectors are not normalized. Self need not rank first for unnormalized
    vectors, and identical vectors may tie. Neither case is an accuracy failure.
    """
    validate_identities(identities)
    state = validate_vectors(vectors, len(identities))
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive")
    for row in query_rows:
        if type(row) is not int or not 0 <= row < len(identities):
            raise ValueError("Query row outside index")
    matrix = np.asarray(vectors, dtype=np.float64)
    row_ids = np.arange(len(matrix))
    queries = []
    for row in query_rows:
        scores = np.einsum("ij,j->i", matrix, matrix[row], optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite IP scores")
        order = np.lexsort((row_ids, -scores))
        labels = order[:min(k, len(matrix))]
        if (len(set(labels.tolist())) != len(labels) or np.any(labels < 0)
                or np.any(labels >= len(identities)) or np.any(np.diff(scores[labels]) > 0)):
            raise ValueError("Invalid structural result IDs/order")
        results = []
        for label in labels:
            video_id, frame_id = identities[int(label)]
            require_video(video_id)
            results.append({"row_id": int(label), "video_id": video_id, "frame_id": frame_id,
                            "score": float(scores[label])})
        boundary = scores[labels[-1]]
        queries.append({"query_row": row, "video_id": identities[row][0], "frame_id": identities[row][1],
                        "self_rank": int(np.flatnonzero(order == row)[0]) + 1,
                        "self_score": float(scores[row]), "results": results,
                        "top_score_ties": int(np.count_nonzero(scores == scores[labels[0]])),
                        "boundary_score_ties": int(np.count_nonzero(scores == boundary)),
                        "boundary_ties_returned": int(np.count_nonzero(scores[labels] == boundary))})
    return {"status": "ok", "backend": "numpy_exhaustive_float64_ip", "dimension": DIMENSION,
            "rows": len(matrix), "metric": "IP", "normalization": "none_raw_visual_vectors",
            "tie_policy": "exact_score_equality_then_ascending_global_row_id",
            "query_policy": "one_middle_mapping_row_per_video_self_retrieval",
            "query_count": len(queries), "k": min(k, len(matrix)), "vectors": state,
            "queries": queries, "semantic_quality_evaluated": False}


def check_video(dataset_root, image_root, video_id, models, decoder):
    require_video(video_id)
    paths = {kind: artifact_path(dataset_root, kind, video_id)
             for kind in ("mapping", "visual_features", "caption_generator", "caption_mapping")}
    frames = local.read_mapping(paths["mapping"])
    identities = [(video_id, frame["frame_id"]) for frame in frames]
    validate_identities(identities)
    if [frame["keyframe_n"] for frame in frames] != list(range(1, len(frames) + 1)):
        raise ValueError("Mapping keyframe row order must be contiguous from 1")
    captions = local.read_captions(paths["caption_generator"], video_id, frames, "caption_generator", models)
    local.read_captions(paths["caption_mapping"], video_id, frames, "caption_mapping", models)
    local.check_features(paths["visual_features"], len(frames), {"dimension": DIMENSION})
    # No mmap remains open, and no dataset array is changed.
    vectors = np.load(paths["visual_features"], allow_pickle=False)
    vector_state = validate_vectors(vectors, len(frames))
    objects = local.check_objects(dataset_root, video_id, frames)
    images = [local.check_image(image_root, video_id, frame, caption, decoder)
              for frame, caption in zip(frames, captions)]
    image_counts = dict(sorted(Counter(image["status"] for image in images).items()))
    status = local.aggregate_status((vector_state["status"], objects["status"],
                                     local.aggregate_status(image["status"] for image in images)))
    detail = {"video_id": video_id, "status": status, "records": len(frames),
              "mapping_unique_ordered": True, "mapping_identity_sha256": digest_json(identities),
              "artifacts": {kind: {"path": path.relative_to(dataset_root).as_posix(),
                                   "sha256": local.sha256(path)} for kind, path in paths.items()},
              "vectors": vector_state,
              "objects": {key: value for key, value in objects.items() if key != "hashes"},
              "object_hashes_sha256": digest_json(objects["hashes"]),
              "images": {"expected": len(frames), "coverage": image_counts,
                         "decoded": sum(image["decoded"] for image in images),
                         "evidence_sha256": digest_json(images),
                         "issues": [image for image in images if image["status"] != "ok"]}}
    return detail, vectors, identities


def run(mode, dataset_root, image_root, output_dir, *, repo_root=REPO_ROOT):
    if mode not in MODES:
        raise ValueError("Only preflight/structural-smoke modes permitted")
    dataset_root = local.safe_path(dataset_root)
    image_root = local.safe_path(image_root)
    if image_root != dataset_root / "data":
        raise local.UnsafePath("Only explicit local dataset/data image root permitted")
    output_dir = local.safe_path(output_dir)
    if output_dir.is_relative_to(dataset_root) or dataset_root.is_relative_to(output_dir):
        raise local.UnsafePath("Output must be outside and not contain dataset input root")
    # Reserve exclusively before payload checks; never overwrite an existing run.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    models_path = local.safe_path(repo_root, "config", "models.json")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    decoder, decoder_state = local.optional_dependency("PIL.Image")
    videos, arrays, identities, query_rows = [], [], [], []
    for video_id in VIDEO_IDS:
        try:
            detail, vectors, pairs = check_video(dataset_root, image_root, video_id, models, decoder)
            query_rows.append(len(identities) + len(pairs) // 2)
            arrays.append(vectors)
            identities.extend(pairs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            detail = {"video_id": video_id, "records": 0, **local.failure(exc)}
        videos.append(detail)
        print(f"{video_id}: {detail['status']} records={detail['records']}", file=sys.stderr, flush=True)
    groups = {group: {"videos": sum(video["video_id"].startswith(group + "_") for video in videos),
                      "records": sum(video["records"] for video in videos
                                     if video["video_id"].startswith(group + "_"))} for group in EXPECTED}
    counts_ok = groups == EXPECTED
    preflight_status = local.aggregate_status(video["status"] for video in videos)
    if not counts_ok:
        preflight_status = "issues"
    index = {"status": "not_started", "query_count": 0, "reason": "preflight_only"}
    if mode == "structural-smoke":
        index["reason"] = "preflight_not_ok"
        if preflight_status == "ok":
            try:
                index = exact_ip(np.concatenate(arrays), identities, query_rows)
            except (ValueError, TypeError) as exc:
                index = {**local.failure(exc), "query_count": 0}
    status = local.aggregate_status((preflight_status, index["status"])) if mode == "structural-smoke" else preflight_status
    report = {
        "schema": "kis_l04_l05_staging_v1", "mode": mode, "status": status,
        "scope": {"groups": list(EXPECTED), "video_ids": list(VIDEO_IDS), "expected": EXPECTED,
                  "expected_videos": 61, "expected_records": 14918},
        "provenance": {"dataset_root": str(dataset_root), "image_root": str(image_root),
                       "python": sys.version, "numpy": np.__version__, "pillow": decoder_state,
                       "source_sha256": local.sha256(__file__), "helper_sha256": local.sha256(local.__file__),
                       "models_sha256": local.sha256(models_path),
                       "visual_embedding_metadata": models.get("visual_embedding", "unknown")},
        "safety": {"discovery": "none_direct_allowlisted_paths_only", "recursive_discovery": False,
                   "excluded_group_access": "none_by_construction_allowlist_rejected_before_io",
                   "scope_validation": "every_video_before_path_construction_and_lstat",
                   "directory_enumeration": False, "dataset_written": False,
                   "production_config_accessed": False, "models_loaded": False,
                   "network_or_drive_api_called": False},
        "summary": {"videos": len(videos), "records": sum(video["records"] for video in videos),
                    "exact_counts_status": "ok" if counts_ok else "issues", "preflight_status": preflight_status,
                    "object_records": sum(video.get("objects", {}).get("records", 0) for video in videos),
                    "decoded_images": sum(video.get("images", {}).get("decoded", 0) for video in videos)},
        "groups": groups, "videos": videos, "index": index,
        "milvus": {"status": "not_started", "reason": "offline_only_no_client_or_service"},
        "accuracy": {"status": "not_evaluated", "Hit@K": "not_evaluated", "MRR": "not_evaluated",
                     "reason": "No L04/L05 query/qrels; self-retrieval is structural only"},
        "limitations": ["Not a semantic benchmark, Milvus test, ingest or production-readiness approval",
                        "Raw visual IP self-retrieval does not require self to rank first; exact ties use row ID",
                        "Object JSON shape/coverage and image decoding/hashes do not establish semantic quality",
                        "Config metadata/hash consistency does not establish historical embedding provenance",
                        "Inputs must remain stable; lstat guards are not an adversarial filesystem sandbox",
                        "No L06 enumeration/read/ingest/query, caption FAISS query, network, model or Drive access"],
    }
    with (output_dir / "report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.mode, args.dataset_root, args.image_root, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"l04_l05_staging: {exc}\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "index_status": report["index"]["status"], "query_count": report["index"]["query_count"]},
                     sort_keys=True, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
