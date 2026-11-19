"""Read-only, explicit L04/L05 local smoke; never discover videos or load models.

Intentionally independent of kis.paths/config: importing those can read the local
Drive image settings. Validators mirror P0 identity/count/hash checks without its
root inventory. Artifacts use canonical group paths; objects additionally support
the explicit legacy objects/<video_id> directory, without any directory scan.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"L04": {"videos": 30, "records": 7471}, "L05": {"videos": 31, "records": 7447}}
VIDEO_IDS = tuple(f"{group}_V{number:03d}" for group, count in (("L04", 30), ("L05", 31))
                  for number in range(1, count + 1))
ARTIFACTS = {
    "mapping": ("mapping", ".csv"),
    "caption_generator": ("caption_generator", ".json"),
    "caption_mapping": ("caption_mapping", ".json"),
    "visual_features": ("clip-features-32", ".npy"),
    "index": ("index", ".index"),
    "index_manifest": ("index", ".meta.json"),
}
PLACEHOLDER_FLAGS = 0x1000 | 0x40000 | 0x400000
REPARSE_FLAG = 0x400


class UnsafePath(ValueError):
    pass


class PlaceholderPath(OSError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, *parts):
    """No traversal, UNC, symlink, junction/reparse or cloud recall, even in parents.

    lstat precedes any payload read. Missing paths are returned for coverage checks.
    Inputs must remain stable during this read-only local run (not a sandbox).
    """
    root = Path(root)
    if ".." in root.parts or str(root).startswith(("\\\\", "//")):
        raise UnsafePath("Traversal or network root rejected")
    for part in parts:
        if not isinstance(part, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in (".", ".."):
            raise UnsafePath("Non-canonical path component")
    path = Path(os.path.abspath(root.joinpath(*parts)))
    if any(re.match(r"L06(?:_|$)", part, re.IGNORECASE) for part in path.parts):
        raise UnsafePath("Excluded group path rejected before filesystem access")
    for parent in (*reversed(path.parents), path):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            break
        attributes = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attributes & REPARSE_FLAG:
            raise UnsafePath("Symlink/reparse path rejected")
        if attributes & PLACEHOLDER_FLAGS:
            raise PlaceholderPath("Cloud placeholder/recall path not opened")
    return path


def canonical_path(dataset_root, kind, video_id):
    if video_id not in VIDEO_IDS:
        raise UnsafePath("Video outside fixed explicit scope")
    directory, suffix = ARTIFACTS[kind]
    return safe_path(dataset_root, directory, video_id.split("_")[0], video_id + suffix)


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError("Expected nonnegative integer identity")
    return int(value)


def read_mapping(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"n", "pts_time", "frame_idx"} <= set(reader.fieldnames or ()):
            raise ValueError("Mapping requires n, pts_time, frame_idx")
        frames = []
        for row in reader:
            number, frame_id, seconds = integer(row["n"]), integer(row["frame_idx"]), float(row["pts_time"])
            if number < 1 or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid mapping identity/timestamp")
            if row.get("fps") not in (None, ""):
                fps = float(row["fps"])
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Invalid mapping fps")
            frames.append({"keyframe_n": number, "frame_id": frame_id, "pts_time": seconds})
    if (not frames or len({f["keyframe_n"] for f in frames}) != len(frames)
            or len({f["frame_id"] for f in frames}) != len(frames)):
        raise ValueError("Empty mapping or duplicate identities")
    if any(a["pts_time"] > b["pts_time"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Mapping timestamp order invalid")
    return frames


def image_relative(video_id, number):
    if video_id not in VIDEO_IDS or integer(number) < 1:
        raise UnsafePath("Invalid image identity")
    return f"data/{video_id.split('_')[0]}/{video_id}/{number:03d}.jpg"


def read_captions(path, video_id, frames, kind, models):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Caption artifact must be an array")
    pairs = []
    for record in records:
        if not isinstance(record, dict) or record.get("video_id") != video_id:
            raise ValueError("Caption video identity mismatch")
        frame_id, number = integer(record.get("frame_id")), integer(record.get("keyframe_n"))
        # Keep parsed identities canonical for the later integer-keyed image lookup.
        record.update(frame_id=frame_id, keyframe_n=number)
        pairs.append((frame_id, number))
        if record.get("image_path") != image_relative(video_id, number):
            raise UnsafePath("Stored image path is not the exact canonical local path")
        if kind == "caption_generator":
            if not (record.get("retrieval_text") or record.get("caption")):
                raise ValueError("Missing searchable caption text")
            generation = record.get("caption_generation")
            if generation:
                expected = models["fast_caption_generation"]
                if any(generation.get(key) != expected[key] for key in ("model", "revision")):
                    raise ValueError("Caption generation model/revision mismatch")
    if pairs != [(frame["frame_id"], frame["keyframe_n"]) for frame in frames]:
        raise ValueError("Mapping/caption identity, count or row order mismatch")
    return records


def check_features(path, count, config):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        shape = list(array.shape)
        if shape != [count, config["dimension"]] or array.dtype.kind != "f":
            raise ValueError("Feature shape/count/dimension/dtype mismatch")
        for start in range(0, count, 256):
            block = np.asarray(array[start:start + 256], dtype=np.float32)
            if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) <= np.finfo(np.float32).eps):
                raise ValueError("Nonfinite or zero visual feature")
        return {"shape": shape, "rows": count, "dimension": shape[1], "dtype": str(array.dtype),
                "finite": True, "nonzero": True}
    finally:
        # Explicit release of read-only mmap on Windows.
        if getattr(array, "_mmap", None) is not None:
            array._mmap.close()


def check_manifest(path, video_id, count, artifacts, config):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Invalid manifest schema")
    if manifest.get("video_id") != video_id or manifest.get("count") != count:
        raise ValueError("Manifest identity/count mismatch")
    files = manifest.get("files")
    names = {"caption_sha256": "caption_generator", "mapping_sha256": "caption_mapping", "index_sha256": "index"}
    if not isinstance(files, dict) or not names.keys() <= files.keys():
        raise ValueError("Manifest required hashes missing")
    # As in P0, mapping_sha256 means caption-mapping JSON, not mapping CSV.
    names.update(mapping_csv_sha256="mapping", features_sha256="visual_features",
                 caption_mapping_sha256="caption_mapping")
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Manifest invalid SHA-256")
        if name not in names or artifacts[names[name]].get("sha256") != expected:
            raise ValueError(f"Manifest hash mismatch/unverifiable: {name}")
    embedding = manifest.get("embedding")
    fields = ("model", "revision", "dimension", "normalized", "metric")
    if not isinstance(embedding, dict) or any(embedding.get(key) != config[key] for key in fields):
        raise ValueError("Manifest embedding config mismatch")
    return {"rows": count, "embedding": embedding, "verified_hashes": files}


def optional_dependency(module):
    try:
        loaded = importlib.import_module(module)
        return loaded, {"status": "available", "version": getattr(loaded, "__version__", "unknown")}
    except Exception as exc:
        return None, {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}


def check_index(path, count, config, faiss):
    index = faiss.read_index(str(path))
    metric = {"IP": faiss.METRIC_INNER_PRODUCT, "L2": faiss.METRIC_L2}.get(config["metric"])
    if (index.ntotal != count or index.d != config["dimension"]
            or index.metric_type != metric or not index.is_trained):
        raise ValueError("FAISS count/dimension/metric/training mismatch")
    # Synthetic unit vector in CAPTION space, not a CLIP visual vector or text query.
    query = np.zeros((1, index.d), dtype=np.float32)
    query[0, 0] = 1
    k = min(3, count)
    distances, labels = index.search(query, k)
    if not k or distances.shape != (1, k) or labels.shape != (1, k) or not np.isfinite(distances).all():
        raise ValueError("FAISS structural search output invalid")
    if np.any(labels < 0) or np.any(labels >= count) or len(set(labels[0].tolist())) != k:
        raise ValueError("FAISS structural search labels invalid")
    ordered = np.diff(distances[0])
    invalid_order = np.any(ordered > 0) if config["metric"] == "IP" else np.any(ordered < 0)
    if invalid_order:
        raise ValueError("FAISS structural search ordering invalid")
    return {"rows": int(index.ntotal), "dimension": int(index.d), "metric": config["metric"],
            "metric_type": int(index.metric_type), "is_trained": bool(index.is_trained),
            "structural_search": "ok", "query": "synthetic_caption_space_unit_vector", "search_k": k}


def aggregate_status(statuses):
    statuses = list(statuses)
    if any(status in {"missing", "invalid", "unsafe", "issues"} for status in statuses):
        return "issues"
    return "ok" if statuses and all(status == "ok" for status in statuses) else "unknown"


def failure(exc):
    status = ("unsafe" if isinstance(exc, UnsafePath) else "unknown" if isinstance(exc, PlaceholderPath)
              else "missing" if isinstance(exc, FileNotFoundError) else "invalid")
    return {"status": status, "reason": f"{type(exc).__name__}: {exc}"}


def check_image(image_root, video_id, frame, caption, decoder):
    relative = image_relative(video_id, frame["keyframe_n"])
    detail = {"path": relative, "status": "unknown", "decoded": False}
    try:
        path = safe_path(image_root, *relative.split("/")[1:])
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError("Image is not a nonempty regular file")
        detail.update(bytes=info.st_size, sha256=sha256(path))
        expected_hash = caption.get("image_sha256") if caption else None
        if expected_hash is not None and (not isinstance(expected_hash, str)
                                          or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)):
            detail["caption_hash_status"] = "invalid"
            raise ValueError("Caption image SHA-256 is malformed")
        detail["caption_hash_status"] = ("unknown" if expected_hash is None else
                                         "ok" if expected_hash == detail["sha256"] else "invalid")
        if detail["caption_hash_status"] == "invalid":
            raise ValueError("Caption/local image SHA-256 mismatch")
        if decoder is None:
            detail["reason"] = "Pillow decoder unavailable; image decode is unknown"
        else:
            with decoder.open(path) as image:
                image.verify()
            with decoder.open(path) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Invalid decoded image dimensions")
                detail.update(dimensions=[image.width, image.height], format=image.format)
            detail.update(status=aggregate_status(("ok", detail["caption_hash_status"])), decoded=True)
            if detail["status"] == "unknown":
                detail["reason"] = "Caption image SHA-256 missing; image hash verification is unknown"
    except Exception as exc:
        detail.update(failure(exc))
    return detail


def object_directory(dataset_root, video_id):
    """Probe only two exact paths for an allowed video; never enumerate objects."""
    if video_id not in VIDEO_IDS:
        raise UnsafePath("Video outside fixed explicit scope")
    candidates = [
        (safe_path(dataset_root, "objects", video_id.split("_")[0], video_id), "grouped_canonical"),
        (safe_path(dataset_root, "objects", video_id), "legacy_explicit"),
    ]
    present = []
    for path, layout in candidates:
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Object directory candidate is not a directory")
        present.append((path, layout))
    if len(present) > 1:
        raise ValueError("Ambiguous grouped/legacy object directories; no payload read")
    return present[0] if present else candidates[0]


def check_objects(dataset_root, video_id, frames):
    if video_id not in VIDEO_IDS:
        raise UnsafePath("Video outside fixed explicit scope")
    try:
        directory, layout = object_directory(dataset_root, video_id)
    except Exception as exc:
        detail = failure(exc)
        return {**detail, "path": None, "layout": "unresolved", "expected": len(frames),
                "records": 0, "coverage": {detail["status"]: len(frames)}, "hashes": [],
                "issues": [dict(detail, keyframe_n=frame["keyframe_n"]) for frame in frames]}
    records, hashes = [], []
    for frame in frames:
        name = f"{frame['keyframe_n']:03d}.json"
        try:
            path = safe_path(directory, name)
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Object record must be a JSON object")
            hashes.append([name, sha256(path)])
            records.append({"status": "ok"})
        except Exception as exc:
            records.append(dict(failure(exc), keyframe_n=frame["keyframe_n"]))
    counts = dict(sorted(Counter(record["status"] for record in records).items()))
    return {"status": aggregate_status(record["status"] for record in records),
            "path": directory.relative_to(dataset_root).as_posix(), "layout": layout, "expected": len(frames),
            "records": counts.get("ok", 0), "coverage": counts,
            "hashes": hashes, "issues": [record for record in records if record["status"] != "ok"]}


def check_video(dataset_root, image_root, video_id, models, faiss, decoder):
    artifacts, frames, captions = {}, [], []
    for kind in ARTIFACTS:
        detail = {"status": "ok"}
        try:
            path = canonical_path(dataset_root, kind, video_id)
            detail.update(path=path.relative_to(dataset_root).as_posix(), sha256=sha256(path))
            if kind == "mapping":
                frames = read_mapping(path)
                detail.update(rows=len(frames), identity_unique=True, timestamp_order=True)
            elif kind in {"caption_generator", "caption_mapping"}:
                records = read_captions(path, video_id, frames, kind, models)
                detail.update(rows=len(records), identity_order_matches=True)
                if kind == "caption_generator":
                    captions = records
            elif kind == "visual_features":
                detail.update(check_features(path, len(frames), models["visual_embedding"]))
            elif kind == "index_manifest":
                detail.update(check_manifest(path, video_id, len(frames), artifacts, models["caption_embedding"]))
            elif faiss is None:
                detail.update(status="unknown", reason="FAISS dependency unavailable")
            else:
                detail.update(check_index(path, len(frames), models["caption_embedding"], faiss))
        except Exception as exc:
            detail.update(failure(exc))
        artifacts[kind] = detail
    artifacts["objects"] = check_objects(dataset_root, video_id, frames)
    by_number = {record["keyframe_n"]: record for record in captions}
    images = [check_image(image_root, video_id, frame, by_number.get(frame["keyframe_n"]), decoder)
              for frame in frames]
    return {"video_id": video_id, "records": len(frames), "artifacts": artifacts, "images": images,
            "readiness": {"artifact": aggregate_status(value["status"] for key, value in artifacts.items()
                                                      if key != "index"),
                          "image": aggregate_status(image["status"] for image in images),
                          "faiss": artifacts["index"]["status"]}}


def summary(videos):
    return {"videos": len(videos), "records": sum(video["records"] for video in videos),
            "artifacts": {kind: dict(sorted(Counter(video["artifacts"][kind]["status"] for video in videos).items()))
                          for kind in (*ARTIFACTS, "objects")},
            "object_records": sum(video["artifacts"]["objects"]["records"] for video in videos),
            "object_layouts": dict(sorted(Counter(video["artifacts"]["objects"]["layout"]
                                                   for video in videos).items())),
            "images": dict(sorted(Counter(image["status"] for video in videos for image in video["images"]).items())),
            "decoded_images": sum(image["decoded"] for video in videos for image in video["images"]),
            "caption_image_hashes": dict(sorted(Counter(image.get("caption_hash_status", "unknown")
                                                        for video in videos for image in video["images"]).items())),
            "readiness": {kind: aggregate_status(video["readiness"][kind] for video in videos)
                          for kind in ("artifact", "image", "faiss")}}


def run(image_root, output_dir, *, dataset_root=None, repo_root=REPO_ROOT):
    repo_root = safe_path(repo_root)
    dataset_root = safe_path(dataset_root if dataset_root is not None else repo_root / "dataset")
    image_root = safe_path(image_root)
    if image_root != dataset_root / "data":
        raise UnsafePath("Only dataset/data is permitted as the local image root")
    output_dir = safe_path(output_dir)
    if output_dir.is_relative_to(dataset_root) or dataset_root.is_relative_to(output_dir):
        raise UnsafePath("Output must be outside and not contain dataset input root")
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    models_path = safe_path(repo_root, "config", "models.json")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    faiss, faiss_state = optional_dependency("faiss")
    decoder, pillow_state = optional_dependency("PIL.Image")
    if decoder is not None:
        pillow_state["version"] = importlib.import_module("PIL").__version__
    videos = []
    for offset, video_id in enumerate(VIDEO_IDS, 1):
        video = check_video(dataset_root, image_root, video_id, models, faiss, decoder)
        videos.append(video)
        print(f"[{offset}/{len(VIDEO_IDS)}] {video_id}: records={video['records']} "
              f"artifact={video['readiness']['artifact']} image={video['readiness']['image']} "
              f"faiss={video['readiness']['faiss']}", file=sys.stderr, flush=True)
    groups = {group: summary([video for video in videos if video["video_id"].startswith(group + "_")])
              for group in EXPECTED}
    counts_match = all(groups[group]["videos"] == expected["videos"]
                       and groups[group]["records"] == expected["records"]
                       for group, expected in EXPECTED.items())
    totals = summary(videos)
    dependency_status = "ok" if faiss is not None and decoder is not None else "unknown"
    totals["readiness"]["dependency"] = dependency_status
    totals["exact_counts_status"] = "ok" if counts_match else "issues"
    report = {
        "schema": "kis_l04_l05_local_smoke_v1", "mode": "offline_local_preflight_and_structural_smoke",
        "scope": {"groups": list(EXPECTED), "video_ids": list(VIDEO_IDS), "expected": EXPECTED,
                  "expected_videos": 61, "expected_records": 14918, "discovery": "none_explicit_paths_only",
                  "excluded_group_access": "none_by_construction_rejected_before_io"},
        "provenance": {"python": sys.version, "numpy": np.__version__, "faiss": faiss_state, "pillow": pillow_state,
                       "dataset_root": str(dataset_root), "image_root": str(image_root),
                       "source_sha256": sha256(__file__), "models_sha256": sha256(models_path),
                       "pipeline_sha256": sha256(safe_path(repo_root, "config", "pipeline.json"))},
        "safety": {"production_scope_changed": False, "dataset_written": False, "models_loaded": False,
                   "network_or_drive_api_called": False, "milvus_used": False, "recursive_discovery": False},
        "summary": totals, "groups": groups, "videos": videos,
        "limitations": ["Structural local smoke only, not a semantic retrieval benchmark or production-readiness approval",
                        "Synthetic caption-space FAISS search does not encode text or measure retrieval relevance",
                        "Config/manifest/hash consistency does not prove historical embedding provenance or semantic accuracy",
                        "Objects use only explicit grouped/legacy video paths; ambiguous/unsafe paths do not pass",
                        "Inputs must remain stable during checks; lstat validation is not an adversarial filesystem sandbox",
                        "No production ingest, Milvus publication, model download or pipeline scope change"],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    with (output_dir / "report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], sort_keys=True, indent=2))
    print(f"Report: {output_dir / 'report.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.image_root, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"l04_l05_local_smoke: {exc}\n")
    # 0 means all smoke checks passed; 1 means report created with issues/unknowns.
    return 0 if report["summary"]["exact_counts_status"] == "ok" and all(
        status == "ok" for status in report["summary"]["readiness"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
