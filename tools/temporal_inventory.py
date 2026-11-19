"""Offline artifact and image inventory; ZIP members are never opened or extracted."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from zipfile import BadZipFile, ZipFile

from kis.config import CAPTION_CONFIG, FAST_CAPTION_CONFIG, VISUAL_CONFIG
from kis.paths import is_keyframe_image, resolve_video_image_dir, resolve_video_object_dir
from tools import p0_audit as audit


ARTIFACT_DIRS = {
    "mapping": "mapping", "features": "clip-features-32",
    "captions": "caption_generator", "caption_mapping": "caption_mapping", "index": "index",
}


def parse_groups(value):
    groups = [part.strip() for part in value.split(",")]
    if not groups or any(not re.fullmatch(r"L\d+", group) for group in groups):
        raise ValueError("groups must be comma-separated Lnn identifiers")
    return tuple(sorted(set(groups)))


def _coverage(expected, present, unknown=()):
    expected, present, unknown = set(expected), set(present), set(unknown)
    unknown -= present
    missing = expected - present - unknown
    status = ("unknown" if not expected or unknown & expected else
              "present" if expected <= present else "partial" if expected & present else "missing")
    return {"status": status, "expected": len(expected), "present": len(expected & present),
            "unknown": len(expected & unknown), "missing": len(missing),
            "missing_keyframe_ns": sorted(missing), "extra_keyframe_ns": sorted(present - expected)}


def local_images(root, video_id, expected):
    """Metadata only: do not open cloud placeholders or hydrate/decode images."""
    present, unknown = set(), set()
    detail = {"root": str(root), "verification": "filesystem_metadata_only_not_decoded"}
    try:
        directory = resolve_video_image_dir(video_id, root)
        detail["directory"] = str(directory)
        for path in sorted(directory.iterdir()) if directory.is_dir() else ():
            if not is_keyframe_image(path):
                continue
            number = int(path.stem)
            info = path.stat()
            # OFFLINE / RECALL_ON_OPEN / RECALL_ON_DATA_ACCESS: never hydrate.
            placeholder = getattr(info, "st_file_attributes", 0) & (0x1000 | 0x40000 | 0x400000)
            if placeholder or path.is_symlink():
                unknown.add(number)
            elif stat.S_ISREG(info.st_mode) and info.st_size > 0:
                present.add(number)
        detail.update(_coverage(expected, present, unknown - present))
    except (OSError, RuntimeError) as exc:
        detail.update(_coverage(expected, (), expected), status="unknown", reason=str(exc))
    return detail


def object_inventory(root, video_id, expected):
    detail = {"status": "missing", "records": 0}
    try:
        directory = resolve_video_object_dir(video_id, root)
        detail["path"] = str(directory)
        present, invalid = set(), []
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            if not path.stem.isascii() or not path.stem.isdigit():
                continue
            detail["records"] += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                invalid.append(path.name)
            else:
                present.add(int(path.stem))
        detail.update(_coverage(expected, present), invalid_records=invalid)
        if invalid:
            detail["status"] = "invalid"
    except (OSError, ValueError, RuntimeError) as exc:
        detail.update(status="invalid", reason=str(exc))
    return detail


def archive_inventory(path, groups):
    """Read central directory only. Unsafe archives never contribute image coverage."""
    result = {"path": str(path), "status": "listed", "entries": 0, "rejected_members": [],
              "artifact_files_by_group": {}, "image_entries": 0,
              "verification": "central_directory_only_payloads_and_crc_not_verified"}
    images, artifacts, signatures, duplicates = {}, {}, [], set()
    seen = set()
    try:
        with ZipFile(path) as archive:
            for info in archive.infolist():
                result["entries"] += 1
                # orig_filename retains invalid characters that ZipInfo may truncate.
                raw_name = info.orig_filename
                name = PurePosixPath(raw_name).as_posix()
                parts = PurePosixPath(name).parts
                # Be conservative across hosts: Windows destinations are case-insensitive.
                destination = name.casefold()
                if destination in seen:
                    duplicates.add(destination)
                seen.add(destination)
                # A directory's final slash is not an empty component; interior // is.
                raw_parts = (raw_name[:-1] if info.is_dir() else raw_name).split("/")
                invalid_components = any(
                    not part or part.endswith((".", " "))
                    or re.search(r'[\x00-\x1f<>:"|?*]', part)
                    or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part, re.IGNORECASE)
                    for part in raw_parts if part != ".")
                mode = info.external_attr >> 16
                if (not parts or name.startswith("/") or ".." in parts or "\\" in raw_name
                        or invalid_components or stat.S_ISLNK(mode)):
                    result["rejected_members"].append(raw_name)
                    continue
                if info.is_dir():
                    continue
                signatures.append((raw_name, info.CRC, info.file_size, info.compress_size))
                if (len(parts) >= 4 and parts[-4] == "data" and parts[-3] in groups
                        and audit.VIDEO_ID_PATTERN.fullmatch(parts[-2])
                        and parts[-2].split("_")[0] == parts[-3] and is_keyframe_image(parts[-1])):
                    if info.file_size > 0:
                        images.setdefault(parts[-2], set()).add(int(PurePosixPath(parts[-1]).stem))
                        result["image_entries"] += 1
                for kind, directory in ARTIFACT_DIRS.items():
                    suffix = {"mapping": ".csv", "features": ".npy", "index": ".index"}.get(kind, ".json")
                    if directory in parts[:-1] and parts[-1].endswith(suffix):
                        video_id = parts[-1][:-len(suffix)]
                        if audit.VIDEO_ID_PATTERN.fullmatch(video_id) and video_id.split("_")[0] in groups:
                            group = video_id.split("_")[0]
                            artifacts.setdefault(group, Counter())[kind] += 1
                if "index" in parts[:-1] and parts[-1].endswith(".meta.json"):
                    group = parts[-1].split("_")[0]
                    if group in groups:
                        artifacts.setdefault(group, Counter())["index_manifest"] += 1
                if (len(parts) >= 3 and parts[-3] == "objects"
                        and audit.VIDEO_ID_PATTERN.fullmatch(parts[-2]) and parts[-1].endswith(".json")):
                    group = parts[-2].split("_")[0]
                    if group in groups:
                        artifacts.setdefault(group, Counter())["objects"] += 1
        result["directory_inventory_sha256"] = hashlib.sha256(
            json.dumps(sorted(signatures), ensure_ascii=False).encode("utf-8")).hexdigest()
    except (OSError, BadZipFile) as exc:
        result.update(status="unknown", reason=str(exc))
    result["rejected_members"].sort()
    result["duplicate_members"] = sorted(duplicates)
    if result["rejected_members"] or duplicates:
        result["status"] = "unsafe"
        images.clear()
        result["image_entries"] = 0
    result["artifact_files_by_group"] = {group: dict(sorted(counts.items()))
                                         for group, counts in sorted(artifacts.items())}
    return result, images


def _summary(records):
    return {
        "videos": len(records), "frames": sum(record["frames"] for record in records),
        "artifact_status": dict(sorted(Counter(record["artifact_status"] for record in records).items())),
        "artifacts": {kind: dict(sorted(Counter(record["artifacts"][kind]["status"] for record in records).items()))
                      for kind in ("mapping", "visual_features", "caption_generator", "caption_mapping",
                                   "index", "index_manifest", "objects")},
        "object_records": sum(record["artifacts"]["objects"]["records"] for record in records),
        "local_images": dict(sorted(Counter(record["images"]["local"]["status"] for record in records).items())),
        "local_image_frames": sum(record["images"]["local"]["present"] for record in records),
        "archive_images": dict(sorted(Counter(record["images"]["archives"]["status"] for record in records).items())),
        "archive_image_frames": sum(record["images"]["archives"]["present"] for record in records),
        "runtime_ready": 0,
    }


def run_inventory(output, config, mapping_root, groups=None, image_roots=(), archives=()):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    groups = parse_groups(",".join(config.video_groups if groups is None else groups))
    roots = {kind: Path(config.caption_root) / directory for kind, directory in ARTIFACT_DIRS.items()}
    roots["mapping"] = Path(mapping_root)
    indexes = {kind: audit._index_root(root) for kind, root in roots.items()}
    video_ids = sorted({Path(name).stem for name in indexes["mapping"]
                        if re.fullmatch(r"L\d+_V\d+\.csv", name) and name.split("_")[0] in groups})
    archive_records, archive_images = [], {}
    archive_unknown = not archives
    for path in sorted(set(map(Path, archives)), key=str):
        detail, images = archive_inventory(path, groups)
        archive_records.append(detail)
        if detail["status"] != "listed":
            archive_unknown = True
            continue
        for video_id, numbers in images.items():
            archive_images.setdefault(video_id, set()).update(numbers)
    records = []
    for video_id in video_ids:
        public, identity = audit._artifact_detail(
            video_id, roots, indexes, groups,
            {"status": "unknown", "reason": "Index contents not read in offline inventory (no FAISS import)"},
            VISUAL_CONFIG, CAPTION_CONFIG, FAST_CAPTION_CONFIG, [])
        expected = set(identity["keyframe_n"])
        objects = object_inventory(Path(config.caption_root) / "objects", video_id, expected)
        public["artifacts"]["objects"] = objects
        statuses = [item["status"] for item in public["artifacts"].values()]
        # Object coverage uses 'present', unlike the reused audit's 'ok'.
        statuses = ["ok" if status == "present" else status for status in statuses]
        artifact_status = audit._artifact_status(statuses, public["issues"])
        if any(status in {"partial", "missing"} for status in statuses):
            artifact_status = "issues"
        local = [local_images(Path(root), video_id, expected) for root in image_roots]
        # Do not combine partial roots: runtime selects one image directory.
        best = max(local, key=lambda item: (item["present"], -item["unknown"]), default=None)
        local_coverage = ({key: best[key] for key in _coverage(expected, ())} if best else
                          _coverage(expected, (), expected))
        archived = _coverage(expected, archive_images.get(video_id, ()), expected if archive_unknown else ())
        if archived["status"] == "present":
            archived["status"] = "listed"
        records.append({"video_id": video_id, "frames": public["identity"]["mapping_rows"],
                        "artifact_status": artifact_status, "artifacts": public["artifacts"],
                        "identity": public["identity"], "issues": public["issues"],
                        "images": {"local": local_coverage, "roots": local, "archives": archived},
                        "runtime_status": "blocked" if local_coverage["status"] in {"missing", "partial"} else "unknown"})
    report = {
        "schema": "kis_offline_preflight_v1", "mode": "offline_inventory_no_models",
        "video_groups": list(groups), "configured_video_groups": list(config.video_groups),
        "production_scope_changed": False,
        "sources": {"mapping_root": str(mapping_root), "caption_root": str(config.caption_root),
                    "image_roots": list(map(str, image_roots)),
                    "source_sha256": {str(Path(__file__).name): audit._sha256_file(__file__),
                                      "temporal_preflight.py": audit._sha256_file(Path(__file__).with_name("temporal_preflight.py")),
                                      "p0_audit.py": audit._sha256_file(audit.__file__)}},
        "summary": _summary(records),
        "groups": {group: _summary([record for record in records if record["video_id"].startswith(group + "_")])
                   for group in groups},
        "missing_mapping_groups": sorted(set(groups) - {video.split("_")[0] for video in video_ids}),
        "archives": archive_records, "videos": records,
        "limitations": [
            "Read-only metadata/artifact audit; no models, network, Drive API, Milvus or retrieval run",
            "Current model config checks local structure, not historical provenance or semantic correctness",
            "Index contents are unknown; file hashes and manifest consistency are checked without FAISS",
            "Local images use filesystem metadata only; no hydration, image decoding or pixel/hash verification",
            "ZIP images are inventory only, not runtime-ready; extract/place in the data root before runtime",
            "Missing L06 images must be supplied separately; artifacts alone do not establish image readiness",
            "Mapping-based video discovery cannot detect entirely absent videos without an external inventory",
            "P0 audits the old saved human holdout run and does not prove L04-L06 retrieval quality",
        ],
    }
    # Exclusive create: never overwrite any report or dataset artifact.
    for root in (Path(config.caption_root), Path(mapping_root), *map(Path, image_roots)):
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError("Inventory output must be outside dataset/image input roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report: {output.resolve()}")
    return report
