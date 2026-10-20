import csv
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath

from kis.paths import MAPPING_DIR, resolve_video_artifact, video_group
from kis.paths import video_caption_manifest_path
from kis.artifact_io import atomic_write_json
from kis.config import CAPTION_CONFIG

SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_data(video_id, caption_path, mapping_path, index_path, count):
    return {
        "schema_version": SCHEMA_VERSION, "video_id": video_id, "count": int(count),
        "files": {"caption_sha256": file_sha256(caption_path),
                  "mapping_sha256": file_sha256(mapping_path), "index_sha256": file_sha256(index_path)},
        "embedding": {"model": CAPTION_CONFIG["model"], "revision": CAPTION_CONFIG["revision"],
                      "dimension": int(CAPTION_CONFIG["dimension"]), "normalized": bool(CAPTION_CONFIG["normalized"]),
                      "metric": CAPTION_CONFIG["metric"]},
    }


def write_caption_manifest(video_id, caption_path, mapping_path, index_path, count, manifest_path=None):
    manifest_path = Path(manifest_path) if manifest_path is not None else video_caption_manifest_path(video_id)
    atomic_write_json(manifest_path, manifest_data(video_id, caption_path, mapping_path, index_path, count))
    return manifest_path


def validate_caption_manifest(video_id, caption_path, mapping_path, index_path, manifest_path, index, count):
    actual = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if actual != manifest_data(video_id, caption_path, mapping_path, index_path, count):
        raise ValueError(f"Caption artifact manifest mismatch for {video_id}: {manifest_path}. Rebuild this video's caption index.")
    if index.d != int(CAPTION_CONFIG["dimension"]):
        raise ValueError(f"Caption index dimension mismatch for {video_id}: {index.d} != {CAPTION_CONFIG['dimension']}")


def load_frame_mapping(video_id: str, mapping_dir=MAPPING_DIR):
    video_group(video_id)
    mapping_dir = Path(mapping_dir)
    csv_path = resolve_video_artifact(mapping_dir, video_id, ".csv")
    mapping = {}

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"n", "frame_idx"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns: n, frame_idx")

        for row_number, row in enumerate(reader, start=2):
            try:
                keyframe_n = int(row["n"])
                frame_id = int(row["frame_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mapping at {csv_path}:{row_number}"
                ) from exc
            if keyframe_n in mapping:
                raise ValueError(
                    f"Duplicate keyframe {keyframe_n} in {csv_path}"
                )
            mapping[keyframe_n] = frame_id

    return mapping


def rebuild_retrieval_text(record):
    base_text = ". ".join(
        text.strip(". ")
        for text in (
            str(record.get("caption", "") or ""),
            str(record.get("details", "") or ""),
        )
        if text.strip()
    )
    ocr_text = str(record.get("ocr_text", "") or "").strip()

    if ocr_text and base_text:
        return f"{base_text}. [OCR]: {ocr_text}"
    if ocr_text:
        return f"[OCR]: {ocr_text}"
    return base_text


def normalize_caption_record(record, video_id, frame_mapping=None):
    if not isinstance(record, dict):
        raise ValueError(f"Caption record for {video_id} must be an object")

    normalized = dict(record)
    if normalized.get("video_id") != video_id:
        raise ValueError(
            f"Caption video_id mismatch: expected {video_id}, "
            f"got {normalized.get('video_id')!r}"
        )

    try:
        keyframe_n = int(normalized["keyframe_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid keyframe_n for {video_id}") from exc
    normalized["keyframe_n"] = keyframe_n

    frame_id = normalized.get("frame_id")
    if frame_id is None:
        if frame_mapping is None or keyframe_n not in frame_mapping:
            raise ValueError(
                f"Cannot resolve frame_id for {video_id} keyframe {keyframe_n}"
            )
        frame_id = frame_mapping[keyframe_n]
    try:
        normalized["frame_id"] = int(frame_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame_id for {video_id}: {frame_id!r}") from exc
    if frame_mapping is not None:
        expected_frame_id = frame_mapping.get(keyframe_n)
        if expected_frame_id is None:
            raise ValueError(
                f"Missing mapping for {video_id} keyframe {keyframe_n}"
            )
        if normalized["frame_id"] != expected_frame_id:
            raise ValueError(
                f"frame_id mismatch for {video_id} keyframe {keyframe_n}: "
                f"{normalized['frame_id']} != {expected_frame_id}"
            )

    image_path = str(normalized.get("image_path", "")).replace("\\", "/")
    parsed_path = PurePosixPath(image_path)
    group = video_group(video_id)
    flat_layout = parsed_path.parts[:2] == ("data", video_id)
    grouped_layout = parsed_path.parts[:3] == ("data", group, video_id)
    if (
        parsed_path.is_absolute()
        or ".." in parsed_path.parts
        or not (flat_layout or grouped_layout)
        or len(parsed_path.parts) < 3
    ):
        raise ValueError(f"Unsafe image_path for {video_id}: {image_path!r}")
    normalized["image_path"] = parsed_path.as_posix()

    if "visual_description" in normalized:
        from kis.structured_caption import render_description, validate_description

        normalized["visual_description"] = validate_description(normalized["visual_description"])
        normalized["caption"], normalized["details"] = render_description(normalized["visual_description"])
        normalized["retrieval_text"] = rebuild_retrieval_text(normalized)
    elif isinstance(normalized.get("caption_generation"), dict) and normalized["caption_generation"].get("backend") == "internvl_dense_v3":
        from kis.light_caption import validate_dense_caption

        normalized["caption"] = validate_dense_caption(normalized.get("caption"))
        normalized["retrieval_text"] = rebuild_retrieval_text(normalized)
    elif "ocr_text" in normalized or not normalized.get("retrieval_text"):
        normalized["retrieval_text"] = rebuild_retrieval_text(normalized)

    if "text_processing" in normalized:
        raise ValueError("Experimental caption view found; rebuild from the original InternVL source")
    return normalized


def caption_identity(record):
    return (
        record["video_id"],
        int(record["frame_id"]),
        int(record["keyframe_n"]),
    )


def validate_unique_caption_records(records, video_id):
    keyframes = set()
    for record in records:
        keyframe_n = int(record["keyframe_n"])
        if keyframe_n in keyframes:
            raise ValueError(
                f"Duplicate keyframe_n for {video_id}: {keyframe_n}"
            )
        keyframes.add(keyframe_n)
