"""Offline pixel OCR extraction/import and immutable SQLite FTS5 indexing.

No caption/index rebuild. Paddle is optional and imported only by explicit
extract; local model dirs and exact package versions are mandatory.
"""

import argparse
from collections import Counter
from contextlib import closing
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from uuid import uuid4

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.ocr import SCHEMA, digest, normalize_text, validate_engine, validate_frame
from kis.paths import KIS_ROOT, MAPPING_DIR, is_keyframe_image, resolve_video_artifact, video_group


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mapping_rows(video_id, mapping_dir):
    path = resolve_video_artifact(mapping_dir, video_id, ".csv")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"n", "frame_idx", "pts_time"}.issubset(reader.fieldnames or []):
            raise ValueError("OCR needs keyframe number, frame index and timestamp mapping")
        rows = [{"keyframe_n": int(row["n"]), "frame_id": int(row["frame_idx"]), "pts_time": float(row["pts_time"])} for row in reader]
    if len({r["keyframe_n"] for r in rows}) != len(rows):
        raise ValueError("Duplicate OCR mapping keyframe number")
    if any(not math.isfinite(r["pts_time"]) or r["pts_time"] < 0 for r in rows):
        raise ValueError("Invalid mapping timestamp")
    counts = Counter(r["frame_id"] for r in rows)
    return {r["keyframe_n"]: r for r in rows if counts[r["frame_id"]] == 1}, path


def verified_image(frame, image_root, mapping_dir):
    validate_frame(dict(frame, status="empty", lines=[]))
    root = Path(image_root).resolve()
    path = (root / frame["image_relative"]).resolve()
    if not path.is_relative_to(root) or not is_keyframe_image(path, frame["keyframe_n"]):
        raise ValueError("OCR input must be a numbered keyframe under the approved image root")
    if digest(path) != frame["image_sha256"]:
        raise ValueError("OCR image checksum mismatch")
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    if (width, height) != (frame["width"], frame["height"]):
        raise ValueError("OCR image dimensions mismatch")
    rows, mapping = mapping_rows(frame["video_id"], mapping_dir)
    expected = rows.get(frame["keyframe_n"])
    if (expected is None or frame["frame_id"] != expected["frame_id"]
            or abs(frame["pts_time"] - expected["pts_time"]) > 1e-6):
        raise ValueError("OCR frame/mapping join missing or ambiguous")
    return path, mapping


def build_index(detections, image_root, mapping_dir, output):
    detections, output = Path(detections), Path(output)
    if output.exists():
        raise FileExistsError("OCR builds are immutable; use a new directory")
    data = read_json(detections)
    if data.get("schema") != SCHEMA or data.get("source_kind") != "pixel_ocr":
        raise ValueError("Only independent pixel OCR artifacts are accepted; caption OCR is not evidence")
    validate_engine(data["engine"])
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("No OCR frame records")
    mapping_hashes, seen_frames, seen_numbers = {}, set(), set()
    for frame in frames:
        validate_frame(frame)
        key, number = (frame["video_id"], frame["frame_id"]), (frame["video_id"], frame["keyframe_n"])
        if key in seen_frames or number in seen_numbers:
            raise ValueError("Duplicate OCR frame identity")
        seen_frames.add(key)
        seen_numbers.add(number)
        _, mapping = verified_image(frame, image_root, mapping_dir)
        mapping_hashes[frame["video_id"]] = digest(mapping)
    input_hash = digest(detections)
    output.mkdir(parents=True, exist_ok=False)
    index_path = output / "index.sqlite3"
    try:
        with closing(sqlite3.connect(index_path)) as db, db:
            db.executescript("""
                PRAGMA foreign_keys=ON;
                CREATE TABLE frames(id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, video_group TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,keyframe_n INTEGER NOT NULL,pts_time REAL NOT NULL,
                    image_sha256 TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,status TEXT NOT NULL,
                    UNIQUE(video_id,frame_id),UNIQUE(video_id,keyframe_n));
                CREATE TABLE lines(id INTEGER PRIMARY KEY,frame INTEGER NOT NULL REFERENCES frames(id),
                    text TEXT NOT NULL,normalized_text TEXT NOT NULL,confidence REAL,bbox TEXT NOT NULL,region TEXT NOT NULL);
                CREATE VIRTUAL TABLE line_fts USING fts5(text, tokenize='unicode61 remove_diacritics 0');
            """)
            line_count = 0
            for frame in frames:
                cursor = db.execute("INSERT INTO frames VALUES(NULL,?,?,?,?,?,?,?,?,?)",
                    (frame["video_id"], video_group(frame["video_id"]), frame["frame_id"], frame["keyframe_n"],
                     frame["pts_time"], frame["image_sha256"], frame["width"], frame["height"], frame["status"]))
                for line in frame["lines"]:
                    normalized = normalize_text(line["text"])
                    row = db.execute("INSERT INTO lines VALUES(NULL,?,?,?,?,?,?)", (cursor.lastrowid, line["text"], normalized,
                        line["confidence"], json.dumps(line["bbox"]), line["region"]))
                    db.execute("INSERT INTO line_fts(rowid,text) VALUES(?,?)", (row.lastrowid, normalized))
                    line_count += 1
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("OCR SQLite integrity check failed")
        if digest(detections) != input_hash:
            raise ValueError("OCR source changed during build")
        for vid, expected in mapping_hashes.items():
            _, path = mapping_rows(vid, mapping_dir)
            if digest(path) != expected:
                raise ValueError("Mapping changed during OCR index build")
        manifest = {"schema": SCHEMA, "source_kind": "pixel_ocr", "build_id": uuid4().hex,
            "engine": data["engine"], "index_sha256": digest(index_path), "input_sha256": input_hash,
            "mapping_sha256": mapping_hashes, "frames": len(frames), "lines": line_count,
            "frame_status_counts": dict(Counter(f["status"] for f in frames)),
            "regions": dict(Counter(line["region"] for f in frames for line in f["lines"])),
            "confidence_is_calibrated": False, "verified": False}
        atomic_write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        atomic_write_json(output / "failure.json", {"type": type(exc).__name__})
        raise


class LocalPaddleOCR:
    """Optional pinned-package Paddle v3 adapter, explicitly local CPU models."""

    def __init__(self, settings):
        validate_engine(settings)
        if settings["backend"] != "paddleocr" or len(settings["models"]) != 2:
            raise ValueError("Paddle adapter needs detection and recognition model manifests")
        for package, expected in (("paddleocr", settings["package_version"]),
                                  ("paddlepaddle", settings["paddle_version"])):
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError("OCR runtime absent: use the isolated pilot notebook; do not modify the main environment") from exc
            if actual != expected:
                raise ValueError(f"Pinned OCR package differs: {package}")
        self.settings = settings
        self.check_models()
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR

        det, rec = settings["models"]
        self.model = PaddleOCR(text_detection_model_name=det["name"], text_detection_model_dir=det["directory"],
            text_recognition_model_name=rec["name"], text_recognition_model_dir=rec["directory"],
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            device="cpu")

    def check_models(self):
        for model in self.settings["models"]:
            root = Path(model["directory"]).resolve()
            for name, expected in model["files_sha256"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root) or digest(path) != expected:
                    raise ValueError("OCR model path/checksum mismatch")
            if not {"inference.json", "inference.pdiparams", "inference.yml"}.issubset(model["files_sha256"]):
                raise ValueError("Incomplete local Paddle inference model")

    def recognize(self, path):
        results = list(self.model.predict(str(path), text_rec_score_thresh=0.0))
        if len(results) != 1:
            raise ValueError("Expected exactly one OCR result per keyframe")
        data = results[0].json
        if isinstance(data, str):
            data = json.loads(data)
        data = data.get("res", data)
        texts, scores, boxes = data["rec_texts"], data["rec_scores"], data["rec_polys"]
        if not len(texts) == len(scores) == len(boxes):
            raise ValueError("OCR text/score/bbox count mismatch")
        return [{"text": text, "confidence": float(score),
                 "bbox": box.tolist() if hasattr(box, "tolist") else box, "region": "unknown"}
                for text, score, box in zip(texts, scores, boxes) if isinstance(text, str) and text.strip()]


def extract(manifest_path, engine_path, image_root, mapping_dir, output, max_frames=12, backend=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite OCR detections")
    manifest = read_json(manifest_path)
    settings = read_json(engine_path)
    if manifest.get("schema") != "kis_ocr_input_v1" or not 1 <= len(manifest.get("frames", [])) <= max_frames <= 256:
        raise ValueError("Explicit bounded OCR input manifest required")
    paths = [verified_image(frame, image_root, mapping_dir)[0] for frame in manifest["frames"]]
    # Model initialization failures are fatal, not an artifact full of negatives.
    backend = backend or LocalPaddleOCR(settings)
    validate_engine(settings)
    frames = []
    for original, path in zip(manifest["frames"], paths):
        frame = dict(original)
        started = perf_counter()
        try:
            frame["lines"] = backend.recognize(path)
            frame["status"] = "ok" if frame["lines"] else "empty"
            validate_frame(frame)
        except Exception as exc:
            frame.update(status="error", lines=[], error_type=type(exc).__name__)
        frame["seconds"] = perf_counter() - started
        frames.append(frame)
    if hasattr(backend, "check_models"):
        backend.check_models()
    result = {"schema": SCHEMA, "source_kind": "pixel_ocr",
        "engine": {**settings, "models": [{k: v for k, v in model.items() if k != "directory"} for model in settings["models"]]},
        "frames": frames, "complete": all(f["status"] != "error" for f in frames),
        "input_manifest_sha256": digest(manifest_path), "verified": False}
    atomic_write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "index"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, default=MAPPING_DIR)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    if args.command == "extract":
        if args.engine is None:
            parser.error("extract requires --engine with pinned packages/local model checksums")
        result = extract(args.input, args.engine, args.image_root, args.mapping_dir, args.output, args.max_frames)
        print(json.dumps({"frames": len(result["frames"]), "complete": result["complete"]}))
        raise SystemExit(0 if result["complete"] else 2)
    else:
        print(json.dumps(build_index(args.input, args.image_root, args.mapping_dir, args.output), indent=2))
