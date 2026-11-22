"""Object metadata lookup and optional evidence boosting."""

import csv
import os
import json
import math
import re
from pathlib import Path
from kis.paths import (
    KIS_ROOT,
    MAPPING_DIR,
    OBJECTS_DIR,
    resolve_video_artifact,
    resolve_video_object_dir,
)

LABEL_MAP_PATH = KIS_ROOT / "config" / "object_labels.json"
LABEL_MAP = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["labels"]
OBJECT_FIELDS = (
    "detection_scores", "detection_class_names", "detection_class_entities",
    "detection_boxes", "detection_class_labels",
)


def validate_object_schema(record):
    if not isinstance(record, dict) or set(record) != set(OBJECT_FIELDS):
        raise ValueError("Object JSON must contain exactly the five provider fields")
    if any(not isinstance(record[field], list) for field in OBJECT_FIELDS):
        raise ValueError("All object fields must be arrays")
    count = len(record["detection_scores"])
    if any(len(record[field]) != count for field in OBJECT_FIELDS):
        raise ValueError("Object arrays must have equal lengths")
    for score, mid, entity, box, label in zip(*(record[field] for field in OBJECT_FIELDS)):
        if not all(isinstance(value, str) for value in (score, mid, entity, label)):
            raise ValueError("Object scores, class names, entities and labels must be strings")
        if not re.fullmatch(r"/m/[A-Za-z0-9_]+", mid) or not entity.strip():
            raise ValueError("Invalid Open Images class name/entity")
        if not label.isdigit() or int(label) < 1:
            raise ValueError("Object class labels must be positive integer strings")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid object confidence")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, str) for value in box):
            raise ValueError("Object boxes must contain four normalized coordinate strings")
        coordinates = [float(value) for value in box]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise ValueError("Object box coordinates must be finite and in [0,1]")
        if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
            raise ValueError("Object boxes must use ymin,xmin,ymax,xmax order")


def provider_object_record(detection, max_detections=100):
    labels = detection.get("detection_class_entities", [])
    scores = detection.get("detection_scores", [])
    boxes = detection.get("detection_boxes", [])
    if not all(isinstance(values, list) for values in (labels, scores, boxes)):
        raise ValueError("Detection fields must be arrays")
    if len(labels) != len(scores) or len(labels) != len(boxes):
        raise ValueError("Detection arrays must have equal lengths")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    entries = [(float(score), LABEL_MAP[str(label).casefold()], box)
               for label, score, box in zip(labels, scores, boxes) if str(label).casefold() in LABEL_MAP]
    entries.sort(key=lambda entry: entry[0], reverse=True)
    output = {field: [] for field in OBJECT_FIELDS}
    for score, mapped, box in entries[:max_detections]:
        output["detection_scores"].append(str(score))
        output["detection_class_names"].append(mapped["mid"])
        output["detection_class_entities"].append(mapped["entity"])
        output["detection_boxes"].append([str(float(value)) for value in box])
        output["detection_class_labels"].append(mapped["label"])
    validate_object_schema(output)
    return output


class ObjectMetadataLookup:
    # shortcut: FIFO cap on parsed records (~3 KB each); raise it if the worker has spare RAM.
    MAX_CACHED_RECORDS = 50000

    def __init__(self, mapping_dir=MAPPING_DIR, objects_dir=OBJECTS_DIR):
        self.mapping_dir = Path(mapping_dir)
        self.objects_dir = Path(objects_dir)
        self.csv_cache = {}
        self._records = {}

    def get_keyframe_n(self, video_id, frame_id):
        numbers = self.csv_cache.get(video_id)
        if numbers is None:
            csv_path = resolve_video_artifact(self.mapping_dir, video_id, ".csv")
            if not os.path.exists(csv_path):
                return None
            numbers = {}
            with open(csv_path, encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    numbers.setdefault(int(row["frame_idx"]), int(row["n"]))  # first row wins, as before
            self.csv_cache[video_id] = numbers
        return numbers.get(int(frame_id))

    def get(self, video_id, frame_id=None):
        if frame_id is None:
            if isinstance(video_id, (tuple, list)) and len(video_id) == 2:
                video_id, frame_id = video_id
            else:
                raise TypeError(
                    "ObjectMetadataLookup.get() requires (video_id, frame_id) "
                    "or a 2-item tuple/list key."
                )
        key = (video_id, int(frame_id))
        if key in self._records:
            return self._records[key]
        record = self._read(video_id, frame_id)
        if len(self._records) >= self.MAX_CACHED_RECORDS:
            self._records.pop(next(iter(self._records)))
        self._records[key] = record
        return record

    def _read(self, video_id, frame_id):
        n = self.get_keyframe_n(video_id, frame_id)
        if n is None:
            return None
        json_path = resolve_video_object_dir(video_id, self.objects_dir) / f"{n:03d}.json"
        if not os.path.exists(json_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prefetch(self, keys, workers=16):
        """Read uncached records in parallel: cold small-file reads are seek-bound, not CPU-bound."""
        from concurrent.futures import ThreadPoolExecutor
        missing = list(dict.fromkeys((vid, int(frame)) for vid, frame in keys if (vid, int(frame)) not in self._records))
        if len(missing) < 2:
            return
        for video_id in {vid for vid, _ in missing}:
            self.get_keyframe_n(video_id, -1)  # load each mapping once, outside the thread pool
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, record in zip(missing, pool.map(lambda key: self._read(*key), missing)):
                if len(self._records) >= self.MAX_CACHED_RECORDS:
                    self._records.pop(next(iter(self._records)))
                self._records[key] = record


DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_OBJECT_WEIGHT = 0.15
OBJECT_ALIASES = {
    "photo": {"photo", "photograph", "picture", "picture frame", "poster"},
    "painting": {"painting", "picture", "picture frame", "poster"},
    "bức ảnh": {"photo", "photograph", "picture", "picture frame"},
    "tranh": {"painting", "picture", "picture frame", "poster"},
    "artisan hand": {"hand", "human hand"},
    "bàn tay nghệ nhân": {"hand", "human hand"},
}


def _normalize_label(value):
    return " ".join(re.sub(r"[^\w\s]", " ", str(value).casefold()).split())


def _detector_labels(entity):
    explicit = entity.get("detector_labels", [])
    if isinstance(explicit, str):
        explicit = [explicit]
    parts = re.split(r"\s*/\s*|\s+or\s+|\s+hoặc\s+", str(entity.get("value", "")), flags=re.IGNORECASE)
    labels = {label for value in [*explicit, *parts] if (label := _normalize_label(value))}
    for label in list(labels):
        labels.update(OBJECT_ALIASES.get(label, ()))
    return labels


def apply_object_filter(candidates, entities, object_lookup,
                        min_confidence=DEFAULT_MIN_CONFIDENCE, object_weight=DEFAULT_OBJECT_WEIGHT):
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if not 0.0 <= object_weight <= 1.0:
        raise ValueError("object_weight must be between 0 and 1")
    object_entities = [entity for entity in entities if entity.get("type") == "object"]
    if not object_entities:
        return candidates
    print(f"[Object Score] Checking {len(object_entities)} object entities")
    prefetch = getattr(object_lookup, "prefetch", None)
    if prefetch is not None:
        prefetch([(candidate["video_id"], candidate["frame_id"]) for candidate in candidates])
    for candidate in candidates:
        key = (candidate["video_id"], candidate["frame_id"])
        retrieval_score = float(candidate.setdefault("retrieval_score", candidate["score"]))
        if not math.isfinite(retrieval_score):
            raise ValueError(f"Non-finite retrieval score for {key}")
        meta = object_lookup.get(key)
        if meta is None:
            candidate["object_score"] = 0.0
            continue
        detected = {}
        for label, raw in zip(meta.get("detection_class_entities", []), meta.get("detection_scores", [])):
            try:
                confidence = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or confidence < min_confidence:
                continue
            label = _normalize_label(label)
            detected[label] = max(min(1.0, max(0.0, confidence)), detected.get(label, 0.0))
        scores = [max((detected.get(label, 0.0) for label in labels), default=0.0)
                  for entity in object_entities if (labels := _detector_labels(entity))]
        candidate["object_score"] = sum(scores) / len(scores) if scores else 0.0
        candidate["score"] = retrieval_score + object_weight * candidate["object_score"] * max(0.0, 1.0 - retrieval_score)
    return sorted(candidates, key=lambda item: item["score"], reverse=True)
