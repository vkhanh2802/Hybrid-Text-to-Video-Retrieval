"""Independent pixel-OCR evidence and bounded lexical retrieval (no OCR model).

Caption OCR fields are deliberately NOT an input. Recognition confidence and
literal matches never verify an action, relation, or continuous sequence.
"""

from copy import deepcopy
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from time import perf_counter
import unicodedata

from kis.paths import MAPPING_DIR, MAX_VIDEO_GROUP, resolve_video_artifact, video_group


SCHEMA = "kis_pixel_ocr_v1"
REGIONS = {"content", "overlay", "unknown"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize_text(text):
    # Preserve diacritics, digits, negation and equation operators. No translation
    # or guessed correction, and do not silently turn Vietnamese into ASCII.
    # NFC, not NFKC: compatibility folding would incorrectly equate x² with x2.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def contains_literal(text, literal):
    text, literal = normalize_text(text), normalize_text(literal)
    if not literal:
        return False
    start = 0
    while (index := text.find(literal, start)) >= 0:
        end = index + len(literal)
        before = index == 0 or not (literal[0].isalnum() and text[index-1].isalnum())
        after = end == len(text) or not (literal[-1].isalnum() and text[end].isalnum())
        if before and after:
            return True
        start = index + 1
    return False


@dataclass(frozen=True)
class OCRConfig:
    mode: str = "off"
    top_k: int = 50
    max_queries: int = 4
    min_confidence: float = 0.5
    weight: float = 0.5
    region: str = "all"

    def __post_init__(self):
        if self.mode not in {"off", "diagnostics", "retrieve"} or self.region not in {"all", *REGIONS}:
            raise ValueError("Invalid OCR mode/region")
        for name, bound in (("top_k", 100), ("max_queries", 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= bound:
                raise ValueError(f"ocr.{name} must be an integer in [1,{bound}]")
        for name in ("min_confidence", "weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"ocr.{name} must be finite in [0,1]")

    def public(self):
        return dict(asdict(self), source="pixel_ocr", verified=False, temporal_support="candidate_seeds_only")


def literal_queries(query, config):
    """Only explicit user text or literal spans; never search translated captions."""
    raw = query.get("raw_query", "")
    manual = query.get("ocr_text")
    found = []
    if isinstance(manual, str) and manual.strip():
        found.append((manual.strip(), "user_ocr_text"))
    if query.get("needs_ocr") is True:
        for text in re.findall(r'["“]([^"”\n]{1,256})["”]', raw):
            found.append((text.strip(), "raw_quoted_text"))
        for entity in query.get("entities", []):
            text = entity.get("value") if isinstance(entity, dict) and entity.get("type") == "text" else None
            if isinstance(text, str) and contains_literal(raw, text):
                found.append((text.strip(), "literal_text_entity"))
    from kis.ocr_routing import raw_text_spans
    found.extend(raw_text_spans(raw))
    result, seen = [], set()
    for text, source in found:
        key = normalize_text(text)
        if key and len(text) <= 256 and any(c.isalnum() for c in key) and key not in seen:
            seen.add(key)
            result.append({"text": text, "source": source})
    return result[:config.max_queries]


def validate_engine(engine):
    if (not isinstance(engine, dict) or engine.get("backend") not in {"paddleocr", "tesseract", "easyocr"}
            or not isinstance(engine.get("package_version"), str) or not engine["package_version"].strip()
            or not isinstance(engine.get("models"), list) or not engine["models"]):
        raise ValueError("OCR requires explicit engine/package/model provenance, not caption metadata")
    for model in engine["models"]:
        if (not isinstance(model.get("name"), str) or not model["name"]
                or not isinstance(model.get("revision"), str) or not model["revision"]):
            raise ValueError("OCR model name/revision missing")
        hashes = model.get("files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("OCR model file hashes missing")
        if any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes.values()):
            raise ValueError("Invalid OCR model checksum")


def validate_frame(frame):
    group = video_group(frame["video_id"])
    if int(group[1:]) > MAX_VIDEO_GROUP:
        raise ValueError("OCR frame outside supported corpus")
    for key, minimum in (("frame_id", 0), ("keyframe_n", 1), ("width", 1), ("height", 1)):
        if isinstance(frame.get(key), bool) or not isinstance(frame.get(key), int) or frame[key] < minimum:
            raise ValueError(f"Invalid OCR frame {key}")
    t = frame.get("pts_time")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
        raise ValueError("Invalid OCR timestamp")
    if not re.fullmatch(r"[a-f0-9]{64}", str(frame.get("image_sha256", ""))):
        raise ValueError("OCR frame requires image checksum")
    lines = frame.get("lines")
    if not isinstance(lines, list) or len(lines) > 4096:
        raise ValueError("Invalid OCR line list")
    if frame.get("status") not in {"ok", "empty", "error"} or (frame["status"] != "ok" and lines):
        raise ValueError("Invalid OCR frame status")
    if frame["status"] == "ok" and not lines:
        raise ValueError("Successful OCR frame must contain at least one nonempty line")
    for line in lines:
        text, confidence, box = line.get("text"), line.get("confidence"), line.get("bbox")
        if not isinstance(text, str) or not text.strip() or len(text) > 4096 or "\x00" in text:
            raise ValueError("Invalid raw OCR text")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("OCR confidence must be unknown or finite in [0,1]")
        if line.get("region") not in REGIONS or not isinstance(box, list) or len(box) != 4:
            raise ValueError("OCR requires explicit content/overlay/unknown region and quadrilateral bbox")
        for point in box:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Invalid OCR polygon")
            for coordinate, bound in zip(point, (frame["width"], frame["height"])):
                if (isinstance(coordinate, bool) or not isinstance(coordinate, (int, float))
                        or not math.isfinite(coordinate) or not 0 <= coordinate <= bound):
                    raise ValueError("OCR bbox outside original image")
        area2 = sum(box[i][0]*box[(i+1) % 4][1] - box[(i+1) % 4][0]*box[i][1] for i in range(4))
        if abs(area2) <= 0:
            raise ValueError("Degenerate OCR bbox")


class OCRRetriever:
    """Read-only SQLite FTS5; lazy integrity checks, no runtime OCR inference."""

    def __init__(self, index_path, video_groups, mapping_dir=MAPPING_DIR):
        self.index_path = Path(index_path)
        self.video_groups = tuple(video_groups)
        for group in self.video_groups:
            if int(video_group(group + "_V001")[1:]) > MAX_VIDEO_GROUP:
                raise ValueError("OCR scope outside corpus")
        if not self.video_groups:
            raise ValueError("OCR scope must not be empty")
        self.mapping_dir = Path(mapping_dir)
        self._manifest = None
        self._stamp = None
        self._mapping_stamps = {}
        self.last_error = None

    def status(self):
        return {"available": self.index_path.is_file() and self.index_path.with_name("manifest.json").is_file(),
                "loaded": self._manifest is not None, "index_path": str(self.index_path),
                "last_error": self.last_error, "source": "pixel_ocr", "verified": False}

    def _check(self):
        manifest_path = self.index_path.with_name("manifest.json")
        if not self.index_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Independent OCR index unavailable; build OCR artifacts first")
        stamp = (self.index_path.stat().st_size, self.index_path.stat().st_mtime_ns,
                 manifest_path.stat().st_size, manifest_path.stat().st_mtime_ns)
        if self._manifest is not None:
            if stamp != self._stamp:
                raise ValueError("OCR index changed; restart service on a validated immutable build")
            return self._manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("source_kind") != "pixel_ocr":
            raise ValueError("Not an independent pixel-OCR index")
        validate_engine(manifest["engine"])
        if digest(self.index_path) != manifest["index_sha256"]:
            raise ValueError("OCR index checksum mismatch")
        self._manifest, self._stamp = manifest, stamp
        return manifest

    def search(self, text, config):
        if not isinstance(text, str) or not text.strip() or len(text) > 256:
            raise ValueError("OCR search needs a nonempty literal <=256 characters")
        if not any(c.isalnum() for c in text):
            return []
        try:
            manifest = self._check()
            params = [normalize_text(text)]
            # Quote the entire literal; user content is never an FTS expression.
            params[0] = '"' + params[0].replace('"', '""') + '"'
            sql = """SELECT f.video_id,f.frame_id,f.keyframe_n,f.pts_time,f.image_sha256,
                            l.text,l.confidence,l.bbox,l.region,bm25(line_fts) AS distance
                     FROM line_fts JOIN lines l ON l.id=line_fts.rowid JOIN frames f ON f.id=l.frame
                     WHERE line_fts MATCH ? AND l.confidence>=? AND f.video_group IN ("""
            sql += ",".join("?" for _ in self.video_groups) + ")"
            params += [config.min_confidence, *self.video_groups]
            if config.region != "all":
                sql += " AND l.region=?"
                params.append(config.region)
            sql += " ORDER BY distance,f.video_id,f.frame_id,l.id LIMIT ?"
            params.append(min(2000, config.top_k * 20))
            with closing(sqlite3.connect(self.index_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                db.execute("PRAGMA query_only=ON")
                db.row_factory = sqlite3.Row
                rows = db.execute(sql, params).fetchall()
            hits, seen = [], set()
            for row in rows:
                if not contains_literal(row["text"], text):
                    continue  # FTS drops operators; literal verification preserves them.
                vid = row["video_id"]
                mapping = resolve_video_artifact(self.mapping_dir, vid, ".csv")
                stamp = (mapping.stat().st_size, mapping.stat().st_mtime_ns)
                if self._mapping_stamps.get(vid) != stamp:
                    if digest(mapping) != manifest["mapping_sha256"].get(vid):
                        raise ValueError("OCR/mapping fingerprint mismatch")
                    self._mapping_stamps[vid] = stamp
                key = (vid, row["frame_id"])
                if key in seen:
                    continue
                seen.add(key)
                hits.append({"video_id": vid, "frame_id": row["frame_id"], "keyframe_n": row["keyframe_n"],
                    "pts_time": row["pts_time"], "score": 1.0 / len(seen),
                    "ocr": {"source": "pixel_ocr", "verified": False, "matched_query": text,
                        "text": row["text"], "confidence": row["confidence"], "region": row["region"],
                        "bbox": json.loads(row["bbox"]), "image_sha256": row["image_sha256"],
                        "bm25_raw": row["distance"], "build_id": manifest["build_id"]}})
                if len(hits) >= config.top_k:
                    break
            self.last_error = None
            return hits
        except (OSError, ValueError, sqlite3.Error, KeyError) as exc:
            self.last_error = type(exc).__name__
            raise


def search_ocr(query, retriever, config):
    started = perf_counter()
    route = literal_queries(query, config) if config.mode != "off" else []
    diag = {"status": "disabled" if config.mode == "off" else "no_literal_text", "queries": route,
            "source": "pixel_ocr", "verified": False, "config": config.public(), "hits": 0}
    hits = {}
    if route and retriever is None:
        diag["status"] = "unavailable"
    elif route:
        try:
            for item in route:
                for rank, hit in enumerate(retriever.search(item["text"], config), 1):
                    key = hit["video_id"], hit["frame_id"]
                    if key not in hits or rank < hits[key][0]:
                        hits[key] = (rank, hit)
            # Literal first. Only try bounded original subphrases after a miss;
            # avoid broad word-OR queries and extra work when the exact text hits.
            if not hits and getattr(retriever, 'supports_candidate_ocr', False) is True:
                from kis.ocr_routing import fallback_fragments
                fallback = []
                seen = {normalize_text(r['text']) for r in route}
                for item in route:
                    for fragment in fallback_fragments(item['text']):
                        key = normalize_text(fragment)
                        if key not in seen:
                            seen.add(key)
                            fallback.append(fragment)
                diag['fallback_queries'] = fallback[:config.max_queries]
                for fragment in diag['fallback_queries']:
                    for rank, hit in enumerate(retriever.search(fragment, config), 1):
                        key = hit['video_id'], hit['frame_id']
                        if key not in hits or rank < hits[key][0]:
                            hit = deepcopy(hit)
                            hit['ocr']['match_kind'] = 'original_subphrase'
                            hits[key] = (rank, hit)
            diag["status"] = "matched" if hits else "no_match"
        except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError) as exc:
            hits.clear()  # Do not rank using only a successful subset of OCR queries.
            diag.update(status="unavailable" if isinstance(exc, FileNotFoundError) else "error",
                        error=type(exc).__name__)
    result = [hit for _, hit in sorted(hits.values(), key=lambda pair: (pair[0], pair[1]["video_id"], pair[1]["frame_id"]))][:config.top_k]
    diag.update(hits=len(result), seconds=perf_counter() - started)
    return result, diag


def fuse_ocr(candidates, hits, config, *, bounded_positive=False):
    if config.mode != "retrieve" or not hits or config.weight == 0:
        return candidates
    values = {(r["video_id"], r["frame_id"]): deepcopy(r) for r in candidates}
    scale = max((max(0., float(r["score"])) for r in candidates), default=0.) if bounded_positive else 1.
    for row in values.values():
        row["pre_ocr_score"] = row["score"]
        row["score"] = float(row["score"]) * (1 if bounded_positive else 1 - config.weight)
    for rank, hit in enumerate(hits, 1):
        key = hit["video_id"], hit["frame_id"]
        row = values.setdefault(key, {**deepcopy(hit), "score": 0.0, "pre_ocr_score": None, "evidence": {}})
        bonus = config.weight * scale * 61 / (60 + rank)
        row["score"] += bonus
        row["ocr"] = deepcopy(hit["ocr"])
        row.setdefault("evidence", {})["ocr"] = {"rank": rank, "contribution": bonus,
                                                    "source": "pixel_ocr", "verified": False}
    for row in values.values():
        row["base_fusion_score"] = row.get("fusion_score")
        row["fusion_score"] = row["score"]
    return sorted(values.values(), key=lambda row: (-row["score"], row["video_id"], row["frame_id"]))
