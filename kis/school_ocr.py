"""Pinned, immutable PUBLIC OCR snapshots for the school CPU worker.

Source JSON has NO engine, confidence, boxes or pixel hashes. Consolidating and
hashing it does not create that provenance. No OCR model or caption is used here.
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from kis.ocr import contains_literal, normalize_text
from kis.paths import resolve_video_artifact, video_group
from kis.public_ocr import FIELDS, PublicOCRRetriever, checked_path, logical_image, unique_object


SCHEMA = "school_public_ocr_snapshot_v1"


def validate_groups(groups):
    if not groups or len(set(groups)) != len(groups) or any(not re.fullmatch(r"L(?:0[1-9]|1[0-9]|2[0-4])", g) for g in groups):
        raise ValueError("Invalid or duplicate OCR groups")


def digest(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


def mappings(root, group, *, allow_frame_aliases=False):
    rows, hashes = {}, {}
    for path in sorted((Path(root) / group).rglob(f"{group}_V*.csv")):
        vid = path.stem
        if not re.fullmatch(group + r"_V[0-9]{3}", vid) or vid in hashes:
            raise ValueError("Duplicate OCR mapping file")
        with checked_path(path).open(encoding="utf-8", newline="") as stream:
            entries = list(csv.DictReader(stream))
        frame_ids = set()
        for row in entries:
            n, frame = int(row["n"]), int(row["frame_idx"])
            if n < 1 or frame < 0 or (vid, n) in rows or (frame in frame_ids and not allow_frame_aliases):
                raise ValueError("Ambiguous OCR mapping identity")
            rows[vid, n] = frame
            frame_ids.add(frame)
        hashes[vid] = digest(path)
    if not rows:
        raise ValueError("No authoritative OCR mapping")
    return rows, hashes


def validate_record(payload, video_id, number, frame):
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict) or set(payload[0]) != FIELDS:
        raise ValueError("Expected exactly one public OCR record, seven fields")
    row = payload[0]
    if (row["video_id"] != video_id or type(row["keyframe_n"]) is not int or row["keyframe_n"] != number
            or row["frame_id"] is not None and (type(row["frame_id"]) is not int or row["frame_id"] != frame)):
        raise ValueError("OCR record identity differs from authoritative mapping")
    logical_image(row["image_path"], video_id, number)
    text, status = row["ocr_text"], row["status"]
    if (not isinstance(text, str) or len(text) > 65536 or "\0" in text
            or not isinstance(row["retrieval_text"], str) or row["retrieval_text"] != text
            or not isinstance(status, str) or status not in {"ok", "empty", "error"} or (status == "ok") != bool(text.strip())):
        raise ValueError("Invalid OCR text/status contract")
    return row


def build_snapshot(dataset, groups, output):
    validate_groups(groups)
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "source_format": "public_json", "verified": False,
              "provenance": "UNVERIFIED_public_OCR_no_image_SHA_or_engine", "groups": {}}
    for group in groups:
        expected, hashes = mappings(dataset / "mapping", group, allow_frame_aliases=True)
        records, seen, states, source_hashes = [], set(), Counter(), {}
        for path in sorted((dataset / "ocr" / group).rglob("*.json")):
            vid, n = path.parent.name, int(path.stem)
            if path.relative_to(dataset / "ocr").as_posix() != f"{group}/{vid}/{n:03d}.json":
                raise ValueError("Noncanonical source OCR path")
            key = (vid, n)
            if key in seen or key not in expected:
                raise ValueError("Duplicate or unmapped source OCR")
            raw = checked_path(path).read_bytes()
            row = validate_record(json.loads(raw, object_pairs_hook=unique_object), vid, n, expected[key])
            source_hashes[path.relative_to(dataset / "ocr").as_posix()] = hashlib.sha256(raw).hexdigest()
            records.append(row)
            seen.add(key)
            states[row["status"]] += 1
        if seen != set(expected):
            raise ValueError("Incomplete OCR group: " + group)
        target = output / f"{group}.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, separators=(",", ":"))
        result["groups"][group] = {"path": target.name, "sha256": digest(target),
                                   "records": len(records), "status_counts": dict(states),
                                   "mapping_sha256": hashes, "source_json_sha256": source_hashes}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return {"root": str(output), "manifest_sha256": digest(output / "MANIFEST.json"),
            "groups": {g: {k: v for k, v in row.items() if k not in {"mapping_sha256", "source_json_sha256"}} for g, row in result["groups"].items()},
            "records": sum(row["records"] for row in result["groups"].values())}


class SchoolOCRSnapshot:
    supports_candidate_ocr = True
    rank_fusion = "bounded_positive_auxiliary"
    _evidence = staticmethod(PublicOCRRetriever._evidence)
    validate_config = PublicOCRRetriever.validate_config
    _validate_candidate_image = PublicOCRRetriever._validate_candidate_image

    def __init__(self, root, manifest_sha256, groups, dataset, frame_rows):
        validate_groups(groups)
        self.root, self.dataset = checked_path(root), Path(dataset)
        self.image_root = self.dataset / "data"
        self.video_groups = tuple(groups)
        self.manifest_path = checked_path(self.root / "MANIFEST.json")
        if digest(self.manifest_path) != manifest_sha256:
            raise ValueError("Pinned OCR manifest differs")
        self.manifest_sha256 = manifest_sha256
        manifest = json.loads(self.manifest_path.read_bytes(), object_pairs_hook=unique_object)
        if manifest.get("schema") != SCHEMA or manifest.get("source_format") != "public_json" or manifest.get("verified") is not False:
            raise ValueError("Not a public OCR snapshot")
        self.bindings, identities = {}, set()
        for r in frame_rows:
            vid, n, frame = r["video_id"], r["keyframe_n"], r["frame_id"]
            if (not isinstance(vid, str) or not re.fullmatch(r"L[0-9]{2}_V[0-9]{3}", vid)
                    or video_group(vid) not in self.video_groups or type(n) is not int or n < 1
                    or type(frame) is not int or frame < 0 or (vid, n) in self.bindings or (vid, frame) in identities):
                raise ValueError("Ambiguous runtime OCR binding")
            self.bindings[vid, n] = frame
            identities.add((vid, frame))
        self.inverse = {(vid, frame): n for (vid, n), frame in self.bindings.items()}
        self.records, self.assets = {}, {self.manifest_path: manifest_sha256}
        self.covered_groups = []
        self.excluded_alias_records = 0
        for group in self.video_groups:
            info = manifest["groups"].get(group)
            if not info:
                continue
            if info["path"] != f"{group}.json":
                raise ValueError("Unsafe OCR bundle name")
            bundle = checked_path(self.root / info["path"])
            if digest(bundle) != info["sha256"]:
                raise ValueError("OCR bundle checksum mismatch")
            self.assets[bundle] = info["sha256"]
            rows = json.loads(bundle.read_bytes(), object_pairs_hook=unique_object)
            if not isinstance(rows, list) or len(rows) != info["records"]:
                raise ValueError("OCR bundle count mismatch")
            expected, mapping_hashes = mappings(self.dataset / "mapping", group, allow_frame_aliases=True)
            frame_counts = Counter((vid, frame) for (vid, _), frame in expected.items())
            if mapping_hashes != info["mapping_sha256"]:
                raise ValueError("OCR mapping inventory mismatch")
            for vid, expected_hash in info["mapping_sha256"].items():
                if video_group(vid) != group:
                    raise ValueError("OCR mapping outside selected group")
                path = checked_path(resolve_video_artifact(self.dataset / "mapping", vid, ".csv", video_groups=self.video_groups))
                if digest(path) != expected_hash:
                    raise ValueError("OCR mapping checksum mismatch")
                self.assets[path] = expected_hash
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("OCR bundle row is not an object")
                vid, n = row.get("video_id"), row.get("keyframe_n")
                if not isinstance(vid, str) or video_group(vid) != group or type(n) is not int:
                    raise ValueError("OCR bundle identity outside scope")
                key = (vid, n)
                frame = expected.get(key)
                if key in self.records or frame is None:
                    raise ValueError("OCR bundle join missing or ambiguous")
                if key not in self.bindings:
                    # Preserve source OCR, but never index aliases deliberately
                    # excluded by ExactVisualIndex. A missing unique binding is
                    # still an error, as is a partially retained alias frame.
                    if frame_counts[vid, frame] < 2 or (vid, frame) in identities:
                        raise ValueError("OCR bundle join missing or ambiguous")
                    self.excluded_alias_records += 1
                elif expected[key] != self.bindings[key]:
                    raise ValueError("OCR bundle join missing or ambiguous")
                validate_record([row], vid, n, frame)
                self.records[key] = row
            if {k for k in self.records if video_group(k[0]) == group} != set(expected):
                raise ValueError("Incomplete OCR snapshot group")
            if dict(Counter(r["status"] for r in rows)) != info["status_counts"]:
                raise ValueError("OCR status counts mismatch")
            self.covered_groups.append(group)
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE ocr_fts USING fts5(text,video_id UNINDEXED,n UNINDEXED,tokenize='unicode61 remove_diacritics 0')")
        self.db.executemany("INSERT INTO ocr_fts VALUES (?,?,?)", [(normalize_text(r["ocr_text"]), vid, n)
                              for (vid, n), r in self.records.items() if r["status"] == "ok" and (vid, n) in self.bindings])
        self.db.execute("PRAGMA query_only=ON")
        self.last_error = None

    def _check(self):
        # Hash a handful of consolidated blobs + CSVs, not 45k tiny files. Same
        # size/restored mtime cannot conceal mutations of consumed assets.
        try:
            for path, expected in self.assets.items():
                checked_path(path)
                if digest(path) != expected:
                    raise ValueError("OCR snapshot changed; restart with a validated build")
        except (OSError, ValueError) as exc:
            self.last_error = type(exc).__name__
            raise

    def status(self):
        return {"source": "public_ocr", "source_format": "public_json", "storage": SCHEMA,
                "available": self.last_error is None, "loaded": True, "verified": False,
                "provenance": "UNVERIFIED_public_OCR_no_image_SHA_or_engine", "build_id": self.manifest_sha256,
                "covered_groups": self.covered_groups, "records": len(self.records),
                "bound_records": len(self.records) - self.excluded_alias_records,
                "searchable_records": sum(r['status'] == 'ok' and key in self.bindings for key, r in self.records.items()),
                "excluded_ambiguous_keyframes": self.excluded_alias_records,
                "missing_groups": [g for g in self.video_groups if g not in self.covered_groups],
                "capabilities": {"confidence": False, "regions": False, "image_sha256": False},
                "warnings": ["Public OCR confidence unknown; default strict confidence threshold is not applied or satisfied.",
                             "OCR coverage is partial; missing/empty/error is unknown, not negative evidence."],
                "last_error": self.last_error}

    def attach_candidates(self, candidates, config):
        self.validate_config(config)
        self._check()
        for candidate in candidates:
            vid, n, frame = candidate["video_id"], candidate.get("keyframe_n"), candidate["frame_id"]
            if video_group(vid) not in self.video_groups or type(frame) is not int:
                raise ValueError("OCR candidate outside runtime scope")
            if n is None:
                n = self.inverse.get((vid, frame))
            if type(n) is not int or self.bindings.get((vid, n)) != frame:
                raise ValueError("OCR candidate mapping identity mismatch")
            if candidate.get("image_path"):
                self._validate_candidate_image(candidate["image_path"], vid, n)
            row = self.records.get((vid, n))
            evidence = self._evidence(row["status"], row["ocr_text"], row["image_path"]) if row else self._evidence("missing")
            evidence.update(keyframe_n=n, build_id=self.manifest_sha256)
            for key in ("matched_query", "bm25_raw"):
                if key in candidate.get("ocr", {}):
                    evidence[key] = candidate["ocr"][key]
            candidate["keyframe_n"], candidate["ocr"] = n, evidence
            if "ocr" in candidate.get("evidence", {}):
                candidate["evidence"]["ocr"].update(source="public_ocr", source_format="public_json", verified=False)

    def search(self, text, config):
        self.validate_config(config)
        self._check()
        if not isinstance(text, str) or not text.strip() or len(text) > 256:
            raise ValueError("OCR requires a bounded original literal")
        if not any(c.isalnum() for c in text):
            return []
        query = '"' + normalize_text(text).replace('"', '""') + '"'
        result = []
        rows = self.db.execute("SELECT video_id,n,bm25(ocr_fts) FROM ocr_fts WHERE ocr_fts MATCH ? ORDER BY bm25(ocr_fts),video_id,CAST(n AS INTEGER) LIMIT ?",
                               (query, min(2000, config.top_k * 20))).fetchall()
        for vid, number, distance in rows:
            n = int(number)
            row = self.records[vid, n]
            if not contains_literal(row["ocr_text"], text):
                continue
            evidence = self._evidence(row["status"], row["ocr_text"], row["image_path"])
            evidence.update(matched_query=text, bm25_raw=distance, build_id=self.manifest_sha256)
            result.append({"video_id": vid, "frame_id": self.bindings[vid, n], "keyframe_n": n,
                           "image_path": row["image_path"], "score": 1.0 / (len(result) + 1), "ocr": evidence})
            if len(result) >= config.top_k:
                break
        return result

    def close(self):
        self.db.close()
