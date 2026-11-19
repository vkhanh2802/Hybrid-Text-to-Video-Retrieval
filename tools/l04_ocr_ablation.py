"""Offline L04-only exploratory known-target lexical OCR A/B, incomplete qrels.

No production imports, indexes, models, configuration, or dataset-root discovery.
The public OCR adapter here is intentionally NOT a production OCR adapter.
"""

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata

from rank_bm25 import BM25Okapi


LABEL = "exploratory known-target, incomplete qrels, within L04, lexical caption-only baseline"
CONFIG = {
    "group": "L04",
    "baseline": "BM25(stored caption retrieval_text)",
    "treatment": "BM25(same caption text + newline + valid public OCR ocr_text)",
    "k1": 1.5,
    "b": 0.75,
    "epsilon": 0.25,
    "tokenizer": "kis.text unicode equivalent: lowercase, NFC, regex [^\\W_]+; no translation/stopwords",
    "ranking": "positive finite scores only; descending score then (video_id, keyframe_n)",
    "metrics": "one known target/query; Hit@1/5/10 and MRR@10; rank is over all positive results",
    "ocr_adapter": "public_array_7_fields_v1; technical ocr-sidecar-v1 count-only",
    "integrity": "fail closed on malformed/duplicate/identity/path/hash mismatch; require local caption images",
}
PUBLIC_FIELDS = {"video_id", "frame_id", "keyframe_n", "image_path", "ocr_text", "retrieval_text", "status"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_path(path):
    """Reject links, junctions, cloud recall, traversal and excluded group paths before reads."""
    path = Path(path)
    if ".." in path.parts or str(path).startswith(("\\\\", "//")):
        raise ValueError("Unsafe path")
    if any(re.fullmatch(r"L\d+(?:_V\d+)?", part, re.I) and not part.startswith("L04") for part in path.parts):
        raise ValueError("Only L04 paths allowed")
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            break
        flags = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or flags & (0x400 | 0x1000 | 0x40000 | 0x400000):
            raise ValueError("Linked/reparse/cloud placeholder path forbidden")
    return path


def video(value):
    if not isinstance(value, str) or not re.fullmatch(r"L04_V\d{3}", value):
        raise ValueError("Only canonical L04 video identities allowed")
    return value


def integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer identity")
    return value


def identity(record, expected_video, mapping):
    if not isinstance(record, dict) or video(record.get("video_id")) != expected_video:
        raise ValueError("Video identity mismatch")
    n = integer(record.get("keyframe_n"))
    if n not in mapping:
        raise ValueError("Missing n -> frame_idx mapping")
    frame = record.get("frame_id")
    if frame is not None and integer(frame, 0) != mapping[n]:
        raise ValueError("frame_id != mapped frame_idx")
    return expected_video, n


def image_path(value, vid, n):
    # Exact canonical depth and matching numeric image stem; never n-1/frame_idx.
    if not isinstance(value, str) or not re.fullmatch(rf"data/L04/{vid}/[0-9]+\.jpg", value):
        raise ValueError("Noncanonical image_path")
    if int(Path(value).stem) != n:
        raise ValueError("image_path/keyframe_n mismatch")
    return value


def read_json(path, manifest, kind):
    path = checked_path(path)
    raw = path.read_bytes()
    manifest.append({"path": path.as_posix(), "kind": kind, "sha256": byte_sha(raw)})
    return json.loads(raw)


def read_mapping(path, manifest):
    path = checked_path(path)
    raw = path.read_bytes()
    manifest.append({"path": path.as_posix(), "kind": "mapping", "sha256": byte_sha(raw)})
    reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
    if not {"n", "frame_idx"}.issubset(reader.fieldnames or []):
        raise ValueError("Mapping requires n and frame_idx")
    result = {}
    for row in reader:
        if not re.fullmatch(r"[0-9]+", row["n"]) or not re.fullmatch(r"[0-9]+", row["frame_idx"]):
            raise ValueError("Malformed mapping integers")
        n, frame = int(row["n"]), int(row["frame_idx"])
        integer(n)
        if n in result or frame in result.values():
            raise ValueError("Duplicate mapping identity")
        result[n] = frame
    return result


def adapt_public(payload, vid, mapping, file_n):
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Per-image public OCR must be a one-record array")
    record = payload[0]
    if not isinstance(record, dict) or set(record) != PUBLIC_FIELDS:
        raise ValueError("Public OCR must have exactly seven fields")
    key = identity(record, vid, mapping)
    if key[1] != file_n:
        raise ValueError("OCR filename identity mismatch")
    image_path(record["image_path"], *key)
    status, text = record["status"], record["ocr_text"]
    if status not in {"ok", "empty", "error"}:
        raise ValueError("Unknown public OCR status")
    if not isinstance(text, str) or not isinstance(record["retrieval_text"], str) or "\x00" in text:
        raise ValueError("Invalid OCR text type")
    if record["retrieval_text"] != text:
        raise ValueError("Public OCR retrieval_text differs from ocr_text")
    if (status == "ok") != bool(text.strip()):
        raise ValueError("OCR status/text contradiction")
    return key, {"status": status, "text": text if status == "ok" else "", "image_path": record["image_path"]}


def count_technical(payload, vid, mapping, file_n):
    """Recognize old technical envelopes strictly, but never feed their text to this A/B."""
    if payload.get("schema") != "ocr-sidecar-v1":
        raise ValueError("Unrecognized technical OCR object")
    # Lazy dependency only if an old object is actually encountered; no schema guessing.
    from jsonschema import Draft202012Validator

    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/ocr-sidecar-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    key = identity(payload, vid, mapping)
    if key[1] != file_n or payload["group"] != "L04":
        raise ValueError("Technical OCR identity mismatch")
    mapped = payload["mapping"]
    if mapped is not None and mapped["frame_idx"] != mapping[key[1]]:
        raise ValueError("Technical frame_idx mismatch")
    if payload["source_path"] is not None:
        image_path(payload["source_path"], *key)
    if payload["image_name"] is not None and payload["image_name"] != Path(payload["source_path"]).name:
        raise ValueError("Technical image_name mismatch")
    return key


def load_corpus(dataset_root):
    """Enumerate only explicit selected roots; no rglob/discovery at dataset root."""
    roots = {kind: checked_path(Path(dataset_root) / kind / "L04")
             for kind in ("caption_generator", "mapping", "data", "ocr")}
    manifest, mappings, candidates = [], {}, {}
    coverage = Counter({key: 0 for key in (
        "caption_records", "caption_blank", "caption_hash_available", "caption_hash_verified",
        "local_images_hashed", "ocr_files", "public_array_files", "technical_object_files",
        "technical_count_only", "technical_status_unknown", "technical_status_missing",
        "technical_status_ok", "technical_status_empty", "technical_status_error",
        "ocr_records", "matched", "missing", "ok", "empty", "error",
        "unknown_status", "duplicate", "blank", "ocr_without_caption", "ocr_image_hash_unverified")})
    caption_paths = sorted(roots["caption_generator"].glob("*.json"))
    if not caption_paths:
        raise ValueError("No L04 caption records")
    for path in caption_paths:
        vid = video(path.stem)
        mappings[vid] = read_mapping(roots["mapping"] / (vid + ".csv"), manifest)
        records = read_json(path, manifest, "caption")
        if not isinstance(records, list) or not records:
            raise ValueError("Caption artifact must be a nonempty array")
        for record in records:
            key = identity(record, vid, mappings[vid])
            if key in candidates:
                raise ValueError("Duplicate caption identity")
            logical = image_path(record.get("image_path"), *key)
            text = record.get("retrieval_text")
            if not isinstance(text, str) or "\x00" in text:
                raise ValueError("Caption retrieval_text must be a string")
            if record.get("ocr_text"):
                raise ValueError("Caption has pre-existing OCR field; not a caption-only baseline")
            local = checked_path(roots["data"] / vid / Path(logical).name)
            sha = digest(local)
            manifest.append({"path": local.as_posix(), "kind": "image", "sha256": sha})
            coverage["local_images_hashed"] += 1
            expected = record.get("image_sha256")
            if expected is not None:
                if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                    raise ValueError("Invalid caption image SHA")
                coverage["caption_hash_available"] += 1
                if sha != expected:
                    raise ValueError("Local image/caption SHA mismatch")
                coverage["caption_hash_verified"] += 1
            candidates[key] = {"video_id": vid, "keyframe_n": key[1], "frame_id": mappings[vid][key[1]],
                               "image_path": logical, "image_sha256": sha, "caption_text": text,
                               "ocr_status": "missing", "ocr_text": ""}
            coverage["caption_records"] += 1
            coverage["caption_blank"] += not bool(text.strip())
    seen = set()
    for directory in sorted(roots["ocr"].iterdir()):
        checked_path(directory)
        if not directory.is_dir():
            raise ValueError("Unexpected file under L04 OCR root")
        vid = video(directory.name)
        if vid not in mappings:
            mappings[vid] = read_mapping(roots["mapping"] / (vid + ".csv"), manifest)
        for path in sorted(directory.iterdir()):
            checked_path(path)
            if not re.fullmatch(r"[0-9]+\.json", path.name) or not path.is_file():
                raise ValueError("Unexpected OCR artifact filename")
            payload = read_json(path, manifest, "ocr")
            coverage["ocr_files"] += 1
            if isinstance(payload, list):
                coverage["public_array_files"] += 1
                key, ocr = adapt_public(payload, vid, mappings[vid], int(path.stem))
            elif isinstance(payload, dict):
                coverage["technical_object_files"] += 1
                key = count_technical(payload, vid, mappings[vid], int(path.stem))
                coverage["technical_count_only"] += 1
                coverage["technical_status_" + payload["status"]] += 1
                ocr = None
            else:
                raise ValueError("Unknown OCR schema")
            if key in seen:
                raise ValueError("Duplicate OCR identity")
            seen.add(key)
            coverage["ocr_records"] += 1
            if key not in candidates:
                coverage["ocr_without_caption"] += 1
                continue
            if ocr is not None:
                if ocr["image_path"] != candidates[key]["image_path"]:
                    raise ValueError("Caption/OCR image path mismatch")
                candidates[key].update(ocr_status=ocr["status"], ocr_text=ocr["text"])
                coverage["matched"] += 1
                coverage["ocr_image_hash_unverified"] += 1
    for doc in candidates.values():
        coverage[doc["ocr_status"]] += 1
        coverage["blank"] += doc["ocr_status"] == "empty"
    coverage["videos"] = len({key[0] for key in candidates})
    coverage["unchanged_document_text"] = sum(not doc["ocr_text"] for doc in candidates.values())
    return [candidates[key] for key in sorted(candidates)], dict(coverage), manifest


def tokenize(text):
    # Same as kis.text.tokenize_caption(mode="unicode"), without heavyweight imports.
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.lower()), flags=re.UNICODE)


class LexicalIndex:
    def __init__(self, documents, with_ocr=False, *, ocr_only=False):
        self.documents = documents
        texts = [doc["ocr_text"] if ocr_only else doc["caption_text"] + ("\n" + doc["ocr_text"] if with_ocr and doc["ocr_text"] else "")
                 for doc in documents]
        tokens = [tokenize(text) for text in texts]
        self.model = (BM25Okapi(tokens, k1=CONFIG["k1"], b=CONFIG["b"], epsilon=CONFIG["epsilon"])
                      if any(tokens) else None)

    def search(self, text):
        # Signature takes only query TEXT, never targets, image paths, IDs, or qrels.
        if self.model is None:
            return []
        scores = self.model.get_scores(tokenize(text))
        positive = [(i, float(score)) for i, score in enumerate(scores) if math.isfinite(score) and score > 0]
        positive.sort(key=lambda item: (-item[1], self.documents[item[0]]["video_id"], self.documents[item[0]]["keyframe_n"]))
        return [{**self.documents[i], "score": score} for i, score in positive]


def measure(results, target):
    key = target["video_id"], target["keyframe_n"]
    rank, score = None, None
    for position, result in enumerate(results, 1):
        if (result["video_id"], result["keyframe_n"]) == key:
            rank, score = position, result["score"]
            break
    return {"rank": rank, "score": score, **{f"hit@{k}": int(rank is not None and rank <= k) for k in (1, 5, 10)},
            "mrr@10": 1 / rank if rank is not None and rank <= 10 else 0.0,
            "positive_results": len(results), "top10": results[:10]}


def evaluate(documents, queries):
    baseline, augmented = LexicalIndex(documents), LexicalIndex(documents, True)
    by_key = {(doc["video_id"], doc["keyframe_n"]): doc for doc in documents}
    rows = []
    for query in queries:
        target = by_key[(query["target"]["video_id"], query["target"]["keyframe_n"])]
        before = measure(baseline.search(query["text"]), query["target"])
        after = measure(augmented.search(query["text"]), query["target"])
        br, ar = before["rank"] or math.inf, after["rank"] or math.inf
        rows.append({"id": query["id"], "kind": query["kind"], "text": query["text"], "language": query["language"],
                     "target": target, "baseline": before, "caption_plus_ocr": after,
                     "rank_delta": before["rank"] - after["rank"] if math.isfinite(br) and math.isfinite(ar) else None,
                     "outcome": "win" if ar < br else "loss" if ar > br else "tie",
                     "mrr10_delta": after["mrr@10"] - before["mrr@10"]})
    metrics = {}
    for subset in ("all", "literal", "scene"):
        selected = [row for row in rows if subset == "all" or row["kind"] == subset]
        if not selected:
            continue
        metrics[subset] = {"queries": len(selected), "rank_outcomes": {
            outcome: sum(row["outcome"] == outcome for row in selected) for outcome in ("win", "tie", "loss")}}
        for arm in ("baseline", "caption_plus_ocr"):
            metrics[subset][arm] = {metric: sum(row[arm][metric] for row in selected) / len(selected)
                                    for metric in ("hit@1", "hit@5", "hit@10", "mrr@10")}
    return rows, metrics, {"baseline_avgdl": baseline.model.avgdl if baseline.model is not None else 0.0,
                           "caption_plus_ocr_avgdl": augmented.model.avgdl if augmented.model is not None else 0.0}


def load_queries(path):
    raw = checked_path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") not in (1, 2) or data.get("group") != "L04" or not data.get("authoring"):
        raise ValueError("Invalid frozen query packet")
    seen = set()
    if not isinstance(data.get("queries"), list) or not data["queries"]:
        raise ValueError("No frozen queries")
    for query in data["queries"]:
        if not isinstance(query.get("id"), str) or query["id"] in seen:
            raise ValueError("Duplicate/invalid query ID")
        seen.add(query["id"])
        kinds = {"literal", "scene", "mixed"} if data["schema_version"] == 2 else {"literal", "scene"}
        if query.get("kind") not in kinds or not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError("Invalid query text/kind")
        vid, n = video(query["target"]["video_id"]), integer(query["target"]["keyframe_n"])
        if not query["source_image"].startswith("dataset/"):
            raise ValueError("Noncanonical source image")
        image_path(query["source_image"].removeprefix("dataset/"), vid, n)
        if not re.fullmatch(r"[a-f0-9]{64}", query["image_sha256"]):
            raise ValueError("Invalid frozen image SHA")
        if not isinstance(query.get("language"), str) or not query["language"].strip():
            raise ValueError("Invalid query language")
        if data["schema_version"] == 2:
            validate_judgments(query)
    return raw, data


LIMITATIONS = [
    LABEL + "; not a multimodal/production benchmark or production readiness evidence.",
    "Eight image-authored queries are exploratory, nonrepresentative and too small for significance/generalization. Qrels mark only known targets; other retrieved images remain unjudged, not negatives.",
    "English scene controls versus mixed English/Vietnamese on-screen literals are not language-controlled. Captions may already describe text; OCR may benefit literal queries for lexical/language reasons.",
    "Public OCR lacks image SHA, engine/model/line confidence provenance: OCR-to-original-image byte equality is UNVERIFIED. Local image/caption SHA checks prove only that separate relationship.",
    "Missing/error/empty/technical-count-only OCR does not remove candidates or imply no text. 'missing' counts missing usable public OCR, including technical-count-only records. Unknown PUBLIC statuses, duplicates, malformed records fail closed, not negative evidence; recognized old technical statuses are counted separately, never indexed.",
    "Appending OCR changes BM25 global IDF and average document length; even unchanged candidate text may score differently. Length normalization/noise can regress scene retrieval.",
    "Ranks exclude zero/negative/nonfinite scores; null rank means no positive result. MRR@10 truncates ranks >10. Exact image rank changes are not general relevance wins/losses.",
    "No tuning, translation, target-ID injection, image embeddings, production indexes, Milvus, LLMs, network, or inference used. Only four explicit dataset L04 roots read; no L05/L06 discovery.",
    "Integrity is lexical-source identity/path/mapping/image-hash validation, not production index/manifest validation; existing retrieval_text is used without reconstruction.",
]


def markdown(report):
    lines = ["# L04 OCR exploratory A/B", "", LABEL, "", "## Provenance", "",
             f"Frozen query SHA-256: `{report['queries_sha256']}`",
             f"Config SHA-256: `{report['config_sha256']}`",
             f"Input manifest SHA-256: `{report['input_manifest_sha256']}`", "",
             "## Known-target metrics", "", "| Subset | Arm | N | Hit@1 | Hit@5 | Hit@10 | MRR@10 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for subset, result in report["metrics"].items():
        for arm in ("baseline", "caption_plus_ocr"):
            m = result[arm]
            lines.append(f"| {subset} | {arm} | {result['queries']} | {m['hit@1']:.4f} | {m['hit@5']:.4f} | {m['hit@10']:.4f} | {m['mrr@10']:.4f} |")
    lines += ["", "## Per-query positive-result ranks", "", "| Query | Kind | Target | OCR | Before rank (score) | After rank (score) | Rank outcome |",
              "|---|---|---|---|---:|---:|---|"]
    for row in report["per_query"]:
        def rank_score(arm):
            result = row[arm]
            return f"{result['rank']} ({result['score']:.6f})" if result["rank"] is not None else "null (null)"
        target = row["target"]
        lines.append(f"| {row['id']} | {row['kind']} | {target['video_id']}/{target['keyframe_n']:03d} | {target['ocr_status']} | {rank_score('baseline')} | {rank_score('caption_plus_ocr')} | {row['outcome']} |")
    lines += ["", "## Coverage", "", "```json", json.dumps(report["coverage"], indent=2), "```", "",
              "## Fixed configuration", "", "```json", json.dumps(report["config"], indent=2), "```", "",
              "## Frozen image authoring and queries", "", report["authoring"]["selection_rule"], "", report["authoring"]["language_policy"]]
    for row in report["per_query"]:
        lines += ["", f"### {row['id']} ({row['kind']})", "", row["text"], "", f"Target caption evidence: {row['target']['caption_text']}", "", f"Target OCR evidence (unverified image-byte provenance): {row['target']['ocr_text']}"]
    lines += ["", "Top-10 identities, scores, caption/OCR evidence for each arm are in report.json; source files and image hashes are in input_manifest.json.", "", "## Limitations", ""]
    lines.extend("- " + limitation for limitation in LIMITATIONS)
    # OCR evidence may itself contain embedded lines/trailing spaces. Preserve raw
    # evidence in JSON, but do not emit accidental Markdown hard-break whitespace.
    return "\n".join(line.rstrip(" \t") for line in "\n".join(lines).splitlines()) + "\n"


ARMS = ("caption", "caption_plus_ocr", "ocr", "rrf")
METRICS = ("hit@1", "hit@5", "hit@10", "mrr@10")
EXCLUDED_VIDEOS = {"L04_V001", "L04_V010", "L04_V020", "L04_V030"}
DEV_PATH = Path("benchmarks/l04_ocr_exploratory_20261111.json")
DEV_SHA = "f60b18f82b5b379414c503f08aae8cec7df325df59f7f0f00d7d8ccaa1ead8db"
FUSION_LIMITATIONS = [
    "New24 test and old8 dev-reference are separate, small, nonrepresentative image-authored sets; neither used for tuning. Not full multimodal/system accuracy or a production gate.",
    "New test qrels include only 120 visually inspected target/neighborhood images (30 positives); all uninspected frames are UNJUDGED, never negatives. Dev qrels preserve exactly one old target each. Known-judged-positive metrics are not complete relevance metrics.",
    "Queries and binary judgments were authored by this assistant from actual images, not independently verified by humans. Full-query relevance requires every stated scene detail and quoted string; readable partial tickers do not satisfy full quoted strings. Judgment ambiguity remains.",
    "Seven literal, nine scene, eight mixed after one same-image literal-to-scene fallback (t13); English and English+Vietnamese are confounded with kind. Tickers often describe a different story than the image; mixed queries are lexical conjunction requests, not inferred event truth.",
    *LIMITATIONS[3:],
    "RRF uses equal weights, k=60, one-based full positive finite component rankings with no topN cutoff. Raw BM25 and RRF scores are not calibrated or comparable across arms. Stable identity tie-breaking can affect repeated broadcast shots.",
]


def doc_key(record):
    return video(record.get("video_id")), integer(record.get("keyframe_n"))


def validate_judgments(query, by_key=None):
    """Validate every inspected label, including negatives; never infer n-1 as frame_idx."""
    target = doc_key(query["target"])
    judgments = query.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        raise ValueError("No inspected judgments")
    seen, positives = set(), set()
    for judgment in judgments:
        key = doc_key(judgment)
        if key in seen:
            raise ValueError("Duplicate qrel identity")
        seen.add(key)
        if (doc_key(judgment["target"]) != target or type(judgment.get("label")) is not int
                or judgment["label"] not in (0, 1) or not isinstance(judgment.get("rationale"), str)
                or not judgment["rationale"].strip()):
            raise ValueError("Invalid qrel label/target/rationale")
        source = judgment.get("source_image")
        if not isinstance(source, str) or not source.startswith("dataset/"):
            raise ValueError("Noncanonical judgment source")
        image_path(source.removeprefix("dataset/"), *key)
        if not re.fullmatch(r"[a-f0-9]{64}", judgment.get("image_sha256", "")):
            raise ValueError("Invalid judgment SHA")
        integer(judgment.get("frame_id"), 0)
        if by_key is not None:
            doc = by_key.get(key)
            if (doc is None or "dataset/" + doc["image_path"] != source
                    or doc["image_sha256"] != judgment["image_sha256"]
                    or doc["frame_id"] != judgment["frame_id"]):
                raise ValueError("Judgment image/hash/mapping mismatch")
        if judgment["label"]:
            positives.add(key)
    if not positives or target not in positives:
        raise ValueError("No positive target qrel")
    return positives


def selection_settings(protocol=None):
    if protocol is None or protocol.get("schema_version") == 1:
        if protocol is not None and (protocol["selection"]["test_videos"] != 12
                                     or protocol["selection"]["test_queries"] != 24):
            raise ValueError("Legacy protocol requires 12 videos/24 queries")
        return 12, EXCLUDED_VIDEOS
    if protocol.get("schema_version") != 2 or protocol.get("group") != "L04":
        raise ValueError("Unsupported selection protocol")
    selection = protocol["selection"]
    count = integer(selection["test_videos"], 2)
    if integer(selection["test_queries"]) != 2 * count:
        raise ValueError("Require two queries per selected video")
    excluded = [video(v) for v in selection["excluded_videos"]]
    if len(set(excluded)) != len(excluded):
        raise ValueError("Duplicate excluded video")
    return count, set(excluded)


def selection_indices(video_names, image_names, protocol=None):
    """Names-only predeclared selection; image_names maps video -> canonical image stems."""
    count, excluded = selection_settings(protocol)
    eligible = sorted(v for v in (video(v) for v in video_names) if v not in excluded)
    if len(set(eligible)) != len(eligible) or len(eligible) < count:
        raise ValueError("Insufficient/duplicate eligible videos")
    selected = []
    inspected = set()
    for i in range(count):
        vid = eligible[i * (len(eligible) - 1) // (count - 1)]
        names = sorted(integer(n) for n in image_names[vid])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate image identity")
        positions = ((len(names) - 1) // 3, 2 * (len(names) - 1) // 3)
        if positions[0] == positions[1] or any(not 2 <= p <= len(names) - 3 for p in positions):
            raise ValueError("Selection requires two interior positions")
        for pos in positions:
            neighbors = [(vid, n) for n in names[pos - 2:pos + 3]]
            if inspected.intersection(neighbors):
                raise ValueError("Overlapping inspected windows")
            inspected.update(neighbors)
            selected.append({"target": (vid, names[pos]), "image_count": len(names), "position": pos,
                             "neighbors": neighbors})
    return selected


def validate_selection(packet, plan, dev_queries, protocol=None, prior_queries=()):
    queries = packet["queries"]
    count, excluded = selection_settings(protocol)
    if len(queries) != 2 * count or len(plan) != 2 * count:
        raise ValueError("Require declared test query count")
    if {doc_key(q["target"])[0] for q in queries} & {doc_key(q["target"])[0] for q in dev_queries}:
        raise ValueError("Test/dev video overlap")
    prior_keys, prior_texts = set(), set()
    for query in prior_queries:
        prior_texts.add(query["text"])
        prior_keys.add(doc_key(query["target"]))
        prior_keys.update(doc_key(j) for j in query.get("judgments", []))
    if prior_queries and not {doc_key(q["target"])[0] for q in prior_queries}.issubset(excluded):
        raise ValueError("Prior target videos not excluded")
    if any(q.get("text") in prior_texts for q in queries):
        raise ValueError("Prior query literal reuse")
    seen = set()
    for index, (query, chosen) in enumerate(zip(queries, plan)):
        key = doc_key(query["target"])
        assigned = ("literal", "scene", "mixed")[index % 3]
        if (key in seen or key != chosen["target"] or key[0] in excluded
                or query.get("split") != "test" or query.get("id") != f"t{index + 1:02}"
                or query.get("image_count") != chosen["image_count"]
                or query.get("target_position_zero_based") != chosen["position"]
                or query.get("assigned_kind") != assigned):
            raise ValueError("Selection/split/assigned kind mismatch")
        seen.add(key)
        if query["kind"] != assigned and not (assigned in {"literal", "mixed"}
                                               and query["kind"] == "scene" and query.get("fallback")):
            raise ValueError("Invalid same-image fallback")
        judgments = query["judgments"]
        if prior_keys.intersection(doc_key(j) for j in judgments):
            raise ValueError("Prior inspected image reuse")
        if ([doc_key(j) for j in judgments] != chosen["neighbors"]
                or [j.get("image_list_offset") for j in judgments] != list(range(-2, 3))):
            raise ValueError("Incomplete/mismatched inspected neighborhood")


def reciprocal_rank_fusion(caption_results, ocr_results):
    """Text-only component outputs, no query targets or qrels; full rankings, k=60."""
    combined = {}
    for arm, results in (("caption", caption_results), ("ocr", ocr_results)):
        seen, rank = set(), 0
        for result in results:
            key = doc_key(result)
            if key in seen:
                raise ValueError("Duplicate RRF component identity")
            seen.add(key)
            score = result["score"]
            if not math.isfinite(score) or score <= 0:
                continue
            rank += 1
            if key not in combined:
                combined[key] = {**result, "score": 0.0, "component_ranks": {"caption": None, "ocr": None},
                                 "component_scores": {"caption": None, "ocr": None}}
            row = combined[key]
            row["score"] += 1.0 / (60 + rank)
            row["component_ranks"][arm] = rank
            row["component_scores"][arm] = score
    return sorted(combined.values(), key=lambda d: (-d["score"], d["video_id"], d["keyframe_n"]))


def measure_positives(results, positives):
    if not positives:
        raise ValueError("Empty positives")
    first = next((result for result in results if doc_key(result) in positives), None)
    absent_vid, absent_n = sorted(positives)[0]
    result = measure(results, first if first is not None else {"video_id": absent_vid, "keyframe_n": absent_n})
    result["first_positive_identity"] = ({"video_id": first["video_id"], "keyframe_n": first["keyframe_n"]}
                                         if first is not None else None)
    return result


def rank_outcome(before, after):
    br, ar = before or math.inf, after or math.inf
    return "win" if ar < br else "loss" if ar > br else "tie"


def aggregate_fusion(rows):
    metrics = {}
    for split in sorted({row["split"] for row in rows}):
        group = [row for row in rows if row["split"] == split]
        subsets = {"all": group}
        for field in ("kind", "language"):
            for value in sorted({row[field] for row in group}):
                subsets[f"{field}/{value}"] = [row for row in group if row[field] == value]
        for name, subset in subsets.items():
            result = {"queries": len(subset)}
            for arm in ARMS:
                result[arm] = {basis: {metric: sum(row[arm][basis][metric] for row in subset) / len(subset)
                                       for metric in METRICS} for basis in ("known_positive", "target")}
            for comparison in ("rrf_vs_caption", "caption_plus_ocr_vs_caption"):
                result[comparison] = {
                    basis: {outcome: sum(row[comparison][basis] == outcome for row in subset)
                            for outcome in ("win", "tie", "loss")} for basis in ("known_positive", "target")}
            metrics[f"{split}/{name}"] = result
    return metrics


def original_unchanged_metrics(rows, query_ids):
    """Explicit unaffected subset, never pooled with corrected18 or dev-reference."""
    if len(query_ids) != 16 or len(set(query_ids)) != 16:
        raise ValueError("Require 16 distinct original unchanged query IDs")
    selected = [row for row in rows if row["split"] == "test" and row["id"] in query_ids]
    if len(selected) != 16 or {row["id"] for row in selected} != set(query_ids):
        raise ValueError("Missing/duplicate original unchanged subset rows")
    return {name.replace("test/", "originalunchanged16/", 1): value
            for name, value in aggregate_fusion(selected).items()}


def review_query_counts(queries):
    return {"queries": len(queries), "positive": sum(j["label"] for q in queries for j in q["judgments"]),
            "negative": sum(1 - j["label"] for q in queries for j in q["judgments"]),
            "kind": dict(Counter(q["kind"] for q in queries)),
            "language": dict(Counter(q["language"] for q in queries))}


def validate_review_correction(raw, packet, protocol):
    """Narrow ROUND1 correction contract: only t06/t10, all target evidence fixed."""
    correction = protocol["correction"]
    if (correction["label"] != "review-corrected18" or packet["authoring"].get("correction") != correction
            or "NOT a fresh held-out" not in correction["disclaimer"]
            or "after v2 evaluation/results were known" not in correction["disclaimer"]):
        raise ValueError("Require transparent post-result review disclaimer")
    expected_paths = {
        "original_packet": "benchmarks/l04_ocr_fusion_new_queries_20261111_v2.json",
        "original_protocol": "benchmarks/l04_ocr_fusion_new_queries_20261111_v2.protocol.json",
        "original_report": "benchmark_results/l04_ocr_fusion_new_queries_20261111_v2_run01/report.json",
    }
    originals = {}
    for field, path in expected_paths.items():
        reference = correction[field]
        original = checked_path(path).read_bytes()
        if reference["path"] != path or byte_sha(original) != reference["sha256"]:
            raise ValueError("Correction original reference/hash mismatch")
        originals[field] = original
    old = json.loads(originals["original_packet"])
    old_protocol = json.loads(originals["original_protocol"])
    if (protocol["selection"] != old_protocol["selection"] or protocol["scoring"] != old_protocol["scoring"]
            or packet["scoring_config_sha256"] != old["scoring_config_sha256"]
            or correction["scoring_config_sha256"] != old["scoring_config_sha256"]):
        raise ValueError("Correction changed selection/scoring configuration")
    before, after = old["queries"], packet["queries"]
    if len(after) != 18 or [q["id"] for q in after] != [q["id"] for q in before]:
        raise ValueError("Correction must keep same ordered 18 query IDs")
    allowed = {"t06", "t10"}
    if (correction["allowed_changed_query_ids"] != ["t06", "t10"]
            or {a["id"] for a, b in zip(after, before) if a != b} != allowed):
        raise ValueError("Only t06/t10 may change")
    # Check actual serialized blocks, not merely normalized JSON equality, for the other16.
    def blocks(content):
        return {json.loads(block)["id"]: block for block in re.findall(rb"(?ms)^    \{\r?\n.*?^    \}", content)}
    old_blocks, new_blocks = blocks(originals["original_packet"]), blocks(raw)
    if len(old_blocks) != 18 or len(new_blocks) != 18:
        raise ValueError("Require 18 auditable serialized query blocks")
    for previous, query in zip(before, after):
        qid = query["id"]
        if qid not in allowed:
            if old_blocks[qid] != new_blocks[qid]:
                raise ValueError("Original unchanged query bytes differ")
            continue
        mutable = {"text", "kind", "language", "fallback", "judgments"}
        if {k: v for k, v in query.items() if k not in mutable} != {
                k: v for k, v in previous.items() if k not in mutable}:
            raise ValueError("Correction changed target identity/evidence")
        if len(query["judgments"]) != 5:
            raise ValueError("Correction requires all five judgments")
        for old_j, new_j in zip(previous["judgments"], query["judgments"]):
            mutable_j = {"label", "rationale", "inspection", "visible_clock"}
            if {k: v for k, v in new_j.items() if k not in mutable_j} != {
                    k: v for k, v in old_j.items() if k not in mutable_j}:
                raise ValueError("Correction changed inspected identity/evidence")
        validate_judgments(query)
    expected_ids = [q["id"] for q in before if q["id"] not in allowed]
    if (correction["subset"]["name"] != "originalunchanged16"
            or correction["subset"]["query_ids"] != expected_ids):
        raise ValueError("Invalid original unchanged16 subset IDs")
    if (correction["before_counts"] != review_query_counts(before)
            or correction["after_counts"] != review_query_counts(after)):
        raise ValueError("Correction before/after count mismatch")
    if {finding["id"] for finding in correction["findings"]} != allowed:
        raise ValueError("Missing correction finding")
    for finding in correction["findings"]:
        qid = finding["id"]
        old_q = next(q for q in before if q["id"] == qid)
        new_q = next(q for q in after if q["id"] == qid)
        if (finding["old_text"] != old_q["text"] or finding["new_text"] != new_q["text"]
                or finding["revised_judgments"] != new_q["judgments"]
                or finding["target_unchanged"] != new_q["target"]):
            raise ValueError("Correction finding/query mismatch")
    prior_report = json.loads(originals["original_report"])
    if prior_report["input_manifest_sha256"] != correction["expected_input_manifest_sha256"]:
        raise ValueError("Correction original input hash mismatch")
    return {**correction, "new_packet_sha256": byte_sha(raw), "unchanged_query_bytes_verified": True}


def evaluate_fusion(documents, test_queries, dev_queries):
    indexes = {"caption": LexicalIndex(documents), "caption_plus_ocr": LexicalIndex(documents, True),
               "ocr": LexicalIndex(documents, ocr_only=True)}
    by_key = {doc_key(doc): doc for doc in documents}
    rows = []
    for split, queries in (("test", test_queries), ("dev-reference", dev_queries)):
        for query in queries:
            positives = (validate_judgments(query, by_key) if split == "test" else {doc_key(query["target"])})
            ranked = {arm: index.search(query["text"]) for arm, index in indexes.items()}
            ranked["rrf"] = reciprocal_rank_fusion(ranked["caption"], ranked["ocr"])
            row = {"id": query["id"], "split": split, "kind": query["kind"], "language": query["language"],
                   "text": query["text"], "target": by_key[doc_key(query["target"])],
                   "known_positives": [list(key) for key in sorted(positives)]}
            for arm, results in ranked.items():
                known = measure_positives(results, positives)
                exact = measure(results, query["target"])
                # Keep evidence once per arm, shared by both measurement bases.
                row[arm] = {"top10": known.pop("top10"), "known_positive": known, "target": exact}
                exact.pop("top10")
                judged = {doc_key(j): j["label"] for j in query.get("judgments", [])}
                if split == "dev-reference":
                    judged = {doc_key(query["target"]): 1}
                for hit in row[arm]["top10"]:
                    hit["judged_label"] = judged.get(doc_key(hit), "UNJUDGED")
            for arm in ("rrf", "caption_plus_ocr"):
                row[arm + "_vs_caption"] = {
                    basis: rank_outcome(row["caption"][basis]["rank"], row[arm][basis]["rank"])
                    for basis in ("known_positive", "target")}
            rows.append(row)
    stats = {arm + "_avgdl": index.model.avgdl if index.model is not None else 0.0 for arm, index in indexes.items()}
    return rows, aggregate_fusion(rows), stats


def fusion_markdown(report):
    lines = ["# L04 frozen caption / OCR / RRF comparison", "", "## Freeze and provenance", ""]
    if report.get("correction_audit"):
        audit = report["correction_audit"]
        lines += [audit["disclaimer"], "", "## ROUND1 corrections (same targets, post-result review)", "",
                  "Original v2 outputs are preserved, not overwritten. Originalunchanged16 is explicitly separate from review-corrected18 and old8 dev-reference.", "",
                  "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", ""]
    for key in ("queries_sha256", "protocol_sha256", "config_sha256", "input_manifest_sha256", "source_manifest_sha256"):
        lines.append(f"- {key}: `{report[key]}`")
    lines += ["", "## Known-judged-positive metrics (first positive, incomplete qrels)", "",
              "| Split/subset | Arm | N | Hit@1 | Hit@5 | Hit@10 | MRR@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in report["metrics"].items():
        for arm in ARMS:
            m = result[arm]["known_positive"]
            lines.append(f"| {name} | {arm} | {result['queries']} | {m['hit@1']:.4f} | {m['hit@5']:.4f} | {m['hit@10']:.4f} | {m['mrr@10']:.4f} |")
    lines += ["", "Exact-target aggregate metrics are recorded separately for every subset in report.json.", "",
              "## Concatenation and RRF versus caption (win/tie/loss by rank)", "",
              "| Split/subset | Comparison | First positive W/T/L | Exact target W/T/L |", "|---|---|---|---|"]
    for name, result in report["metrics"].items():
        for comparison in ("caption_plus_ocr_vs_caption", "rrf_vs_caption"):
            def wtl(basis):
                return "/".join(str(result[comparison][basis][key]) for key in ("win", "tie", "loss"))
            lines.append(f"| {name} | {comparison} | {wtl('known_positive')} | {wtl('target')} |")
    lines += ["", "## Per-query ranks: first positive / exact chosen target", "",
              "| Split | ID | Kind | Query | Target | Caption | Caption+OCR | OCR | RRF |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in report["per_query"]:
        ranks = [f"{row[a]['known_positive']['rank']} / {row[a]['target']['rank']}" for a in ARMS]
        target = row["target"]
        text = row["text"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['split']} | {row['id']} | {row['kind']} | {text} | {target['video_id']}/{target['keyframe_n']:03d} | " + " | ".join(ranks) + " |")
    lines += ["", "## Observed changes (not a production recommendation)", ""] + report["observations"]
    for title, value in (("Candidate coverage", report["coverage"]), ("Qrels coverage", report["qrels_coverage"]),
                         ("Frozen scoring settings", report["config"]), ("Authoring sequence", report["authoring"])):
        lines += ["", "## " + title, "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## Frozen queries and inspected evidence", ""]
    for row in report["per_query"]:
        lines += [f"### {row['split']} / {row['id']} ({row['kind']}; {row['language']})", "", row["text"], "",
                  f"Target: {row['target']['image_path']}", ""]
    lines += ["Every top10 identity, raw score, caption/OCR evidence, judgment status, and RRF component rank/score is in report.json. All new judgments and rationale/source/image SHA are in queries.frozen.json; old targets are preserved in dev_queries.frozen.json. Uninspected results are UNJUDGED. No qrels entered scoring.",
              "", "## Limitations", ""] + ["- " + text for text in report["limitations"]]
    return "\n".join(line.rstrip(" \t") for line in "\n".join(lines).splitlines()) + "\n"


def run_fusion(dataset_root, queries_path, output, raw, packet):
    # Read and authenticate ALL frozen experiment specifications before any corpus payload.
    protocol_path = checked_path(packet["protocol_path"])
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    config = protocol["scoring"]
    if (byte_sha(protocol_raw) != packet["protocol_sha256"]
            or byte_sha(json_bytes(config)) != packet["scoring_config_sha256"]):
        raise ValueError("Frozen protocol/config hash mismatch")
    if (any(config[key] != CONFIG[key] for key in ("k1", "b", "epsilon"))
            or config["tokenizer"] != "lowercase, NFC, regex [^\\W_]+; no translation/stopwords"
            or config["arms"] != list(ARMS) or config["rrf"]["k"] != 60
            or config["rrf"]["caption_weight"] != 1.0 or config["rrf"]["ocr_weight"] != 1.0):
        raise ValueError("Unsupported frozen scoring settings")
    if packet["dev_reference"] != {"path": DEV_PATH.as_posix(), "sha256": DEV_SHA}:
        raise ValueError("Invalid frozen dev reference")
    dev_raw, dev_packet = load_queries(DEV_PATH)
    if byte_sha(dev_raw) != DEV_SHA:
        raise ValueError("Legacy dev packet hash mismatch")
    selection_settings(protocol)
    prior_queries, prior_raws = [], []
    preservation = protocol.get("preservation_manifest", [])
    for entry in preservation:
        if digest(checked_path(entry["path"])) != entry["sha256"]:
            raise ValueError("Prior artifact preservation hash mismatch")
    if protocol["schema_version"] == 2:
        references = protocol["selection"]["prior_packets"]
        if {r["path"] for r in references} != {DEV_PATH.as_posix(), "benchmarks/l04_ocr_fusion_20261111_v1.json"}:
            raise ValueError("Require both prior query packets")
        for reference in references:
            prior_raw, prior_packet = load_queries(reference["path"])
            if byte_sha(prior_raw) != reference["sha256"]:
                raise ValueError("Prior query packet hash mismatch")
            prior_raws.append((reference["path"], prior_raw))
            prior_queries.extend(prior_packet["queries"])
        old_protocol = json.loads(checked_path("benchmarks/l04_ocr_fusion_20261111_v1.protocol.json").read_bytes())
        if config != old_protocol["scoring"]:
            raise ValueError("Scoring must match prior frozen four-arm configuration exactly")
    correction_audit = validate_review_correction(raw, packet, protocol) if protocol.get("correction") else None
    root = checked_path(Path(dataset_root) / "data/L04")
    image_names = {}
    for directory in sorted(root.iterdir()):
        checked_path(directory)
        vid = video(directory.name)
        names = []
        for path in sorted(directory.iterdir()):
            checked_path(path)
            if not re.fullmatch(r"[0-9]+\.jpg", path.name) or not path.is_file():
                raise ValueError("Noncanonical L04 image inventory")
            names.append(int(path.stem))
        image_names[vid] = names
    plan = selection_indices(image_names, image_names, protocol)
    validate_selection(packet, plan, dev_packet["queries"], protocol, prior_queries)
    if protocol["schema_version"] == 2:
        declared = protocol["selection"]["frozen_plan"]
        for query, chosen, frozen in zip(packet["queries"], plan, declared):
            if (doc_key(frozen) != chosen["target"] or frozen["position"] != chosen["position"]
                    or frozen["image_count"] != chosen["image_count"]
                    or frozen["assigned_kind"] != query["assigned_kind"]
                    or frozen["neighbors"] != [j["source_image"] for j in query["judgments"]]):
                raise ValueError("Frozen names-only selection mismatch")
        if len(declared) != len(plan):
            raise ValueError("Frozen selection count mismatch")
    documents, coverage, manifest = load_corpus(dataset_root)
    by_key = {doc_key(doc): doc for doc in documents}
    universe = {(vid, n) for vid, names in image_names.items() for n in names}
    if len(documents) != 7471 or coverage["videos"] != 30 or universe != set(by_key):
        raise ValueError("Expected unchanged 7471-image/30-video universe")
    for query in packet["queries"] + dev_packet["queries"]:
        target = by_key.get(doc_key(query["target"]))
        if (target is None or target["image_sha256"] != query["image_sha256"]
                or "dataset/" + target["image_path"] != query["source_image"]):
            raise ValueError("Frozen query image evidence mismatch")
        if "judgments" in query:
            integer(query["frame_id"], 0)
            validate_judgments(query, by_key)
            mapping_path = checked_path(Path(dataset_root) / "mapping/L04" / (target["video_id"] + ".csv"))
            if query["frame_id"] != target["frame_id"] or query["mapping_sha256"] != digest(mapping_path):
                raise ValueError("Frozen query mapping mismatch")
    rows, metrics, stats = evaluate_fusion(documents, packet["queries"], dev_packet["queries"])
    if correction_audit:
        if byte_sha(json_bytes(manifest)) != correction_audit["expected_input_manifest_sha256"]:
            raise ValueError("Correction input universe/bytes changed")
        ids = correction_audit["subset"]["query_ids"]
        subset_metrics = original_unchanged_metrics(rows, ids)
        prior_report = json.loads(checked_path(correction_audit["original_report"]["path"]).read_bytes())
        old_rows = {row["id"]: row for row in prior_report["per_query"] if row["split"] == "test"}
        for row in rows:
            if row["split"] == "test" and row["id"] in ids and row != old_rows[row["id"]]:
                raise ValueError("Original unchanged16 result regression")
        if subset_metrics != original_unchanged_metrics(list(old_rows.values()), ids):
            raise ValueError("Original unchanged16 metric regression")
        metrics.update(subset_metrics)
        correction_audit.update(new_protocol_sha256=byte_sha(protocol_raw),
                                originalunchanged16_results_unchanged=True,
                                originalunchanged16_metrics_unchanged=True)
    # Also execute the unmodified legacy evaluator as a numeric regression check.
    _, legacy_metrics, _ = evaluate(documents, dev_packet["queries"])
    previous_path = checked_path("benchmark_results/l04_ocr_ab_20261111_run02/report.json")
    previous_raw = previous_path.read_bytes()
    previous = json.loads(previous_raw)
    if previous["queries_sha256"] != DEV_SHA or previous["metrics"] != legacy_metrics:
        raise ValueError("Previous frozen two-arm metric regression")
    for subset in ("all", "literal", "scene"):
        name = "dev-reference/" + ("all" if subset == "all" else "kind/" + subset)
        for old, new in (("baseline", "caption"), ("caption_plus_ocr", "caption_plus_ocr")):
            if legacy_metrics[subset][old] != metrics[name][new]["target"]:
                raise ValueError("Legacy two-arm metric regression")
    prior_four_arm_unchanged = None
    if protocol["schema_version"] == 2:
        prior_four_arm = json.loads(checked_path("benchmark_results/l04_ocr_fusion_20261111_run01/report.json").read_bytes())
        for name, result in metrics.items():
            if name.startswith("dev-reference/"):
                for arm in ARMS:
                    if result[arm] != prior_four_arm["metrics"][name][arm]:
                        raise ValueError("Previous four-arm dev-reference metric regression")
        prior_four_arm_unchanged = True
    import numpy
    import rank_bm25

    sources = [{"path": Path(path).as_posix(), "sha256": digest(Path(path))} for path in
               (Path(__file__), Path("tests/test_l04_ocr_ablation.py"), rank_bm25.__file__, numpy.__file__,
                numpy.core._multiarray_umath.__file__)]
    preserved = [{"path": p.as_posix(), "sha256": digest(p)} for p in
                 (DEV_PATH, *sorted(previous_path.parent.iterdir())) if p.is_file()]
    dependencies = {name: version(name) for name in ("rank-bm25", "numpy", "jsonschema")}
    input_raw, source_raw = json_bytes(manifest), json_bytes({"files": sources, "dependencies": dependencies,
                                                            "preserved_legacy_artifacts": preserved,
                                                            "preserved_prior_experiment_artifacts": preservation,
                                                            "python": sys.version, "executable": sys.executable})
    observations = []
    for split in ("test", "dev-reference"):
        selected = [row for row in rows if row["split"] == split]
        deltas = [(row["rrf"]["known_positive"]["mrr@10"] - row["caption"]["known_positive"]["mrr@10"], row["id"])
                  for row in selected]
        gains = sorted(((d, q) for d, q in deltas if d > 0), reverse=True)
        losses = sorted((d, q) for d, q in deltas if d < 0)
        observations.append(f"- {split}: RRF first-positive MRR@10 gains: {gains}; regressions: {losses}. These few queries drive the aggregate, not broad system accuracy.")
        scenes = [(row["id"], row["rrf_vs_caption"]["known_positive"], row["caption"]["known_positive"]["rank"],
                   row["rrf"]["known_positive"]["rank"]) for row in selected if row["kind"] == "scene"]
        observations.append(f"- {split} scene rank changes (ID, outcome, caption, RRF): {scenes}. OCR absence/noise, different language, repetitive anchor scenes, and incomplete qrels limit interpretation.")
        for arm in ("caption_plus_ocr", "rrf"):
            for kind in ("literal", "scene"):
                changes = [(r["id"], r[arm + "_vs_caption"]["known_positive"],
                            r["caption"]["known_positive"]["rank"], r[arm]["known_positive"]["rank"])
                           for r in selected if r["kind"] == kind]
                observations.append(f"- {split} {kind}, {arm} vs caption (ID, outcome, first-positive ranks): {changes}.")
    test_queries = packet["queries"]
    inspected = sum(len(q["judgments"]) for q in test_queries)
    positives = sum(j["label"] for q in test_queries for j in q["judgments"])
    kinds = dict(Counter(q["kind"] for q in test_queries))
    languages = dict(Counter(q["language"] for q in test_queries))
    limitations = [
        f"Fresh {len(test_queries)} test queries and old {len(dev_packet['queries'])} dev-reference queries are separate small nonrepresentative sets, never pooled or used for tuning. Not full multimodal/system accuracy or a production gate.",
        f"Test qrels cover only {inspected} directly inspected images ({positives} positives); all other images are UNJUDGED, not negatives. First-judged-positive metrics are not complete relevance metrics.",
        FUSION_LIMITATIONS[2],
        f"Actual kind counts: {kinds}; language counts: {languages}; same-image fallbacks: {[q['id'] for q in test_queries if q.get('fallback')]}. Language and query kind are confounded. Ticker/scene conjunctions do not imply shared event truth.",
        *FUSION_LIMITATIONS[4:],
    ]
    if correction_audit:
        limitations[0] = correction_audit["disclaimer"] + " Old8 dev-reference is separate, never pooled. Not full multimodal/system accuracy or a production gate."
    report = {"label": "L04 frozen image-grounded lexical four-arm comparison; incomplete qrels",
              "queries_sha256": byte_sha(raw), "protocol_sha256": byte_sha(protocol_raw),
              "config_sha256": byte_sha(json_bytes(config)), "dev_queries_sha256": byte_sha(dev_raw),
              "input_manifest_sha256": byte_sha(input_raw), "source_manifest_sha256": byte_sha(source_raw),
              "implementation_sha256": digest(Path(__file__)), "authoring": packet["authoring"], "config": config,
              "coverage": coverage, "corpus_statistics": stats, "metrics": metrics, "per_query": rows,
              "legacy_two_arm_metrics": legacy_metrics, "observations": observations, "limitations": limitations,
              "previous_report_sha256": byte_sha(previous_raw), "previous_two_arm_metrics_unchanged": True,
              "previous_four_arm_dev_metrics_unchanged": prior_four_arm_unchanged,
              "qrels_coverage": {"test_queries": len(test_queries), "dev_reference_queries": len(dev_packet["queries"]), "test_inspected": inspected,
                                 "test_positive": positives, "test_negative": inspected - positives,
                                 "kind_counts": kinds, "language_counts": languages,
                                 "per_test_query_unjudged": len(documents) - 5, "dev_positives_per_query": 1,
                                 "dev_unjudged_per_query": len(documents) - 1},
              "invocation": {"argv": sys.argv, "dataset_root": str(dataset_root), "queries": str(queries_path), "output": str(output)}}
    if correction_audit:
        report["label"] = "L04 review-corrected18 after ROUND1/results; originalunchanged16 separate; incomplete qrels"
        report["correction_audit"] = correction_audit
    if (checked_path(queries_path).read_bytes() != raw or protocol_path.read_bytes() != protocol_raw
            or checked_path(DEV_PATH).read_bytes() != dev_raw
            or any(checked_path(path).read_bytes() != content for path, content in prior_raws)):
        raise ValueError("Frozen specifications changed during run")
    for entry in preservation:
        if digest(checked_path(entry["path"])) != entry["sha256"]:
            raise ValueError("Prior artifact changed during run")
    report["prior_artifacts_bytes_unchanged"] = bool(preservation)
    output.mkdir(parents=True, exist_ok=False)
    for name, contents in (("queries.frozen.json", raw), ("protocol.frozen.json", protocol_raw),
                           ("dev_queries.frozen.json", dev_raw), ("config.json", json_bytes(config)),
                           ("input_manifest.json", input_raw), ("source_manifest.json", source_raw),
                           ("report.json", json_bytes(report)), ("report.md", fusion_markdown(report).encode("utf-8"))):
        (output / name).write_bytes(contents)
    artifact_manifest = [{"path": p.name, "sha256": digest(p)} for p in sorted(output.iterdir())]
    (output / "artifacts_manifest.json").write_bytes(json_bytes(artifact_manifest))
    return report


def run(dataset_root, queries_path, output):
    output = checked_path(output)
    if output.exists():
        raise FileExistsError("Output already exists; use a new unique run directory")
    raw, packet = load_queries(queries_path)  # Freeze/read before any caption/OCR payload.
    if packet["schema_version"] == 2:
        return run_fusion(dataset_root, queries_path, output, raw, packet)
    query_sha = byte_sha(raw)
    documents, coverage, manifest = load_corpus(dataset_root)
    by_key = {(doc["video_id"], doc["keyframe_n"]): doc for doc in documents}
    for query in packet["queries"]:
        target = by_key[(query["target"]["video_id"], query["target"]["keyframe_n"])]
        if (target["image_sha256"] != query["image_sha256"]
                or "dataset/" + target["image_path"] != query["source_image"]):
            raise ValueError("Frozen query image evidence mismatch")
    rows, metrics, stats = evaluate(documents, packet["queries"])
    config = dict(CONFIG, rank_bm25_version=version("rank-bm25"))
    input_raw = json_bytes(manifest)
    report = {"label": LABEL, "queries_sha256": query_sha, "config": config,
              "config_sha256": byte_sha(json_bytes(config)), "input_manifest_sha256": byte_sha(input_raw),
              "implementation_sha256": digest(Path(__file__)), "authoring": packet["authoring"],
              "coverage": coverage, "corpus_statistics": stats, "metrics": metrics, "per_query": rows,
              "limitations": LIMITATIONS}
    if checked_path(queries_path).read_bytes() != raw:
        raise ValueError("Frozen query file changed during run")
    output.mkdir(parents=True, exist_ok=False)
    for name, contents in (("queries.frozen.json", raw), ("config.json", json_bytes(config)),
                           ("input_manifest.json", input_raw), ("report.json", json_bytes(report)),
                           ("report.md", markdown(report).encode("utf-8"))):
        (output / name).write_bytes(contents)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=("L04",))
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    report = run(args.dataset_root, args.queries, args.output)
    print(json.dumps({"label": report["label"], "output": str(args.output), "queries_sha256": report["queries_sha256"],
                      "coverage": report["coverage"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
