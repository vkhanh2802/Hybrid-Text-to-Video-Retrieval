"""Prepare a blind, human-authored keyframe holdout; validate/freeze only.

Never reads caption corpus or retrieval scores. No models or evaluation here.
Prior benchmark JSON is scanned only for video exposure IDs and file hashes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from kis.artifact_io import atomic_write_json
from kis.paths import KIS_ROOT


FROZEN_SOURCES = ("kis/condition_evidence.py", "kis/temporal.py", "kis/event_retrieval.py",
                  "kis/candidate_pool.py", "kis/query_plan.py", "config/pipeline.json", "config/models.json")
VIDEO_ID = re.compile(r"\bL\d{2}_V\d{3}\b")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_exposure(roots):
    videos, files = set(), {}
    def add(value):
        if isinstance(value, str) and VIDEO_ID.fullmatch(value):
            videos.add(value)
    def walk(item):
        if isinstance(item, dict):
            for key, value in item.items():
                # Search streams rank the existing corpus and are not labels.
                # Preflight/registries likewise do not expose visual judgments.
                # Exclude targets, downloaded samples and explicitly selected/
                # locally scored pools, not every ID in a global search log.
                if key in {"samples", "queries", "judgments", "excluded_samples"} and isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            add(row.get("video_id"))
                if key == "query" and isinstance(value, dict):
                    add(value.get("video_id"))
                if key in {"candidate_videos", "selected_videos", "scored_videos"} and isinstance(value, list):
                    for video_id in value:
                        add(video_id)
                if key in {"video_stats", "condition_stats", "caption_sha256_by_video"} and isinstance(value, dict):
                    for video_id in value:
                        add(video_id)
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)
    for root in roots:
        for path in sorted(Path(root).rglob("*.json")):
            # Fail closed if a previous record is unreadable, rather than silently
            # claiming a video is unseen. Do not echo potentially sensitive text.
            try:
                walk(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Unreadable prior benchmark JSON: {path.name}") from exc
            files[str(path.relative_to(KIS_ROOT)) if path.is_relative_to(KIS_ROOT) else path.name] = sha(path)
    return sorted(videos), files


def verify_protocol(packet):
    packet = Path(packet)
    protocol = json.loads((packet / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != "kis_human_holdout_protocol_v1":
        raise ValueError("Unknown holdout protocol")
    for name, fingerprint in protocol["source_sha256"].items():
        if sha(KIS_ROOT / name) != fingerprint:
            raise ValueError(f"Frozen implementation changed: {name}. Do not tune on holdout.")
    if sha(packet / "exclusions.json") != protocol["exclusions_sha256"]:
        raise ValueError("Exposure inventory changed")
    return protocol


def build_packet(packet):
    packet = Path(packet)
    verify_protocol(packet)
    manifest_path = packet / "images" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(sample["split"] != "holdout" for sample in manifest["samples"]):
        raise ValueError("Human packet must be holdout only")
    excluded = set(json.loads((packet / "exclusions.json").read_text(encoding="utf-8"))["excluded_video_ids"])
    if any(sample["video_id"] in excluded for sample in manifest["samples"]):
        raise ValueError("Previously exposed video entered holdout")
    if (packet / "review.html").exists():
        raise FileExistsError("Review packet already exists; do not overwrite human work")
    public_samples = []
    for sample in manifest["samples"]:
        sheet = packet / "images" / sample["contact_sheet"]
        public_samples.append({"sample_id": sample["sample_id"], "video_id": sample["video_id"],
            "contact_sheet": "images/" + sample["contact_sheet"], "contact_sheet_sha256": sha(sheet),
            "frames": sample["frames"]})
    data = {"schema": "kis_human_holdout_annotations_v1", "manifest_sha256": sha(manifest_path),
            "protocol_sha256": sha(packet / "protocol.json"), "reviewer": "", "image_only_confirmed": False,
            "samples": [{**sample, "status": "pending", "skip_reason": "", "notes": "", "events": []}
                        for sample in public_samples]}
    atomic_write_json(packet / "annotation_template.json", data)
    # Escape HTML delimiters inside embedded JSON even if a future annotation
    # contains hostile text. User text is rendered with textContent/value only.
    embedded = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = (KIS_ROOT / "tools" / "templates" / "human_holdout.html").read_text(encoding="utf-8")
    (packet / "review.html").write_text(template.replace("__PACKET_JSON__", embedded), encoding="utf-8")
    atomic_write_json(packet / "packet_integrity.json", {"manifest_sha256": sha(manifest_path),
        "protocol_sha256": sha(packet / "protocol.json"), "template_sha256": sha(packet / "annotation_template.json"),
        "contact_sheets": {sample["contact_sheet"]: sample["contact_sheet_sha256"] for sample in public_samples}})
    print(json.dumps({"review_page": str((packet / "review.html").resolve()), "videos": len(public_samples),
                      "images_available": sum(f["status"] == "downloaded" for s in public_samples for f in s["frames"]),
                      "evaluation_run": False}, indent=2))


def prepare(packet, count=12, seed=2026110203):
    packet = Path(packet)
    if packet.exists():
        raise FileExistsError("Choose a new packet directory")
    videos, inventory = prior_exposure((KIS_ROOT / "benchmarks", KIS_ROOT / "benchmark_results"))
    packet.mkdir(parents=True)
    atomic_write_json(packet / "exclusions.json", {"excluded_video_ids": videos, "source_sha256": inventory,
        "scope": "Prior labeled/sample targets and selected/local-scoring pools; not corpus-wide retrieval seed hits or artifact preflight IDs"})
    sources = {name: sha(KIS_ROOT / name) for name in FROZEN_SOURCES}
    atomic_write_json(packet / "implementation_snapshot.json", {
        name: (KIS_ROOT / name).read_text(encoding="utf-8") for name in FROZEN_SOURCES})
    protocol = {"schema": "kis_human_holdout_protocol_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "await_human_labels_before_any_scoring", "seed": seed, "requested_videos": count,
        "split": "holdout", "sampling": "uniform unseen videos, uniform anchor, +/-2 existing unambiguous keyframes",
        "excluded_videos": len(videos), "exclusions_sha256": sha(packet / "exclusions.json"), "source_sha256": sources,
        "compared_modes": ["off", "semantic_guard"], "thresholds": {"visual": 0.25, "semantic": 0.45},
        "annotation": "Human authors events and frame judgments from images only; English text must be approved before freeze",
        "unknown_policy": "Explicit unknown, not negative; unreviewed blocks freeze",
        "query_policy": "One image event/window; optional 2-3 ordered events only when separate shots justify it. No forced clothing or sequence labels.",
        "primary_metrics": ["macro_tpr", "macro_fpr", "macro_balanced_accuracy", "eligible_query_count"],
        "sequence_metrics": ["known_frame_hit_at_20", "known_path_hit_at_20", "target_video_top1"],
        "adoption_gate": "No automatic promotion. Require both classes on >=8 event queries, >=4 annotated sequences, lower macro FPR without lower TPR, and no known-path/frame-hit regression. Even passing is only a small-sample descriptive result requiring review.",
        "limitations": "New target/annotation holdout, not a never-retrieved corpus: global seed-hit/preflight IDs are not exclusions. Sparse qrels incomplete; video IDs do not guarantee content disjointness; unavailable images not replaced; random sampling may yield no eligible garment cases.",
        "holdout_retuning_forbidden": True, "models_required_for_preparation": False}
    atomic_write_json(packet / "protocol.json", protocol)
    from tools.drive_groundtruth import run
    run(packet / "images", count, seed, [packet / "exclusions.json"], split_mode="holdout")
    build_packet(packet)


def validate_annotations(packet, data, require_english=False):
    packet = Path(packet)
    verify_protocol(packet)
    integrity = json.loads((packet / "packet_integrity.json").read_text(encoding="utf-8"))
    manifest_path = packet / "images" / "manifest.json"
    if (sha(manifest_path) != integrity["manifest_sha256"] or
            sha(packet / "protocol.json") != integrity["protocol_sha256"]):
        raise ValueError("Frozen packet metadata changed")
    for name, fingerprint in integrity["contact_sheets"].items():
        if sha(packet / name) != fingerprint:
            raise ValueError("Contact sheet changed; image observations no longer match packet")
    if (data.get("schema") != "kis_human_holdout_annotations_v1" or
            data.get("manifest_sha256") != integrity["manifest_sha256"] or
            data.get("protocol_sha256") != integrity["protocol_sha256"]):
        raise ValueError("Annotations belong to another packet")
    if not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip() or data.get("image_only_confirmed") is not True:
        raise ValueError("Reviewer name/alias and image-only human confirmation required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {s["sample_id"]: s for s in manifest["samples"]}
    available = {(s["sample_id"], f["keyframe_n"]): f for s in manifest["samples"] for f in s["frames"]
                 if f["status"] == "downloaded"}
    seen, queries, skipped, translations = set(), [], [], []
    for sample in data.get("samples", []):
        sample_id = sample.get("sample_id")
        if sample_id not in expected or sample_id in seen or sample.get("video_id") != expected[sample_id]["video_id"]:
            raise ValueError("Missing, duplicate or foreign sample identity")
        seen.add(sample_id)
        if sample.get("status") == "skipped":
            if not str(sample.get("skip_reason", "")).strip():
                raise ValueError(f"{sample_id}: skipped sample needs a reason")
            skipped.append({"sample_id": sample_id, "reason": sample["skip_reason"]})
            continue
        events = sample.get("events", [])
        if sample.get("status") != "reviewed" or not 1 <= len(events) <= 3:
            raise ValueError(f"{sample_id}: review 1-3 events or skip with reason")
        event_rows, event_texts = [], []
        for index, event in enumerate(events):
            query_id = f"{sample_id}_event_{index+1}"
            if not str(event.get("text_vi", "")).strip():
                raise ValueError(f"{query_id}: Vietnamese observation required")
            english = event.get("text_en", "")
            if not isinstance(english, str) or not english.strip() or event.get("english_approved") is not True:
                translations.append(query_id)
            judgments, identities = [], set()
            for judgment in event.get("judgments", []):
                identity = (judgment.get("sample_id"), judgment.get("keyframe_n"))
                frame = available.get(identity)
                if frame is None or identity in identities or judgment.get("judgment") not in {"positive", "negative", "unknown"}:
                    raise ValueError(f"{query_id}: duplicate, unavailable or unreviewed frame")
                identities.add(identity)
                judgments.append({**frame, "sample_id": identity[0], "judgment": judgment["judgment"]})
            own = {key for key in available if key[0] == sample_id}
            if not own.issubset(identities):
                raise ValueError(f"{query_id}: every available frame in its window must be explicitly judged")
            positives = [j for j in judgments if j["judgment"] == "positive" and j["sample_id"] == sample_id]
            if not positives:
                raise ValueError(f"{query_id}: needs a visible positive in its own window")
            event_rows.append(judgments)
            event_texts.append(english.strip() if isinstance(english, str) else "")
            queries.append({"query_id": query_id, "split": "holdout", "video_id": sample["video_id"],
                "text_vi": event["text_vi"].strip(), "event_text": event_texts[-1], "judgments": judgments,
                "annotation_source": "human_visual_observation", "human_verified": True})
        if len(events) > 1:
            # Verify at least one observed path with strictly later timestamps,
            # bounded by the unchanged runtime's 60s gap, not just list order.
            times = sorted(j["pts_time"] for j in event_rows[0] if j["judgment"] == "positive" and j["sample_id"] == sample_id)
            for judgments in event_rows[1:]:
                times = sorted(j["pts_time"] for j in judgments if j["judgment"] == "positive" and j["sample_id"] == sample_id
                               and any(0 < j["pts_time"] - t <= 60 for t in times))
            if not times:
                raise ValueError(f"{sample_id}: no strictly ordered positive keyframe path within 60s gaps")
    if seen != set(expected):
        raise ValueError("Every sampled video must be reviewed or explicitly skipped")
    if not queries:
        raise ValueError("No reviewed queries; cannot evaluate an empty holdout")
    if require_english and translations:
        raise ValueError("English translation still needs human approval: " + ", ".join(translations))
    return {"queries": queries, "skipped_samples": skipped, "translations_pending": translations,
            "reviewed_windows": len(expected) - len(skipped), "eligible_pair_queries": sum(
                {"positive", "negative"}.issubset({j["judgment"] for j in q["judgments"]}) for q in queries)}


def freeze(packet, annotations, output):
    if Path(output).exists():
        raise FileExistsError("Frozen labels cannot be overwritten")
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    result = validate_annotations(packet, data, require_english=True)
    atomic_write_json(output, {"schema": "kis_human_holdout_labels_v1", "annotation_source": "human_visual_observation",
        "human_verified": True, "reviewer": data["reviewer"], "caption_used_for_annotation": False,
        "manifest_sha256": data["manifest_sha256"], "protocol_sha256": data["protocol_sha256"],
        "annotations_sha256": sha(annotations), "human_annotations": data, **result})
    print(json.dumps({"frozen_labels": str(Path(output).resolve()), "sha256": sha(output),
                      "queries": len(result["queries"]), "evaluation_run": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--output-dir", type=Path, required=True)
    prepare_cmd.add_argument("--count", type=int, default=12)
    prepare_cmd.add_argument("--seed", type=int, default=2026110203)
    for phase in ("validate", "freeze", "build"):
        cmd = sub.add_parser(phase)
        cmd.add_argument("--packet", type=Path, required=True)
        if phase != "build":
            cmd.add_argument("--annotations", type=Path, required=True)
        if phase == "freeze":
            cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir, args.count, args.seed)
    elif args.command == "build":
        build_packet(args.packet)
    elif args.command == "freeze":
        freeze(args.packet, args.annotations, args.output)
    else:
        result = validate_annotations(args.packet, json.loads(args.annotations.read_text(encoding="utf-8")))
        print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))
