"""Check temporal artifacts without Docker, model loading or inference."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import VISUAL_CONFIG, load_runtime_config
from kis.event_retrieval import LocalEventScorer
from kis.paths import DATA_DIR, KIS_ROOT, MAPPING_DIR, resolve_video_artifact


def check_video(video_id, caption_root):
    result = {"video_id": video_id, "status": "error"}
    try:
        frames, vectors = LocalEventScorer._video_data(video_id)
        if not frames:
            raise ValueError("Empty temporal mapping")
        result.update(frames=len(frames), first_seconds=min(f["pts_time"] for f in frames),
                      last_seconds=max(f["pts_time"] for f in frames), visual_available=vectors is not None)
        counts = Counter(frame["frame_id"] for frame in frames)
        ambiguous = sorted(frame_id for frame_id, count in counts.items() if count > 1)
        excluded = sum(counts[frame_id] for frame_id in ambiguous)
        result.update(ambiguous_frame_ids=ambiguous, ambiguous_frame_rows_excluded=excluded,
                      warnings=["Repeated frame IDs are excluded from temporal scoring"] if ambiguous else [])
        if excluded == len(frames):
            raise ValueError("No unambiguous temporal frames")
        if vectors is not None:
            if vectors.shape[1] != int(VISUAL_CONFIG["dimension"]):
                raise ValueError("Visual dimension does not match pinned model")
            # Bounded copies; do not materialize the full corpus feature matrix.
            for start in range(0, len(vectors), 256):
                block = np.asarray(vectors[start:start + 256], dtype=np.float32)
                if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) <= np.finfo(np.float32).eps):
                    raise ValueError("Non-finite or zero local visual embedding")
        path = resolve_video_artifact(Path(caption_root) / "caption_generator", video_id, ".json")
        captions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        if not isinstance(captions, list):
            raise ValueError("Caption artifact must be an array")
        identities = {int(frame["frame_id"]): int(frame["keyframe_n"]) for frame in frames}
        seen = set()
        searchable = 0
        for caption in captions:
            frame_id, keyframe_n = int(caption["frame_id"]), int(caption["keyframe_n"])
            if caption["video_id"] != video_id or identities.get(frame_id) != keyframe_n or frame_id in seen:
                raise ValueError("Caption/local mapping identity mismatch or duplicate")
            seen.add(frame_id)
            searchable += bool(caption.get("retrieval_text") or caption.get("caption"))
        result.update(caption_records=len(captions), searchable_captions=searchable,
                      caption_frame_coverage=len(captions) / len(frames))
        if vectors is None and not searchable:
            raise ValueError("Neither visual embeddings nor searchable captions available")
        result["status"] = "ready"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    finally:
        # Release mmap handles for Windows; do not retain a stale corpus cache.
        LocalEventScorer._video_data.cache_clear()
    return result


def run(output, config=None, mapping_root=MAPPING_DIR, *, groups=None, image_roots=None, archives=()):
    config = config or load_runtime_config()
    if groups is not None or image_roots is not None or archives:
        from tools.temporal_inventory import run_inventory

        return run_inventory(output, config, mapping_root, groups,
                             (DATA_DIR,) if image_roots is None else image_roots, archives)
    groups = set(config.video_groups)
    video_ids = sorted({path.stem for path in Path(mapping_root).rglob("L*_V*.csv")
                        if path.stem.split("_")[0] in groups})
    records = [check_video(video_id, config.caption_root) for video_id in video_ids]
    report = {
        "mode": "artifact_only_no_models", "video_groups": list(config.video_groups),
        "limitations": ["Does not verify embedding provenance or semantic correctness",
                        "Does not connect to Milvus, load encoders, or measure retrieval quality",
                        "Caption vectors/index manifests are not loaded or verified",
                        "Video discovery is mapping-based; missing entire videos require an external inventory"],
        "summary": {
            "videos": len(records), "ready": sum(item["status"] == "ready" for item in records),
            "errors": sum(item["status"] == "error" for item in records),
            "frames": sum(item.get("frames", 0) for item in records),
            "visual_only_videos": sum(item["status"] == "ready" and not item.get("searchable_captions") for item in records),
            "videos_with_warnings": sum(bool(item.get("warnings")) for item in records),
            "ambiguous_frame_rows_excluded": sum(item.get("ambiguous_frame_rows_excluded", 0) for item in records),
            "missing_mapping_groups": sorted(groups - {vid.split("_")[0] for vid in video_ids}),
        },
        "videos": records,
    }
    atomic_write_json(output, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {Path(output).resolve()}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=KIS_ROOT / "benchmark_results/temporal/preflight.json")
    parser.add_argument("--groups", help="Offline audit scope only, e.g. L04,L05,L06; does not change config")
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_DIR)
    parser.add_argument("--image-root", type=Path, action="append", help="Repeatable metadata-only image root")
    parser.add_argument("--archive", type=Path, action="append", default=[], help="Repeatable ZIP central-directory inventory only")
    args = parser.parse_args(argv)
    try:
        from tools.temporal_inventory import parse_groups

        report = run(args.output, mapping_root=args.mapping_root,
                     groups=parse_groups(args.groups) if args.groups is not None else None,
                     image_roots=args.image_root, archives=args.archive)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"temporal_preflight: {exc}\n")
    if report.get("schema") == "kis_offline_preflight_v1":
        # A successful inventory is not a runtime-readiness assertion.
        return 0
    return 1 if report["summary"]["errors"] or report["summary"]["missing_mapping_groups"] or not report["summary"]["videos"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
