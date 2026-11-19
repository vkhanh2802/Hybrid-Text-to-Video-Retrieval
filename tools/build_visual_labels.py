"""Freeze visually reviewed observations against the downloaded sample manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from kis.artifact_io import atomic_write_json
from kis.paths import KIS_ROOT


def build(manifest_path, observations_path, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    if Path(output).exists():
        raise FileExistsError("Labels are frozen; choose a versioned output, do not overwrite after retrieval")
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    queries = []
    for observation in observations["samples"]:
        sample = samples[observation["sample_id"]]
        if sample["video_id"] != observation["video_id"]:
            raise ValueError("Observation/manifest video mismatch")
        by_number = {frame["keyframe_n"]: frame for frame in sample["frames"] if frame["status"] == "downloaded"}
        for task in ("image", "sequence"):
            label = observation[task]
            positives = label["relevant_keyframes"] if task == "image" else label["event_relevant_keyframes"][0]
            anchor_n = sample["anchor"]["keyframe_n"] if task == "image" else positives[0]
            if anchor_n not in positives:
                raise ValueError("Random anchor must remain a judged positive")
            all_positive = positives if task == "image" else [n for event in label["event_relevant_keyframes"] for n in event]
            if any(n not in by_number for n in all_positive):
                raise ValueError("Cannot label an image that was not fetched/viewed")
            anchor = by_number[anchor_n]
            query = {"query_id": sample["sample_id"] + "_" + task, "task": task,
                     "video_id": sample["video_id"], "split": sample["split"],
                     "frame_id": anchor["frame_id"], "anchor_keyframe_n": anchor_n,
                     "image_sha256": anchor["image_sha256"], "relevant_keyframes": positives,
                     "annotation_source": "ai_visual_observation", "human_verified": False,
                     "reviewed_images": [{key: by_number[n][key] for key in ("keyframe_n", "frame_id", "pts_time", "image_sha256")}
                                         for n in sorted(set(all_positive))],
                     **{field: label[field] for field in ("raw_query", "english", "keywords")}}
            if task == "sequence":
                if len(label["events"]) != len(label["event_relevant_keyframes"]):
                    raise ValueError("Every event needs independently observed positive keyframes")
                query.update(events=label["events"], event_relevant_keyframes=label["event_relevant_keyframes"])
            queries.append(query)
    labels = {"schema_version": 1, "annotation_source": "ai_visual_observation", "human_verified": False,
              "caption_used_for_annotation": False,
              "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              "observations_sha256": hashlib.sha256(Path(observations_path).read_bytes()).hexdigest(),
              "excluded_samples": observations["excluded_samples"], "queries": queries}
    atomic_write_json(output, labels)
    print(json.dumps({"queries": len(queries), "output": str(Path(output).resolve()),
                      "label_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=KIS_ROOT / "benchmarks/drive_observations_20261102.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.observations, args.output)
