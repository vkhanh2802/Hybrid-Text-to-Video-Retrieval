"""Deterministically sample Drive keyframe windows for image-authored judgments.

Read-only Drive access via the existing machine-local adapter. Persist only
contact sheets and image hashes in a gitignored benchmark directory, not keys,
Drive IDs, original image files, captions, or automatic relevance labels.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageOps

from kis.artifact_io import atomic_write_json
from kis.config import load_runtime_config
from kis.paths import KIS_ROOT, MAPPING_DIR, local_image_settings


def sample_windows(mapping_root, groups, seed=20261102, count=8, radius=2, exclude_video_ids=(), split_mode="dev_holdout"):
    if split_mode not in {"dev_holdout", "holdout"}:
        raise ValueError("Unknown sampling split mode")
    excluded = set(exclude_video_ids)
    paths = {}
    for path in sorted(Path(mapping_root).rglob("L*_V*.csv")):
        if path.stem.split("_")[0] not in groups or path.stem in excluded:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate mapping: {path.stem}")
        paths[path.stem] = path
    if count < 2 or count > len(paths) or radius < 0:
        raise ValueError("Choose 2..available videos and nonnegative radius")
    rng = random.Random(seed)
    videos = rng.sample(sorted(paths), count)
    windows = []
    for index, video_id in enumerate(videos):
        with paths[video_id].open(encoding="utf-8", newline="") as file:
            rows = [{"video_id": video_id, "keyframe_n": int(row["n"]),
                     "frame_id": int(row["frame_idx"]), "pts_time": float(row["pts_time"])}
                    for row in csv.DictReader(file)]
        counts = Counter(row["frame_id"] for row in rows)
        rows = sorted((row for row in rows if counts[row["frame_id"]] == 1), key=lambda row: (row["pts_time"], row["keyframe_n"]))
        if not rows:
            raise ValueError(f"No unambiguous frames: {video_id}")
        anchor = rng.randrange(len(rows))
        windows.append({"sample_id": f"drive_{index + 1:02d}", "video_id": video_id,
                         "split": "holdout" if split_mode == "holdout" else "dev" if index < count // 2 else "holdout",
                        "anchor": rows[anchor], "frames": rows[max(0, anchor-radius):anchor+radius+1]})
    return windows


def fetch_window(window, source, output):
    sheet = Image.new("RGB", (1440, 660), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{window['sample_id']} | {window['split']} | {window['video_id']} | IMAGE OBSERVATION ONLY", fill="black")
    records = []
    for index, frame in enumerate(window["frames"]):
        record = dict(frame)
        x, y = (index % 3) * 480, 32 + (index // 3) * 312
        try:
            data, mime = source.read(frame["video_id"], frame["keyframe_n"])
            with Image.open(BytesIO(data)) as original:
                record.update(image_sha256=hashlib.sha256(data).hexdigest(), width=original.width,
                              height=original.height, mime=mime, status="downloaded")
                thumb = ImageOps.contain(original.convert("RGB"), (472, 272))
                sheet.paste(thumb, (x, y))
                thumb.close()
            del data
        except Exception as exc:
            # Avoid credential/HTTP response text in persisted reports.
            record.update(status="fetch_failed", error_type=type(exc).__name__)
            draw.text((x+10, y+40), "FETCH FAILED: " + type(exc).__name__, fill="black")
        draw.text((x+8, y+275), f"n={frame['keyframe_n']} frame={frame['frame_id']} t={frame['pts_time']:.3f}s", fill="black")
        records.append(record)
    name = window["sample_id"] + ".jpg"
    sheet.save(output / name, quality=92)
    sheet.close()
    return {**window, "frames": records, "contact_sheet": name,
            "label_status": "unreviewed", "caption_accessed": False}


def run(output, count=8, seed=20261102, exclude_files=(), split_mode="dev_holdout"):
    from kis.local_images import DriveImages

    settings = local_image_settings()
    folder = os.environ.get("KIS_DRIVE_FOLDER_ID") or settings.get("drive_folder_id")
    if not folder:
        raise RuntimeError("Configure the existing Drive image source first; no credentials are printed")
    source = DriveImages(folder, credentials_file=settings.get("credentials_file"))
    config = load_runtime_config()
    excluded = set()
    for filename in exclude_files:
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        excluded.update(data.get("excluded_video_ids", []))
        for item in [*data.get("samples", []), *data.get("queries", [])]:
            if item.get("video_id"):
                excluded.add(item["video_id"])
    windows = sample_windows(MAPPING_DIR, set(config.video_groups), seed=seed, count=count,
                             exclude_video_ids=excluded, split_mode=split_mode)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "seed": seed, "sampling": "uniform videos then uniform anchor row; +/-2 unambiguous rows",
                "split_mode": split_mode,
                "excluded_video_ids": sorted(excluded),
                "source": "google_drive_readonly", "label_provenance": "unreviewed, no captions or retrieval scores used",
                "limitations": "Video-disjoint split is not content-duplicate-disjoint; sparse keyframes do not prove continuous action",
                "samples": windows}
    atomic_write_json(output / "sample_plan.json", manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda window: fetch_window(window, source, output), windows))
    manifest["samples"] = records
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output.resolve()), "videos": len(records),
                      "downloaded_images": sum(frame["status"] == "downloaded" for record in records for frame in record["frames"]),
                      "failed_images": sum(frame["status"] != "downloaded" for record in records for frame in record["frames"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    parser.add_argument("--split-mode", choices=("dev_holdout", "holdout"), default="dev_holdout")
    args = parser.parse_args()
    run(args.output_dir, args.count, args.seed, args.exclude_file, args.split_mode)
