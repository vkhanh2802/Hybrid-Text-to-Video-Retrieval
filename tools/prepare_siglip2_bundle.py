"""Pack keyframes, resized for SigLIP2-384, into one ZIP for a Kaggle Dataset.

SigLIP2-384's own preprocess squashes every image to 384x384 (no crop), so the
bundle stores that size directly: about half the bytes of aspect-kept JPEGs.

Runs on the school CPU (stdlib + Pillow only). Row order follows each mapping
CSV exactly, so the Kaggle notebook can write NPY files aligned with the
existing clip-features-32 layout. Missing images are listed, never invented.

    python tools/prepare_siglip2_bundle.py --dataset /path/dataset \
        --output /path/siglip2-bundle.zip --groups L01 L02 ... L21
"""

import argparse
import csv
import io
import json
from multiprocessing import Pool
from pathlib import Path
import zipfile

from PIL import Image

SUFFIXES = (".jpg", ".jpeg", ".png")


def find_image(dataset, video_id, number):
    directory = dataset / "data" / video_id.split("_")[0] / video_id
    for suffix in SUFFIXES:
        path = directory / f"{number:03d}{suffix}"
        if path.is_file():
            return path
    return None


def resize_jpeg(job):
    path, size, quality = job
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        buffer = io.BytesIO()
        # 4:4:4 chroma: q90 4:2:0 shifted SigLIP2 vectors to cos ~0.95 vs the original;
        # q95 4:4:4 measured cos >= 0.995 on L01_V001 samples.
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def mapping_files(dataset, groups):
    files = {}
    for group in groups:
        paths = sorted((dataset / "mapping" / group).rglob(f"{group}_V*.csv"))
        if not paths:
            raise SystemExit(f"Missing mapping group {group}")
        for path in paths:
            if path.stem in files:
                raise SystemExit(f"Duplicate mapping for {path.stem}: {files[path.stem]} and {path}")
            files[path.stem] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--size", type=int, default=384, help="Square side used by the model preprocess")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-videos", type=int, help="Smoke test: only the first N videos")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new output path")

    videos = mapping_files(args.dataset, args.groups)
    names = sorted(videos)[: args.limit_videos] if args.limit_videos else sorted(videos)
    manifest = {"schema": 1, "size": args.size, "quality": args.quality,
                "groups": args.groups, "videos": {}}
    total = missing_total = 0
    partial = args.output.with_name(args.output.name + ".partial")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_STORED) as bundle, Pool(args.workers) as pool:
        for count, video_id in enumerate(names, 1):
            text = videos[video_id].read_text(encoding="utf-8")
            numbers = [int(row["n"]) for row in csv.DictReader(io.StringIO(text))]
            bundle.writestr(f"mapping/{video_id}.csv", text)
            paths = [find_image(args.dataset, video_id, n) for n in numbers]
            missing = [n for n, p in zip(numbers, paths) if p is None]
            jobs = [(p, args.size, args.quality) for p in paths if p is not None]
            present = [n for n, p in zip(numbers, paths) if p is not None]
            for n, data in zip(present, pool.imap(resize_jpeg, jobs, chunksize=16)):
                bundle.writestr(f"images/{video_id}/{n:03d}.jpg", data)
            manifest["videos"][video_id] = {"rows": len(numbers), "missing": missing}
            total += len(numbers)
            missing_total += len(missing)
            if count % 25 == 0 or count == len(names):
                print(f"{count}/{len(names)} videos, {total} rows, {missing_total} missing", flush=True)
        manifest.update(rows=total, missing=missing_total)
        bundle.writestr("manifest.json", json.dumps(manifest, indent=1))
    partial.rename(args.output)
    print(json.dumps({"output": str(args.output), "videos": len(names), "rows": total,
                      "missing": missing_total, "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
