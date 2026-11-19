"""Embed every mapped keyframe with SigLIP2-384 (GPU fp16 or CPU), resumable per video.

Same output contract as notebooks/kaggle_siglip2_embed.ipynb:
<output>/<group>/<video>/<video>.npy, float16, L2-normalized, rows in mapping
CSV order. A finished video is written atomically and skipped on restart.
A video with a missing image is not written; it is listed in FAILED.json.

Images come from <dataset>/data/<group>/<video>/<n>.jpg, or straight from the
competition ZIPs (members keyframes/<video>/<n>.jpg) with --zips, no extraction.

    python tools/embed_siglip2.py --dataset dataset --output dataset/siglip2-features \
        --groups L22 L23 --zips dataset/data/L22.zip dataset/data/L23.zip
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np  # before torch, also in spawned loader workers (Windows OpenMP order)

MODEL_REPO = "timm/ViT-B-16-SigLIP2-384"
MODEL_REVISION = "369333c629e1ab4eb90806c8e13df3054c3c1e63"
SUFFIXES = (".jpg", ".jpeg", ".png")
ZIP_MEMBER = re.compile(r"(?:.*/)?(L\d+_V\d+)/(\d+)\.(?:jpe?g|png)", re.IGNORECASE)


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


def zip_index(zips):
    index = {}
    for archive in zips:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                match = ZIP_MEMBER.fullmatch(name)
                if match:
                    key = (match.group(1), int(match.group(2)))
                    if key in index:
                        raise SystemExit(f"Duplicate image {key} in {index[key][0]} and {archive}")
                    index[key] = (str(archive), name)
    return index


def find_image(dataset, video_id, number):
    directory = dataset / "data" / video_id.split("_")[0] / video_id
    for suffix in SUFFIXES:
        path = directory / f"{number:03d}{suffix}"
        if path.is_file():
            return str(path)
    return None


def single_thread_worker(_):
    import torch
    torch.set_num_threads(1)  # N loader processes x all cores each would thrash the CPU


class Frames:
    """Map-style dataset; each loader process opens its own ZIP handles."""

    def __init__(self, sources, preprocess):
        self.sources, self.preprocess, self.handles = sources, preprocess, {}

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        from PIL import Image
        source = self.sources[i]
        if isinstance(source, tuple):
            archive, member = source
            if archive not in self.handles:
                self.handles[archive] = zipfile.ZipFile(archive)
            stream = io.BytesIO(self.handles[archive].read(member))
        else:
            stream = source
        with Image.open(stream) as image:
            return self.preprocess(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--zips", nargs="*", type=Path, default=[], help="Read images from these ZIPs")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=8, help="CPU torch threads")
    parser.add_argument("--loaders", type=int, default=8, help="Image decode processes")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()

    import torch
    import open_clip
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=[
        "open_clip_config.json", "open_clip_model.safetensors", "tokenizer*", "special_tokens_map.json"])
    model, preprocess = open_clip.create_model_from_pretrained("local-dir:" + model_dir)
    model = model.to(device).eval()
    fp16 = device == "cuda"
    images = zip_index(args.zips) if args.zips else None
    print(f"device={device} fp16={fp16} zip_images={len(images) if images is not None else '-'}", flush=True)

    videos = mapping_files(args.dataset, args.groups)
    names = sorted(videos)[: args.limit_videos] if args.limit_videos else sorted(videos)
    args.output.mkdir(parents=True, exist_ok=True)
    failed, plan, sources = {}, [], []
    for video_id in names:
        target = args.output / video_id.split("_")[0] / video_id / f"{video_id}.npy"
        if target.is_file():
            continue
        with videos[video_id].open(encoding="utf-8", newline="") as stream:
            numbers = [int(row["n"]) for row in csv.DictReader(stream)]
        found = [images.get((video_id, n)) if images is not None else find_image(args.dataset, video_id, n)
                 for n in numbers]
        missing = [n for n, s in zip(numbers, found) if s is None]
        if missing:
            failed[video_id] = missing
            print(f"SKIP {video_id}: {len(missing)}/{len(numbers)} missing images", flush=True)
            continue
        plan.append((video_id, target, len(numbers)))
        sources += found
    print(f"todo: {len(plan)} videos, {len(sources)} images", flush=True)

    # One loader for every pending row: Windows spawns loader processes slowly,
    # so a loader per video spends more time starting than decoding.
    loader = DataLoader(Frames(sources, preprocess), batch_size=args.batch, num_workers=args.loaders,
                        pin_memory=device == "cuda", worker_init_fn=single_thread_worker,
                        prefetch_factor=4 if args.loaders else None)
    buffer, started, done_rows = [], time.time(), 0
    pending = iter(plan)
    current = next(pending, None)

    def flush_ready():
        nonlocal current, done_rows
        while current is not None and sum(len(b) for b in buffer) >= current[2]:
            video_id, target, rows = current
            stacked = np.concatenate(buffer)
            vectors, rest = stacked[:rows].astype(np.float16), stacked[rows:]
            buffer[:] = [rest] if len(rest) else []
            if vectors.shape != (rows, 768) or not np.isfinite(vectors).all():
                raise SystemExit(f"{video_id}: bad output {vectors.shape}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with partial.open("wb") as stream:
                np.save(stream, vectors)
            os.replace(partial, target)
            done_rows += rows
            print(f"{video_id} {rows} rows | {done_rows}/{len(sources)} | "
                  f"{done_rows / (time.time() - started):.1f} img/s", flush=True)
            current = next(pending, None)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        for pixels in loader:
            buffer.append(model.encode_image(pixels.to(device, non_blocking=True), normalize=True).float().cpu().numpy())
            flush_ready()
    if current is not None:
        raise SystemExit(f"Loader ended before {current[0]} was complete")

    per_video = {}
    for video_id in names:
        target = args.output / video_id.split("_")[0] / video_id / f"{video_id}.npy"
        if target.is_file():
            per_video[video_id] = hashlib.sha256(target.read_bytes()).hexdigest()
    tag = "-".join(sorted(set(args.groups))) if len(args.groups) <= 3 else f"{args.groups[0]}-{args.groups[-1]}"
    (args.output / f"FAILED_{tag}.json").write_text(json.dumps(failed, indent=1))
    manifest = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "open_clip": open_clip.__version__,
                "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "dimension": 768, "dtype": "float16", "normalized": True, "metric": "IP",
                "preprocess": str(preprocess), "groups": args.groups, "videos_expected": len(names),
                "videos_written": len(per_video), "failed_videos": len(failed), "sha256": per_video}
    (args.output / f"SIGLIP2_MANIFEST_{tag}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("videos_expected", "videos_written", "failed_videos")}), flush=True)


if __name__ == "__main__":
    main()
