"""GPU entry point for Kaggle; keeps model stages sequential to release VRAM."""

import argparse
import gc
import json
from pathlib import Path
import zipfile

from tools.audit_artifact_formats import audit_video_formats
from tools.prepare_video_data import VideoDatasetBuilder, KeyframeConfig, configure_device, discover_videos
from tools.kaggle_keyframes_caption import restart_caption_runtime_if_changed


def release_models():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run(video_dir, output_root, video_ids=None, smoke_seconds=None,
        visual_batch=8, object_batch=8, caption_batch=16, seed=42,
        caption_backend="internvl", caption_4bit=None, restart_captions=False):
    print(configure_device("cuda", threads=4), flush=True)
    video_dir, output_root = Path(video_dir), Path(output_root)
    if output_root.resolve().is_relative_to(Path("/kaggle/input")):
        raise ValueError("Output must be under /kaggle/working, not read-only /kaggle/input")
    if smoke_seconds is not None:
        output_root = output_root / f"smoke_{smoke_seconds:g}s_seed_{seed}"
    videos = discover_videos(video_dir, video_ids)
    builder = VideoDatasetBuilder(output_root, device="cuda")
    config = KeyframeConfig(frame_gap_min=80, frame_gap_max=160, random_seed=seed,
                            max_duration_seconds=smoke_seconds)
    for video in videos.values():
        builder.extract_keyframes(video, config)
    builder.encode_visual_features(videos, batch_size=visual_batch)
    release_models()
    builder.generate_objects(videos, batch_size=object_batch)
    release_models()
    if restart_captions:
        from kis.structured_caption import caption_generation_config
        from tools.structured_caption_generator import resolve_caption_backend_runtime

        runtime = resolve_caption_backend_runtime(caption_backend, "cuda", caption_4bit)
        config = caption_generation_config(caption_backend, runtime["use_4bit"], runtime["compute_dtype"])
        for video_id in videos:
            restart_caption_runtime_if_changed(builder, video_id, config, force=True)
    builder.generate_captions(videos, batch_size=caption_batch, backend=caption_backend, use_4bit=caption_4bit)
    release_models()
    builder.build_caption_indexes(videos, batch_size=32)
    release_models()
    for video_id in videos:
        print(builder.validate_video(video_id), flush=True)
        formats = audit_video_formats(output_root, video_id)
        if any(report["status"] != "OK" for report in formats):
            raise ValueError(formats)
        print(json.dumps(formats, ensure_ascii=False), flush=True)
    archive = output_root.parent / f"{output_root.name}-artifacts.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        for video_id in videos:
            for path in builder.artifact_paths(video_id).values():
                files = sorted(path.rglob("*")) if path.is_dir() else [path]
                for file in files:
                    if file.is_file():
                        target.write(file, file.relative_to(output_root).as_posix())
            backups = output_root / "preparation" / "caption_backups" / video_id
            if backups.is_dir():
                for file in sorted(backups.rglob("*.json")):
                    target.write(file, file.relative_to(output_root).as_posix())
    print(f"Download: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/kis_output"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--smoke-seconds", type=float)
    parser.add_argument("--visual-batch", type=int, default=8)
    parser.add_argument("--object-batch", type=int, default=8)
    parser.add_argument("--caption-batch", type=int, default=16)
    parser.add_argument("--caption-backend", choices=("internvl", "qwen"), default="internvl")
    parser.add_argument("--restart-captions", action="store_true")
    parser.add_argument("--caption-4bit", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.video_dir, args.output_root, args.video_ids, args.smoke_seconds,
        args.visual_batch, args.object_batch, args.caption_batch, args.seed,
        args.caption_backend, args.caption_4bit, args.restart_captions)
