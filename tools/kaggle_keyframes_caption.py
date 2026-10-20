"""Kaggle runner: video -> keyframes/mapping -> captions; no retrieval stages."""

import argparse
import gc
import json
from pathlib import Path
from uuid import uuid4
import zipfile

from kis.artifact_io import atomic_write_json
from tools.prepare_video_data import KeyframeConfig, VideoDatasetBuilder, configure_device, discover_videos
from kis.structured_caption import caption_generation_config
from tools.structured_caption_generator import resolve_caption_backend_runtime


def restart_caption_runtime_if_changed(builder, video_id, generation_config, force=False):
    """Archive only runtime-incompatible captions; preserve keyframes and IDs."""
    state = builder._state(video_id)
    previous = state.get("caption_generation")
    caption_path = builder.artifact_paths(video_id)["captions"] if force else None
    if (previous is None or previous == generation_config) and not force:
        return
    if force and previous is None and not caption_path.is_file():
        return
    runtime_fields = {"precision", "compute_dtype", "inference_revision"}
    if not force and (not isinstance(previous, dict) or (
        {key: value for key, value in previous.items() if key not in runtime_fields}
        != {key: value for key, value in generation_config.items() if key not in runtime_fields}
    )):
        raise ValueError(f"Caption model/prompt changed for {video_id}; use --restart-captions or a new output root")
    backup = builder.root / "preparation" / "caption_backups" / video_id / uuid4().hex
    atomic_write_json(backup / "preparation.json", state)
    caption_path = builder.artifact_paths(video_id)["captions"]
    if caption_path.is_file():
        caption_path.replace(backup / caption_path.name)
    state.setdefault("caption_generation_history", []).append({
        "config": previous, "backup": backup.relative_to(builder.root).as_posix(),
    })
    state.pop("caption_generation", None)
    builder._save_state(video_id, state)
    print(f"[{video_id}] Caption runtime changed; prior checkpoint archived at {backup}", flush=True)


def archive_output(builder, video_ids, status):
    root = builder.root
    suffix = "" if status == "complete" else "-partial"
    archive = root.parent / f"{root.name}-keyframes-captions{suffix}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        for video_id in video_ids:
            paths = builder.artifact_paths(video_id)
            for kind in ("images", "mapping", "captions", "preparation"):
                path = paths[kind]
                files = sorted(path.rglob("*")) if path.is_dir() else [path]
                for file in files:
                    if file.is_file():
                        target.write(file, file.relative_to(root).as_posix())
            backups = root / "preparation" / "caption_backups" / video_id
            if backups.is_dir():
                for file in sorted(backups.rglob("*.json")):
                    target.write(file, file.relative_to(root).as_posix())
        target.write(root / "caption_run.json", "caption_run.json")
    print(f"Download: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


def run(video_dir, output_root, video_ids=None, smoke_seconds=None,
        caption_batch=16, seed=42, caption_backend="internvl", caption_4bit=None, restart_captions=False):
    print(configure_device("cuda", threads=4), flush=True)
    output_root = Path(output_root)
    if output_root.resolve().is_relative_to(Path("/kaggle/input")):
        raise ValueError("Output must be under /kaggle/working, not read-only /kaggle/input")
    if smoke_seconds is not None:
        output_root = output_root / f"smoke_{smoke_seconds:g}s_seed_{seed}"
    videos = discover_videos(video_dir, video_ids)
    builder = VideoDatasetBuilder(output_root, device="cuda")
    runtime = resolve_caption_backend_runtime(caption_backend, "cuda", caption_4bit)
    caption_4bit = runtime["use_4bit"]
    generation_config = caption_generation_config(caption_backend, caption_4bit, runtime["compute_dtype"])
    print(f"[Caption runtime] {runtime}", flush=True)
    status = {"stages": ["keyframes", "captions"], "status": "running",
              "video_ids": sorted(videos), "caption_generation": generation_config}
    atomic_write_json(builder.root / "caption_run.json", status)
    try:
        config = KeyframeConfig(frame_gap_min=80, frame_gap_max=160, random_seed=seed,
                                max_duration_seconds=smoke_seconds)
        for video in videos.values():
            builder.extract_keyframes(video, config)
        for video_id in videos:
            restart_caption_runtime_if_changed(builder, video_id, generation_config, force=restart_captions)
        print("[Stages] keyframes -> captions", flush=True)
        builder.generate_captions(videos, batch_size=caption_batch, backend=caption_backend, use_4bit=caption_4bit)
        for video_id in videos:
            rows = builder.mapping_rows(video_id)
            captions = builder._caption_records(video_id)
            print(f"[{video_id}] {len(rows)} keyframes; {len(captions)} captions; OK", flush=True)
    except Exception as exc:
        status.update(status="partial", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        atomic_write_json(builder.root / "caption_run.json", status)
        try:
            archive_output(builder, videos, "partial")
        except Exception as archive_error:
            print(f"Could not create partial ZIP: {archive_error}; checkpoint files remain in {builder.root}", flush=True)
        raise
    status["status"] = "complete"
    atomic_write_json(builder.root / "caption_run.json", status)
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return archive_output(builder, videos, "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/kis_output"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--smoke-seconds", type=float)
    parser.add_argument("--caption-batch", type=int, default=16)
    parser.add_argument("--caption-backend", choices=("internvl", "qwen"), default="internvl")
    parser.add_argument("--restart-captions", action="store_true", help="archive prior captions and regenerate; keep existing keyframes")
    parser.add_argument("--caption-4bit", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Accept existing notebook arguments, without executing those stages.
    parser.add_argument("--visual-batch", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--object-batch", type=int, default=8, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.video_dir, args.output_root, args.video_ids, args.smoke_seconds,
        args.caption_batch, args.seed, args.caption_backend, args.caption_4bit, args.restart_captions)
