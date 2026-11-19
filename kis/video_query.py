"""Bounded short-video queries using the existing SigLIP image embedding space."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from kis.config import VISUAL_CONFIG
from kis.result_pool import complete_frame_pool


MIN_VIDEO_SECONDS = 1.0
MAX_VIDEO_SECONDS = 3.0
MAX_VIDEO_BYTES = 20 * 1024 * 1024
DRES_MAX_VIDEO_SECONDS = 60.0
DRES_MAX_VIDEO_BYTES = 24 * 1024 * 1024
SAMPLE_FPS = 2
MAX_SAMPLE_FRAMES = 6
SAMPLE_SIZE = 256
MAX_FRAME_PIXELS = 3840 * 2160
MAX_VIDEO_FPS = 120
DECODE_TIMEOUT_SECONDS = 20
DURATION_TOLERANCE = 0.001
VIDEO_TEXT_WEIGHT = 0.7
VIDEO_MIME_TYPES = frozenset({
    "application/octet-stream", "video/mp4", "video/quicktime", "video/webm",
    "video/x-matroska", "video/x-msvideo", "video/avi", "video/x-m4v", "video/3gpp",
})
# Do not let an uploaded playlist fetch URLs or arbitrary demuxers open resources.
INPUT_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi"


def fuse_video_text_results(text_results, video_results, *, top_k=100,
                            text_weight=VIDEO_TEXT_WEIGHT, k=60):
    """Weighted rank fusion: text pipeline and video scores have different scales."""
    if (top_k <= 0 or k < 1 or not math.isfinite(text_weight) or not 0 <= text_weight <= 1):
        raise ValueError("Invalid video/text fusion settings")
    candidates = {}
    for channel, results, weight in (("text", text_results, text_weight),
                                     ("video", video_results, 1 - text_weight)):
        seen = set()
        for item in results:
            key = (item["video_id"], int(item["frame_id"]))
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            raw_score = float(item.get("final_score", item["score"]))
            if not math.isfinite(raw_score):
                raise ValueError("Video/text fusion received a non-finite score")
            if weight == 0:
                continue
            if key not in candidates:
                candidate = deepcopy(item)
                # Neither branch may carry stale VLM or pre-hybrid ranking scores.
                for field in ("vlm_raw_score", "vlm_error", "text_ranking_score", "norm_score", "raw_score"):
                    candidate.pop(field, None)
                candidate.update(score=0.0, final_score=0.0, fusion_score=0.0,
                                 vlm_status="unscored", evidence={})
                candidates[key] = candidate
            candidate = candidates[key]
            contribution = weight * (k + 1) / (k + rank)
            candidate["fusion_score"] += contribution
            candidate["evidence"]["text_pipeline" if channel == "text" else "video"] = {
                "rank": rank, "raw_score": raw_score, "weight": weight,
                "rrf_contribution": contribution, "evidence": deepcopy(item.get("evidence", {}))}
    for item in candidates.values():
        item["score"] = item["final_score"] = item["fusion_score"]
        item["evidence"]["text_fusion"] = {
            "mode": "weighted_rrf", "text_weight": text_weight,
            "video_weight": 1 - text_weight, "k": k}
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def video_query_status():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    return {
        "available": not missing, "missing_dependencies": missing,
        "min_seconds": MIN_VIDEO_SECONDS, "max_seconds": MAX_VIDEO_SECONDS,
        "max_bytes": MAX_VIDEO_BYTES, "sample_fps": SAMPLE_FPS,
        "max_sample_frames": MAX_SAMPLE_FRAMES,
        "official_max_seconds": DRES_MAX_VIDEO_SECONDS, "official_max_bytes": DRES_MAX_VIDEO_BYTES,
        "retrieval": "siglip_image_mean_pool", "temporal_verified": False,
        "text_fusion": {"available": True, "text_weight": VIDEO_TEXT_WEIGHT,
                        "video_weight": 1 - VIDEO_TEXT_WEIGHT,
                        "mode": "weighted_rrf", "k": 60, "full_text_pipeline": True},
    }


def _run_media_tool(command):
    try:
        result = subprocess.run(command, capture_output=True, timeout=DECODE_TIMEOUT_SECONDS,
                                check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ValueError("Video processing timed out; check clip size and FFmpeg runtime") from None
    except OSError:
        raise RuntimeError("Cannot start FFmpeg/ffprobe; check the server installation") from None
    if result.returncode != 0:
        # Decoder stderr may contain local paths or untrusted file metadata.
        raise ValueError("Cannot decode this video; use a valid MP4, MOV, WebM, MKV or AVI clip")
    return result.stdout


def _video_metadata(payload, *, min_seconds=MIN_VIDEO_SECONDS, max_seconds=MAX_VIDEO_SECONDS):
    try:
        info = json.loads(payload)
        stream = info["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        raw_duration = stream.get("duration")
        if raw_duration in (None, "N/A"):
            raw_duration = info.get("format", {}).get("duration")
        duration = float(raw_duration)
        numerator, denominator = str(stream["avg_frame_rate"]).split("/")
        fps = float(numerator) / float(denominator)
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
        raise ValueError("Cannot determine the video's duration, frame rate or dimensions") from None
    if not math.isfinite(duration) or not (
            min_seconds - DURATION_TOLERANCE <= duration <= max_seconds + DURATION_TOLERANCE):
        raise ValueError(f"Video duration must be between {min_seconds:g} and {max_seconds:g} seconds")
    if width <= 0 or height <= 0 or width * height > MAX_FRAME_PIXELS:
        raise ValueError("Video resolution must not exceed 3840 x 2160 pixels (or portrait equivalent)")
    if not math.isfinite(fps) or not 0 < fps <= MAX_VIDEO_FPS:
        raise ValueError("Video frame rate must be greater than 0 and at most 120 FPS")
    return {"duration_seconds": duration, "width": width, "height": height, "fps": fps}


@dataclass
class DecodedVideoQuery:
    images: list
    metadata: dict


@contextmanager
def decode_video_query(data, *, official=False):
    """Delete the temporary upload before inference and always close decoded images."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("Choose a nonempty video file")
    max_bytes = DRES_MAX_VIDEO_BYTES if official else MAX_VIDEO_BYTES
    if len(data) > max_bytes:
        raise ValueError(f"Video file must not exceed {max_bytes // (1024 * 1024)} MiB")
    status = video_query_status()
    if not status["available"]:
        raise RuntimeError("Video input requires FFmpeg and ffprobe on the server PATH")
    images = []
    try:
        with tempfile.TemporaryDirectory(prefix="kis-video-query-") as directory:
            # Never use a client filename/path, URL, or an artifact directory.
            path = Path(directory) / "query.video"
            path.write_bytes(data)
            input_options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", INPUT_FORMATS]
            probe = _run_media_tool([
                shutil.which("ffprobe"), "-v", "error", *input_options,
                "-select_streams", "v:0", "-show_entries",
                "stream=width,height,duration,avg_frame_rate:format=duration", "-of", "json", str(path),
            ])
            metadata = (_video_metadata(probe, min_seconds=0.1, max_seconds=DRES_MAX_VIDEO_SECONDS)
                        if official else _video_metadata(probe))
            sample_fps = min(SAMPLE_FPS, MAX_SAMPLE_FRAMES / metadata["duration_seconds"]) if official else SAMPLE_FPS
            raw = _run_media_tool([
                shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
                *input_options, "-threads", "1", "-max_pixels", str(MAX_FRAME_PIXELS),
                "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", f"fps={sample_fps},scale={SAMPLE_SIZE}:{SAMPLE_SIZE}:force_original_aspect_ratio=increase,crop={SAMPLE_SIZE}:{SAMPLE_SIZE}",
                # One extra sample detects a falsely short duration; never silently trim.
                "-frames:v", str(MAX_SAMPLE_FRAMES + 1), "-threads", "1",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ])
        frame_bytes = SAMPLE_SIZE * SAMPLE_SIZE * 3
        count, remainder = divmod(len(raw), frame_bytes)
        if not count or remainder:
            raise ValueError("Video has no complete decodable frames")
        if count > MAX_SAMPLE_FRAMES:
            raise ValueError("Decoded video exceeds the six-frame sampling budget")
        for offset in range(0, len(raw), frame_bytes):
            images.append(Image.frombytes("RGB", (SAMPLE_SIZE, SAMPLE_SIZE), raw[offset:offset + frame_bytes]))
        del raw
        metadata.update(sampled_frames=count, sample_fps=sample_fps,
                        query_source="dres_official" if official else "manual_upload",
                        aggregation="mean_normalized_image_embedding", upload_stored=False,
                        temporal_verified=False)
        yield DecodedVideoQuery(images, metadata)
    finally:
        for image in images:
            image.close()


def retrieve_video_query(images, encoder, retriever, caption_retriever, *, top_k=20,
                         text_query="", text_weight=VIDEO_TEXT_WEIGHT,
                         require_full_results=False, pool_diagnostics=None):
    """Video appearance query, optionally fused with a text embedding.

    The text side uses the same SigLIP space, so a weighted sum stays meaningful.
    Captions hydrate display metadata and never become a text query.
    The DRES service uses the video-only form here, then fuses the full text
    pipeline's ranked results with fuse_video_text_results instead.
    """
    if not images or len(images) > MAX_SAMPLE_FRAMES or top_k <= 0:
        raise ValueError("Invalid video sample count or retrieval limit")
    if not isinstance(text_query, str) or not isinstance(text_weight, (int, float)) or not 0 <= text_weight <= 1:
        raise ValueError("Invalid video text query or fusion weight")
    text_query = text_query.strip()
    if not text_query and text_weight != VIDEO_TEXT_WEIGHT:
        text_weight = VIDEO_TEXT_WEIGHT
    vectors = np.asarray(encoder.encode_images(images), dtype=np.float32)
    if vectors.shape != (len(images), int(VISUAL_CONFIG["dimension"])) or not np.isfinite(vectors).all():
        raise RuntimeError("Image encoder returned invalid video query embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise RuntimeError("Image encoder returned a zero video query embedding")
    pooled = (vectors / norms).mean(axis=0)
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise RuntimeError("Cannot pool the video's image embeddings")
    combined = pooled / norm
    if text_query:
        text_vector = np.asarray(encoder.encode_text(text_query), dtype=np.float32)
        if text_vector.shape != (int(VISUAL_CONFIG["dimension"]),) or not np.isfinite(text_vector).all():
            raise RuntimeError("Text encoder returned an invalid video query embedding")
        text_norm = np.linalg.norm(text_vector)
        if text_norm <= np.finfo(np.float32).eps:
            raise RuntimeError("Text encoder returned a zero video query embedding")
        combined = text_weight * (text_vector / text_norm) + (1 - text_weight) * combined
        combined_norm = np.linalg.norm(combined)
        if not np.isfinite(combined_norm) or combined_norm <= np.finfo(np.float32).eps:
            raise RuntimeError("Cannot fuse the video and text query embeddings")
        combined = combined / combined_norm
    hits = retriever.search(combined, top_k=top_k)
    if require_full_results:
        hits, info = complete_frame_pool(hits,
            lambda depth: retriever.search(combined, top_k=depth), target=top_k, initial_k=top_k)
        if pool_diagnostics is not None:
            pool_diagnostics.update(info)
    results, seen = [], set()
    for hit in hits:
        key = (hit["video_id"], int(hit["frame_id"]))
        if key in seen:
            continue
        seen.add(key)
        score = float(hit["score"])
        if not math.isfinite(score):
            raise RuntimeError("Visual retrieval returned a non-finite score")
        evidence = {"visual": {"raw_score": score, "metric": "cosine",
                               "query_source": ("video_image_mean_pool+text_embedding" if text_query
                                                else "video_image_mean_pool")}}
        if text_query:
            evidence["text_fusion"] = {"mode": "weighted_embedding_sum", "text_weight": text_weight,
                                       "video_weight": 1 - text_weight,
                                       "text_query_chars": len(text_query)}
        candidate = dict(hit, score=score, final_score=score, visual_score=score,
                         vlm_status="skipped_video_query", text_rerank_status="not_applicable",
                         evidence=evidence)
        metadata = caption_retriever.get_caption(*key) if caption_retriever is not None else None
        if metadata:
            for field in ("retrieval_text", "caption", "keyframe_n", "image_path", "image_sha256"):
                if field in metadata:
                    candidate[field] = metadata[field]
        results.append(candidate)
    return sorted(results, key=lambda item: item["final_score"], reverse=True)[:top_k]
