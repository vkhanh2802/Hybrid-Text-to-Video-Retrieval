"""Shared model settings and the active dataset scope."""

from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path

from kis.paths import KIS_ROOT, MAX_VIDEO_GROUP, video_group
from kis.temporal import TemporalConfig
from kis.text_reranker import TextRerankerConfig
from kis.ocr import OCRConfig

MODEL_MANIFEST_PATH = KIS_ROOT / "config" / "models.json"
MODEL_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))

VISUAL_CONFIG = MODEL_MANIFEST["visual_embedding"]
CAPTION_CONFIG = MODEL_MANIFEST["caption_embedding"]
VLM_CONFIG = MODEL_MANIFEST["vlm_reranker"]
CAPTION_GENERATION_CONFIG = MODEL_MANIFEST["caption_generation"]
FAST_CAPTION_CONFIG = MODEL_MANIFEST["fast_caption_generation"]
QUERY_STRUCTURING_CONFIG = MODEL_MANIFEST["query_structuring"]


def milvus_model_properties():
    return {
        "kis.model": VISUAL_CONFIG["model"],
        "kis.pretrained": VISUAL_CONFIG["pretrained"],
        "kis.hf_repo": VISUAL_CONFIG["hf_repo"],
        "kis.hf_revision": VISUAL_CONFIG["hf_revision"],
        "kis.dimension": str(VISUAL_CONFIG["dimension"]),
        "kis.normalized": str(VISUAL_CONFIG["normalized"]).lower(),
        "kis.metric": VISUAL_CONFIG["metric"],
    }


@dataclass(frozen=True)
class RuntimeConfig:
    caption_root: Path
    video_groups: tuple
    retrieval_profile: str
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    text_reranker: TextRerankerConfig = field(default_factory=TextRerankerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    ocr_index: Path | None = None
    ocr_format: str = "strict"
    ocr_root: Path | None = None
    ocr_manifest_sha256: str | None = None
    # Groups whose captions came from another generator (no internvl_dense_v3 tag).
    legacy_caption_groups: tuple = ()


def load_runtime_config(config_path=None, environ=None):
    env = os.environ if environ is None else environ
    path = Path(config_path) if config_path else KIS_ROOT / "config" / "pipeline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported runtime config version")
    root = Path(env.get("KIS_CAPTION_ROOT") or data["caption_root"])
    if not root.is_absolute():
        root = KIS_ROOT / root
    groups = tuple(part.strip() for part in env["KIS_VIDEO_GROUPS"].split(",") if part.strip()) if "KIS_VIDEO_GROUPS" in env else tuple(data["video_groups"])
    if not groups:
        raise ValueError("Specify video groups covered by the active caption corpus")
    for group in groups:
        video_group(f"{group}_V001")
        if int(group[1:]) > MAX_VIDEO_GROUP:
            raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}: {group}")
    temporal = dict(data.get("temporal", {}))
    if env.get("KIS_DISABLE_TEMPORAL", "").lower() in {"1", "true", "yes"}:
        temporal["enabled"] = False
    text_reranker = dict(data.get("text_reranker", {}))
    if env.get("KIS_TEXT_RERANKER_MODE"):
        text_reranker["mode"] = env["KIS_TEXT_RERANKER_MODE"]
    ocr = dict(data.get("ocr", {}))
    if env.get("KIS_OCR_MODE"):
        ocr["mode"] = env["KIS_OCR_MODE"]
    if env.get("KIS_OCR_WEIGHT"):
        ocr["weight"] = float(env["KIS_OCR_WEIGHT"])
    ocr_index = Path(env.get("KIS_OCR_INDEX") or data.get("ocr_index") or root / "ocr" / "index.sqlite3")
    if not ocr_index.is_absolute():
        ocr_index = KIS_ROOT / ocr_index
    ocr_format = env.get("KIS_OCR_FORMAT", data.get("ocr_format", "strict"))
    if ocr_format not in {"strict", "public_json", "school_snapshot"}:
        raise ValueError("KIS_OCR_FORMAT must be strict, public_json or school_snapshot")
    public_root = env.get("KIS_OCR_ROOT") or data.get("ocr_root")
    if ocr_format in {"public_json", "school_snapshot"} and not public_root:
        raise ValueError("public OCR requires explicit KIS_OCR_ROOT")
    pin = env.get("KIS_OCR_MANIFEST_SHA256") or data.get("ocr_manifest_sha256")
    if ocr_format == "school_snapshot" and not re.fullmatch(r"[a-f0-9]{64}", pin or ""):
        raise ValueError("school_snapshot requires pinned KIS_OCR_MANIFEST_SHA256")
    if public_root:
        public_root = Path(public_root)
        if not public_root.is_absolute():
            public_root = KIS_ROOT / public_root
        public_root = Path(os.path.abspath(public_root))
    return RuntimeConfig(root.resolve(), groups, env.get("KIS_RETRIEVAL_PROFILE") or data["retrieval_profile"],
                         TemporalConfig(**temporal), TextRerankerConfig(**text_reranker), OCRConfig(**ocr), ocr_index.resolve(),
                          ocr_format, public_root, pin, tuple(data.get("legacy_caption_groups", ())))


def load_caption_runtime(config):
    from kis.text import CaptionRetriever

    caption = CaptionRetriever(artifact_root=config.caption_root, video_groups=config.video_groups)
    video_ids = sorted({item["video_id"] for item in caption.caption_data})
    for item in caption.caption_data:
        if (item.get("caption_generation", {}).get("backend") != "internvl_dense_v3"
                and video_group(item["video_id"]) not in config.legacy_caption_groups):
            raise ValueError("The active pipeline requires the imported InternVL captions")
        if "text_processing" in item:
            raise ValueError("Use original caption v2; rebuild with python -m tools.ingest")
    return caption, video_ids
