"""Shared model settings and the active dataset scope."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

from kis.paths import KIS_ROOT, MAX_VIDEO_GROUP, video_group

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
    return RuntimeConfig(root.resolve(), groups, env.get("KIS_RETRIEVAL_PROFILE") or data["retrieval_profile"])


def load_caption_runtime(config):
    from kis.text import CaptionRetriever

    caption = CaptionRetriever(artifact_root=config.caption_root, video_groups=config.video_groups)
    video_ids = sorted({item["video_id"] for item in caption.caption_data})
    for item in caption.caption_data:
        if item.get("caption_generation", {}).get("backend") != "internvl_dense_v3":
            raise ValueError("The active pipeline requires the imported InternVL captions")
        if "text_processing" in item:
            raise ValueError("Use original caption v2; rebuild with python -m tools.ingest")
    return caption, video_ids
