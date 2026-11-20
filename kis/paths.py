import os
import json
from pathlib import Path, PurePosixPath
import re


KIS_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = KIS_ROOT / "dataset"
MAX_VIDEO_GROUP = 24  # L22-L24 caption/index staged 2026-11-19
CAPTION_DIR = DATASET_ROOT / "caption_generator"
CAPTION_MAPPING_DIR = DATASET_ROOT / "caption_mapping"
INDEX_DIR = DATASET_ROOT / "index"


def local_image_settings(settings_path=None):
    settings = Path(settings_path) if settings_path is not None else KIS_ROOT / "config" / "images.local.json"
    return json.loads(settings.read_text(encoding="utf-8")) if settings.is_file() else {}


def configured_data_dir(environ=None, settings_path=None):
    env = os.environ if environ is None else environ
    configured = env.get("KIS_DATA_ROOT")
    if not configured:
        configured = local_image_settings(settings_path).get("data_root")
    if not configured:
        return DATASET_ROOT / "data"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = KIS_ROOT / path
    return path.resolve()


DATA_DIR = configured_data_dir()
OBJECTS_DIR = DATASET_ROOT / "objects"
MAPPING_DIR = DATASET_ROOT / "mapping"
# kis.config imports this module, so read the visual features dir from the manifest directly.
VISUAL_FEATURES_NAME = json.loads((KIS_ROOT / "config" / "models.json").read_text(encoding="utf-8"))[
    "visual_embedding"]["features_dir"]
CLIP_FEATURES_DIR = DATASET_ROOT / VISUAL_FEATURES_NAME

VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_keyframe_image(path, number=None):
    path = Path(path)
    return (path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isascii() and path.stem.isdigit()
            and (number is None or int(path.stem) == int(number)))


def safe_filename_component(value, default="output") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return sanitized or default


def video_group(video_id: str) -> str:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")
    return video_id.split("_", 1)[0]


def canonical_video_artifact_path(
    root: Path,
    video_id: str,
    suffix: str,
) -> Path:
    return canonical_video_directory(root, video_id) / f"{video_id}{suffix}"


def canonical_video_directory(root: Path, video_id: str) -> Path:
    return Path(root) / video_group(video_id) / video_id


def resolve_video_artifact(
    root: Path,
    video_id: str,
    suffix: str,
    video_groups=None,
) -> Path:
    root = Path(root)
    if video_groups is None and "KIS_VIDEO_GROUPS" in os.environ:
        video_groups = tuple(g.strip() for g in os.environ["KIS_VIDEO_GROUPS"].split(",") if g.strip())
    if video_groups is not None and video_group(video_id) not in video_groups:
        raise ValueError(f"Artifact outside selected groups: {video_id}")
    canonical = canonical_video_artifact_path(root, video_id, suffix)
    candidates = []

    for path in (canonical, root / f"{video_id}{suffix}"):
        if path.is_file():
            candidates.append(path)

    candidates.extend(
        path
        for path in ((root / video_group(video_id)).rglob(f"{video_id}{suffix}")
                     if video_groups is not None else root.rglob(f"{video_id}{suffix}"))
        if path.is_file()
    )
    unique_candidates = sorted({path.resolve() for path in candidates})

    if len(unique_candidates) > 1:
        locations = ", ".join(str(path) for path in unique_candidates)
        raise RuntimeError(
            f"Duplicate artifacts for {video_id}{suffix}: {locations}"
        )
    if unique_candidates:
        return unique_candidates[0]

    return canonical


def resolve_video_image_dir(video_id: str, image_root=DATA_DIR) -> Path:
    group = video_group(video_id)
    image_root = Path(image_root)
    candidates = [
        path
        for path in (image_root / video_id, image_root / group / video_id)
        if path.is_dir()
    ]
    unique_candidates = sorted({path.resolve() for path in candidates})

    if len(unique_candidates) > 1:
        locations = ", ".join(str(path) for path in unique_candidates)
        raise RuntimeError(
            f"Duplicate image directories for {video_id}: {locations}"
        )
    if unique_candidates:
        return unique_candidates[0]

    return canonical_video_directory(image_root, video_id)


def discover_video_image_dirs(image_root=DATA_DIR, video_groups=None):
    image_root = Path(image_root)
    if video_groups is None and "KIS_VIDEO_GROUPS" in os.environ:
        video_groups = tuple(g.strip() for g in os.environ["KIS_VIDEO_GROUPS"].split(",") if g.strip())
    discovered = {}

    def register(video_id, image_dir):
        if not any(is_keyframe_image(path) and path.is_file() for path in image_dir.iterdir()):
            return
        existing = discovered.get(video_id)
        if existing is not None and existing.resolve() != image_dir.resolve():
            raise RuntimeError(
                f"Duplicate image directories for {video_id}: "
                f"{existing}, {image_dir}"
            )
        discovered.setdefault(video_id, image_dir)

    if not image_root.is_dir():
        return discovered

    for child in sorted(image_root.iterdir()):
        if (VIDEO_ID_PATTERN.fullmatch(child.name)
                and (video_groups is None or video_group(child.name) in video_groups) and child.is_dir()):
            register(child.name, child)

    for child in sorted(image_root.iterdir()):
        if (re.fullmatch(r"L\d+", child.name)
                and (video_groups is None or child.name in video_groups) and child.is_dir()):
            for subdir in sorted(child.iterdir()):
                if (subdir.is_dir() and VIDEO_ID_PATTERN.fullmatch(subdir.name)
                        and video_group(subdir.name) == child.name):
                    register(subdir.name, subdir)

    return discovered


def video_image_dir(video_id: str) -> Path:
    return resolve_video_image_dir(video_id)


def resolve_stored_image_path(image_path, data_root=DATA_DIR):
    path = Path(image_path)
    if path.is_absolute():
        return path
    parts = PurePosixPath(str(image_path).replace("\\", "/")).parts
    if parts and parts[0] == "data":
        return Path(data_root).joinpath(*parts[1:])
    return KIS_ROOT / path


def video_csv_path(video_id: str) -> Path:
    return resolve_video_artifact(MAPPING_DIR, video_id, ".csv")


def video_clip_path(video_id: str) -> Path:
    return resolve_video_artifact(CLIP_FEATURES_DIR, video_id, ".npy")


def video_caption_path(video_id: str) -> Path:
    return canonical_video_artifact_path(CAPTION_DIR, video_id, ".json")


def video_caption_index_path(video_id: str) -> Path:
    return canonical_video_artifact_path(INDEX_DIR, video_id, ".index")


def video_caption_manifest_path(video_id: str) -> Path:
    return canonical_video_artifact_path(INDEX_DIR, video_id, ".meta.json")


def video_caption_mapping_path(video_id: str) -> Path:
    return canonical_video_artifact_path(
        CAPTION_MAPPING_DIR,
        video_id,
        ".json",
    )


def video_object_dir(video_id: str) -> Path:
    return canonical_video_directory(OBJECTS_DIR, video_id)


def resolve_video_object_dir(video_id: str, objects_dir=OBJECTS_DIR) -> Path:
    objects_dir = Path(objects_dir)
    canonical = canonical_video_directory(objects_dir, video_id)
    legacy = objects_dir / video_id
    candidates = [path for path in (canonical, legacy) if path.is_dir()]
    unique_candidates = sorted({path.resolve() for path in candidates})

    if len(unique_candidates) > 1:
        locations = ", ".join(str(path) for path in unique_candidates)
        raise RuntimeError(
            f"Duplicate object directories for {video_id}: {locations}"
        )
    if unique_candidates:
        return unique_candidates[0]
    return canonical


