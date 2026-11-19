"""Package offline tools without datasets, model weights or secrets."""

import argparse
from pathlib import Path
import zipfile

from kis.paths import KIS_ROOT


COMMON = (
    "kis/__init__.py", "kis/artifact_io.py", "kis/paths.py", "kis/config.py", "kis/temporal.py",
    "kis/objects.py", "kis/caption_schema.py", "kis/text_reranker.py", "kis/ocr.py",
    "kis/structured_caption.py", "kis/light_caption.py", "config/models.json",
    "config/object_labels.json", "tools/__init__.py", "tools/prepare_video_data.py",
    "tools/structured_caption_generator.py",
)
FULL_SOURCES = {name: name for name in (*COMMON, "tools/kaggle_prepare.py",
    "tools/kaggle_keyframes_caption.py", "tools/kaggle_prepare_keyframes.py", "tools/audit_artifact_formats.py",
    "tools/ingest.py", "kis/visual.py")}
FULL_SOURCES.update({"requirements-kaggle.txt": "config/requirements-kaggle.txt", "KAGGLE.md": "docs/KAGGLE.md"})
FILES = tuple(FULL_SOURCES)
KEYFRAMES_CAPTION_FILES = {name: name for name in COMMON}
KEYFRAMES_CAPTION_FILES.update({
    "tools/kaggle_prepare.py": "tools/kaggle_keyframes_caption.py",
    "requirements-kaggle.txt": "config/requirements-kaggle-caption.txt",
    "KAGGLE.md": "docs/KAGGLE.md",
})


def package(destination, profile="full"):
    if profile not in {"full", "keyframes-caption"}:
        raise ValueError("Unknown Kaggle bundle profile")
    sources = FULL_SOURCES if profile == "full" else KEYFRAMES_CAPTION_FILES
    for name in sources.values():
        if not (KIS_ROOT / name).is_file():
            raise FileNotFoundError(KIS_ROOT / name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for destination_name, source_name in sources.items():
            archive.write(KIS_ROOT / source_name, f"kis_kaggle/{destination_name}")
    print(f"Created {destination.resolve()} ({profile}) with {len(sources)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "keyframes-caption"), default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or KIS_ROOT / "dist" / ("kis-kaggle-code.zip" if args.profile == "full" else "kis-kaggle-keyframes-caption.zip")
    package(destination, args.profile)
