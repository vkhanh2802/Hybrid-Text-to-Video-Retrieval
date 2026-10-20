"""Import caption v2 and build its indexes; keyframe images are optional."""

import argparse
import json
from pathlib import Path
import re

from kis.artifact_io import atomic_write_json, publish_staged_files, temporary_path, write_json
from kis.caption_schema import load_frame_mapping, normalize_caption_record, validate_unique_caption_records, write_caption_manifest
from kis.paths import CAPTION_DIR, KIS_ROOT, MAPPING_DIR, canonical_video_artifact_path, video_group
from kis.config import CAPTION_CONFIG, load_runtime_config


def collect_caption_files(caption_dir=CAPTION_DIR):
    files = {}
    for path in sorted(Path(caption_dir).rglob("L*_V*.json")):
        if not path.is_file():
            continue
        if path.stem in files:
            raise RuntimeError(f"Duplicate caption files for {path.stem}: {files[path.stem]}, {path}")
        files[path.stem] = path
    return files


def load_normalized_captions(video_id, caption_path, mapping_dir=MAPPING_DIR):
    records = json.loads(Path(caption_path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Caption file must be a non-empty array: {caption_path}")
    mapping = load_frame_mapping(video_id, mapping_dir)
    records = [normalize_caption_record(item, video_id, mapping) for item in records]
    validate_unique_caption_records(records, video_id)
    texts = [item.get("retrieval_text") or item.get("caption", "") for item in records]
    if not any(texts):
        return records, None
    if not all(texts):
        raise ValueError(f"Some caption records have no searchable text: {caption_path}")
    return records, texts


def write_caption_artifacts(video_id, records, embeddings, project_root=KIS_ROOT, caption_subdir="caption_generator"):
    import faiss

    root = Path(project_root)
    mapping = [{"video_id": item["video_id"], "frame_id": int(item["frame_id"]),
                "keyframe_n": int(item["keyframe_n"]), "image_path": item.get("image_path", "")} for item in records]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    caption_path = canonical_video_artifact_path(root / caption_subdir, video_id, ".json")
    mapping_path = canonical_video_artifact_path(root / "caption_mapping", video_id, ".json")
    index_path = canonical_video_artifact_path(root / "index", video_id, ".index")
    staged = [(path, temporary_path(path)) for path in (caption_path, mapping_path, index_path)]
    try:
        write_json(staged[0][1], records)
        write_json(staged[1][1], mapping)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(staged[2][1]))
        publish_staged_files(staged)
    finally:
        for _, path in staged:
            path.unlink(missing_ok=True)
    manifest = write_caption_manifest(video_id, caption_path, mapping_path, index_path, len(records),
                                      manifest_path=index_path.with_suffix(".meta.json"))
    return index_path, mapping_path, manifest


def build_caption_index(video_ids=None, project_root=KIS_ROOT, device=None, batch_size=32,
                        mapping_dir=None, caption_subdir="caption_generator"):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    root = Path(project_root)
    files = collect_caption_files(root / caption_subdir)
    if video_ids is not None:
        missing = sorted(set(video_ids) - set(files))
        if missing:
            raise FileNotFoundError(f"Missing caption files: {', '.join(missing)}")
        files = {video_id: files[video_id] for video_id in sorted(set(video_ids))}
    if not files:
        raise FileNotFoundError(f"No per-video captions found in {root / caption_subdir}")
    model = SentenceTransformer(CAPTION_CONFIG["model"], revision=CAPTION_CONFIG["revision"], device=device)
    built, total, skipped = 0, 0, []
    for video_id, path in files.items():
        if path.resolve() != canonical_video_artifact_path(root / caption_subdir, video_id, ".json").resolve():
            raise RuntimeError(f"Legacy caption path detected for {video_id}: {path}. Import with python -m tools.ingest first.")
        records, texts = load_normalized_captions(video_id, path, mapping_dir or root / "mapping")
        if texts is None:
            skipped.append(video_id)
            continue
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embeddings)
        write_caption_artifacts(video_id, records, embeddings, root, caption_subdir)
        built += 1
        total += len(records)
        print(f"[{video_id}] indexed {len(records)} captions")
    return {"built": built, "records": total, "skipped": skipped}


def caption_sources(source, groups):
    source = Path(source)
    directories = [source / "captions", source / "caption_generator"]
    directories += [path for path in source.iterdir() if path.is_dir() and re.fullmatch(r"L\d+", path.name)]
    return sorted({
        path for directory in directories for path in directory.rglob("L*_V*.json")
        if path.is_file() and video_group(path.stem) in groups
    })


def index_is_current(root, video_id, records):
    import faiss
    from kis.text import CaptionRetriever

    paths = [canonical_video_artifact_path(root / directory, video_id, suffix) for directory, suffix in (
        ("captions", ".json"), ("caption_mapping", ".json"), ("index", ".index"), ("index", ".meta.json"),
    )]
    if not all(path.is_file() for path in paths):
        return False
    try:
        mapping = json.loads(paths[1].read_text(encoding="utf-8"))
        index = faiss.read_index(str(paths[2]))
        return CaptionRetriever._validate_triplet(video_id, records, mapping, index, paths)
    except (OSError, ValueError, RuntimeError):
        return False


def ingest(source=None, config=None, rebuild_visual=False):
    config = config or load_runtime_config(environ={})
    root = config.caption_root
    files = caption_sources(source or root, set(config.video_groups))
    if not files:
        raise FileNotFoundError(f"No caption v2 files found in {source or root}")
    prepared, originals = {}, {}
    for path in files:
        video_id = path.stem
        mapping = load_frame_mapping(video_id, MAPPING_DIR)
        records = [normalize_caption_record(item, video_id, mapping) for item in json.loads(path.read_text(encoding="utf-8"))]
        validate_unique_caption_records(records, video_id)
        if {item["keyframe_n"] for item in records} != set(mapping):
            raise ValueError(f"Incomplete caption coverage: {video_id}")
        if any(item.get("caption_generation", {}).get("backend") != "internvl_dense_v3" for item in records):
            raise ValueError(f"Expected caption v2: {video_id}")
        if video_id in prepared and prepared[video_id] != records:
            raise ValueError(f"Conflicting caption v2 files: {video_id}")
        prepared[video_id] = records
        originals.setdefault(video_id, []).append(path)

    pending = []
    for video_id, records in prepared.items():
        target = canonical_video_artifact_path(root / "captions", video_id, ".json")
        if not target.is_file() or json.loads(target.read_text(encoding="utf-8")) != records:
            atomic_write_json(target, records)
        for path in originals[video_id]:
            if path.resolve() != target.resolve() and path.resolve().is_relative_to(root):
                path.unlink()
                parent = path.parent
                while parent != root and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
        if not index_is_current(root, video_id, records):
            pending.append(video_id)

    if pending:
        build_caption_index(pending, project_root=root, mapping_dir=MAPPING_DIR,
                            device="cpu", batch_size=64, caption_subdir="captions")
    if rebuild_visual:
        from tools.build_milvus import run
        run("rebuild")
    summary = {"caption_root": str(root), "videos": len(prepared),
               "records": sum(map(len, prepared.values())), "indexes_built": len(pending),
               "caption_groups": sorted({video_group(video_id) for video_id in prepared}),
               "visual_groups": list(config.video_groups)}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="import an extracted caption dataset")
    parser.add_argument("--rebuild-visual", action="store_true", help="replace Milvus with features through L21")
    args = parser.parse_args()
    ingest(args.source, rebuild_visual=args.rebuild_visual)
