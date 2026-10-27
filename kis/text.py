"""Lexical and semantic search over the same versioned caption corpus."""

import json
from pathlib import Path
import re
import unicodedata

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from kis.caption_schema import validate_caption_manifest
from kis.config import CAPTION_CONFIG
from kis.paths import DATASET_ROOT, MAX_VIDEO_GROUP, resolve_video_artifact, video_group


def tokenize_caption(text, mode="unicode"):
    text = str(text).lower()
    if mode == "legacy":
        return text.split()
    if mode != "unicode":
        raise ValueError("tokenizer mode must be legacy or unicode")
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, caption_data, tokenizer_mode="legacy"):
        self.caption_data = caption_data
        self.tokenizer_mode = tokenizer_mode
        self.corpus_tokens = [tokenize_caption(item.get("retrieval_text", ""), tokenizer_mode) for item in caption_data]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search_bm25(self, query, top_k=100):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = tokenize_caption(query, self.tokenizer_mode)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            if scores[idx] <= 0:
                continue
            item = self.caption_data[idx]
            result = {"video_id": item["video_id"], "frame_id": item["frame_id"],
                      "score": float(scores[idx]), "retrieval_text": item.get("retrieval_text", "")}
            result.update({field: item[field] for field in ("keyframe_n", "image_path") if item.get(field) is not None})
            results.append(result)
        return results


class CaptionRetriever:
    @staticmethod
    def _collect_artifact_map(directory, pattern, groups):
        artifacts = {}
        for path in sorted(directory.rglob(pattern)):
            if not path.is_file():
                continue
            video_id = path.name.removesuffix(".meta.json") if path.name.endswith(".meta.json") else path.stem
            if video_group(video_id) not in groups:
                continue
            if video_id in artifacts:
                raise RuntimeError(f"Duplicate artifacts for {video_id}: {artifacts[video_id]}, {path}")
            artifacts[video_id] = path
        return artifacts

    @staticmethod
    def _load_json_list(path, name):
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{name} must be a JSON array: {path}")
        return records

    @staticmethod
    def _validate_triplet(video_id, captions, mapping, index, paths):
        caption_path, mapping_path, index_path, manifest_path = paths
        if not captions:
            raise ValueError(f"Caption file is empty: {caption_path}")
        count = len(captions)
        if len(mapping) != count or index.ntotal != count:
            raise ValueError(f"Artifact count mismatch for {video_id}: captions={count}, mapping={len(mapping)}, index={index.ntotal}")
        for position, (caption, mapped) in enumerate(zip(captions, mapping)):
            identity = tuple(caption.get(key) for key in ("video_id", "frame_id", "keyframe_n"))
            if identity[0] != video_id or identity != tuple(mapped.get(key) for key in ("video_id", "frame_id", "keyframe_n")):
                raise ValueError(f"Artifact identity mismatch for {video_id} at row {position}: {caption_path}, {mapping_path}")
        validate_caption_manifest(video_id, caption_path, mapping_path, index_path, manifest_path, index, count)
        return any(item.get("retrieval_text") or item.get("caption") for item in captions)

    def __init__(self, video_id=None, json_path=None, index_path=None, mapping_path=None,
                 video_groups=None, artifact_root=None, caption_subdir="caption_generator"):
        groups = set(video_groups or (f"L{n:02d}" for n in range(1, MAX_VIDEO_GROUP + 1)))
        if any(int(video_group(f"{group}_V001")[1:]) > MAX_VIDEO_GROUP for group in groups):
            raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}")
        if video_id and int(video_group(video_id)[1:]) > MAX_VIDEO_GROUP:
            raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}: {video_id}")
        root = Path(artifact_root) if artifact_root is not None else DATASET_ROOT
        specs = ((root / caption_subdir, ".json"), (root / "caption_mapping", ".json"),
                 (root / "index", ".index"), (root / "index", ".meta.json"))
        single_video = video_id is not None or json_path is not None
        if video_id is not None:
            paths_by_video = {video_id: tuple(resolve_video_artifact(directory, video_id, suffix) for directory, suffix in specs)}
        elif json_path is not None:
            if index_path is None or mapping_path is None:
                raise ValueError("Caption paths must be provided")
            records = self._load_json_list(json_path, "Caption artifact")
            if not records:
                raise ValueError(f"Caption file is empty: {json_path}")
            vid = records[0].get("video_id")
            if int(video_group(vid)[1:]) > MAX_VIDEO_GROUP:
                raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}: {vid}")
            paths_by_video = {vid: (json_path, mapping_path, index_path, Path(index_path).with_name(f"{vid}.meta.json"))}
        else:
            maps = [self._collect_artifact_map(directory, f"L*_V*{suffix}", groups) for directory, suffix in specs]
            ids = set(maps[0])
            if not ids or any(set(artifacts) != ids for artifacts in maps[1:]):
                raise FileNotFoundError("Incomplete caption artifacts: captions, mappings, indexes and manifests must cover the same videos")
            paths_by_video = {vid: tuple(artifacts[vid] for artifacts in maps) for vid in sorted(ids)}

        self.video_id = video_id
        self.caption_data, self.mapping = [], []
        self.index = None
        loaded_paths = []
        for vid, paths in paths_by_video.items():
            captions = self._load_json_list(paths[0], "Caption artifact")
            mapped = self._load_json_list(paths[1], "Caption mapping")
            index = faiss.read_index(str(paths[2]))
            if not self._validate_triplet(vid, captions, mapped, index, paths):
                if single_video:
                    raise ValueError(f"Caption artifact has no searchable text: {paths[0]}")
                continue
            if single_video:
                self.index = index
            else:
                if self.index is None:
                    self.index = faiss.IndexFlatIP(index.d)
                self.index.add(index.reconstruct_n(0, index.ntotal))
            self.caption_data.extend(captions)
            self.mapping.extend(mapped)
            loaded_paths.append(paths)
        if self.index is None:
            raise FileNotFoundError("No searchable caption indexes found")
        for position, attribute in enumerate(("json_path", "mapping_path", "index_path", "manifest_path")):
            setattr(self, attribute, loaded_paths[0][position] if single_video else [paths[position] for paths in loaded_paths])
        self._caption_by_identity = {(item["video_id"], int(item["frame_id"])): item for item in self.caption_data}
        self.model = SentenceTransformer(CAPTION_CONFIG["model"], revision=CAPTION_CONFIG["revision"], device="cpu")
        print(f"[Caption] Loaded {self.index.ntotal} caption vectors")

    def get_caption(self, video_id, frame_id):
        return self._caption_by_identity.get((video_id, int(frame_id)))

    def encode_query(self, query):
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def search(self, query, top_k=50):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores, ids = self.index.search(self.encode_query(query), min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.mapping):
                continue
            item, caption = self.mapping[idx], self.caption_data[idx]
            result = {"video_id": item["video_id"], "frame_id": int(item["frame_id"]), "score": float(score),
                      "retrieval_text": caption.get("retrieval_text", caption.get("caption", ""))}
            result.update({key: item[key] for key in ("keyframe_n", "image_path") if item.get(key) is not None})
            results.append(result)
        return results

    def search_with_text(self, query, top_k=20):
        return [{key: item[key] for key in ("video_id", "frame_id", "score", "retrieval_text")}
                for item in self.search(query, top_k)]
