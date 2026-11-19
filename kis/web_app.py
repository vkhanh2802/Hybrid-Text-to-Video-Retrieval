import csv
import hashlib
import logging
import math
import os

import numpy as np
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Literal
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr

from kis.paths import (DATA_DIR, KIS_ROOT, MAPPING_DIR, VIDEO_ID_PATTERN, is_keyframe_image, local_image_settings,
                       resolve_video_artifact, video_clip_path, video_image_dir)
from kis.caption_schema import load_frame_mapping
from kis.solve_kis import _resolve_candidate_image_path, solve_kis, rerank_candidates_with_vlm
from kis.dres import MAX_QUERY_TEXT_CHARS, MAX_VIDEO_DESCRIPTION_CHARS
from kis.ranking import get_search_profile
from kis.config import load_runtime_config, load_caption_runtime
from kis.structure_query import MAX_QUERY_CHARS, create_structured_query
from kis.llm import PROVIDER_PRESETS, load_query_api_config
from kis.text_reranker import LazyCrossEncoder
from kis.ocr import OCRRetriever
from kis.result_pool import WEB_RESULT_LIMIT, require_frame_count
from kis.video_query import (
    MAX_VIDEO_BYTES, VIDEO_MIME_TYPES, VIDEO_TEXT_WEIGHT, decode_video_query,
    retrieve_video_query, fuse_video_text_results, video_query_status,
)
from kis.dres_api import create_dres_router
from kis.planner_fallback import load_fallback_config, plan_query


WEB_DIR = KIS_ROOT / "web"
logger = logging.getLogger(__name__)
WEB_LLM_LIMIT = 20
SIMILAR_RESULT_LIMIT = 30  # image-to-image: only the nearest frames


class QueryAPISettings(BaseModel):
    model_config = {"extra": "forbid"}
    provider: str | None = Field(default=None, min_length=1, max_length=64)
    protocol: Literal["openai", "gemini", "anthropic"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=256)
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    response_format: Literal["auto", "json_object", "json_schema", "text"] | None = None

    def overrides(self):
        values = self.model_dump(exclude_none=True)
        if self.api_key is not None:
            values["api_key"] = self.api_key.get_secret_value()
        return values


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    llm: QueryAPISettings | None = None
    temporal: bool | None = None
    ocr_text: str | None = Field(default=None, min_length=1, max_length=256)
    ocr_region: Literal["all", "content", "overlay", "unknown"] | None = None


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def serialize_search_results(results, object_lookup, image_resolver=None):
    image_resolver = image_resolver or _resolve_candidate_image_path
    serialized = []

    for rank, candidate in enumerate(results, start=1):
        image_path = None
        try:
            image_path = image_resolver(candidate, object_lookup)
        except (OSError, RuntimeError, ValueError):
            pass

        keyframe_n = candidate.get("keyframe_n")
        if keyframe_n is None and image_path is not None:
            try:
                keyframe_n = int(Path(image_path).stem)
            except ValueError:
                pass

        video_id = str(candidate["video_id"])
        frame_id = int(candidate["frame_id"])
        keyframe_n = int(keyframe_n) if keyframe_n is not None else None
        image_url = (
            f"/api/images/{video_id}/{keyframe_n}"
            if image_path is not None and keyframe_n is not None
            else None
        )

        serialized.append(
            {
                "rank": rank,
                "video_id": video_id,
                "image_number": keyframe_n,
                "frame_id": frame_id,
                "image_url": image_url,
                "caption": str(candidate.get("retrieval_text") or ""),
                "caption_raw": str(candidate.get("caption") or ""),
                "evidence": candidate.get("evidence", {}),
                "score": _finite_float(candidate.get("final_score")),
                "fusion_score": _finite_float(candidate.get("fusion_score")),
                "visual_score": _finite_float(candidate.get("visual_score")),
                "bm25_score": _finite_float(candidate.get("bm25_score")),
                "semantic_score": _finite_float(candidate.get("semantic_score")),
                "object_score": _finite_float(candidate.get("object_score")),
                "retrieval_score": _finite_float(candidate.get("score")),
                "text_ranking_score": _finite_float(candidate.get("text_ranking_score")),
                "text_logit": _finite_float(candidate.get("text_logit")),
                "text_rerank_status": candidate.get("text_rerank_status", "disabled"),
                "ocr": candidate.get("ocr"),
                "vlm_score": _finite_float(candidate.get("vlm_raw_score")),
                "vlm_error": candidate.get("vlm_error"),
                "vlm_status": candidate.get("vlm_status", "disabled"),
                "vlm_ocr_used": candidate.get("vlm_ocr_used", False),
                "temporal": serialize_temporal_evidence(candidate.get("temporal"), object_lookup, image_resolver),
            }
        )

    return serialized


def serialize_temporal_evidence(temporal, object_lookup, image_resolver):
    if temporal is None:
        return None
    events = []
    for event in temporal["events"]:
        item = dict(event, image_url=None)
        if event["frame_id"] is not None:
            try:
                path = image_resolver(dict(event), object_lookup)
                if path is not None and event.get("keyframe_n") is not None:
                    item["image_url"] = f"/api/images/{event['video_id']}/{int(event['keyframe_n'])}"
            except (OSError, RuntimeError, ValueError):
                pass
        events.append(item)
    return dict(temporal, events=events)


@lru_cache(maxsize=4096)
def mapped_keyframes(video_id):
    try:
        return frozenset(load_frame_mapping(video_id))
    except (OSError, ValueError, RuntimeError):
        return frozenset()


def frame_vector(video_id, frame_id):
    """Stored visual vector of one frame: row i of the features NPY is row i of the mapping CSV."""
    with resolve_video_artifact(MAPPING_DIR, video_id, ".csv").open(encoding="utf-8", newline="") as handle:
        rows = [i for i, row in enumerate(csv.DictReader(handle)) if int(row["frame_idx"]) == frame_id]
    if len(rows) != 1:
        raise ValueError("Frame không có (hoặc trùng) trong mapping; không tìm tương tự")
    vector = np.asarray(np.load(video_clip_path(video_id), mmap_mode="r", allow_pickle=False)[rows[0]], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Vector của frame không hợp lệ")
    return vector


@lru_cache(maxsize=4096)
def find_keyframe_image(video_id, image_number):
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        return None
    if image_number < 0 or image_number not in mapped_keyframes(video_id):
        return None

    image_dir = video_image_dir(video_id)
    if not image_dir.is_dir():
        return None

    for image_path in image_dir.iterdir():
        if is_keyframe_image(image_path, image_number) and image_path.is_file():
            return image_path
    return None


class LazyVLMReranker:
    """Do not load Qwen for temporal queries (or candidates without images)."""

    supports_ocr = True

    def __init__(self):
        self.model = None
        self.load_error = None

    def _load(self):
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        if self.model is None:
            try:
                from kis.vlm_reranker import VLMReranker
                import torch
                threads = torch.get_num_threads()
                try:
                    # The Windows loader hit a native torch_cpu.dll crash under
                    # concurrent CPU weight preparation. Serialize load only;
                    # restore inference threads even when model loading fails.
                    torch.set_num_threads(1)
                    self.model = VLMReranker()
                finally:
                    torch.set_num_threads(threads)
            except Exception as exc:
                self.load_error = str(exc)
                raise
        return self.model

    @property
    def supports_batch(self):
        return self.model is None or getattr(self.model, "supports_batch", False) is True

    @property
    def last_batch_stats(self):
        value = getattr(self.model, "last_batch_stats", {})
        return value if isinstance(value, dict) else {}

    @property
    def last_batch_errors(self):
        value = getattr(self.model, "last_batch_errors", [])
        return value if isinstance(value, list) else []

    def score_batch(self, image_paths, query, captions=None, *, ocrs=None):
        model = self._load()
        if ocrs is not None and any(ocr is not None for ocr in ocrs) and getattr(model, "supports_ocr", False) is not True:
            raise ValueError("VLM scorer does not support separate OCR evidence")
        return model.score_batch(image_paths, query, captions, ocrs=ocrs)

    def score(self, image_path, query, caption=None, *, ocr=None):
        self._load()
        if ocr is not None:
            if getattr(self.model, "supports_ocr", False) is not True:
                raise ValueError("VLM scorer does not support separate OCR evidence")
            return self.model.score(image_path, query, caption=caption, ocr=ocr)
        return self.model.score(image_path, query, caption=caption)


class SearchService:
    def __init__(self):
        self._operation_lock = Lock()
        self.state = "cold"
        self.last_error = None
        self.runtime_config = load_runtime_config()
        self.text_reranker = LazyCrossEncoder(self.runtime_config.text_reranker)
        self.ocr_retriever = OCRRetriever(self.runtime_config.ocr_index or self.runtime_config.caption_root / "ocr/index.sqlite3",
                                          self.runtime_config.video_groups)
        if self.runtime_config.ocr_format == "public_json":
            from kis.public_ocr import PublicOCRRetriever

            self.ocr_retriever = PublicOCRRetriever(self.runtime_config.ocr_root, self.runtime_config.video_groups)
            if self.runtime_config.ocr.mode != "off":
                self.ocr_retriever.validate_config(self.runtime_config.ocr)
        self.profile = get_search_profile(self.runtime_config.retrieval_profile)
        self.caption_artifact_root = self.runtime_config.caption_root
        self.video_groups = self.runtime_config.video_groups
        self.vlm_enabled = os.environ.get("KIS_DISABLE_VLM", "").lower() not in {
            "1",
            "true",
            "yes",
        }
        self.query_structurer = None
        self.last_query_planner = None
        self.visual_encoder = None
        self.visual_retriever = None
        self.caption_retriever = None
        self.bm25_retriever = None
        self.object_lookup = None
        self.vlm_reranker = None
        self.image_source = None
        self.remote_worker = None
        from kis.remote_worker import RemoteWorker, worker_settings
        remote = worker_settings()
        if remote:
            self.remote_worker = RemoteWorker(remote["url"], remote["token"], self.video_groups)
            self.image_source = self.remote_worker
        if self.runtime_config.ocr_format == "school_snapshot":
            if self.remote_worker is None:
                raise ValueError("school_snapshot requires the remote school CPU worker")
            from kis.public_ocr import PublicOCRRetriever
            PublicOCRRetriever.validate_config(self.ocr_retriever, self.runtime_config.ocr)
        image_settings = local_image_settings()
        folder_id = os.environ.get("KIS_DRIVE_FOLDER_ID") or (
            image_settings.get("drive_folder_id") if image_settings.get("image_source") == "drive_api" else None)
        if folder_id and self.remote_worker is None:
            try:
                from kis.local_images import DriveImages
            except ModuleNotFoundError as exc:
                raise RuntimeError("Direct Drive images require the machine-local kis/local_images.py module") from exc
            self.image_source = DriveImages(folder_id, credentials_file=image_settings.get("credentials_file"))

    def _vlm_batch_stats(self):
        value = getattr(self.vlm_reranker, "last_batch_stats", {})
        return value if isinstance(value, dict) else {}

    def _ocr_status(self):
        if self.remote_worker is not None:
            info = self.remote_worker.info or {}
            return {**self.runtime_config.ocr.public(), **info.get("ocr", {
                "available": False, "loaded": False, "status": "awaiting_worker_handshake"}),
                "execution": "school_cpu"}
        return {**self.runtime_config.ocr.public(), **self.ocr_retriever.status()}

    def status(self):
        try:
            query_api = load_query_api_config().public()
        except (OSError, TypeError, ValueError):
            query_api = {"configured": False, "error": "Invalid query API configuration"}
        try:
            fallback_status = load_fallback_config().public()
        except (OSError, ValueError):
            fallback_status = {"enabled": False, "error": "Invalid planner fallback configuration"}
        return {
            "state": self.state,
            "input_modes": ["text", "video"],
            "result_limit": WEB_RESULT_LIMIT,
            "llm_scoring_limit": WEB_LLM_LIMIT,
            "frontend_input_source": "dres_official",
            "video_input": video_query_status(),
            "vlm_enabled": self.vlm_enabled,
            "last_error": self.last_error,
            "retrieval_profile": self.profile.name,
            "temporal": self.runtime_config.temporal.public(),
            "ocr": self._ocr_status(),
            "text_reranker": dict(self.runtime_config.text_reranker.public(),
                                  loaded=self.text_reranker.model is not None,
                                  load_error=self.text_reranker.load_error),
            "caption_artifact_root": str(self.caption_artifact_root),
            "video_groups": list(self.video_groups),
            "caption_backend": "internvl_dense_v3",
            "image_source": ("school_worker" if self.remote_worker is not None else
                             "drive_api" if self.image_source is not None else "local_files"),
            "retrieval_backend": "school_worker" if self.remote_worker is not None else "local_milvus",
            "worker": self.remote_worker.info if self.remote_worker is not None else None,
            "image_root": None if self.image_source is not None else str(DATA_DIR),
            "temporary_image_cleanup": self.image_source is not None,
            "query_planner_configured": query_api["configured"],
            "query_api": query_api,
            "query_planner_fallback": fallback_status,
            "last_query_planner": self.last_query_planner,
            "caption_vectors": self.caption_retriever.index.ntotal if self.caption_retriever is not None else None,
            "caption_groups": (self.remote_worker.info.get("caption_groups", []) if self.remote_worker is not None and self.remote_worker.info
                               else sorted({item["video_id"].split("_")[0] for item in self.caption_retriever.caption_data}) if self.caption_retriever is not None else []),
        }

    def _load(self):
        if self.state == "ready":
            return

        self.state = "loading"
        self.last_error = None
        try:
            if self.remote_worker is not None:
                from kis.config import MODEL_MANIFEST
                self.remote_worker.connect(self.runtime_config, MODEL_MANIFEST)
                self.visual_encoder = self.visual_retriever = self.caption_retriever = self.object_lookup = self.remote_worker
                self.vlm_reranker = LazyVLMReranker() if self.vlm_enabled else None
                self.state = "ready"
                return
            from kis.text import BM25Retriever
            from kis.objects import ObjectMetadataLookup
            from kis.paths import MAPPING_DIR, OBJECTS_DIR
            from kis.visual import SiglipRetriever, SiglipEncoder
            caption_retriever, video_ids = load_caption_runtime(self.runtime_config)
            if self.runtime_config.ocr_format == "public_json":
                self.ocr_retriever.set_video_ids(video_ids)
            visual_encoder = SiglipEncoder(device="cpu")
            visual_retriever = SiglipRetriever(video_groups=self.video_groups, nprobe=self.profile.nprobe)
            bm25_retriever = BM25Retriever(caption_retriever.caption_data, tokenizer_mode=self.profile.tokenizer_mode)
            object_lookup = ObjectMetadataLookup(
                mapping_dir=MAPPING_DIR,
                objects_dir=OBJECTS_DIR,
            )
            vlm_reranker = None
            if self.vlm_enabled:
                vlm_reranker = LazyVLMReranker()

            self.visual_encoder = visual_encoder
            self.visual_retriever = visual_retriever
            self.caption_retriever = caption_retriever
            self.bm25_retriever = bm25_retriever
            self.object_lookup = object_lookup
            self.vlm_reranker = vlm_reranker
            self.state = "ready"
        except Exception as exc:
            self.state = "failed"
            self.last_error = str(exc)
            raise

    def similar(self, video_id, frame_id):
        """Image-to-image: the chosen frame's own stored visual vector as the query."""
        if not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id.split("_")[0] not in self.video_groups:
            raise ValueError("Video nằm ngoài phạm vi tìm kiếm")
        started_at = perf_counter()
        vector = frame_vector(video_id, frame_id)
        with self._operation_lock:
            self._load()
            results = self.visual_retriever.search(vector, top_k=SIMILAR_RESULT_LIMIT)
            for row in results:
                row["final_score"] = row["visual_score"] = row["score"]
                metadata = self.caption_retriever.get_caption(row["video_id"], row["frame_id"]) or {}
                for field in ("retrieval_text", "caption", "keyframe_n", "image_path"):
                    if row.get(field) is None and field in metadata:
                        row[field] = metadata[field]
            resolver = self.image_source.reference if self.image_source is not None else None
            serialized = serialize_search_results(results, self.object_lookup, image_resolver=resolver)
        return {"input_type": "similar", "source": {"video_id": video_id, "frame_id": frame_id},
                "raw_query": f"Tương tự {video_id} · frame {frame_id}", "query_variants": [],
                "elapsed_seconds": round(perf_counter() - started_at, 2), "vlm_enabled": False,
                "result_limit": SIMILAR_RESULT_LIMIT, "temporal_results": [], "results": serialized}

    def search(self, raw_query, llm=None, temporal=None, ocr_text=None, ocr_region=None):
        started_at = perf_counter()
        with self._operation_lock:
            results, structured_query, diagnostics, query_api = self._retrieve_text(
                raw_query, llm, temporal, ocr_text, ocr_region)
            resolver = self.image_source.reference if self.image_source is not None else None
            serialized = serialize_search_results(results, self.object_lookup, image_resolver=resolver)
            require_frame_count(serialized, WEB_RESULT_LIMIT)
            temporal_results = serialize_search_results(
                diagnostics.get("temporal_results", []), self.object_lookup, image_resolver=resolver)

        return {
            "input_type": "text",
            "query_id": structured_query["query_id"],
            "raw_query": structured_query["raw_query"],
            "query_variants": structured_query["query_variants"],
            "elapsed_seconds": round(perf_counter() - started_at, 2),
            "vlm_enabled": self.vlm_enabled,
            "result_limit": WEB_RESULT_LIMIT,
            "llm_scoring_limit": WEB_LLM_LIMIT,
            "llm_scored_count": sum(item["vlm_status"] == "reranked" and item["vlm_score"] is not None for item in serialized),
            "retrieval_profile": self.profile.name,
            "caption_artifact_root": str(self.caption_artifact_root),
            "video_groups": list(self.video_groups),
            "caption_backend": "internvl_dense_v3",
            "query_api": {"provider": query_api.provider, "protocol": query_api.protocol, "model": query_api.model},
            "query_planner": diagnostics.get("query_planner"),
            "stage_seconds": diagnostics.get("stage_seconds", {}),
            "vlm_batch": diagnostics.get("vlm_batch", {}),
            "temporal": diagnostics.get("temporal", {}),
            "temporal_results": temporal_results,
            "result_pool": diagnostics.get("result_pool", {"status": "complete", "target": WEB_RESULT_LIMIT}),
            "text_reranker": diagnostics.get("text_reranker", {}),
            "ocr": diagnostics.get("ocr", {}),
            "fusion_method": self.profile.policy.cross_fusion,
            "results": serialized,
        }

    def _retrieve_text(self, raw_query, llm=None, temporal=None, ocr_text=None, ocr_region=None,
                       *, rerank_top_k=None):
        """Shared text pipeline. Caller holds _operation_lock; hybrid defers VLM."""
        if ocr_text is not None and (not isinstance(ocr_text, str) or not ocr_text.strip() or len(ocr_text) > 256):
            raise ValueError("ocr_text must be nonempty and <=256 characters")
        ocr_config = self.runtime_config.ocr if ocr_region is None else replace(self.runtime_config.ocr, region=ocr_region)
        if self.runtime_config.ocr_format == "public_json" and ocr_config.mode != "off":
            self.ocr_retriever.validate_config(ocr_config)
        if self.runtime_config.ocr_format == "school_snapshot" and ocr_config.mode != "off":
            from kis.public_ocr import PublicOCRRetriever
            PublicOCRRetriever.validate_config(self.ocr_retriever, ocr_config)
        from kis.structure_query import create_query_structurer

        query_api = load_query_api_config(overrides=llm)
        transient = llm is not None
        query_structurer = None if transient else self.query_structurer
        if query_structurer is not None:
            query_api = getattr(query_structurer, "config", query_api)

        def primary_plan():
            nonlocal query_structurer
            if query_structurer is None:
                query_structurer = create_query_structurer(query_api)
                if not transient:
                    self.query_structurer = query_structurer
            return create_structured_query(raw_query, output_path=None, generator=query_structurer)

        try:
            load_started = perf_counter()
            self._load()
            load_seconds = perf_counter() - load_started
            planner_started = perf_counter()
            structured_query, used_api, planner_info = plan_query(
                raw_query, query_api, primary_plan, allow_fallback=not transient)
            query_api = used_api
            self.last_query_planner = planner_info
            planner_seconds = perf_counter() - planner_started
            if ocr_text is not None:
                structured_query["ocr_text"] = ocr_text.strip()
                structured_query["needs_ocr"] = True
        finally:
            if transient and hasattr(query_structurer, "close"):
                query_structurer.close()
        diagnostics = {}
        if self.remote_worker is not None:
            rpc_started = perf_counter()
            remote_result = self.remote_worker.retrieve(
                structured_query, temporal=temporal, ocr_region=ocr_region,
                limit=None if self.vlm_enabled else WEB_RESULT_LIMIT)
            rpc_seconds = perf_counter() - rpc_started
            diagnostics = remote_result["diagnostics"]
            diagnostics["query_planner"] = planner_info
            stages = diagnostics.setdefault("stage_seconds", {})
            # OCR may overlap retrieval streams: summing CPU stages would
            # double-count that work. New workers provide measured wall time.
            stages["remote_transport_overhead"] = max(0.0, rpc_seconds - diagnostics.get(
                'retrieval_wall_seconds', sum(stages.values())))
            rerank_started = perf_counter()
            budget = min(WEB_LLM_LIMIT, self.profile.rerank_top_k) if rerank_top_k is None else min(WEB_LLM_LIMIT, rerank_top_k)
            results = rerank_candidates_with_vlm(
                remote_result["candidates"], structured_query["raw_query"],
                self.vlm_reranker if budget else None, self.object_lookup, top_k=budget,
                vlm_weight=0.5, verified_first=self.profile.policy.verified_first,
                image_source=self.image_source,
                base_score_field="text_ranking_score" if self.runtime_config.text_reranker.mode == "rerank" else "score",
                ocr_query=structured_query)[:WEB_RESULT_LIMIT]
            if self.remote_worker.transport_failed:
                raise RuntimeError("School worker connection failed during image reranking; search not completed")
            require_frame_count(results, WEB_RESULT_LIMIT)
            stages.update(vlm=perf_counter() - rerank_started, runtime_load=load_seconds, query_planning=planner_seconds)
            diagnostics["vlm_batch"] = self._vlm_batch_stats() if budget else {}
            return results, structured_query, diagnostics, query_api
        results = solve_kis(
            structured_query=structured_query,
            visual_retriever=self.visual_retriever,
            visual_encoder=self.visual_encoder,
            text_bm25_retriever=self.bm25_retriever,
            text_semantic_retriever=self.caption_retriever,
            object_lookup=self.object_lookup,
            vlm_reranker=self.vlm_reranker if rerank_top_k != 0 else None,
            retrieval_top_k=WEB_RESULT_LIMIT,
            rerank_top_k=(min(WEB_LLM_LIMIT, self.profile.rerank_top_k)
                          if rerank_top_k is None else rerank_top_k),
            result_top_k=WEB_RESULT_LIMIT,
            temporal_as_auxiliary=True,
            require_full_results=True,
            vlm_weight=0.5,
            policy=self.profile.policy,
            diagnostics=diagnostics,
            image_source=self.image_source,
            text_reranker_config=self.runtime_config.text_reranker,
            text_reranker=self.text_reranker,
            ocr_config=ocr_config,
            ocr_retriever=self.ocr_retriever,
            temporal_config=(self.runtime_config.temporal if temporal is None
                             else replace(self.runtime_config.temporal, enabled=temporal)),
        )
        diagnostics.setdefault("stage_seconds", {}).update(
            runtime_load=load_seconds, query_planning=planner_seconds)
        diagnostics["query_planner"] = planner_info
        return results, structured_query, diagnostics, query_api

    def search_video(self, data, *, official=False, description="", hint_text=""):
        if not isinstance(description, str) or len(description) > MAX_VIDEO_DESCRIPTION_CHARS:
            raise ValueError("Text bổ sung cho video không được quá 2000 ký tự")
        description = description.strip()
        if not isinstance(hint_text, str) or len(hint_text) > MAX_QUERY_TEXT_CHARS:
            raise ValueError("Text đề DRES vượt giới hạn")
        hint_text = hint_text.strip()
        text_query = "\n".join(part for part in (hint_text[:MAX_QUERY_TEXT_CHARS], description) if part)
        if text_query and not official:
            raise ValueError("Text bổ sung chỉ dùng với video đề DRES chính thức")
        if len(text_query) > MAX_QUERY_CHARS:
            raise ValueError("Tổng text đề DRES và text bổ sung không được quá 8000 ký tự; hãy rút ngắn text bổ sung")
        started_at = perf_counter()
        structured_query, diagnostics, query_api = None, {}, None
        text_seconds = fusion_seconds = 0.0
        with self._operation_lock:
            decode_started = perf_counter()
            decoder = decode_video_query(data, official=True) if official else decode_video_query(data)
            with decoder as clip:
                decode_seconds = perf_counter() - decode_started
                try:
                    load_started = perf_counter()
                    self._load()
                    load_seconds = perf_counter() - load_started
                except Exception as exc:
                    logger.exception("Cannot initialize video search runtime")
                    raise RuntimeError("Video search unavailable; check Docker/Milvus, caption artifacts and model cache") from exc
                retrieval_started = perf_counter()
                video_pool = {}
                results = retrieve_video_query(clip.images, self.visual_encoder,
                    self.visual_retriever, self.caption_retriever, top_k=WEB_RESULT_LIMIT,
                    require_full_results=True, pool_diagnostics=video_pool)
                retrieval_seconds = perf_counter() - retrieval_started
                if text_query:
                    text_started = perf_counter()
                    text_results, structured_query, diagnostics, query_api = self._retrieve_text(
                        text_query, rerank_top_k=0)
                    text_seconds = perf_counter() - text_started
                    fusion_started = perf_counter()
                    results = fuse_video_text_results(text_results, results, top_k=WEB_RESULT_LIMIT)
                    fusion_seconds = perf_counter() - fusion_started
                    # Video-only fused candidates may not be in the text pool.
                    # Attach OCR AFTER 70/30 fusion, BEFORE the sole Qwen pass.
                    if (self.remote_worker is not None and self.runtime_config.ocr_format == "school_snapshot"
                            and self.runtime_config.ocr.mode == "retrieve"):
                        evidence_started = perf_counter()
                        diagnostics.setdefault("ocr", {})["post_fusion_evidence"] = self.remote_worker.attach_ocr(
                            results[:WEB_LLM_LIMIT])
                        diagnostics["ocr"]["post_fusion_evidence_rpc_seconds"] = perf_counter() - evidence_started
                rerank_started = perf_counter()
                if text_query:
                    if self.vlm_enabled and self.vlm_reranker is not None:
                        results = rerank_candidates_with_vlm(results, text_query, self.vlm_reranker,
                            self.object_lookup, top_k=WEB_LLM_LIMIT, image_source=self.image_source)
                    else:
                        for candidate in results:
                            candidate["vlm_status"] = "disabled"
                rerank_seconds = perf_counter() - rerank_started
                if self.remote_worker is not None and self.remote_worker.transport_failed:
                    raise RuntimeError("School worker connection failed during video search; search not completed")
                resolver = self.image_source.reference if self.image_source is not None else None
                serialized = serialize_search_results(results, self.object_lookup, image_resolver=resolver)
                require_frame_count(serialized, WEB_RESULT_LIMIT)
                temporal_results = serialize_search_results(
                    diagnostics.get("temporal_results", []), self.object_lookup, image_resolver=resolver)
                video = dict(clip.metadata)
                scored_count = sum(item["vlm_status"] == "reranked" and item["vlm_score"] is not None for item in serialized)
                guidance_status = ("not_requested" if not text_query else "disabled" if not self.vlm_enabled
                                   or self.vlm_reranker is None else "reranked" if scored_count else "no_valid_scores")
        guidance = {"status": guidance_status,
                    "role": "full_text_pipeline_and_video_rank_fusion",
                    "retrieval_changed": bool(text_query),
                    "text_fusion": {"enabled": bool(text_query),
                                    "mode": "weighted_rrf" if text_query else "not_requested",
                                    "k": 60 if text_query else None,
                                    "text_weight": VIDEO_TEXT_WEIGHT if text_query else 0.0,
                                    "video_weight": 1 - VIDEO_TEXT_WEIGHT if text_query else 1.0,
                                    "text_query_chars": len(text_query)}}
        return {
            "input_type": "video", "query_id": "v_" + hashlib.sha256(
                data + (b"\0" + text_query.encode() if text_query else b"")).hexdigest()[:12],
            "raw_query": text_query,
            "query_variants": structured_query["query_variants"] if structured_query else [], "video_query": video,
            "supplemental_description": description, "hint_text": hint_text,
            "combined_text_query": text_query,
            "video_text_guidance": guidance,
            "elapsed_seconds": round(perf_counter() - started_at, 2),
            "vlm_enabled": bool(text_query and self.vlm_enabled),
            "retrieval_profile": "video_full_text_rrf_0.7" if text_query else "video_visual",
            "text_retrieval_profile": self.profile.name if text_query else None,
            "result_limit": WEB_RESULT_LIMIT, "llm_scoring_limit": WEB_LLM_LIMIT if text_query else 0,
            "result_pool": {"status": "complete", "target": WEB_RESULT_LIMIT, "returned": len(serialized),
                            "video_expansion": video_pool, "text": diagnostics.get("result_pool")},
            "llm_scored_count": scored_count,
            "caption_artifact_root": str(self.caption_artifact_root),
            "video_groups": list(self.video_groups), "caption_backend": "internvl_dense_v3",
            "query_api": ({"provider": query_api.provider, "protocol": query_api.protocol, "model": query_api.model}
                           if query_api else None),
            "query_planner": diagnostics.get("query_planner"),
            "fusion_method": ("video_text_pipeline_weighted_rrf" if text_query else "visual_mean_pool")
                              + ("+text_vlm_rerank" if scored_count else ""),
            "stage_seconds": {"video_decode": decode_seconds, "visual": retrieval_seconds,
                              "runtime_load": load_seconds, "text_pipeline": text_seconds,
                              "hybrid_fusion": fusion_seconds, "vlm": rerank_seconds},
            "text_stage_seconds": diagnostics.get("stage_seconds", {}),
            "vlm_batch": self._vlm_batch_stats() if text_query else {},
            "temporal": diagnostics.get("temporal", {"status": "not_applicable", "reason": "video_appearance_query"}),
            "temporal_results": temporal_results,
            "ocr": diagnostics.get("ocr", {"status": "not_applicable", "reason": "video_appearance_query"}),
            "text_reranker": diagnostics.get("text_reranker", {"status": "not_applicable", "reason": "video_appearance_query"}),
            "results": serialized,
        }


service = SearchService()
app = FastAPI(title="KIS Framefinder", version="1.0.0")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
app.include_router(create_dres_router(lambda: service.video_groups, search_service_provider=lambda: service))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return service.status()


@app.get("/api/query-providers")
def api_query_providers():
    return {"providers": [{"provider": name, "protocol": values[0], "base_url": values[1],
                            "api_key_env": values[2]} for name, values in PROVIDER_PRESETS.items()]}


@app.post("/api/search")
async def api_search(request: SearchRequest):
    try:
        if request.ocr_text is not None or request.ocr_region is not None:
            return await run_in_threadpool(service.search, request.query,
                request.llm.overrides() if request.llm is not None else None,
                request.temporal, request.ocr_text, request.ocr_region)
        if request.temporal is not None:
            return await run_in_threadpool(service.search, request.query,
                                          request.llm.overrides() if request.llm is not None else None,
                                          request.temporal)
        if request.llm is None:
            return await run_in_threadpool(service.search, request.query)
        return await run_in_threadpool(service.search, request.query, request.llm.overrides())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc


class SimilarRequest(BaseModel):
    video_id: str = Field(pattern=r"^L\d+_V\d+$", max_length=32)
    frame_id: int = Field(strict=True, ge=0)


@app.post("/api/similar")
async def api_similar(request: SimilarRequest):
    try:
        return await run_in_threadpool(service.similar, request.video_id, request.frame_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/search/video", openapi_extra={
    "requestBody": {"required": True, "content": {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        "video/mp4": {"schema": {"type": "string", "format": "binary"}},
    }},
})
async def api_search_video(request: Request):
    """Raw binary upload: no multipart dependency or persistent upload directory."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in VIDEO_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Send raw video bytes with a video Content-Type or application/octet-stream")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdigit():
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        if int(content_length) > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=413, detail="Video file must not exceed 20 MiB")
    upload = bytearray()
    async for chunk in request.stream():
        if len(upload) + len(chunk) > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=413, detail="Video file must not exceed 20 MiB")
        upload.extend(chunk)
    if not upload:
        raise HTTPException(status_code=422, detail="Choose a nonempty video file")
    data = bytes(upload)
    del upload
    try:
        return await run_in_threadpool(service.search_video, data)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video search failed")
        raise HTTPException(status_code=500, detail="Video search failed; check server logs") from exc


@app.get("/api/keyframes/{video_id}", include_in_schema=False)
def api_keyframes(video_id: str):
    # Mapping rows for stepping the DRES selection to neighbouring keyframes.
    if not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id.split("_")[0] not in service.video_groups:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        mapping = load_frame_mapping(video_id)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc
    return {"video_id": video_id, "keyframes": [{"image_number": n, "frame_id": mapping[n]} for n in sorted(mapping)]}


@app.get("/api/images/{video_id}/{image_number}", include_in_schema=False)
def api_image(video_id: str, image_number: int):
    headers = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}
    if (not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id.split("_")[0] not in service.video_groups
            or image_number < 0 or image_number not in mapped_keyframes(video_id)):
        raise HTTPException(status_code=404, detail="Image not found")
    if service.image_source is not None:
        try:
            data, mime = service.image_source.read(video_id, image_number)
            return Response(content=data, media_type=mime, headers=headers)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Cannot read image source; check connection and access") from exc
    image_path = find_keyframe_image(video_id, image_number)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path,
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("kis.web_app:app", host="127.0.0.1", port=8000, workers=1)
