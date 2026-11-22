"""Resident school CPU retrieval/image worker. Loopback + per-install token only."""

import argparse
import base64
from collections import Counter
from dataclasses import replace
import csv
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
from time import perf_counter

from kis.paths import VIDEO_ID_PATTERN, VISUAL_FEATURES_NAME, resolve_video_artifact
from kis.remote_worker import runtime_signature
from kis.result_pool import WEB_RESULT_LIMIT


class ExactVisualIndex:
    def __init__(self, root, groups):
        import numpy as np
        self.rows, self.image_numbers, plans = [], {}, []
        self.ambiguous = []
        for group in groups:
            paths = sorted((root / "mapping" / group).rglob(f"{group}_V*.csv"))
            if not paths:
                raise ValueError("Missing worker mapping group: " + group)
            for mapping in paths:
                with mapping.open(encoding="utf-8", newline="") as stream:
                    entries = list(csv.DictReader(stream))
                counts = Counter(int(r["frame_idx"]) for r in entries)
                keep = []
                self.image_numbers[mapping.stem] = {int(r["n"]) for r in entries}
                for i, row in enumerate(entries):
                    frame = int(row["frame_idx"])
                    if counts[frame] > 1:
                        self.ambiguous.append((mapping.stem, frame))
                        continue
                    keep.append(i)
                    self.rows.append({"video_id": mapping.stem, "frame_id": frame, "keyframe_n": int(row["n"])})
                feature = resolve_video_artifact(root / VISUAL_FEATURES_NAME, mapping.stem, ".npy", video_groups=groups)
                plans.append((feature, len(entries), keep))
        self.matrix = np.empty((len(self.rows), 768), dtype=np.float32)
        offset = 0
        for feature, count, keep in plans:
            vectors = np.load(feature, allow_pickle=False)
            if vectors.shape != (count, 768) or not np.isfinite(vectors).all():
                raise ValueError("Worker visual/mapping mismatch")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= np.finfo(np.float32).eps):
                raise ValueError("Worker zero embedding")
            self.matrix[offset:offset + len(keep)] = (vectors / norms)[keep]
            offset += len(keep)
        self.row_of = {(row["video_id"], row["frame_id"]): i for i, row in enumerate(self.rows)}

    def search(self, vector, top_k=100):
        import numpy as np
        if type(top_k) is not int or not 1 <= top_k <= 6400:
            raise ValueError("Invalid worker retrieval depth")
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (768,) or not np.isfinite(vector).all() or not math.isfinite(float(norm)) or norm <= 0:
            raise ValueError("Invalid worker visual vector")
        scores = self.matrix @ (vector / norm)
        # Partial top-k, then a stable sort of only those rows (ties stay in row order).
        top = np.arange(len(scores)) if top_k >= len(scores) else np.sort(np.argpartition(-scores, top_k - 1)[:top_k])
        indices = top[np.argsort(-scores[top], kind="stable")]
        return [dict(self.rows[int(i)], score=float(scores[i])) for i in indices]

    def similar(self, video_id, frame_id, top_k):
        """Image-to-image: the stored vector of one worker frame as the query."""
        if not isinstance(video_id, str) or type(frame_id) is not int:
            raise ValueError("Invalid similar-frame identity")
        position = self.row_of.get((video_id, frame_id))
        if position is None:
            raise FileNotFoundError("Frame not in worker visual index")
        return self.search(self.matrix[position], top_k)


def exclude_ambiguous_captions(captions, ambiguous):
    """Apply the visual index's existing ambiguity policy to text streams too.

    All source captions/indexes remain intact; only the resident retrieval view
    changes. Otherwise a newly imported alias could invalidate OCR pool joins.
    """
    import faiss
    ambiguous = set(ambiguous)
    keep = [i for i, r in enumerate(captions.caption_data) if (r['video_id'], r['frame_id']) not in ambiguous]
    removed = len(captions.caption_data) - len(keep)
    if not removed:
        return 0
    index = faiss.IndexFlatIP(captions.index.d)
    # Keep memory bounded instead of reconstructing the whole caption corpus.
    for start in range(0, len(captions.caption_data), 4096):
        stop = min(start + 4096, len(captions.caption_data))
        selected = [i - start for i in keep if start <= i < stop]
        if selected:
            index.add(captions.index.reconstruct_n(start, stop - start)[selected])
    captions.caption_data = [captions.caption_data[i] for i in keep]
    captions.mapping = [captions.mapping[i] for i in keep]
    captions.index = index
    captions._caption_by_identity = {(r['video_id'], int(r['frame_id'])): r for r in captions.caption_data}
    captions._caption_position = {(r['video_id'], int(r['frame_id'])): i for i, r in enumerate(captions.caption_data)}
    return removed


class WorkerRuntime:
    def __init__(self, dataset):
        import numpy
        import torch
        import faiss
        from kis.config import load_runtime_config, load_caption_runtime, MODEL_MANIFEST
        from kis.text import BM25Retriever
        from kis.visual import SiglipEncoder
        from kis.objects import ObjectMetadataLookup
        from kis.ranking import get_search_profile
        from kis.ocr import OCRRetriever
        from kis.text_reranker import LazyCrossEncoder
        begin = perf_counter()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        faiss.omp_set_num_threads(4)
        self.config = replace(load_runtime_config(), caption_root=Path(dataset))
        self.lock = threading.Lock()
        self.captions, _ = load_caption_runtime(self.config)
        self.encoder = SiglipEncoder(device="cpu")
        self.visual = ExactVisualIndex(Path(dataset), self.config.video_groups)
        caption_source_count = self.captions.index.ntotal
        excluded_captions = exclude_ambiguous_captions(self.captions, self.visual.ambiguous)
        self.bm25 = BM25Retriever(self.captions.caption_data, tokenizer_mode=get_search_profile(self.config.retrieval_profile).tokenizer_mode)
        self.lookup = ObjectMetadataLookup(mapping_dir=Path(dataset) / "mapping", objects_dir=Path(dataset) / "objects")
        self.text_reranker = LazyCrossEncoder(self.config.text_reranker)
        if self.config.ocr_format == "school_snapshot":
            from kis.school_ocr import SchoolOCRSnapshot
            self.ocr = SchoolOCRSnapshot(self.config.ocr_root, self.config.ocr_manifest_sha256,
                                        self.config.video_groups, dataset, self.visual.rows)
            self.ocr.validate_config(self.config.ocr)
        elif self.config.ocr_format != "strict":
            raise ValueError("Public OCR requires an explicitly deployed worker source; refusing silent change")
        else:
            self.ocr = OCRRetriever(Path(dataset) / "ocr/index.sqlite3", self.config.video_groups)
        self.dataset = Path(dataset).resolve()
        self._thumbnails = {}
        self.info = {"protocol": 1, "state": "ready", "models": MODEL_MANIFEST,
                      "result_limit": WEB_RESULT_LIMIT,
                      "runtime": runtime_signature(self.config), "caption_vectors": self.captions.index.ntotal,
                     "caption_source_vectors": caption_source_count,
                     "excluded_ambiguous_caption_rows": excluded_captions,
                     "caption_groups": sorted({row["video_id"].split("_")[0] for row in self.captions.caption_data}),
                     "visual_vectors": len(self.visual.rows), "visual_backend": "exact_npy_cosine",
                     "ambiguous_frame_ids_excluded": sorted(set(self.visual.ambiguous)),
                     "missing_image_groups": [g for g in self.config.video_groups if not (self.dataset / "data" / g).is_dir()],
                      "load_seconds": perf_counter() - begin}
        self.info["ocr"] = self.ocr.status()

    def ocr_evidence(self, payload):
        if self.config.ocr_format != "school_snapshot" or self.config.ocr.mode != "retrieve":
            raise ValueError("Snapshot OCR not enabled")
        rows = payload["candidates"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise ValueError("Invalid OCR evidence budget")
        if any(not isinstance(r, dict) or set(r) != {"video_id", "frame_id", "keyframe_n"} for r in rows):
            raise ValueError("OCR evidence expects identities only")
        started = perf_counter()
        try:
            self.ocr.attach_candidates(rows, self.config.ocr)
            status = {"status": "attached", "seconds": perf_counter() - started}
        except (OSError, ValueError, RuntimeError):
            for row in rows:
                row["ocr"] = dict(self.ocr._evidence("error"), build_id=self.config.ocr_manifest_sha256)
            status = {"status": "error", "seconds": perf_counter() - started}
        return {"candidates": rows, "status": status}

    def retrieve(self, payload):
        request_started = perf_counter()
        from kis.solve_kis import solve_kis
        from kis.ranking import get_search_profile
        structured = payload["structured_query"]
        if (not isinstance(structured, dict) or not isinstance(structured.get("raw_query"), str)
                or not 1 <= len(structured["raw_query"]) <= 8000):
            raise ValueError("Invalid full structured query")
        temporal = payload.get("temporal")
        if temporal is not None and type(temporal) is not bool:
            raise ValueError("Invalid temporal flag")
        limit = payload.get("limit")
        if limit is not None and (type(limit) is not int or not WEB_RESULT_LIMIT <= limit <= 20000):
            raise ValueError("Invalid candidate limit")
        config = self.config
        tc = config.temporal if temporal is None else replace(config.temporal, enabled=temporal)
        oc = config.ocr if payload.get("ocr_region") is None else replace(config.ocr, region=payload["ocr_region"])
        diagnostics, captured = {}, []

        class CapturePool:
            def select(self, candidates):
                captured.extend(candidates)
                return candidates

        from concurrent.futures import ThreadPoolExecutor
        from kis.ocr import search_ocr, literal_queries
        # One request still holds runtime.lock. Only the pinned snapshot's
        # read-only FTS connection is accessed by this one background thread.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ocr-retrieval') as pool:
            future = (pool.submit(search_ocr, structured, self.ocr, oc)
                      if config.ocr_format == 'school_snapshot' and oc.mode != 'off'
                      and literal_queries(structured, oc) else None)
            solve_kis(structured, self.visual, self.encoder, self.bm25, self.captions, self.lookup,
                      retrieval_top_k=WEB_RESULT_LIMIT, result_top_k=WEB_RESULT_LIMIT, rerank_top_k=0, require_full_results=True,
                      candidate_selector=CapturePool(), temporal_as_auxiliary=True, temporal_config=tc,
                      policy=get_search_profile(config.retrieval_profile).policy, diagnostics=diagnostics,
                      text_reranker_config=config.text_reranker, text_reranker=self.text_reranker,
                      ocr_config=oc, ocr_retriever=self.ocr, ocr_future=future)
        if not WEB_RESULT_LIMIT <= len(captured) <= 20000:
            raise RuntimeError("Invalid complete pre-Qwen pool")
        if limit is not None:
            del captured[limit:]  # Qwen off: the unused pool tail only costs VPN transfer.
        # Only original top20 can be sent to Qwen. Keep the full retrieval pool
        # and fusion evidence, but avoid transferring unused full OCR strings.
        for row in captured[20:]:
            row.pop('ocr', None)
        diagnostics.get('ocr', {}).update(candidate_prompt_evidence_limit=20)
        diagnostics['retrieval_wall_seconds'] = perf_counter() - request_started
        return {"candidates": captured, "diagnostics": {k: v for k, v in diagnostics.items()
                if k in {"stage_seconds", "temporal", "temporal_results", "text_reranker", "ocr", "result_pool", "cross_fusion", "retrieval_wall_seconds"}}}

    def image(self, video_id, number):
        if (not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id.split("_")[0] not in self.config.video_groups
                or number not in self.visual.image_numbers.get(video_id, set())):
            raise FileNotFoundError("Unmapped image")
        directory = self.dataset / "data" / video_id.split("_")[0] / video_id
        for suffix in (".jpg", ".jpeg", ".png"):
            path = directory / f"{number:03d}{suffix}"
            if path.is_file() and path.resolve().is_relative_to(self.dataset / "data"):
                if not 0 < path.stat().st_size <= 24 * 1024 ** 2:
                    raise ValueError("Image too large")
                return path.read_bytes(), "image/png" if suffix == ".png" else "image/jpeg"
        raise FileNotFoundError("Image absent from school dataset")

    THUMBNAIL_WIDTH = 384  # result cards are ~210 px wide; 384 stays sharp on 2x screens
    MAX_THUMBNAILS = 4096  # ~20 KB each

    def thumbnail(self, video_id, number):
        """Small JPEG for the result grid: ~8x fewer bytes over the VPN than the keyframe."""
        key = (video_id, number)
        cached = self._thumbnails.get(key)
        if cached is not None:
            return cached
        from PIL import Image
        data, _ = self.image(video_id, number)
        with Image.open(BytesIO(data)) as image:
            image.draft("RGB", (self.THUMBNAIL_WIDTH, self.THUMBNAIL_WIDTH))  # JPEG decodes at reduced scale
            image = image.convert("RGB")
            image.thumbnail((self.THUMBNAIL_WIDTH, self.THUMBNAIL_WIDTH), Image.BICUBIC)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=80)
        reply = buffer.getvalue(), "image/jpeg"
        if len(self._thumbnails) >= self.MAX_THUMBNAILS:
            self._thumbnails.pop(next(iter(self._thumbnails)), None)
        self._thumbnails[key] = reply
        return reply


def handler(runtime, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No task content or headers in logs.

        def reply(self, status, value, mime="application/json"):
            body = json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
            if len(body) > 24 * 1024 ** 2:
                raise ValueError("Response too large")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized"})
                return
            if self.command == "GET":
                if self.path == "/status":
                    info = dict(runtime.info)
                    if hasattr(runtime, "ocr"):
                        info["ocr"] = runtime.ocr.status()
                    self.reply(200, info)
                    return
                match = re.fullmatch(r"/(images|thumbs)/(L\d+_V\d+)/(\d+)", self.path)
                if not match:
                    raise FileNotFoundError()
                read = runtime.thumbnail if match[1] == "thumbs" else runtime.image
                data, mime = read(match[2], int(match[3]))
                self.reply(200, data, mime)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 12 * 1024 ** 2 or self.headers.get_content_type() != "application/json":
                raise ValueError("Invalid request size/type")
            payload = json.loads(self.rfile.read(size))
            with runtime.lock:
                if self.path == "/retrieve":
                    value = runtime.retrieve(payload)
                elif self.path == "/ocr-evidence":
                    value = runtime.ocr_evidence(payload)
                elif self.path in {"/visual-search", "/similar"}:
                    rows = (runtime.visual.search(payload["vector"], payload["top_k"]) if self.path == "/visual-search"
                            else runtime.visual.similar(payload["video_id"], payload["frame_id"], payload["top_k"]))
                    for row in rows:
                        metadata = runtime.captions.get_caption(row["video_id"], row["frame_id"])
                        if metadata:
                            row.update({k: metadata[k] for k in ("retrieval_text", "caption", "image_path") if k in metadata})
                    value = {"results": rows}
                elif self.path == "/encode-text":
                    text = payload["text"]
                    if not isinstance(text, str) or not 1 <= len(text) <= 8000:
                        raise ValueError("Invalid text")
                    value = {"vector": runtime.encoder.encode_text(text).tolist()}
                elif self.path == "/encode-images":
                    from PIL import Image
                    encoded = payload["images"]
                    if not isinstance(encoded, list) or not 1 <= len(encoded) <= 6:
                        raise ValueError("Video sample budget exceeded")
                    images = []
                    try:
                        for data in encoded:
                            raw = base64.b64decode(data, validate=True)
                            with Image.open(BytesIO(raw)) as image:
                                if image.width * image.height > 1024 ** 2:
                                    raise ValueError("Sample too large")
                                images.append(image.convert("RGB"))
                        value = {"vectors": runtime.encoder.encode_images(images).tolist()}
                    finally:
                        for image in images:
                            image.close()
                else:
                    raise FileNotFoundError()
            self.reply(200, value)

        def handle_request(self):
            self.connection.settimeout(180)
            try:
                self.dispatch()
            except FileNotFoundError:
                self.reply(404, {"error": "not_found"})
            except (ValueError, KeyError, TypeError):
                self.reply(422, {"error": "invalid_request"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.reply(503, {"error": "worker_failed"})

        do_GET = do_POST = handle_request
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--ready-fd", type=int)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise ValueError("Worker token missing")
    try:
        runtime = WorkerRuntime(args.dataset)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(runtime, token))
        info = dict(runtime.info, pid=os.getpid(), port=server.server_port)
        if args.ready_fd is not None:
            with os.fdopen(args.ready_fd, "w") as stream:
                stream.write(json.dumps(info) + "\n")
            args.ready_fd = None
        print(json.dumps({"state": "ready", "pid": os.getpid(), "port": server.server_port}), flush=True)
        server.serve_forever()
    except Exception as exc:
        if args.ready_fd is not None:
            with os.fdopen(args.ready_fd, "w") as stream:
                stream.write(json.dumps({"state": "failed", "error_type": type(exc).__name__}) + "\n")
        raise


if __name__ == "__main__":
    main()
