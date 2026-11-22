"""Loopback-only, authenticated school worker client. No provider/DRES secrets.

Errors are explicit: a lost tunnel never silently switches corpus or image source.
"""

from contextlib import contextmanager
from io import BytesIO
import json
import math
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from kis.paths import KIS_ROOT, VIDEO_ID_PATTERN
from kis.result_pool import WEB_RESULT_LIMIT


def worker_settings(environ=None, path=None):
    env = os.environ if environ is None else environ
    path = Path(path) if path else KIS_ROOT / "config/worker.local.json"
    settings = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    url = env.get("KIS_WORKER_URL", settings.get("url", ""))
    token = env.get("KIS_WORKER_TOKEN", settings.get("token", ""))
    return {"url": url, "token": token} if url else None


def runtime_signature(config):
    result = {"video_groups": list(config.video_groups), "retrieval_profile": config.retrieval_profile,
            "temporal": config.temporal.public(), "ocr": config.ocr.public(),
             "text_reranker": config.text_reranker.public(), "ocr_format": config.ocr_format}
    if config.ocr_format == "school_snapshot":
        result["ocr_manifest_sha256"] = config.ocr_manifest_sha256
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RemoteWorker:
    source_name = "school_worker"
    MAX_RESPONSE = 24 * 1024 * 1024

    def __init__(self, url, token, groups, timeout=180):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username or parts.password or parts.path not in {"", "/"}
                or parts.query or parts.fragment or not parts.port):
            raise ValueError("Worker URL must be an explicit loopback HTTP port behind SSH")
        if not isinstance(token, str) or len(token) < 32 or any(c in token for c in "\r\n"):
            raise ValueError("Worker requires a private authentication token")
        self.url, self._token, self.groups = url.rstrip("/"), token, tuple(groups)
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), NoRedirect())
        self.info = None
        self.captions = {}
        self._state = threading.local()

    @property
    def transport_failed(self):
        # Per thread: an image fetch failing in another request must not fail this search.
        return getattr(self._state, "failed", False)

    @transport_failed.setter
    def transport_failed(self, value):
        self._state.failed = value

    def request(self, route, payload=None, *, image=False):
        data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = Request(self.url + route, data=data, headers={
            "Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(self.MAX_RESPONSE + 1)
                mime = response.headers.get_content_type()
        except HTTPError as exc:
            code = exc.code
            exc.close()
            if image and code == 404:
                raise FileNotFoundError("Image absent from school dataset") from None
            self.transport_failed = True
            raise RuntimeError(f"School worker rejected request (HTTP {code}); no fallback applied") from None
        except (URLError, OSError, TimeoutError):
            self.transport_failed = True
            raise RuntimeError("School worker unavailable; check VPN, SSH tunnel and worker") from None
        if len(body) > self.MAX_RESPONSE:
            raise RuntimeError("School worker response exceeds the bounded payload")
        if image:
            if mime not in {"image/jpeg", "image/png"} or not body:
                raise RuntimeError("Invalid school image response")
            return body, mime
        if mime != "application/json":
            raise RuntimeError("Invalid school worker response type")
        return json.loads(body)

    def connect(self, config, models):
        info = self.request("/status")
        if (info.get("protocol") != 1 or info.get("state") != "ready"
                or info.get("result_limit") != WEB_RESULT_LIMIT
                or info.get("runtime") != runtime_signature(config) or info.get("models") != models):
            raise RuntimeError("School worker scope/config/model mismatch")
        self.info = info
        return info

    def validate_candidates(self, values, *, complete=False):
        if not isinstance(values, list) or len(values) > 20000:
            raise RuntimeError("Invalid school candidate pool")
        seen = set()
        for row in values:
            vid, frame, number = row.get("video_id"), row.get("frame_id"), row.get("keyframe_n")
            if (not isinstance(vid, str) or not VIDEO_ID_PATTERN.fullmatch(vid)
                    or vid.split("_")[0] not in self.groups or type(frame) is not int or frame < 0
                    or type(number) is not int or number < 1
                    or not math.isfinite(float(row.get("score", float("nan"))))):
                raise RuntimeError("Invalid/out-of-scope school candidate")
            key = (vid, frame)
            if key in seen:
                raise RuntimeError("Duplicate school candidate identity")
            seen.add(key)
            if len(self.captions) >= 8192:
                self.captions.pop(next(iter(self.captions)))
            self.captions[key] = row
        if complete and len(values) < WEB_RESULT_LIMIT:
            raise RuntimeError(f"School worker returned fewer than {WEB_RESULT_LIMIT} distinct frames")
        return values

    def retrieve(self, structured, *, temporal=None, ocr_region=None, limit=None):
        self.transport_failed = False
        payload = {"structured_query": structured, "temporal": temporal, "ocr_region": ocr_region}
        if limit is not None:
            payload["limit"] = limit  # older workers ignore it and return the full pool
        result = self.request("/retrieve", payload)
        result["candidates"] = self.validate_candidates(result["candidates"], complete=True)
        return result

    def attach_ocr(self, candidates):
        # Send identities only: never ship captions, hints or query pixels back.
        identities = [{k: row[k] for k in ("video_id", "frame_id", "keyframe_n")} for row in candidates]
        reply = self.request("/ocr-evidence", {"candidates": identities})
        evidence = reply.get("candidates")
        pin = (self.info or {}).get("runtime", {}).get("ocr_manifest_sha256")
        if not isinstance(evidence, list) or len(evidence) != len(candidates):
            raise RuntimeError("Invalid OCR evidence reply count")
        for expected, row in zip(identities, evidence):
            ocr = row.get("ocr", {})
            if (any(row.get(k) != v for k, v in expected.items()) or ocr.get("source_format") != "public_json"
                    or ocr.get("verified") is not False or ocr.get("status") not in {"ok", "empty", "error", "missing"}
                    or not isinstance(ocr.get("text"), str) or len(ocr["text"]) > 65536
                    or ocr.get("build_id") != pin or any(ocr.get(k) is not None for k in ("engine", "confidence", "image_sha256"))):
                raise RuntimeError("Invalid/unpinned OCR evidence reply")
        for candidate, row in zip(candidates, evidence):
            candidate["ocr"] = row["ocr"]
        return reply.get("status", {})

    def reference(self, candidate, lookup=None):
        vid, number = candidate.get("video_id"), candidate.get("keyframe_n")
        if not isinstance(vid, str) or not VIDEO_ID_PATTERN.fullmatch(vid) or vid.split("_")[0] not in self.groups:
            raise ValueError("Image outside school worker scope")
        if number is None:
            cached = self.captions.get((vid, candidate.get("frame_id")), {})
            number = cached.get("keyframe_n")
        if number is None:
            return None
        if type(number) is not int or number < 1:
            raise ValueError("Invalid school image number")
        candidate["keyframe_n"] = number
        return Path(f"{number:03d}.jpg")

    def read(self, video_id, number):
        self.reference({"video_id": video_id, "keyframe_n": number})
        return self.request(f"/images/{video_id}/{number}", image=True)

    @contextmanager
    def open_candidate(self, candidate, lookup):
        if self.reference(candidate, lookup) is None:
            yield None
            return
        data, _ = self.read(candidate["video_id"], candidate["keyframe_n"])
        with BytesIO(data) as stream:
            yield stream

    def encode_text(self, text):
        import numpy as np
        return np.asarray(self.request("/encode-text", {"text": text})["vector"], dtype=np.float32)

    def encode_images(self, images):
        import base64
        import numpy as np
        if not 1 <= len(images) <= 6:
            raise ValueError("Video sample budget exceeded")
        self.transport_failed = False
        encoded = []
        for image in images:
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")  # Lossless, no query pixel changes.
                encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return np.asarray(self.request("/encode-images", {"images": encoded})["vectors"], dtype=np.float32)

    def search(self, vector, top_k=100):
        result = self.request("/visual-search", {"vector": vector.tolist(), "top_k": top_k})
        return self.validate_candidates(result["results"])

    def similar(self, video_id, frame_id, top_k):
        result = self.request("/similar", {"video_id": video_id, "frame_id": frame_id, "top_k": top_k})
        return self.validate_candidates(result["results"])

    def get_caption(self, video_id, frame_id):
        return self.captions.get((video_id, int(frame_id)))

    def get_keyframe_n(self, video_id, frame_id):
        row = self.get_caption(video_id, frame_id)
        return row.get("keyframe_n") if row else None

    @property
    def index(self):
        return SimpleNamespace(ntotal=self.info["caption_vectors"] if self.info else 0)

    @property
    def caption_data(self):
        return []  # Status uses the explicit remote handshake; no local corpus load.
