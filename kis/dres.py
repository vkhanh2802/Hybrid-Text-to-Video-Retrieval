"""DRES v2 client and explicit, one-use submission previews. No persistent secrets."""

import base64
import binascii
import csv
import hashlib
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import logging
import os
from pathlib import Path
import re
import secrets
from threading import Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urljoin, quote
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from uuid import UUID

from kis.llm import _api_ssl_context
from kis.result_pool import WEB_RESULT_LIMIT, require_frame_count
from kis.paths import KIS_ROOT, MAPPING_DIR, VIDEO_ID_PATTERN, resolve_video_artifact

logger = logging.getLogger("kis.dres")


COOKIE_NAME = "kis_dres_connection"
CONNECTION_TTL = 8 * 60 * 60
PREVIEW_TTL = 60
MAX_RESPONSE_BYTES = 512 * 1024
MAX_HINT_BYTES = 32 * 1024 * 1024
MAX_HINT_MEDIA_BYTES = 24 * 1024 * 1024
MAX_VIDEO_DESCRIPTION_CHARS = 2000
# Must not exceed the text-query cap used by the structuring pipeline.
MAX_QUERY_TEXT_CHARS = 8000


class DRESError(RuntimeError):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def expects_text_answer(snapshot):
    task = snapshot.get("task") or {}
    # Client task info exposes a type label, not its full submission filters.
    return bool(re.search(r"question|answer|qakis|\bqa\b", str(task.get("taskType", "")), re.I))


def evaluation_id(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Run ID phải là UUID hợp lệ") from None


def load_dres_settings(environ=None, settings_path=None):
    env = os.environ if environ is None else environ
    path = Path(settings_path) if settings_path else KIS_ROOT / "config/dres.local.json"
    local = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    base = str(env.get("KIS_DRES_BASE_URL") or local.get("base_url") or "").rstrip("/")
    run = str(env.get("KIS_DRES_RUN_ID") or local.get("evaluation_id") or "")
    if base:
        parts = urlsplit(base)
        if (not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
                or parts.scheme not in {"https", "http"}
                or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})
                or any(part in {".", ".."} for part in parts.path.split("/"))):
            raise ValueError("DRES URL phải dùng HTTPS (HTTP chỉ cho localhost), không chứa credentials hoặc query")
    return {"base_url": base, "evaluation_id": evaluation_id(run) if run else "", "configured": bool(base)}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DRESClient:
    def __init__(self, base_url, session="", opener=None):
        self.base_url = base_url.rstrip("/")
        self._session = ""
        self._hint_key = None
        self._hint_sequence = []
        if session:
            self.set_session(session)
        self.opener = opener or build_opener(HTTPSHandler(context=_api_ssl_context()), _NoRedirect())

    def set_session(self, value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,256}", value):
            raise ValueError("Session ID không hợp lệ")
        self._session = value

    def _request(self, method, path, body=None, *, max_bytes=MAX_RESPONSE_BYTES, media_type=None):
        headers = {"Accept": media_type or "application/json", "Cache-Control": "no-store"}
        if self._session:
            # DRES's documented SESSIONID cookie avoids putting tokens in URL logs.
            headers["Cookie"] = "SESSIONID=" + self._session
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + "/api/v2/" + path, data=data, headers=headers, method=method)
        submitting = path.startswith("submit/")
        try:
            with self.opener.open(request, timeout=15) as response:
                raw = response.read(max_bytes + 1)
                status = response.status
                if media_type and response.headers.get_content_type() != media_type:
                    raise DRESError("DRES trả định dạng media không được hỗ trợ")
        except HTTPError as exc:
            # Log upstream status + short body (never headers/cookie) so a refused submit can be diagnosed.
            try:
                upstream = exc.read(400).decode("utf-8", "replace")
            except Exception:
                upstream = ""
            exc.close()
            if self._session:
                upstream = upstream.replace(self._session, "[session]")
            logger.warning("DRES %s %s -> HTTP %s %s", method, path.split("/")[0], exc.code, " ".join(upstream.split()))
            messages = {401: "DRES session hết hạn hoặc thông tin đăng nhập không hợp lệ",
                        403: "Tài khoản không có quyền thực hiện thao tác này trên DRES",
                        404: "Không tìm thấy run/task hoặc tài khoản không được tham gia run này",
                        412: "DRES từ chối bài nộp (trùng bài, giới hạn lượt hoặc quy tắc task)"}
            if submitting:
                try:
                    refusal = json.loads(upstream)
                except ValueError:
                    refusal = None
                if isinstance(refusal, dict) and refusal.get("status") is False and refusal.get("description"):
                    # Explicit {"status": false} = DRES rejected it (e.g. media item not in the
                    # run's collection): nothing was recorded, so another frame can be sent at once.
                    reason = " ".join(str(refusal["description"]).split())[:300]
                    raise DRESError(f"DRES từ chối bài, chưa ghi nhận: {reason} Có thể chọn frame khác và nộp lại.",
                                    exc.code if exc.code in {400, 412} else 422) from None
            if submitting and exc.code >= 500:
                raise DRESError("Không rõ DRES đã ghi nhận bài chưa. Kiểm tra trang DRES trước khi nộp lại; không tự retry.") from None
            raise DRESError(messages.get(exc.code, f"DRES trả HTTP {exc.code}; kiểm tra trạng thái run/task"),
                            exc.code if exc.code in {400, 401, 403, 404, 412} else 502) from None
        except (URLError, TimeoutError, OSError) as exc:
            logger.warning("DRES %s %s -> %s: %s", method, path.split("/")[0], type(exc).__name__, exc)
            message = ("Mất kết nối khi gửi bài; chưa rõ DRES đã nhận chưa. Kiểm tra trang DRES, không tự retry."
                       if submitting else "Không kết nối được DRES; kiểm tra mạng hoặc chứng chỉ TLS")
            raise DRESError(message) from None
        if len(raw) > max_bytes:
            raise DRESError("DRES response vượt giới hạn; không tự gửi lại thao tác")
        if media_type:
            return status, raw
        try:
            return status, json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            logger.warning("DRES %s %s -> HTTP %s non-JSON body %r", method, path.split("/")[0], status, raw[:200])
            if submitting and 200 <= status < 300:
                # HTTP 2xx means DRES took the submission; an odd body must not make the user resubmit.
                return status, {"status": True, "submission": None}
            raise DRESError("DRES không trả JSON hợp lệ; không tự gửi lại thao tác") from None

    def login(self, username, password):
        _, user = self._request("POST", "login", {"username": username, "password": password})
        self.set_session(user.get("sessionId", ""))
        return {key: user.get(key) for key in ("username", "role")}

    def user(self):
        _, user = self._request("GET", "user")
        if not isinstance(user, dict) or not user.get("username"):
            raise DRESError("DRES chưa xác thực tài khoản", 401)
        return {key: user.get(key) for key in ("username", "role")}

    def snapshot(self, run_id):
        run_id = evaluation_id(run_id)
        _, runs = self._request("GET", "client/evaluation/list")
        if not isinstance(runs, list) or any(not isinstance(item, dict) for item in runs):
            raise DRESError("DRES trả danh sách evaluation không hợp lệ")
        run = next((item for item in runs if item.get("id") == run_id), None)
        if run is None:
            raise DRESError("Run ID không nằm trong các evaluation mà tài khoản được tham gia", 404)
        _, state = self._request("GET", f"evaluation/{run_id}/state")
        if not isinstance(state, dict):
            raise DRESError("DRES trả trạng thái evaluation không hợp lệ")
        task = None
        if state.get("taskId"):
            _, task = self._request("GET", f"client/evaluation/currentTask/{run_id}")
        snapshot = {"evaluation_id": run_id, "name": run.get("name"),
                "evaluation_status": state.get("evaluationStatus"),
                "task_id": state.get("taskId"), "task_status": state.get("taskStatus"),
                "task_template_id": state.get("taskTemplateId"), "time_elapsed": state.get("timeElapsed"),
                "time_left": state.get("timeLeft"), "task": task,
                "can_submit": (state.get("evaluationStatus") == "ACTIVE"
                               and state.get("taskStatus") == "RUNNING" and bool(state.get("taskId")))}
        snapshot["answer_mode"] = "text" if expects_text_answer(snapshot) else "frame"
        return snapshot

    def task_content(self, snapshot, refresh=False):
        """Only current, released viewer hints; never request target/answer APIs."""
        template = snapshot.get("task_template_id")
        key = (snapshot["evaluation_id"], snapshot.get("task_id"), template)
        if key != self._hint_key or not snapshot["can_submit"]:
            self._hint_key, self._hint_sequence = None, []
        if not snapshot["can_submit"] or not template:
            return {"status": "waiting", "elements": []}
        # A manual refresh re-reads the sequence so newly added sentences appear.
        if self._hint_key != key or refresh:
            _, hint = self._request("GET", f"evaluation/{key[0]}/template/task/{quote(template, safe='')}/hint",
                                    max_bytes=MAX_HINT_BYTES)
            if (not isinstance(hint, dict) or hint.get("taskId") != template
                    or not isinstance(hint.get("sequence"), list) or len(hint["sequence"]) > 256):
                raise DRESError("DRES trả nội dung đề không hợp lệ")
            self._hint_key, self._hint_sequence = key, hint["sequence"]
        # Guard task changes/countdown during a slow hint request. Offsets are seconds.
        _, state = self._request("GET", f"evaluation/{key[0]}/state")
        if (state.get("taskId") != key[1] or state.get("taskTemplateId") != template
                or state.get("taskStatus") != "RUNNING" or state.get("evaluationStatus") != "ACTIVE"):
            self._hint_key, self._hint_sequence = None, []
            raise DRESError("Task đã đổi hoặc dừng; đang chờ đề mới", 409)
        elapsed = state.get("timeElapsed")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            return {"status": "waiting", "elements": []}
        # DRES reveals TEXT hints as additive sentences over time. Keep every
        # released sentence in order; an empty TEXT element starts a new chain.
        text_chain, media = [], {}
        for index, element in enumerate(self._hint_sequence):
            if not isinstance(element, dict):
                continue
            kind, offset = element.get("contentType"), element.get("offset")
            if (kind not in {"TEXT", "VIDEO", "IMAGE"} or not isinstance(offset, (int, float))
                    or offset < 0 or offset > elapsed):
                continue
            if kind == "TEXT":
                content = element.get("content")
                if isinstance(content, str) and content.strip():
                    text_chain.append((index, element))
                else:
                    text_chain = []
            elif kind not in media or offset >= media[kind][1]["offset"]:
                media[kind] = (index, element)
        elements = []
        for index, element in text_chain:
            elements.append({"type": "TEXT", "index": index, "offset": element["offset"],
                             "text": str(element.get("content"))})
        for kind, (index, element) in media.items():
            if not element.get("content") and not element.get("contentUrl"):
                continue  # An empty content element clears the previous hint of this type.
            elements.append({"type": kind, "index": index, "offset": element["offset"],
                             "media_url": f"/api/dres/task/media/{quote(key[1], safe='')}/{index}"})
        elements.sort(key=lambda item: (item["offset"], item["index"]))
        return {"status": "available" if elements else "waiting", "elements": elements}

    def task_media(self, snapshot, task_id, index):
        if snapshot.get("task_id") != task_id:
            raise DRESError("Media thuộc task cũ", 409)
        content = self.task_content(snapshot)
        item = next((item for item in content["elements"] if item["index"] == index and item["type"] in {"VIDEO", "IMAGE"}), None)
        if item is None:
            raise DRESError("Media chưa được phát hoặc không còn trong đề hiện tại", 404)
        element = self._hint_sequence[index]
        mime = "video/mp4" if item["type"] == "VIDEO" else "image/jpeg"
        if element.get("contentUrl"):
            # Do not fetch arbitrary URLs, forward credentials to another host, or expose URL tokens.
            if not isinstance(element["contentUrl"], str):
                raise DRESError("URL media đề DRES không hợp lệ")
            target = urlsplit(urljoin(self.base_url + "/", element["contentUrl"]))
            base = urlsplit(self.base_url)
            expected = f"evaluation/{snapshot['evaluation_id']}/template/task/{quote(snapshot['task_template_id'], safe='')}/hint/"
            # DRES behind /dres emits root-relative /api/v2 URLs. Always rebuild
            # on the configured base, never follow the upstream URL verbatim.
            prefixes = (base.path.rstrip("/") + "/api/v2/", "/api/v2/")
            prefix = next((value for value in prefixes if target.path.startswith(value + expected)), None)
            if (target.scheme != base.scheme or target.netloc != base.netloc or target.query or target.fragment
                    or prefix is None
                    or not re.fullmatch(r"\d+/media", target.path[len(prefix + expected):])):
                raise DRESError("URL media đề DRES không hợp lệ")
            _, raw = self._request("GET", target.path[len(prefix):], max_bytes=MAX_HINT_MEDIA_BYTES, media_type=mime)
            _, state = self._request("GET", f"evaluation/{snapshot['evaluation_id']}/state")
            if (state.get("taskId") != task_id or state.get("taskTemplateId") != snapshot["task_template_id"]
                    or state.get("taskStatus") != "RUNNING" or state.get("evaluationStatus") != "ACTIVE"):
                self._hint_key, self._hint_sequence = None, []
                raise DRESError("Task đã đổi khi tải media; không phát đề cũ", 409)
        else:
            encoded = element.get("content", "")
            if not isinstance(encoded, str) or len(encoded) > (MAX_HINT_MEDIA_BYTES * 4 // 3 + 4):
                raise DRESError("Media đề vượt giới hạn")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise DRESError("Media đề không đúng base64") from None
        if not raw or len(raw) > MAX_HINT_MEDIA_BYTES:
            raise DRESError("Media đề rỗng hoặc vượt giới hạn")
        return raw, mime

    def submit(self, run_id, payload):
        code, result = self._request("POST", f"submit/{evaluation_id(run_id)}", payload)
        return {"http_status": code, "accepted": result.get("status") is True,
                "verdict": result.get("submission"),
                # Do not expose arbitrary upstream strings or claim acceptance means correctness.
                "pending": code == 202 or result.get("submission") == "INDETERMINATE"}


MAX_FRAME_OFFSET = 250  # +-10 s at 25 fps around the chosen keyframe


def mapped_submission(video_id, image_number, frame_id, groups, mapping_dir=MAPPING_DIR, frame_offset=0):
    if (not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id.split("_")[0] not in groups
            or type(image_number) is not int or image_number < 1
            or type(frame_id) is not int or frame_id < 0
            or type(frame_offset) is not int or abs(frame_offset) > MAX_FRAME_OFFSET):
        raise ValueError("Video/frame nằm ngoài scope hoặc không hợp lệ")
    path = resolve_video_artifact(mapping_dir, video_id, ".csv", video_groups=groups)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["n"]) == image_number]
    if len(rows) != 1 or int(rows[0]["frame_idx"]) != frame_id:
        raise ValueError("Video, số ảnh và frame ID không khớp mapping; không nộp")
    try:
        timestamp = Decimal(rows[0]["pts_time"])
        if not timestamp.is_finite() or timestamp < 0:
            raise InvalidOperation
        if frame_offset:
            # Shift whole video frames from the mapped keyframe: pts + offset / fps.
            fps = Decimal(rows[0]["fps"])
            if not fps.is_finite() or fps <= 0:
                raise InvalidOperation
            timestamp += Decimal(frame_offset) / fps
            if timestamp < 0 or frame_id + frame_offset < 0:
                raise ValueError("Frame sau khi lùi nằm trước đầu video; không nộp")
        milliseconds = int((timestamp * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (KeyError, InvalidOperation):
        raise ValueError("Mapping không có timestamp hợp lệ; không suy thời gian từ số ảnh/frame") from None
    result = {"video_id": video_id, "image_number": image_number, "frame_id": frame_id,
              "timestamp_seconds": float(timestamp), "timestamp_ms": milliseconds,
              "answer": {"mediaItemName": video_id, "start": milliseconds, "end": milliseconds}}
    if frame_offset:
        result.update(frame_offset=frame_offset, submitted_frame_id=frame_id + frame_offset)
    return result


@dataclass
class DRESConnection:
    client: DRESClient = field(repr=False)
    run_id: str
    user: dict
    expires: float = field(default_factory=lambda: monotonic() + CONNECTION_TTL)
    previews: dict = field(default_factory=dict, repr=False)
    lock: Lock = field(default_factory=Lock, repr=False)

    def status(self):
        return {"connected": True, "user": self.user, "base_url": self.client.base_url,
                "run": self.client.snapshot(self.run_id)}

    def task(self, refresh=False):
        with self.lock:
            result = self.status()
            try:
                result["content"] = self.client.task_content(result["run"], refresh=refresh)
            except DRESError as exc:
                if exc.status_code in {401, 409}:
                    raise
                result["content"] = {"status": "unavailable", "elements": [], "message": str(exc)}
            return result

    def task_media(self, task_id, index):
        with self.lock:
            return self.client.task_media(self.client.snapshot(self.run_id), task_id, index)

    def search_task(self, task_id, search_service, video_description=""):
        """Resolve the official input on the server; reject results for a stale task/hint."""
        if not isinstance(video_description, str) or len(video_description) > MAX_VIDEO_DESCRIPTION_CHARS:
            raise ValueError("Text bổ sung cho video không được quá 2000 ký tự")
        video_description = video_description.strip()
        with self.lock:
            snapshot = self.client.snapshot(self.run_id)
            if not snapshot["can_submit"] or snapshot.get("task_id") != task_id:
                raise DRESError("Task không còn RUNNING hoặc đã đổi; hãy làm mới đề", 409)
            # Every user search is also a fresh hint read, without reconnecting.
            content = self.client.task_content(snapshot, refresh=True)
            elements = content["elements"]
            fingerprint = hashlib.sha256(json.dumps(elements, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            text = "\n".join(item["text"] for item in elements if item["type"] == "TEXT" and item.get("text"))
            if len(text) > MAX_QUERY_TEXT_CHARS:
                raise DRESError("Toàn bộ text đề DRES vượt 8000 ký tự; không cắt ngầm nội dung đề", 422)
            video = next((item for item in elements if item["type"] == "VIDEO"), None)
            visual_task = "visual" in str((snapshot.get("task") or {}).get("taskType", "")).lower()
            if video is not None and (visual_task or not text):
                data, _ = self.client.task_media(snapshot, task_id, video["index"])
                kind = "video"
            elif text.strip():
                if video_description:
                    raise DRESError("Text bổ sung chỉ dùng khi đề chính thức là video", 422)
                kind = "text"
            else:
                raise DRESError("DRES chưa phát đề text/video có thể tìm kiếm; không dùng input tự nhập thay thế", 409)
        # Status polls must be able to observe task changes while the model is busy.
        if kind == "video":
            # Official DRES text (revealed so far) and the optional user hint share
            # the text side of the hybrid query; video stays the official anchor.
            kwargs = {"official": True}
            if text.strip():
                kwargs["hint_text"] = text
            if video_description:
                kwargs["description"] = video_description
            result = (search_service.search_video(data, **kwargs) if len(kwargs) > 1
                      else search_service.search_video(data, official=True))
        else:
            result = search_service.search(text)
        with self.lock:
            current = self.client.snapshot(self.run_id)
            if not current["can_submit"] or current.get("task_id") != task_id:
                raise DRESError("DRES đã chuyển câu trong khi model xử lý; đã bỏ kết quả của đề cũ", 409)
            latest = self.client.task_content(current, refresh=True)
            latest_fingerprint = hashlib.sha256(json.dumps(latest["elements"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if latest_fingerprint != fingerprint:
                raise DRESError("Nội dung đề DRES đã thay đổi trong khi xử lý; hãy tìm lại theo đề mới", 409)
        require_frame_count(result["results"], WEB_RESULT_LIMIT)
        return dict(result, dres_content=content, dres_run=current,
                    dres_task={"task_id": task_id, "task_name": (snapshot.get("task") or {}).get("name"),
                                      "input_type": kind, "source": "dres_official", "hint_fingerprint": fingerprint,
                                      "has_supplemental_description": bool(video_description),
                                      "query_text_chars": len(text)})

    def preview(self, selection, groups, task_id=None):
        with self.lock:
            snapshot = self.client.snapshot(self.run_id)
            if task_id is not None and task_id != snapshot.get("task_id"):
                raise DRESError("Frame được chọn thuộc task cũ; chưa chuẩn bị bài nộp", 409)
            if not snapshot["can_submit"]:
                raise DRESError("DRES chưa có task đang RUNNING; không thể chuẩn bị nộp", 409)
            if expects_text_answer(snapshot):
                raise DRESError("Task hiện tại yêu cầu đáp án chữ (QA); dùng ô đáp án QA thay vì gửi frame", 409)
            mapped = mapped_submission(groups=groups, **selection)
            payload = {"answerSets": [{"taskId": snapshot["task_id"], "answers": [mapped["answer"]]}]}
            token = secrets.token_urlsafe(24)
            self.previews.clear()
            self.previews[token] = {"kind": "frame", "expires": monotonic() + PREVIEW_TTL,
                "selection": selection, "mapped": mapped, "payload": payload, "task_id": snapshot["task_id"]}
            return {"submission_type": "frame", "preview_id": token, "expires_in_seconds": PREVIEW_TTL, "selection": mapped,
                    "payload": payload, "run": snapshot, "evaluation_id": self.run_id}

    def preview_text(self, text, task_id=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Đáp án QA phải có từ 1 đến 4000 ký tự")
        with self.lock:
            snapshot = self.client.snapshot(self.run_id)
            if task_id is not None and task_id != snapshot.get("task_id"):
                raise DRESError("Đáp án thuộc task cũ; chưa chuẩn bị bài nộp", 409)
            if not snapshot["can_submit"] or not expects_text_answer(snapshot):
                raise DRESError("Chưa có task QA đang RUNNING; làm mới task trước khi chuẩn bị đáp án chữ", 409)
            payload = {"answerSets": [{"taskId": snapshot["task_id"], "answers": [{"text": text.strip()}]}]}
            token = secrets.token_urlsafe(24)
            self.previews.clear()
            self.previews[token] = {"kind": "text", "expires": monotonic() + PREVIEW_TTL,
                                   "payload": payload, "task_id": snapshot["task_id"]}
            return {"submission_type": "text", "preview_id": token, "expires_in_seconds": PREVIEW_TTL,
                    "selection": {"answer_text": text.strip()}, "payload": payload,
                    "run": snapshot, "evaluation_id": self.run_id}

    def submit_preview(self, token, groups):
        with self.lock:
            # Consume BEFORE any remote work: double clicks and uncertain errors never retry.
            preview = self.previews.pop(token, None)
            if preview is None or preview["expires"] < monotonic():
                raise DRESError("Xác nhận đã dùng hoặc hết hạn; hãy chọn và kiểm tra lại kết quả", 409)
            current = self.client.snapshot(self.run_id)
            if not current["can_submit"] or current["task_id"] != preview["task_id"]:
                raise DRESError("Task đã thay đổi hoặc dừng; bài chưa được gửi. Kiểm tra task mới", 409)
            if preview["kind"] == "frame":
                if expects_text_answer(current):
                    raise DRESError("Task yêu cầu đáp án chữ; không gửi frame", 409)
                mapped = mapped_submission(groups=groups, **preview["selection"])
                if mapped != preview["mapped"]:
                    raise DRESError("Mapping đã thay đổi; bài chưa được gửi. Chọn lại kết quả", 409)
            elif not expects_text_answer(current):
                raise DRESError("Task không còn yêu cầu đáp án chữ; bài chưa được gửi", 409)
            return self.client.submit(self.run_id, preview["payload"])


class DRESConnections:
    def __init__(self):
        self._connections = {}
        self._lock = Lock()

    def add(self, connection, old_token=None):
        with self._lock:
            self._connections = {key: item for key, item in self._connections.items() if item.expires > monotonic()}
            self._connections.pop(old_token, None)
            if len(self._connections) >= 32:
                raise DRESError("Quá nhiều phiên DRES local; ngắt bớt phiên rồi thử lại", 503)
            token = secrets.token_urlsafe(32)
            self._connections[token] = connection
            return token

    def get(self, token):
        with self._lock:
            connection = self._connections.get(token)
            if connection is None or connection.expires <= monotonic():
                self._connections.pop(token, None)
                raise DRESError("Chưa kết nối hoặc phiên DRES local đã hết hạn", 401)
            return connection

    def remove(self, token):
        with self._lock:
            self._connections.pop(token, None)
