"""Same-origin UI bridge to one server-configured DRES endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, SecretStr, model_validator

from kis.dres import (
    COOKIE_NAME, CONNECTION_TTL, DRESClient, DRESConnection, DRESConnections,
    DRESError, evaluation_id, load_dres_settings,
    MAX_VIDEO_DESCRIPTION_CHARS,
)
from kis.result_pool import IncompleteResultPool
from kis.planner_fallback import QueryPlanningError


class DRESConnectRequest(BaseModel):
    model_config = {"extra": "forbid"}
    evaluation_id: str = Field(min_length=1, max_length=64)
    session_id: SecretStr | None = None
    username: str | None = Field(default=None, min_length=1, max_length=128)
    password: SecretStr | None = None

    @model_validator(mode="after")
    def one_auth_method(self):
        if self.session_id is not None:
            if self.username is not None or self.password is not None:
                raise ValueError("Chỉ dùng session ID hoặc username/password, không dùng đồng thời")
        elif not self.username or self.password is None:
            raise ValueError("Nhập session ID hoặc username và password")
        return self


class DRESSelection(BaseModel):
    model_config = {"extra": "forbid"}
    video_id: str = Field(pattern=r"^L\d+_V\d+$", max_length=32)
    image_number: int = Field(strict=True, ge=1)
    frame_id: int = Field(strict=True, ge=0)
    frame_offset: int = Field(default=0, strict=True, ge=-250, le=250)
    task_id: str | None = Field(default=None, min_length=1, max_length=128)


class DRESSubmitRequest(BaseModel):
    model_config = {"extra": "forbid"}
    preview_id: str = Field(min_length=16, max_length=128)
    confirmed: bool = Field(strict=True)


class DRESTextAnswer(BaseModel):
    model_config = {"extra": "forbid"}
    answer_text: str = Field(min_length=1, max_length=4000)
    task_id: str | None = Field(default=None, min_length=1, max_length=128)


class DRESTaskSearch(BaseModel):
    model_config = {"extra": "forbid"}
    task_id: str = Field(min_length=1, max_length=128)
    video_description: str = Field(default="", max_length=MAX_VIDEO_DESCRIPTION_CHARS)


class _PrivateValidationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic's default error response includes the original input.
                raise HTTPException(status_code=422, detail="Dữ liệu DRES không hợp lệ; kiểm tra các trường và chỉ chọn một cách đăng nhập") from None

        return private_handler


def _same_origin(request: Request):
    if request.headers.get("x-kis-dres") != "1":
        raise HTTPException(status_code=403, detail="Missing same-origin DRES request header")
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="Cross-origin DRES requests are not allowed")


def create_dres_router(groups_provider, connections=None, client_factory=DRESClient, search_service_provider=None):
    store = connections if connections is not None else DRESConnections()
    router = APIRouter(prefix="/api/dres", dependencies=[Depends(_same_origin)], route_class=_PrivateValidationRoute)

    def get_connection(request):
        return store.get(request.cookies.get(COOKIE_NAME))

    def fail(exc):
        if isinstance(exc, DRESError):
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
        if isinstance(exc, OSError):
            raise HTTPException(status_code=422, detail="Không đọc được cấu hình/mapping local; bài chưa được gửi") from None
        raise HTTPException(status_code=422, detail=str(exc)) from None

    @router.get("/config")
    def config(response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            return load_dres_settings()
        except (OSError, ValueError):
            raise HTTPException(status_code=503, detail="Cấu hình máy chủ DRES không hợp lệ") from None

    @router.post("/connect")
    def connect(body: DRESConnectRequest, request: Request, response: Response):
        try:
            settings = load_dres_settings()
            if not settings["configured"]:
                raise DRESError("Cấu hình KIS_DRES_BASE_URL trên server trước khi kết nối", 503)
            run_id = evaluation_id(body.evaluation_id)
            client = client_factory(settings["base_url"])
            if body.session_id is not None:
                client.set_session(body.session_id.get_secret_value().strip())
                user = client.user()
            else:
                user = client.login(body.username, body.password.get_secret_value())
            connection = DRESConnection(client, run_id, user)
            result = connection.status()  # Read-only validation BEFORE storing a connection.
            token = store.add(connection, request.cookies.get(COOKIE_NAME))
        except (DRESError, ValueError, OSError) as exc:
            fail(exc)
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", max_age=CONNECTION_TTL, path="/api/dres")
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.get("/status")
    def status(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            return get_connection(request).status()
        except DRESError as exc:
            fail(exc)

    @router.post("/disconnect")
    def disconnect(request: Request, response: Response):
        store.remove(request.cookies.get(COOKIE_NAME))
        response.delete_cookie(COOKIE_NAME, path="/api/dres", httponly=True, samesite="strict")
        response.headers["Cache-Control"] = "no-store"
        return {"connected": False}

    @router.get("/task")
    def task(request: Request, response: Response, refresh: bool = False):
        response.headers["Cache-Control"] = "no-store"
        try:
            return get_connection(request).task(refresh=refresh)
        except DRESError as exc:
            fail(exc)

    @router.get("/task/media/{task_id}/{index}")
    def task_media(task_id: str, index: int, request: Request):
        try:
            raw, mime = get_connection(request).task_media(task_id, index)
            return Response(content=raw, media_type=mime, headers={
                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        except DRESError as exc:
            fail(exc)

    @router.post("/search")
    def search(body: DRESTaskSearch, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if search_service_provider is None:
            raise HTTPException(status_code=503, detail="Chưa cấu hình model tìm kiếm")
        try:
            return get_connection(request).search_task(body.task_id, search_service_provider(), body.video_description)
        except (DRESError, ValueError, OSError) as exc:
            fail(exc)
        except IncompleteResultPool as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except QueryPlanningError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except RuntimeError:
            raise HTTPException(status_code=503, detail="Model chưa sẵn sàng; kiểm tra Groq, Docker/Milvus và model cache") from None

    @router.post("/preview")
    def preview(body: DRESSelection, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            return get_connection(request).preview(body.model_dump(exclude={"task_id"}), groups_provider(), task_id=body.task_id)
        except (DRESError, ValueError, OSError) as exc:
            fail(exc)

    @router.post("/submit")
    def submit(body: DRESSubmitRequest, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if body.confirmed is not True:
            raise HTTPException(status_code=422, detail="Cần xác nhận trước khi gửi bài")
        try:
            return get_connection(request).submit_preview(body.preview_id, groups_provider())
        except (DRESError, ValueError, OSError) as exc:
            fail(exc)

    @router.post("/preview/text")
    def preview_text(body: DRESTextAnswer, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            return get_connection(request).preview_text(body.answer_text, task_id=body.task_id)
        except (DRESError, ValueError, OSError) as exc:
            fail(exc)

    return router
