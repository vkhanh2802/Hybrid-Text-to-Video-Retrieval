"""Validated Groq planning with an explicit, stateless OpenCode fallback.

Only query text and the public schema are sent to OpenCode. No agent session,
tools, DRES submission, or credential export is involved.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter
from types import SimpleNamespace

from kis.llm import QueryAPIError, QueryOutputError, parse_query_json
from kis.paths import KIS_ROOT


class QueryPlanningError(RuntimeError):
    """A safe user-facing planner error, with no raw provider/CLI output."""


@dataclass(frozen=True)
class FallbackConfig:
    enabled: bool = False
    cli_path: str = ""
    model: str = ""
    timeout_seconds: float = 90

    def public(self):
        return {"enabled": self.enabled, "provider": "opencode" if self.enabled else None,
                "model": self.model or "service-default", "timeout_seconds": self.timeout_seconds}


def load_fallback_config(path=None):
    path = Path(path) if path else KIS_ROOT / "config/planner-fallback.local.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(data, dict) or set(data) - {"enabled", "cli_path", "model", "timeout_seconds"}:
        raise ValueError("Invalid planner fallback settings")
    enabled = data.get("enabled", False)
    timeout = data.get("timeout_seconds", 90)
    model, cli = data.get("model", ""), data.get("cli_path", "")
    if (type(enabled) is not bool or type(timeout) not in (int, float) or not 1 <= timeout <= 180
            or not isinstance(model, str) or not isinstance(cli, str)):
        raise ValueError("Invalid planner fallback settings")
    if model and not re.fullmatch(r"[a-zA-Z0-9_-]+/[^\s\x00-\x1f]+", model):
        raise ValueError("OpenCode fallback model must be providerID/modelID")
    return FallbackConfig(enabled, cli, model, timeout)


def failure_reason(error):
    if isinstance(error, QueryAPIError):
        return f"http_{error.status}"
    if isinstance(error, (ValueError, QueryOutputError)):
        return "invalid_output"
    text = str(error).lower()
    if "cannot resolve" in text or "getaddrinfo" in text:
        return "dns"
    if "timed out" in text or isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    return "unavailable"


class OpenCodeQueryStructurer:
    def __init__(self, config):
        self.config = config
        self.model = config.model or "service-default"

    def generate(self, query):
        from kis.structure_query import SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, canonicalize_model_query

        cli = self.config.cli_path or shutil.which("opencode")
        if not cli or not Path(cli).is_file() or Path(cli).suffix.lower() in {".cmd", ".bat"}:
            raise QueryPlanningError("OpenCode fallback: không tìm thấy CLI thực thi; kiểm tra cấu hình local.")
        prompt = (SYSTEM_INSTRUCTION
                  + "\nReturn exactly ONE COMPLETE JSON object matching this schema. No markdown or explanation.\n"
                  + json.dumps(QUERY_RESPONSE_SCHEMA, separators=(",", ":"))
                  + "\nRAW_QUERY_JSON:\n" + json.dumps(query, ensure_ascii=False))
        body = {"prompt": prompt}
        if self.config.model:
            provider, model = self.config.model.split("/", 1)
            body["model"] = {"providerID": provider, "id": model}
        args = [str(cli), "api", "post", "/api/experimental/generate/text", "--data",
                json.dumps(body, ensure_ascii=False)]
        if os.name == "nt" and len(subprocess.list2cmdline(args)) > 30000:
            raise QueryPlanningError("OpenCode fallback: query vượt giới hạn lệnh Windows; hãy rút ngắn query.")
        # CLI owns service discovery/auth. Never forward KIS, Groq or DRES secrets.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("KIS_", "GROQ_", "GEMINI_", "DRES_"))
               and k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODEL_API_KEY", "META_API_KEY"}}
        try:
            result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=self.config.timeout_seconds,
                                    env=env, shell=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            raise QueryPlanningError("OpenCode fallback: quá thời gian chờ; chưa tạo được query.") from None
        except OSError:
            raise QueryPlanningError("OpenCode fallback: không khởi chạy được CLI.") from None
        if result.returncode:
            match = re.search(r"\bHTTP\s+(\d{3})\b", result.stderr or "")
            code = f" (HTTP {match.group(1)})" if match else ""
            raise QueryPlanningError(f"OpenCode fallback: service/model không khả dụng{code}.")
        if len(result.stdout) > 2 * 1024 * 1024:
            raise QueryPlanningError("OpenCode fallback: phản hồi vượt giới hạn.")
        try:
            reply = json.loads(result.stdout)
            text = reply["data"]["text"]
            generated = parse_query_json(text)
            return canonicalize_model_query(generated)
        except (KeyError, TypeError, ValueError, QueryOutputError):
            raise QueryPlanningError("OpenCode fallback: không trả về structured query JSON hợp lệ.") from None


FALLBACK_PROVIDERS = {"groq", "gemini"}  # hosted primaries whose outages the OpenCode fallback covers


def plan_query(raw_query, primary_config, primary_call, *, allow_fallback=True):
    from kis.structure_query import create_structured_query, validate_raw_query

    raw_query = validate_raw_query(raw_query)  # Invalid user input must not trigger model calls.
    info = {"primary": {"provider": primary_config.provider, "model": primary_config.model},
            "used": {"provider": primary_config.provider, "model": primary_config.model},
            "fallback_used": False, "reason": None, "primary_seconds": 0.0, "fallback_seconds": 0.0}
    started = perf_counter()
    try:
        structured = primary_call()
    except (RuntimeError, ValueError, OSError) as error:
        info["primary_seconds"] = round(perf_counter() - started, 3)
        reason = failure_reason(error)
        if not allow_fallback or primary_config.provider not in FALLBACK_PROVIDERS:
            raise
        try:
            fallback = load_fallback_config()
        except (OSError, ValueError):
            raise QueryPlanningError(f"{primary_config.provider} không tạo được query; cấu hình OpenCode fallback không hợp lệ.") from None
        if not fallback.enabled:
            raise
        fallback_started = perf_counter()
        try:
            generator = OpenCodeQueryStructurer(fallback)
            structured = create_structured_query(raw_query, output_path=None, generator=generator)
        except (RuntimeError, ValueError, OSError):
            raise QueryPlanningError(
                f"{primary_config.provider} không tạo được query ({reason}); OpenCode fallback cũng thất bại. "
                "Kiểm tra kết nối/model của OpenCode rồi thử lại; hệ thống chưa chạy truy hồi.") from None
        info.update(fallback_used=True, reason=reason,
                    used={"provider": "opencode", "model": generator.model},
                    fallback_seconds=round(perf_counter() - fallback_started, 3))
        return structured, SimpleNamespace(provider="opencode", protocol="opencode", model=generator.model), info
    info["primary_seconds"] = round(perf_counter() - started, 3)
    return structured, primary_config, info
