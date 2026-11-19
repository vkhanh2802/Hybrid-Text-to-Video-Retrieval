from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kis.llm import QueryAPIError, load_query_api_config
from kis.planner_fallback import (
    FallbackConfig, OpenCodeQueryStructurer, QueryPlanningError,
    load_fallback_config, plan_query,
)
from kis.structure_query import create_structured_query, default_query_id
from tests.test_llm import api_generated_query


class PlannerFallbackTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(provider="groq", protocol="openai", model="openai/gpt-oss-20b")
        self.fallback = FallbackConfig(True, "test-opencode.exe", "openai/gpt-6.1-sol", 90)

    def test_settings_are_opt_in_and_reject_secrets_or_bad_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            self.assertFalse(load_fallback_config(path).enabled)
            path.write_text(json.dumps({"enabled": True, "model": self.fallback.model}))
            self.assertTrue(load_fallback_config(path).enabled)
            for value in ({"api_key": "must-not-be-stored"}, {"enabled": "true"},
                          {"timeout_seconds": 0}, {"model": "no-provider"}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_fallback_config(path)

    def test_primary_success_never_loads_or_calls_fallback(self):
        expected = {"raw_query": "query"}
        primary = Mock(return_value=expected)
        with patch("kis.planner_fallback.load_fallback_config") as load:
            result, used, info = plan_query("query", self.config, primary)
        self.assertIs(result, expected)
        self.assertIs(used, self.config)
        self.assertFalse(info["fallback_used"])
        load.assert_not_called()

    def test_dns_auth_rate_limit_timeout_and_invalid_output_use_validated_fallback(self):
        errors = [(RuntimeError("Cannot resolve groq API hostname; check base URL and DNS"), "dns"),
                  (RuntimeError("groq API connection timed out"), "timeout"),
                  (QueryAPIError("groq", 401, "private rejected key"), "http_401"),
                  (QueryAPIError("groq", 429, "rate limited"), "http_429"),
                  (ValueError("invalid JSON"), "invalid_output")]
        for error, reason in errors:
            with self.subTest(reason=reason), \
                    patch("kis.planner_fallback.load_fallback_config", return_value=self.fallback), \
                    patch.object(OpenCodeQueryStructurer, "generate", return_value=api_generated_query()) as generate:
                result, used, info = plan_query("original input", self.config, Mock(side_effect=error))
            self.assertEqual(result["raw_query"], "original input")
            self.assertEqual(result["query_id"], default_query_id("original input"))
            self.assertEqual(used.provider, "opencode")
            self.assertEqual(info["reason"], reason)
            self.assertTrue(info["fallback_used"])
            self.assertNotIn("private rejected key", json.dumps(info))
            generate.assert_called_once_with("original input")

    def test_gemini_primary_outage_also_uses_fallback(self):
        gemini = SimpleNamespace(provider="gemini", protocol="gemini", model="gemini-flash-lite-latest")
        with patch("kis.planner_fallback.load_fallback_config", return_value=self.fallback),                 patch.object(OpenCodeQueryStructurer, "generate", return_value=api_generated_query()):
            _, used, info = plan_query("query", gemini, Mock(side_effect=RuntimeError("Gemini query API failed: 503 UNAVAILABLE")))
        self.assertEqual(used.provider, "opencode")
        self.assertTrue(info["fallback_used"])

    def test_disabled_explicit_override_and_non_groq_do_not_fall_back(self):
        for enabled, allowed, provider in [(False, True, "groq"), (True, False, "groq"),
                                           (True, True, "custom")]:
            config = SimpleNamespace(provider=provider, model="test")
            with patch("kis.planner_fallback.load_fallback_config", return_value=replace(self.fallback, enabled=enabled)), \
                    patch.object(OpenCodeQueryStructurer, "generate") as generate:
                with self.assertRaisesRegex(RuntimeError, "primary failed"):
                    plan_query("query", config, Mock(side_effect=RuntimeError("primary failed")), allow_fallback=allowed)
            generate.assert_not_called()

    def test_invalid_user_input_never_calls_primary_or_fallback(self):
        primary = Mock()
        for query in ("", "x" * 8001, None):
            with self.assertRaises((ValueError, TypeError)):
                plan_query(query, self.config, primary)
        primary.assert_not_called()

    def test_fallback_invalid_schema_fails_closed_with_no_secret_leak(self):
        with patch("kis.planner_fallback.load_fallback_config", return_value=self.fallback), \
                patch.object(OpenCodeQueryStructurer, "generate", return_value={"query_variants": ["incomplete"]}):
            with self.assertRaises(QueryPlanningError) as error:
                plan_query("query", self.config, Mock(side_effect=RuntimeError("private-token")))
        self.assertNotIn("private-token", str(error.exception))

    def test_each_query_retries_primary_instead_of_sticking_to_fallback(self):
        primary = Mock(side_effect=[RuntimeError("Cannot resolve groq"), {"raw_query": "second"}])
        with patch("kis.planner_fallback.load_fallback_config", return_value=self.fallback), \
                patch.object(OpenCodeQueryStructurer, "generate", return_value=api_generated_query()) as generate:
            first = plan_query("first", self.config, primary)
            second = plan_query("second", self.config, primary)
        self.assertTrue(first[2]["fallback_used"])
        self.assertFalse(second[2]["fallback_used"])
        self.assertEqual(primary.call_count, 2)
        self.assertEqual(generate.call_count, 1)

    def test_cli_uses_stateless_endpoint_without_shell_or_forwarded_credentials(self):
        response = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "data": {"text": json.dumps(api_generated_query())}}))
        env = {"PATH": "safe-path", "GROQ_API_KEY": "groq-secret", "KIS_WORKER_TOKEN": "worker-secret",
               "DRES_SESSION": "dres-secret", "OPENAI_API_KEY": "other-secret"}
        with patch.dict("os.environ", env, clear=True), patch.object(Path, "is_file", return_value=True), \
                patch("kis.planner_fallback.subprocess.run", return_value=response) as run:
            result = create_structured_query("full input", output_path=None, generator=OpenCodeQueryStructurer(self.fallback))
        args, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(args[1:4], ["api", "post", "/api/experimental/generate/text"])
        body = json.loads(args[-1])
        self.assertEqual(set(body), {"prompt", "model"})
        self.assertEqual(body["model"], {"providerID": "openai", "id": "gpt-6.1-sol"})
        self.assertIn('"full input"', body["prompt"])
        self.assertEqual(kwargs["env"], {"PATH": "safe-path"})
        self.assertFalse(kwargs["shell"])
        self.assertEqual(result["raw_query"], "full input")

    def test_cli_failures_are_bounded_and_do_not_expose_stdout_stderr(self):
        responses = [SimpleNamespace(returncode=1, stderr="HTTP 502 secret-token", stdout="private-output"),
                     SimpleNamespace(returncode=0, stderr="", stdout='{"data":{"text":"not JSON secret-token"}}'),
                     subprocess.TimeoutExpired("opencode", 90)]
        for response in responses:
            with patch.object(Path, "is_file", return_value=True), \
                    patch("kis.planner_fallback.subprocess.run", side_effect=response if isinstance(response, Exception) else None,
                          return_value=response):
                with self.assertRaises(QueryPlanningError) as error:
                    OpenCodeQueryStructurer(self.fallback).generate("query")
            self.assertNotIn("secret-token", str(error.exception))
            self.assertNotIn("private-output", str(error.exception))

    def test_missing_or_shell_cli_is_rejected_before_subprocess(self):
        for path in ("missing.exe", "opencode.cmd", "opencode.bat"):
            with patch.object(Path, "is_file", return_value=path != "missing.exe"), \
                    patch("kis.planner_fallback.subprocess.run") as run:
                with self.assertRaises(QueryPlanningError):
                    OpenCodeQueryStructurer(replace(self.fallback, cli_path=path)).generate("query")
            run.assert_not_called()

    def test_web_fallback_runs_only_for_planner_and_preserves_twenty_vlm_budget(self):
        from kis import web_app
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            service = web_app.SearchService()
        service._load = Mock()
        candidates = [{"video_id": "L01_V001", "frame_id": i, "keyframe_n": i + 1,
                       "score": .8, "final_score": .8} for i in range(200)]
        with patch("kis.structure_query.create_query_structurer", side_effect=RuntimeError("Cannot resolve groq")), \
                patch("kis.planner_fallback.load_fallback_config", return_value=self.fallback), \
                patch.object(OpenCodeQueryStructurer, "generate", return_value=api_generated_query()), \
                patch.object(web_app, "solve_kis", return_value=candidates) as solve:
            result = service.search("original query")
        self.assertEqual(len(result["results"]), 200)
        self.assertTrue(result["query_planner"]["fallback_used"])
        self.assertEqual(result["query_api"]["provider"], "opencode")
        self.assertEqual(solve.call_args.kwargs["structured_query"]["raw_query"], "original query")
        self.assertEqual(solve.call_args.kwargs["rerank_top_k"], 20)
        service.query_structurer = SimpleNamespace(config=load_query_api_config(overrides={"api_key": "test-key"}))
        with patch.object(web_app, "create_structured_query", return_value={"raw_query": "query"}), \
                patch.object(web_app, "solve_kis", side_effect=RuntimeError("CPU tunnel failed")), \
                patch.object(OpenCodeQueryStructurer, "generate") as fallback:
            with self.assertRaisesRegex(RuntimeError, "CPU tunnel failed"):
                service.search("query")
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
