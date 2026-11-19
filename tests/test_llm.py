from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
import os
import shutil
import socket
import ssl
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import certifi

from kis.llm import HTTPQueryStructurer, QueryAPIError, load_query_api_config, parse_query_json
from kis.structure_query import QUERY_RESPONSE_EXAMPLE, QUERY_RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, create_query_structurer, create_structured_query
from kis.paths import KIS_ROOT
from tests.test_structure_query import BIRD_QUERY, FakeGenerator, bird_generated_query, valid_generated_query


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return BytesIO(json.dumps(response).encode("utf-8"))


def api_generated_query():
    value = valid_generated_query()
    value["stream_queries"] = {stream: ["an artisan draws a portrait"] for stream in ("visual", "bm25", "semantic")}
    return value


def completion(value=None, **extra):
    if value is None:
        value = api_generated_query()
    return {"choices": [{"message": {"content": json.dumps(value)}, **extra}]}


def groq_generation_error():
    body = {"error": {
        "message": "Generated JSON does not match the expected schema. missing properties: visual_description, entities, needs_ocr, needs_asr, question, events, temporal_constraints, stream_queries",
        "code": "json_validate_failed",
        "failed_generation": json.dumps({"query_variants": ["a person beside a bicycle"]}),
    }}
    return HTTPError("https://example.com", 400, "Bad request", {}, BytesIO(json.dumps(body).encode()))


class QueryAPIConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.settings = Path(self.directory.name) / "llm.local.json"
        self.manifest = patch("kis.llm.QUERY_STRUCTURING_CONFIG", {"provider": "gemini", "model": "test-gemini"})
        self.manifest.start()
        self.addCleanup(self.manifest.stop)
        self.addCleanup(self.directory.cleanup)

    def config(self, env=None, **overrides):
        return load_query_api_config(overrides, environ=env or {}, settings_path=self.settings)

    def test_legacy_gemini_key_and_public_status_are_compatible(self):
        config = self.config({"GEMINI_API_KEY": "private-gemini-key"})
        self.assertTrue(config.configured)
        self.assertEqual((config.provider, config.model, config.api_key), ("gemini", "test-gemini", "private-gemini-key"))
        self.assertNotIn("private-gemini-key", repr(config))
        self.assertNotIn("private-gemini-key", json.dumps(config.public()))

    def test_changing_provider_does_not_inherit_gemini_model_or_key(self):
        config = self.config({"GEMINI_API_KEY": "google-secret", "KIS_LLM_PROVIDER": "groq", "GROQ_API_KEY": "groq-secret"})
        self.assertEqual((config.protocol, config.model, config.api_key), ("openai", "", "groq-secret"))
        with self.assertRaisesRegex(ValueError, "query model"):
            config.require_ready()

    def test_file_environment_and_per_call_precedence(self):
        self.settings.write_text(json.dumps({"provider": "groq", "model": "file-model", "api_key": "file-key"}))
        config = self.config({"KIS_LLM_MODEL": "env-model", "KIS_LLM_API_KEY": "env-key"}, model="request-model")
        self.assertEqual((config.provider, config.model, config.api_key), ("groq", "request-model", "env-key"))

    def test_custom_provider_and_complete_deployment_endpoint(self):
        config = self.config(provider="my-platform", model="model-x", api_key="custom-key",
                             base_url="https://example.com/deployments/x/chat/completions?api-version=2024-10-21")
        self.assertEqual(config.protocol, "openai")
        self.assertTrue(config.configured)
        azure = self.config(provider="azure", model="deployment-name", api_key="azure-key",
                            base_url="https://example.com/deployments/x/chat/completions?api-version=2024-10-21")
        self.assertEqual((azure.auth_header, azure.auth_prefix), ("api-key", ""))

    def test_endpoint_override_cannot_forward_server_credentials_or_headers(self):
        self.settings.write_text(json.dumps({"provider": "groq", "model": "server-model", "api_key": "secret",
                                             "headers": {"X-Private": "private-header"}}))
        config = self.config({"GROQ_API_KEY": "other-secret"}, base_url="https://other.example/v1")
        self.assertFalse(config.configured)
        self.assertEqual((config.api_key, config.headers), ("", {}))

    def test_provider_override_uses_its_own_key_not_generic_server_key(self):
        env = {"KIS_LLM_API_KEY": "server-google-key", "GROQ_API_KEY": "correct-groq-key"}
        config = self.config(env, provider="groq", model="request-model")
        self.assertEqual(config.api_key, "correct-groq-key")
        self.assertEqual(self.config(env, provider="openai", model="request-model").api_key, "")

    def test_explicit_key_environment_and_keyless_local_models(self):
        config = self.config({"MY_MODEL_KEY": "my-key"}, provider="custom", model="model", api_key_env="MY_MODEL_KEY",
                             base_url="https://example.com/v1")
        self.assertEqual(config.api_key, "my-key")
        local = self.config(provider="ollama", model="local-model")
        local.require_ready()
        self.assertEqual(local.api_key, "")

    def test_protocol_change_replaces_authentication_defaults(self):
        config = self.config(protocol="anthropic", base_url="https://example.com/v1", api_key="key")
        self.assertEqual((config.auth_header, config.auth_prefix), ("x-api-key", ""))

    def test_null_temperature_disables_the_parameter_from_file_or_overrides(self):
        self.settings.write_text(json.dumps({"temperature": None}))
        self.assertIsNone(self.config().temperature)
        self.settings.write_text("{}")
        self.assertIsNone(self.config(temperature=None).temperature)

    def test_invalid_settings_fail_before_any_api_request(self):
        for options in ({"protocol": "unsupported"}, {"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
                        {"max_tokens": 0}, {"base_url": "file:///credentials"},
                        {"base_url": "https://user:secret@example.com"},
                        {"base_url": "https://example.com?api_key=secret"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.config(**options)


class QueryTransportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_query_api_config(
            {"provider": "custom", "model": "mock-model", "api_key": "secret-key", "base_url": "https://example.com/v1", "response_format": "auto"},
            environ={}, settings_path=Path(self.directory.name) / "absent.json",
        )

    def generator(self, opener, **options):
        return HTTPQueryStructurer(replace(self.config, **options), SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, opener, example=QUERY_RESPONSE_EXAMPLE)

    def test_openai_request_keeps_untrusted_query_in_user_message(self):
        opener = FakeOpener(completion())
        result = create_structured_query("người vẽ chân dung", output_path=None, generator=self.generator(opener))
        request, timeout = opener.calls[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(request.get_header("User-agent"), "KIS-Framefinder/1.0")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["model"], "mock-model")
        self.assertNotIn("người vẽ chân dung", body["messages"][0]["content"])
        self.assertIn("người vẽ chân dung", body["messages"][1]["content"])
        self.assertEqual(result["raw_query"], "người vẽ chân dung")
        self.assertEqual(timeout, 60)

    def test_https_client_loads_explicit_ca_bundle_and_keeps_verification(self):
        verified_context = ssl.create_default_context(cafile=certifi.where())
        def reject_windows_store(**kwargs):
            if not kwargs.get("cafile"):
                raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA] malformed Windows certificate")
            return verified_context

        with patch.dict("os.environ", {}, clear=True), \
             patch("kis.llm.ssl.create_default_context", side_effect=reject_windows_store) as create_context:
            generator = HTTPQueryStructurer(self.config, SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA)
        create_context.assert_called_once_with(cafile=certifi.where())
        handler = next(item for item in generator.opener.handlers if hasattr(item, "_context"))
        self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handler._context.check_hostname)

    def test_custom_ca_bundle_is_respected_and_bad_bundle_is_reported(self):
        with patch.dict("os.environ", {"SSL_CERT_FILE": "company-ca.pem"}), \
             patch("kis.llm.ssl.create_default_context", side_effect=OSError("invalid CA file")) as create_context:
            with self.assertRaisesRegex(RuntimeError, "SSL_CERT_FILE"):
                HTTPQueryStructurer(self.config, SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA)
        create_context.assert_called_once_with(cafile="company-ca.pem")

    def test_network_errors_distinguish_tls_dns_and_timeout(self):
        cases = ((URLError(ssl.SSLCertVerificationError("certificate invalid")), "HTTPS certificate"),
                 (URLError(socket.gaierror("hostname not found")), "hostname"),
                 (URLError(TimeoutError()), "timed out"), (TimeoutError(), "timed out"))
        for error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.generator(FakeOpener(error)).generate("query")

    def test_strict_schema_supports_dynamic_attributes_without_changing_pipeline_shape(self):
        result = api_generated_query()
        result["entities"][0]["attributes"] = [{"key": "action", "value": "drawing a portrait"}]
        result["entities"][1]["attributes"] = []
        opener = FakeOpener(completion(result))
        generator = self.generator(opener, response_format="json_schema")
        structured = create_structured_query("query", output_path=None, generator=generator)
        wire = json.loads(opener.calls[0][0].data)["response_format"]["json_schema"]
        self.assertTrue(wire["strict"])
        self.assertEqual(wire["schema"]["properties"]["entities"]["items"]["properties"]["attributes"]["type"], "array")
        self.assertEqual(QUERY_RESPONSE_SCHEMA["properties"]["entities"]["items"]["properties"]["attributes"]["type"], "object")
        self.assertEqual(structured["entities"][0]["attributes"], {"action": "drawing a portrait"})

    def test_groq_gpt_oss_auto_mode_requests_strict_entity_objects(self):
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            with self.subTest(model=model):
                result = api_generated_query()
                for entity in result["entities"]:
                    entity["attributes"] = [{"key": key, "value": value} for key, value in entity["attributes"].items()]
                opener = FakeOpener(completion(result))
                generator = self.generator(opener, provider="groq", model=model, response_format="auto")
                structured = create_structured_query("query", output_path=None, generator=generator)
                wire = json.loads(opener.calls[0][0].data)["response_format"]
                self.assertEqual(wire["type"], "json_schema")
                self.assertTrue(wire["json_schema"]["strict"])
                self.assertEqual(wire["json_schema"]["schema"]["properties"]["entities"]["items"]["type"], "object")
                self.assertEqual(structured["entities"], valid_generated_query()["entities"])

    def test_groq_auto_strict_mode_is_not_downgraded_when_schema_is_rejected(self):
        error = HTTPError("https://example.com", 400, "Bad request", {}, BytesIO(b'{"error":{"message":"response_format schema rejected"}}'))
        opener = FakeOpener(error, completion())
        with self.assertRaises(QueryAPIError):
            self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="auto").generate("query")
        self.assertEqual(len(opener.calls), 1)

    def test_groq_failed_generation_is_repaired_once_with_complete_output(self):
        opener = FakeOpener(groq_generation_error(), completion())
        generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="auto")
        result = create_structured_query("người bên xe đạp", output_path=None, generator=generator)
        self.assertEqual(result["raw_query"], "người bên xe đạp")
        self.assertEqual(result["entities"], valid_generated_query()["entities"])
        self.assertEqual(len(opener.calls), 2)
        first, repaired = [json.loads(request.data) for request, _ in opener.calls]
        self.assertTrue(first["response_format"]["json_schema"]["strict"])
        self.assertEqual(repaired["response_format"]["type"], "json_schema")
        self.assertTrue(repaired["response_format"]["json_schema"]["strict"])
        self.assertIn("COMPLETE JSON object", repaired["messages"][0]["content"])
        self.assertIn("Complete example for a DIFFERENT query", repaired["messages"][0]["content"])
        self.assertIn("stream_queries", repaired["messages"][0]["content"])
        self.assertNotIn("người bên xe đạp", repaired["messages"][0]["content"])
        self.assertIn("người bên xe đạp", repaired["messages"][1]["content"])

    def test_groq_partial_success_is_repaired_with_the_same_schema(self):
        opener = FakeOpener(completion({"query_variants": ["partial result"]}), completion())
        generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="auto")
        result = create_structured_query("query", output_path=None, generator=generator)
        self.assertEqual(result["entities"], valid_generated_query()["entities"])
        self.assertEqual(len(opener.calls), 2)
        bodies = [json.loads(request.data) for request, _ in opener.calls]
        self.assertEqual(bodies[0]["response_format"], bodies[1]["response_format"])
        for body in bodies:
            marker = "Complete example for a DIFFERENT query; use its structure, never copy its facts:\n"
            text = body["messages"][0]["content"].split(marker, 1)[1]
            example, _ = json.JSONDecoder().raw_decode(text)
            self.assertIsInstance(example["entities"][0]["attributes"], list)
        self.assertIsInstance(QUERY_RESPONSE_EXAMPLE["entities"][0]["attributes"], dict)

    def test_bird_query_fifth_entity_is_canonical_on_first_or_repair_response(self):
        for first_error in (False, True):
            with self.subTest(repair=first_error):
                responses = [completion(bird_generated_query())]
                if first_error:
                    responses.insert(0, groq_generation_error())
                opener = FakeOpener(*responses)
                generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="json_schema")
                result = create_structured_query(BIRD_QUERY, output_path=None, generator=generator)
                self.assertEqual(result["raw_query"], BIRD_QUERY)
                self.assertEqual(len(result["entities"]), 5)
                self.assertEqual(result["entities"][4], {"type": "concept", "value": "vivid red eyes", "attributes": {}})
                self.assertEqual(len(opener.calls), 2 if first_error else 1)
                self.assertTrue(all(json.loads(request.data)["response_format"]["type"] == "json_schema" for request, _ in opener.calls))

    def test_complete_failed_generation_can_be_recovered_locally_without_another_call(self):
        for first_error in (False, True):
            with self.subTest(second_attempt=first_error):
                body = {"error": {"code": "json_validate_failed", "message": "Generated JSON does not match the expected schema.",
                                  "failed_generation": json.dumps(bird_generated_query(), ensure_ascii=False)}}
                error = HTTPError("https://example.com", 400, "error", {}, BytesIO(json.dumps(body).encode()))
                responses = [error]
                if first_error:
                    responses.insert(0, groq_generation_error())
                opener = FakeOpener(*responses)
                generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="json_schema")
                result = create_structured_query(BIRD_QUERY, output_path=None, generator=generator)
                self.assertEqual(result["entities"][4]["value"], "vivid red eyes")
                self.assertEqual(len(opener.calls), 2 if first_error else 1)

    def test_empty_model_completion_has_one_structured_repair(self):
        opener = FakeOpener({"choices": [{"message": {"content": ""}}]}, completion())
        generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="json_schema")
        result = create_structured_query("query", output_path=None, generator=generator)
        self.assertEqual(result["entities"], valid_generated_query()["entities"])
        self.assertEqual(len(opener.calls), 2)

    def test_groq_invalid_repair_does_not_publish_or_retry_again(self):
        malformed = api_generated_query()
        malformed["needs_ocr"] = "not-a-boolean"
        for second in (completion(malformed), completion(valid_generated_query()), groq_generation_error()):
            with self.subTest(second=type(second).__name__):
                output = Path(self.directory.name) / "query.json"
                output.write_text('{"state":"old"}')
                opener = FakeOpener(groq_generation_error(), second)
                generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="auto")
                with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                    create_structured_query("query", output_path=output, generator=generator)
                self.assertEqual(json.loads(output.read_text()), {"state": "old"})
                self.assertEqual(len(opener.calls), 2)

    def test_groq_auth_quota_or_invalid_schema_errors_do_not_trigger_repair(self):
        for status, code, message in ((401, "invalid_api_key", "Invalid key"),
                                      (429, "rate_limit_exceeded", "Quota exceeded"),
                                      (400, "invalid_json_schema", "response_format schema rejected")):
            with self.subTest(status=status):
                error = HTTPError("https://example.com", status, "error", {}, BytesIO(json.dumps({"error": {"code": code, "message": message}}).encode()))
                opener = FakeOpener(error, completion())
                generator = self.generator(opener, provider="groq", model="openai/gpt-oss-20b", response_format="auto")
                with self.assertRaises(QueryAPIError):
                    create_structured_query("query", output_path=None, generator=generator)
                self.assertEqual(len(opener.calls), 1)

    def test_failed_generation_metadata_is_redacted_and_not_printed(self):
        body = {"error": {"code": "json_validate_failed", "message": "invalid model JSON",
                          "failed_generation": '{"query_variants":["secret-key"]}'}}
        error = HTTPError("https://example.com", 400, "error", {}, BytesIO(json.dumps(body).encode()))
        with self.assertRaises(QueryAPIError) as caught:
            self.generator(FakeOpener(error)).generate("query")
        self.assertEqual(caught.exception.code, "json_validate_failed")
        self.assertNotIn("secret-key", caught.exception.failed_generation)
        self.assertNotIn("query_variants", str(caught.exception))

    def test_anthropic_requests_one_structured_tool_result(self):
        opener = FakeOpener({"content": [{"type": "tool_use", "name": "structure_query", "input": valid_generated_query()}]})
        generator = self.generator(opener, protocol="anthropic", auth_header="x-api-key", auth_prefix="")
        self.assertEqual(generator.generate("query"), valid_generated_query())
        request = opener.calls[0][0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.com/v1/messages")
        self.assertEqual(request.get_header("X-api-key"), "secret-key")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(body["tools"][0]["input_schema"], QUERY_RESPONSE_SCHEMA)
        self.assertEqual(body["tool_choice"]["name"], "structure_query")

    def test_full_endpoint_custom_auth_and_completion_token_parameter(self):
        opener = FakeOpener(completion())
        generator = self.generator(opener, base_url="https://example.com/v1/chat/completions?api-version=1",
                                   auth_header="api-key", auth_prefix="", token_parameter="max_completion_tokens", temperature=None)
        generator.generate("query")
        request = opener.calls[0][0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions?api-version=1")
        self.assertEqual(request.get_header("Api-key"), "secret-key")
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("temperature", body)

    def test_auto_falls_back_only_when_json_mode_is_rejected(self):
        error = HTTPError("https://example.com", 400, "Bad request", {}, BytesIO(b'{"error":{"message":"response_format unsupported"}}'))
        opener = FakeOpener(error, completion())
        self.generator(opener).generate("query")
        self.assertEqual(len(opener.calls), 2)
        self.assertNotIn("response_format", json.loads(opener.calls[1][0].data))

    def test_quota_error_is_not_retried_and_does_not_expose_key(self):
        error = HTTPError("https://example.com", 429, "Rate limited", {}, BytesIO(b'{"error":{"message":"quota exhausted for secret-key"}}'))
        opener = FakeOpener(error)
        with self.assertRaises(QueryAPIError) as caught:
            self.generator(opener).generate("query")
        self.assertEqual(caught.exception.status, 429)
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertEqual(len(opener.calls), 1)

    def test_text_only_mode_and_fenced_content_blocks(self):
        response = {"choices": [{"message": {"content": [{"type": "text", "text": "```json\n" + json.dumps(valid_generated_query()) + "\n```"}]}}]}
        opener = FakeOpener(response)
        self.assertEqual(self.generator(opener, response_format="text").generate("query"), valid_generated_query())
        self.assertNotIn("response_format", json.loads(opener.calls[0][0].data))

    def test_truncated_invalid_or_empty_output_is_rejected(self):
        for response in (completion(finish_reason="length"), {"choices": []}, {"choices": [{"message": {"content": ""}}]}):
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.generator(FakeOpener(response)).generate("query")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            parse_query_json('{"unfinished":')
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_query_json("[]")

    def test_factory_selects_native_gemini_and_default_generation_uses_factory(self):
        with patch("kis.structure_query.GeminiQueryStructurer") as gemini:
            create_query_structurer(replace(self.config, protocol="gemini"))
        self.assertEqual(gemini.call_args.kwargs["api_key"], "secret-key")
        with patch("kis.structure_query.create_query_structurer", return_value=FakeGenerator(deepcopy(valid_generated_query()))) as factory:
            result = create_structured_query("query", output_path=None)
        factory.assert_called_once_with()
        self.assertEqual(result["raw_query"], "query")


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "requires Windows PowerShell")
class GroqLauncherTest(unittest.TestCase):
    def test_launcher_reads_windows_user_key_without_inherited_process_key(self):
        presence = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-Command",
            "if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable('GROQ_API_KEY','User'))) { exit 2 }"],
            capture_output=True, text=True, timeout=30)
        if presence.returncode == 2:
            self.skipTest("No Windows User Groq key configured; test never writes real credentials")
        self.assertEqual(presence.returncode, 0, presence.stderr)
        with tempfile.TemporaryDirectory(prefix="kis saved key test ") as directory:
            probe = Path(directory) / "probe.cmd"
            probe.write_text(
                '@echo off\npowershell.exe -NoProfile -Command "'
                "$saved = [Environment]::GetEnvironmentVariable('GROQ_API_KEY','User'); "
                "if ([string]::IsNullOrWhiteSpace($saved)) { exit 10 }; "
                "if ($env:GROQ_API_KEY -cne $saved.Trim()) { exit 11 }; "
                "if ($env:KIS_LLM_API_KEY -cne $env:GROQ_API_KEY) { exit 12 }; "
                "'Saved key loaded without printing it'" + '"\nexit /b %ERRORLEVEL%\n', encoding="ascii",
            )
            env = dict(os.environ)
            env.pop("GROQ_API_KEY", None)
            env.pop("KIS_LLM_API_KEY", None)
            command = ("function Read-Host { throw 'Must not prompt when a User key is configured' }; "
                       f"& '{KIS_ROOT / 'start_kis.ps1'}' -Provider groq -Python '{probe}'")
            result = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
                                    env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "Saved key loaded without printing it")

    def test_launcher_uses_groq_environment_key_or_masked_prompt(self):
        with tempfile.TemporaryDirectory(prefix="kis launch test ") as directory:
            probe = Path(directory) / "probe.cmd"
            probe.write_text(
                '@echo off\n'
                'powershell.exe -NoProfile -Command "'
                "if ($env:KIS_LLM_PROVIDER -ne 'groq') { exit 10 }; "
                "if ($env:KIS_LLM_MODEL -ne 'openai/gpt-oss-20b') { exit 11 }; "
                "if ($env:KIS_LLM_BASE_URL -ne 'https://api.groq.com/openai/v1') { exit 12 }; "
                "if ($env:KIS_LLM_RESPONSE_FORMAT -ne 'json_schema') { exit 16 }; "
                "if ($env:GROQ_API_KEY -ne 'launch-test-key') { exit 13 }; "
                "if ($env:KIS_LLM_API_KEY -ne $env:GROQ_API_KEY) { exit 14 }; "
                "if ($env:EXPECT_PROMPT -ne $env:TEST_PROMPT_USED) { exit 15 }; "
                "'Groq launcher OK'" + '"\nexit /b %ERRORLEVEL%\n', encoding="ascii",
            )
            for existing_key in (True, False):
                with self.subTest(existing_key=existing_key):
                    env = dict(os.environ)
                    env.pop("GROQ_API_KEY", None)
                    env["KIS_LLM_PROVIDER"] = "gemini"
                    env["KIS_LLM_RESPONSE_FORMAT"] = "json_object"
                    env["KIS_LLM_API_KEY"] = "old-test-key"
                    env["EXPECT_PROMPT"] = "no" if existing_key else "yes"
                    env["TEST_PROMPT_USED"] = "no"
                    if existing_key:
                        env["GROQ_API_KEY"] = "launch-test-key"
                    command = (
                        "function Read-Host { param([string]$Prompt, [switch]$AsSecureString); "
                        "if (-not $AsSecureString) { throw 'Prompt must mask the key' }; "
                        "$env:TEST_PROMPT_USED = 'yes'; "
                        "ConvertTo-SecureString 'launch-test-key' -AsPlainText -Force }; "
                        f"& '{KIS_ROOT / 'start_kis.ps1'}' -Provider groq -Python '{probe}'"
                        + (" -PromptForKey" if not existing_key else "")
                    )
                    result = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
                                            env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Groq launcher OK", result.stdout)
                    self.assertNotIn("launch-test-key", result.stdout)

    def test_check_api_mode_uses_cli_no_save_without_starting_web(self):
        with tempfile.TemporaryDirectory(prefix="kis api check ") as directory:
            probe = Path(directory) / "probe.cmd"
            probe.write_text(
                '@echo off\n'
                'if not "%1"=="-m" exit /b 10\n'
                'if not "%2"=="kis.structure_query" exit /b 11\n'
                'if not "%3"=="--no-save" exit /b 12\n'
                'echo Query API check OK\nexit /b 0\n', encoding="ascii",
            )
            env = {**os.environ, "GROQ_API_KEY": "launch-test-key"}
            command = f"& '{KIS_ROOT / 'start_kis.ps1'}' -Provider groq -Python '{probe}' -CheckApi"
            result = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Query API check OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
