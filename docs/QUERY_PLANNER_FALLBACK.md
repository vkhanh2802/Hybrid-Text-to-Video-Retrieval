# Groq primary, OpenCode fallback

The text search path tries the configured Groq planner first on **every query**.
If Groq fails to create a validated structured query, the explicitly enabled
OpenCode fallback gets one attempt. DNS/network errors, timeout, HTTP rejection,
and exhausted JSON/schema validation failure can trigger fallback. CPU/image/
Qwen failures do not. Explicit per-request provider overrides do not silently
fall back. Invalid/oversized user input never reaches either model.

## Machine-local configuration

Create `config/planner-fallback.local.json` (gitignored) using
`config/planner-fallback.example.json`. Fields:

- `enabled`: boolean; defaults to false when the local file is absent.
- `cli_path`: path to the actual OpenCode executable. Empty uses `opencode`
  from PATH. On Windows use an `.exe`, not a shell `.cmd`/`.bat` wrapper.
- `model`: an existing connected `providerID/modelID`. Empty uses the OpenCode
  service's base default, which need not be the model in the current chat.
- `timeout_seconds`: 1–180, default 90. Groq retains its own timeout/repair budget.

No Muse token, Groq key, or OpenCode provider key belongs in this file.
Groq remains configured via `kis.llm.load_query_api_config`; do not change
`config/models.json` or the CPU worker model manifest to enable this fallback.

Choose a fallback model that reliably emits the full KIS schema. A small model
can return JSON that is syntactically valid but fails schema/canonicalization.
The current local configuration uses `groq/openai/gpt-oss-120b` through
OpenCode; OpenCode's transport still works when KIS's direct Groq DNS lookup
fails. It is still not independent of Groq the vendor, so a total Groq outage
will disable both planners.

## Transport and validation

The adapter uses the installed OpenCode V2 CLI's authenticated service discovery:

```text
opencode api post /api/experimental/generate/text --data <JSON>
```

The body contains only `prompt` and optional `model: {providerID, id}`. The prompt
contains the KIS instructions/schema and the exact text query. The stateless
generation route does not create an agent session or request agent tools.
Provider authentication stays in OpenCode; KIS does not extract its credentials.
KIS/Groq/DRES secrets are removed from the child environment. The command uses
`shell=False`, a bounded timeout and a bounded accepted response size. Raw CLI
stdout/stderr and prompts are not included in user-facing error messages.

OpenCode returns `data.text`; KIS parses, canonicalizes and validates it through
the same `create_structured_query` path as primary planning. There is no fallback
from failed schema validation to arbitrary plain-text retrieval. Failure of both
planners is explicit and stops before retrieval.

This avoids the Groq provider dependency on a fallback request, **not Internet
dependency in general**. OpenCode service and the selected provider/model must
be available. Generation uses that provider's allowance/billing. The endpoint
is experimental; confirmed against the installed OpenCode 2.2.5 OpenAPI contract.

## Reporting

Text and text-guided video responses include `query_planner`:

```json
{
  "primary": {"provider": "groq", "model": "openai/gpt-oss-20b"},
  "used": {"provider": "opencode", "model": "providerID/modelID"},
  "fallback_used": true,
  "reason": "dns",
  "primary_seconds": 0.0,
  "fallback_seconds": 0.0
}
```

Times above are illustrative. `query_api` reports the planner actually used.
`/api/status` reports the fallback configuration and last successful planning
metadata. The frontend displays the provider and fallback reason explicitly.
Pure-video queries still use no text planner. The 200-frame result contract,
20-image Qwen budget, 70/30 text-video fusion and DRES submission safeguards remain.

Changing the fallback local file is read on the next fallback attempt. Deploying
new Python code requires restarting KIS web, not the CPU worker or OpenCode.
