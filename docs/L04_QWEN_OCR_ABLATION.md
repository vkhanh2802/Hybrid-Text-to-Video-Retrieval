# L04 local OFFLINE Qwen OCR A/B — 2026-11-12

## Actual run outcome: PARTIAL — no accuracy claim

The real local worker **loaded the pinned model successfully** and passed the
first A/B pair smoke. It then stopped fail-closed after **74/640 valid arm scores
(37 complete pairs; 37 A + 37 B)**. This is not a completed benchmark.

The **75th attempted call**, query **t02**, baseline candidate position **18**,
arm **B**, global pair index **37**, returned:

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

That is the exact raw output: 32 exclamation marks, using all 32 generated-token
slots, not a JSON score. The whole-response JSON parser raised
`JSONDecodeError: Expecting value: line 1 column 1 (char 0)` and the worker
raised `RuntimeError: Invalid strict JSON score; stopped, raw preserved, no zero fallback`.
The worker exited 1; the run CLI/background command exited 2. `report` succeeded
and wrote `status: PARTIAL_NO_ACCURACY_CLAIM`, **`metrics: null`**. Full-query
coverage is one completed query (t01); t02 and the other planned queries cannot
support the primary all16 comparison. No partial accuracy metrics were computed.

This observed failure is **invalid model output**, not a recorded CUDA OOM or
VRAM-gate failure. Its underlying generation cause is not established. The code,
prompt, query set, token limit, quantization and scoring configuration were **not
changed or retried after observing it**. Although `--resume` exists for an
interruption, automatically retrying this deterministic invalid response is not
a remedy and was not attempted. Any independently justified future experiment
must use a new frozen protocol/output rather than silently replacing this run.

Actual model load: **18.9184 s**. Model memory footprint: **2,347,450,960 bytes**.
Free CUDA memory before loading: **5,404,360,704 bytes**; frozen safe budget:
**4,733,272,064 bytes**. The peak reserved memory among valid calls was
**2,847,932,416 bytes**. Completed-call runtime totals were **48.2259 s for A**
and **49.1298 s for B**; these are partial-run runtime facts, not accuracy or
benefit estimates. The failed B call took 2.8874 s and had 664 input tokens.
Its failure record retains `score: null`, never an artificial zero.

Audit artifacts in the output directory: `hardware.json`, `smoke.json`,
`checkpoint.json` (74 strict-valid raw outputs), `failed_call.json` (exact invalid
output), `worker_process.json` (full stdout/stderr), and `report.json`/`report.md`.
The existing frozen packet, protocol, config, pool, input and preservation hashes
remain unchanged. **There is no valid conclusion that Qwen OCR B helps or hurts.**

This is an isolated experimental harness, **not a production pipeline change**.
It uses only the existing `focusguard_ai` environment and exact existing cache.
It does not install/update packages, change the environment, download weights,
call APIs, build indexes, kill processes, or use `KMP_DUPLICATE_LIB_OK`.

## Frozen comparison

- Primary set: **originalunchanged16**, the explicit IDs in
  `benchmarks/l04_qwen_ocr_ablation_20261112_v1.protocol.json`.
  The original t06/t10 are excluded, not rewritten. The other query objects and
  every qrel are checked against the original v2 packet and retained unchanged.
- These are already-seen assistant-authored queries with incomplete qrels,
  **not new held-out data**. Unjudged results are not negatives.
- Read only the four explicit `dataset/{caption_generator,mapping,data,ocr}/L04`
  roots via the existing guarded adapter. No other dataset group is discovered.
- Retrieval is caption-only BM25, positive finite scores only, stable identity
  ties, **top20 once**, with no OCR retrieval, label/target injection or separate
  per-arm pool. There are 320 frozen query/candidate pairs, 640 planned arm scores.
- A = query + image + bounded stored caption. B adds only one separate bounded
  OCR JSON block, including its diagnostic status. Same system instructions,
  caption/query bounds, images, processor limits, scoring scale and generation.
  Captions can naturally mention visible text; they are not scrubbed/re-authored.
- System instructions treat all quoted data as untrusted, not instructions or
  ground truth. The model must judge visible support for the full query, not
  text overlap alone. No identities, filenames, ranks or qrels enter the prompt.
- OCR statuses missing/empty/error are unknown evidence, not negatives; they
  never remove a baseline candidate. Public OCR has **UNVERIFIED image-byte
  provenance** because it has no upstream image SHA. Local image/caption SHA
  verification does not establish that separate relationship.

## Model and hardware gates

Python:
`C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe`

Exact local model directory:
`D:\AIChallenge\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3`

- Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Installed bitsandbytes NF4, double quantization, FP16 computation; CUDA GPU0,
  batch1; no CPU/full precision/alternative-model fallback.
- Explicit min/max image pixels **65,536 / 262,144**, overriding the cached
  processor's much larger default. Caption/query/OCR bounds each 2,000 characters.
  Full OCR text, its canonical JSON-string SHA, original length and truncation
  diagnostics are retained in the frozen pools.
- Greedy generation, 32 new tokens, one beam, seed 20261112. Strict whole-response
  JSON parser accepts only `{"score": <finite number in [0,1]>}`. No booleans,
  duplicate keys, extra fields, markdown, trailing text or numeric fallback.
- Require at least 4,608 MiB actual CUDA free memory immediately before loading;
  budget = actual free minus 640 MiB. Check post-load reserve and first scored
  pair before continuing. A first-pair smoke is **included**, not hidden warmup.
- Even global pair indices run A then B; odd indices run B then A. Model loads
  once. Each call uses CUDA synchronization and records elapsed time, input/output
  token lengths, image grid, allocated/reserved peak memory, exact raw response.
- Any import, NF4, load, inference, OOM or JSON error stops the run, not a loop
  recording hundreds of artificial zero scores.

### Observed compatibility probes

Read-only `nvidia-smi` initially showed RTX 3050 6GB Laptop GPU, driver **581.86**,
6,144 MiB total, 1,184 MiB used, 4,819 MiB free. The recorded preflight later
showed 1,250 MiB used / 4,753 MiB free. CUDA's free-memory accounting differs:
the isolated runtime probe reported **5,404,360,704 bytes free**, total
6,441,926,656 bytes, compute capability 8.6.

Installed versions (metadata-only, no submodule `find_spec`):

| Package | Version |
|---|---|
| torch | 2.7.1+cu118 |
| numpy | 2.4.4 |
| transformers | 5.15.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.2 |
| Pillow | 11.1.0 |
| rank-bm25 | 0.2.2 |

The driver's CUDA support is **not** the Torch build. Torch reports CUDA **11.8**.
`import torch` first in a clean `-I` child actually exited **3**, OpenMP Error #15
(duplicate `libiomp5md.dll`). The error is retained verbatim in `preflight.json`.
The safe existing repository convention, **`import numpy` before `import torch`**,
succeeded in a separate isolated child. It also successfully imported the Qwen
APIs and executed an actual small bitsandbytes NF4 CUDA quantization/matmul.
This successful order is explicit in both worker and tests; it is not error
suppression. Inherited `PYTHONPATH`, `PYTHONHOME` and the unsafe KMP flag are
removed **only from the child environment**, never edited globally.

Both shards resolve and are readable: 3,982,649,232 and 3,526,688,744 bytes
(7,509,337,976 bytes total). All indexed tensor names and header offsets are
checked against file lengths. Metadata/tokenizer/index files have SHA-256 hashes.
This is **not a full tensor-content SHA verification**. No weights are copied
into the repository. Worker HF offline flags, `local_files_only=True`, disabled
remote code, and an audit hook rejecting socket operations prevent network use.

## Run commands (PowerShell)

Run from the repository root. Every command uses the existing Python explicitly:

```powershell
$py = 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
& $py -B -m unittest tests.test_l04_qwen_ocr_ablation tests.test_l04_ocr_ablation
& $py -B tools/l04_qwen_ocr_ablation.py --help
& $py -B tools/l04_qwen_ocr_ablation.py freeze
& $py -B tools/l04_qwen_ocr_ablation.py preflight
& $py -B tools/l04_qwen_ocr_ablation.py run
& $py -B tools/l04_qwen_ocr_ablation.py report
```

Default output is the unique local directory
`benchmark_results/l04_qwen_ocr_ablation_20261112_run01/`.
Freeze refuses to overwrite an existing directory. For an entirely new run,
pass a **new** `--output-dir` consistently to each command. Existing packs/reports
are never rewritten; `preservation.json` records and verifies their hashes.

Resume only a matching interrupted run:

```powershell
& $py -B tools/l04_qwen_ocr_ablation.py run --resume
& $py -B tools/l04_qwen_ocr_ablation.py report
```

Do **not** rerun preflight after a run context exists. Its hardware/import/cache
record is immutable and part of the context hash. Resume requires identical
frozen packet, input hashes, code, prompt, config, package versions, revision and
cache metadata. Each atomic per-arm checkpoint key binds those values and the
full candidate input. Incomplete A/B pairs can resume but cannot produce accuracy
metrics; invalid/error checkpoints fail closed. A changed runtime/code requires
a new output directory and a fresh freeze, not mixing results.

## Evaluation and artifacts

Only a complete 640-valid-score run produces ranking/accuracy metrics. A blocked
or partial run reports genuine completed counts and `metrics: null`, not fake
performance. See `worker_process.json` for exact load/inference failure output.
`failed_call.json` preserves an invalid model response if one occurs.

Compare the frozen caption-BM25 ranking, Qwen A and Qwen B. Qwen ranks by strict
score, then original BM25 position, then identity; no lexical score blending.
For all16 and each query kind, report known-judged-positive and exact-target
Hit@1/5/10 and MRR@10 separately. PoolRecall@20/irrecoverable misses and conditional
metrics on covered queries are distinct from primary end-to-end fixed-pool
metrics over all16. Per-query ranks and B-vs-A gains/losses are retained, along
with raw responses, latency/token summaries and counterbalanced order.

Important packet artifacts: `freeze.json`, `protocol.json`, `config.json`,
`queries.private.json`, `pools.json`, `input_manifest.json`, `coverage.json`,
`source_manifest.json`, `preservation.json`, `preflight.json`, `run_context.json`,
optional `hardware.json`, `smoke.json`, `checkpoint.json`, `worker_process.json`,
`report.json` and `report.md`. `benchmark_results/` is already gitignored; outputs
remain local artifacts, not staged source files.

Input manifest SHA: `9e2d3681a90f91f209c2f4b987640da202faddde01b14718800948b488daf1a6`.
Config SHA: `80a61db1360ae03305055d9065f567d13bf60a32a594b14806a9e39e7b51ce22`.
Pool SHA: `77fe4d84e73e314db7ac69eb7e805eb2043db0d2548ac50a2f3ec786844b0e80`.

`pool_diagnostics.json` is an additional **derived diagnostic**, not a changed
freeze or a Qwen result. The immutable top20 pools cover a known positive and
the exact target for **9/16 queries (56.25%)**. The seven irrecoverable query IDs
are t01, t03, t04, t07, t08, t11 and t13. Do not inject their targets or switch to
OCR retrieval after observing this ceiling. All 320 candidate instances have
public OCR status `ok`; the full corpus has 7,471 matched public OCR records,
all with unverified upstream image-byte provenance, and 7,471 independently
verified caption/local-image SHAs. These are coverage facts, not evidence that
Qwen or OCR improves ranking. The derived diagnostic binds the frozen pool and
private-query hashes and does not enter the worker prompt or task keys.

Tests use explicitly **SYNTHETIC** fixtures, not real benchmark scores. They
exercise strict JSON parsing, prompt isolation/escaping, bounds, unchanged
caption/pixel settings, caption-only top20/no injection, missing OCR retention,
counterbalancing, context/hash/resume rejection, incomplete-pair blocking,
deterministic ties, conditional versus all-query metrics, fail-closed preflight,
cache mismatch checks and forbidden dataset roots. No heavy model imports occur
at module top level. Black, Ruff and mypy were not installed; nothing was installed
to make them available. Syntax/JSON/whitespace checks and unit tests are used.
The expanded combined unittest command passed **54 tests**. `git diff --check`
also passed (only existing line-ending warnings), as did direct AST, JSON,
new-file whitespace, frozen-input/source and preservation-manifest checks.
