# Experimental L04 end-to-end OCR retrieval + prompt A/B

This isolated harness calls the **real `kis.solve_kis.solve_kis`** through dependency
injection. It does not enable or edit production configuration, rebuild indexes,
modify captions/images, or weaken OCR validation. No environment files are edited.

## Frozen design

Protocol: `benchmarks/l04_e2e_ocr_ablation_20261112_v1.protocol.json`.
The two smallest IDs of each **actual** query kind in originalunchanged16 are
**t01, t02, t03, t04, t05, t09**. Exact unchanged queries, qrels, image hashes,
CSV identities and evidence are copied to `frozenqueries.json` before ranking.
These are already-seen, assistant-authored, incomplete judgments, **not holdout**.

Both arms use one identical cached Groq planner output per query, original caption
BM25 (Unicode), all thirty existing per-video caption FAISS indexes (384), actual
local pinned MiniLM encoder, actual SigLIP text encoding (768), server-filtered
read-only staging Milvus, the production **`rrf`** profile (nprobe128, RRF60,
visual/BM25/semantic weights .5/.2/.3), top100 retrieval and Qwen top20, weight .5.
Encoders run on CPU sequentially and are released before CUDA-only Qwen. There
is no CPU Qwen fallback. Explicit temporal-disabled single-image profile is
identical across arms; any actual planner sequence blocks, not silently rewritten.

- **A:** original caption retrieval and production Qwen numeric scoring prompt.
- **B:** adds independent **OCR-only** BM25 retrieval, production OCR rank fusion
  weight .5/top50, and a separate quoted untrusted OCR evidence block to Qwen.
  Captions never change. This measures the **combined retrieval + prompt effect**;
  candidate pools may differ. Pool recall/overlap must be reported separately.

Public OCR arrays do not have upstream image SHA/engine provenance: they remain
**experimental UNVERIFIED / `verified=False`**. Local caption/image hashes are
checked, but do not magically verify upstream OCR. The isolated adapter differs
from production SQLite FTS and confidence/provenance validation. Missing/error
OCR is unknown, not negative; incomplete corpus coverage blocks this experiment.

## Gates and execution

```powershell
$py = 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
& $py -B -m tools.l04_e2e_ocr_ablation freeze
& $py -B -m tools.l04_e2e_ocr_ablation preflight
# Continue ONLY if preflight GO:
& $py -B -m tools.l04_e2e_ocr_ablation diagnose
# Continue ONLY if diagnose GO:
& $py -B -m tools.l04_e2e_ocr_ablation run
& $py -B -m tools.l04_e2e_ocr_ablation report
```

Default unique output: `benchmark_results/l04_e2e_ocr_ablation_20261112_run01`.
Freeze refuses an existing directory. Source/input/preservation manifests are
immutable after freeze; changed code or inputs require a new explicit protocol/run.
Model snapshot files are fully SHA256-hashed in preflight, including weights.

Only latest explicitly documented staging collection
`kis_l04_l05_stg_20261108_02` at loopback19530 is permitted. Read-only client
allowlist is describe_collection/index, get_load_state, filtered query/search.
No production collection access, collection load, index build, aliases, writes,
installs, downloads, L05/L06 filesystem discovery or fetched L05 rows. Existing
index coverage/load state and L04-filtered count must pass before any batch.

Qwen uses the approved exact local snapshot, `local_files_only=True`, NF4 double
quant, FP16 compute and explicitly FP16 nonquantized parameters, SDPA, deterministic
max8 tokens, strict whole-output integer0..10. The original production prompt is
extracted without importing its environment-mutating loader. Experimental
differences (explicit dtype/offline loader, fatal parsing/finite checks, B OCR
block) are frozen in the protocol. **Not** the prior JSON0..1/max32 scorer.

Before any planner/API/ranking, controlled diagnosis revisits only old failed
t02/candidate18 with A/B sequentially, up to two repetitions. All processor input
shapes/dtypes/finite checks, raw model-logit finite counts/dtypes, generated IDs,
raw outputs, prompts, scores and latency are saved in `rawQwen/`. Invalid first
pair stops before repetition or batch. No arbitrary zero, forced-choice ranking,
quantization workaround, qrel input, or parser relaxation is allowed.

Groq uses existing production credential resolution, frozen gpt-oss20b/schema/
temperature0/max4096. Exactly one request/query, no repair retries; attempted-call
intent persists before transport. Only public config/response hash/model/usage/
latency are reported, never key/headers or credential file contents.

`run_arm` verifies actual provider invocation and solver diagnostics, explicit
fusion/OCR/temporal states, twenty candidate images and twenty valid Qwen scores.
`solve_kis`'s normally swallowed VLM/OCR errors block the comparison. Raw failures
are preserved. Checkpoints bind model/config/input/prompt/source context; invalid
checkpoints cannot become zero. Partial reports have **null accuracy metrics**.

## Validation

```powershell
& $py -B -m unittest tests.test_l04_e2e_ocr_ablation tests.test_solve_kis tests.test_ocr_pipeline
```

New tests are offline **synthetic safety tests**, never fabricated benchmark
results: real solver DI invocation, mandatory stages/failures, strict parsing,
same planner input across A/B, canonical CSV mapping, read-only Milvus/server
scope, unverified OCR, missing-image/error handling and partial-metric refusal.
No production hook was added.

Actual gate results and exact root traceback belong in the new run's
`preflight.json`, `diagnose.json`, `failure.json`, `report.json` and `report.md`.
No Qwen/OCR improvement or production accuracy claim is permitted unless all
six queries complete both arms. The previous failed ablation is left unchanged.

## Executed result: 2026-11-12 run01 — BLOCKED, no accuracy

Freeze and real preflight completed. Staging was already Loaded and fully
indexed14,918/14,918, pending0, with exact server-filtered L04 count7,471. All30
384-d caption FAISS artifacts and all30 768-d visual sources validated, including
staging source hashes. Pinned model cache files were fully hashed. CUDA free
memory immediately before model load was5,404,360,704 bytes.

The controlled old t02/candidate18 input was re-scored **once per arm** with the
new frozen production numeric prompt/max8 protocol. Both A and B returned
`!!!!!!!!`, with151,936 NaNs in every raw generation-logit step. Inputs and stored
parameter tensors were finite. Configuration: CUDA NF4, FP16 compute/nonquantized
parameters, SDPA; raw logits float32. The failure is numerical before parsing;
the exact failing layer/dtype/library cause remains unresolved. No second
repeat, alternate scorer/dtype, parser relaxation, fabricated score or batch.

Separately, the existing production planner loader reports Groq
`configured=False` in this process. No credential extraction or environment
change was attempted. **Zero Groq requests, zero valid diagnostic scores, zero
of240 ranking scores, zero of12 query arms; metrics null.** Real encoder/search/
RRF batch execution is untested, not claimed complete from synthetic tests.

Detailed gates, raw prompts/outputs/finite-logit diagnostics, exact tracebacks,
runtime model/package/config details and preservation checks are under
`benchmark_results/l04_e2e_ocr_ablation_20261112_run01/`. `report.md` and the
`execution_audit` appended to `report.json` include the independent credential
gate and post-run command/preservation audit; a bare rerun of the generic report
command regenerates its core report, not these appended audit annotations.

Tests: recommended three-module command30 PASS; additional corpus/OCR/visual
tests74 PASS; planner/transport tests52 PASS. Black, Ruff, Mypy unavailable;
AST/JSON/whitespace and `git diff --check` PASS. No production files changed.

Session-scope caveat: initial AGENTS discovery used a repository-wide recursive
`Get-ChildItem -Filter AGENTS.md -Recurse -File`. It returned no file but may
enumerate excluded dataset directories. Thus the harness's explicit L04 access
construction does **not** certify that the entire session avoided excluded
directory enumeration. This deviation is recorded in the run report; no full
benchmark or accuracy claim was produced.
