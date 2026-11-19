# P0 audit of a frozen retrieval run

## 1. Purpose

`tools/p0_audit.py` produces a deterministic, read-only audit of one saved
retrieval run. It never loads an encoder, caption model, VLM, planner or live
Milvus client, and it never reruns retrieval. All conclusions come from saved
rows, saved raw evidence, frozen labels and local artifact files.

The audit answers a narrower question than a benchmark run:

- Which stage of the saved funnel lost each explicit human positive?
- Are the local artifacts internally consistent?
- Which inputs can be reproduced from filesystem hashes?
- Which states remain unknown because the saved run did not capture them?

Missing, malformed or out-of-scope inputs are reported explicitly. They are
never silently converted into a zero recall denominator.

## 2. Command

```powershell
python -m tools.p0_audit `
  --source-run benchmark_results/human_holdout_v3_evaluation_20261102 `
  --labels benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json `
  --mapping-root dataset/mapping `
  --caption-root dataset `
  --output benchmark_results/p0_audit_20261107_v6.json
```

Exit codes:

- `0`: the report was written, including when the audit found issues.
- `2`: fatal input or schema error; no report was written.

The output path is opened in exclusive-create mode. The tool refuses to
overwrite the report or any saved input row, raw evidence file, label file or
captured context file.

## 3. Inputs

Required inputs:

- `--source-run`: directory containing `context.json`, `drive_*.json` saved
  result rows and `raw_evidence/<unit_id>_<video_id>.json` files.
- `--labels`: frozen human label JSON with schema
  `kis_human_holdout_labels_v1`, `annotation_source=human_visual_observation`,
  `human_verified=true` and `caption_used_for_annotation=false`.
- `--mapping-root`: frame mapping CSV root (`<group>/<video_id>.csv`).
- `--caption-root`: root containing `clip-features-32/`, `caption_generator/`,
  `caption_mapping/` and `index/`.
- `--output`: new report path.

The label hash must match `context.json.label_sha256`. Protocol and annotation
provenance must agree when both files carry them. A mismatch is fatal rather
than audited as a soft issue.

The captured `context.json.temporal_config` must be complete: `enabled`,
`max_events`, `event_top_k`, `max_videos`, `candidate_pooling`,
`candidate_rrf_k`, `max_frames_per_video`, `max_gap_seconds`, `gap_penalty`,
`min_visual_similarity`, `min_semantic_similarity` and `condition_evidence`.
A missing or partial captured temporal config is a fatal schema error (exit
`2`); current defaults are never substituted for missing historical fields.

## 4. Report schema

`schema` is `kis_p0_audit_v1`. Top-level sections:

| Section | Contents |
|---|---|
| `provenance` | Tool, label, context, row, raw-evidence and artifact hashes; hash-verified captured model config; separate non-historical current-config snapshot; explicit unknown live-service states. |
| `inventory` | Input counts, label judgment counts, paths and artifact/FAISS summary. |
| `artifacts` | Per-video artifact checks: mapping, features, caption generator, caption mapping, index and index manifest. |
| `queries` | One report per explicit event query or reviewed sequence, with per-mode stage details. |
| `stages` | Aggregated stage funnel by task and mode. |
| `summary` | Overall status, counts, issue histogram and unresolved-unit total. |
| `issues` | Deduplicated `{code, detail}` audit issues. |
| `limitations` | Explicit scope boundaries of the audit. |

All JSON is written with sorted keys, UTF-8 encoding, two-space indentation and
one trailing newline. No timestamps, random values or runtime measurements are
embedded.

Audit issues are deduplicated by `(code, detail)` after every producer has
contributed and then sorted deterministically. The top-level `issues` array,
the `summary.issues.total` / `by_code` histogram and the overall status all use
that same deduplicated list, so histogram counts always sum to the reported
total.

## 5. Stage funnel

Stages are ordered from retrieval seed lists to saved results:

```text
visual_hits
bm25_hits
semantic_hits
union_hits
candidate_pool      (video level)
local_frames
support             (visual >= 0.25 OR semantic >= 0.45)
temporal_feasibility
saved_top_1
saved_top_5
saved_top_20
```

Stage units and identities:

| Stage | Unit | Identity / test |
|---|---|---|
| `visual_hits`, `bm25_hits`, `semantic_hits` | Explicit positive frame | Saved seed stream contains canonical `(video_id, frame_id)`. |
| `union_hits` | Explicit positive frame | Present in at least one seed stream. |
| `candidate_pool` | Distinct positive video | Video was selected into the saved candidate pool. |
| `local_frames` | Explicit positive frame | Frame is present in saved raw evidence for the unit/video. |
| `support` | Explicit positive frame | Saved raw evidence actions exactly match the canonical event text/actions and the saved visual or semantic score passes its configured threshold. |
| `temporal_feasibility` | Sequence query | A complete ordered path over valid explicit positive identities fits the captured temporal configuration. Event-frame queries are `not_applicable`. |
| `saved_top_1`, `saved_top_5`, `saved_top_20` | Query | Saved result list contains a canonical positive at the cutoff. Event-frame hits use canonical `(video_id, frame_id)`; a sequence hit requires a complete non-fallback path whose events match the frozen explicit positive identities and valid local raw evidence (see below). |

Raw evidence is accepted only when its ordered `actions` array exactly equals
the canonical label event text (event-frame queries) or the full ordered
`query.actions` list (sequences). A mismatch creates a
`raw_evidence_actions_mismatch` issue. Stages that use captured raw scores,
including `support` and `temporal_feasibility`, remain unresolved for that
evidence, and the query status is not reported as `ok`. For sequences, a
missing, invalid or action-mismatched raw record for the target video also keeps
`saved_top_1/5/20` unresolved in both modes, with `saved_metrics_match: null`;
it is neither a sequence success nor a negative miss. Ordered raw actions are
checked before evaluating any saved sequence path.

For event-frame rows, `events` must be an array containing exactly one object
whose `action` equals the frozen label `event_text`. A missing array, a
different event count, a non-object entry or a wrong action is a
`saved_row_query_mismatch` (`label_mismatch`); the row's streams and saved
results are not used for credit in that case.

A saved sequence result counts as a hit only when all of the following hold:

- `temporal.status` is `retrieved_complete`; fallback and partial results never
  count as sequence success;
- `temporal.coverage` is exactly `1.0`, without a tolerance (for example,
  `0.9999999995` and `1.0000000005` are both invalid);
- every event status is `retrieved`, with canonical `video_id`, non-negative
  `frame_id`, positive `keyframe_n` and finite non-negative `pts_time`;
- every event matches one frozen explicit positive of that event on the full
  canonical identity `(video_id, frame_id, keyframe_n, pts_time)`, with
  `abs(event.pts_time - positive.pts_time) <= 1e-6`;
- event timestamps are strictly increasing and each consecutive gap is within
  the captured `max_gap_seconds`;
- local raw evidence for the sequence target video is present and valid, its
  ordered actions exactly match canonical `sequence.actions`, and every event
  identity also matches it. Missing, invalid or action-mismatched raw evidence
  leaves `saved_top_1/5/20` unresolved with reason `raw_evidence_missing`,
  `raw_evidence_invalid` or `raw_evidence_actions_mismatch`; it is never a
  sequence success.

Canonical frozen positives, grouped by `event_index`, are the source of truth
for identity. Valid local raw evidence is also mandatory before a saved
sequence path can be credited. A raw-evidence gap keeps the query denominator
and is not converted into a negative miss. If raw evidence and the saved result
agree with each other but both use an identity that is not a frozen positive,
the path is `saved_sequence_path_invalid` and never counts as success.

A structurally malformed complete path creates a
`saved_sequence_path_invalid` issue and does not count as a hit. A complete path
whose events do not match explicit positive identities is never a hit. With
valid raw evidence it is reported as `saved_sequence_path_invalid`; without
valid raw evidence the saved-path outcome remains unresolved.

For `temporal_feasibility`, when one event has several explicit positive frames
and some of those alternatives are absent from local raw evidence, the audit
still tries the remaining valid positives. A valid complete path is reported as
feasible and the unavailable alternatives are recorded separately in
`temporal_missing_positives` plus `positive_local_frame_missing` /
`positive_local_identity_mismatch` issues. Only when no complete path is found
does the missing alternative evidence leave feasibility unresolved.

If a saved row is missing, task-mismatched or query-mismatched, every stage
still emits explicit units with status `unresolved`, the correct positive-frame
/ positive-video / query denominator, and a mismatch reason. A missing row uses
`saved_row_missing`; a task mismatch uses `saved_row_task_mismatch`; an
event-frame query mismatch uses `saved_row_query_mismatch`; a sequence event
mismatch uses `sequence_event_order_mismatch`. Label-mismatch queries keep
status `label_mismatch` while retaining all stage denominators in the aggregate,
so they never disappear from the funnel or become zero-recall misses. The
event-frame `temporal_feasibility` stage remains `not_applicable`.

Every per-stage detail carries:

- `status`: `known`, `partial`, `unresolved`, or `not_applicable`.
- `covered`: units that passed the stage.
- `resolved`: units whose outcome is known, whether pass or fail.
- `total`: explicit units considered by that stage.
- `unresolved`: units whose outcome could not be determined.
- `reasons`: unresolved-unit histogram.

Aggregate `coverage` is `covered / resolved`. An unresolved unit is excluded
from the denominator and shown separately. If nothing is resolved, `coverage`
is `null`; it is never reported as `0.0`.

For sequence `saved_top_1/5/20`, missing, invalid or action-mismatched raw evidence
still contributes one to aggregate `queries`, `total`, `unresolved` and
`queries_with_unresolved`, with no `covered` or `resolved` credit. An
all-unresolved aggregate has status `unknown` and `coverage: null`; a mix of
resolved and unresolved queries has
status `partial`, retains every query in `total`, and computes coverage only over
`resolved`. Any raw-record status other than `ok` is treated as invalid evidence,
including `unknown`.

Positive denominators come only from explicit human `positive` judgments.
`negative`, `unknown`, unreviewed, missing and out-of-scope values never become
positive denominators. A query with no explicit positive is reported as
`no_explicit_positive`, not as a hit or a miss.

### Guard mode

Saved raw evidence contains captured visual and semantic scores and frames, but
not the per-frame semantic-guard removal mask. Therefore `semantic_guard`
support is always `unknown` with reason `guard_mask_absent`. Saved result
metrics for both modes are still audited against their own result lists.

## 6. Artifact checks

Per-video checks:

- Mapping CSV parses as integers/decimal numbers, has unique frame identities,
  non-decreasing timestamps and a valid order.
- Visual feature array is 2-D, finite, non-zero-norm and dimensionally consistent
  with the mapping and manifest.
- Caption generator and caption mapping row identities match the frame mapping
  row order.
- Caption image hashes match the frozen positive labels when both are present.
- Caption generation backend/revision is consistent within the video and with
  the captured fast-caption configuration.
- Index and manifest hashes match local files. The manifest must provide
  `caption_sha256`, `mapping_sha256` and `index_sha256`, each as a 64-character
  lowercase SHA-256. A missing or malformed hash makes the manifest invalid.
  The manifest `mapping_sha256` refers to the caption-mapping JSON, not the
  retrieval frame-mapping CSV.
- Historical dimension/model/revision expectations come only from a captured
  models config whose hash is pinned by the saved context and matches the local
  config file. If that captured config is absent or unverified, config-dependent
  comparisons are skipped as unknown. The current checkout configuration is
  never substituted for the historical run configuration.
- When FAISS is unavailable, index contents are `unknown`; the index file is
  still hashed and its manifest hash is checked.
- `caption_artifact_sha256` keys from the saved context are normalized to
  forward slashes before comparison with local caption paths.

Artifact status is `issues` if any artifact is missing, ambiguous, invalid or
raises a cross-artifact/identity issue; `ok` if every audited artifact is valid;
otherwise `unknown`.

## 7. Provenance and unknowns

The audit hashes:

- the label file and saved `context.json`;
- every saved result row and raw-evidence file;
- every audited local artifact;
- the audit tool itself.

`captured_model_config` records the models config pinned by the saved context
(`context.models_sha256` or a `config/models.json` source hash) and is
`verified` only when the local file matches that hash. A mismatch is
`unverified`, an absent hash is `unknown`, and neither case supplies expected
model/dimension/revision values to artifact checks.

`current_config_snapshot` is a separate section explicitly marked
`historical: false`. It is the current checkout configuration, not proof of the
configuration used by the saved run, and it never changes historical artifact
status.

Live Milvus, planner and VLM states are explicitly `unknown`. The saved run did
not capture a query trace, planner trace or VLM trace, and a read-only audit
cannot reconstruct them.

## 8. Status and exit behavior

- Per-query status: `ok`, `missing_row`, `label_mismatch`, `no_explicit_positive`,
  `out_of_scope` or `unresolved` (including raw-evidence action mismatches).
- Overall summary status: `issues` if any audit issue exists, else `unknown` if
  any stage unit or artifact remains unresolved, else `ok`.
- Audit issues do not change the exit code. A report that documents defects is
  still a successful audit and exits `0`.
- Unreadable labels, a label/context hash mismatch, an unsupported schema, a
  missing source run or an existing output file exits `2`.

## 9. 2026-11-07 reference run

The regenerated report for the stricter audit is
`benchmark_results/p0_audit_20261107_v6.json`. Existing v5 and earlier reports
and frozen labels are retained unchanged. The v6 report is byte-identical to a
second CLI run written to a temporary output. Its summary, query outcomes and
issue list match v5: the saved inputs do not exercise the action-mismatch or
near-1.0 coverage defects, which are covered by regression tests. The report
hash differs from v5 because the audit tool's source hash changed.

- SHA-256: `8a7cd1e70d091d4e46846430bd285344b3b779d7482704cc61da2ef419a3e73d`
- Summary: `status=issues`, 36 `ok` queries, 222 unresolved stage units,
  61 issues.

Observed input inventory:

- 36 saved rows: 26 event-frame queries and 10 reviewed sequences.
- 432 raw-evidence files covering 230 candidate videos.
- 80 explicit positive frames.
- 42 positive / 86 negative / 2 unknown event judgments.

The report exits `0` while recording real audit issues:

- The captured `config/models.json` hash verifies against the local file, so
  historical model/dimension/revision checks use the captured configuration;
  `current_config_snapshot` remains separate and non-historical.
- 13 visual-only `L21` candidates have no local caption, caption-mapping, index
  or index-manifest copies. They remain auditable for visual artifacts only.
- 3 `L21` mappings contain a duplicate `frame_id=0`.
- FAISS is unavailable in the audit environment, so 219 index-content statuses
  are `unknown`; index files are still hashed.
- Semantic-guard support is `unknown` for all queries because the guard mask was
  not saved.
- 2 of 10 sequences have no raw record for the target video. Their
  `saved_top_1/5/20` stages are `unresolved` with reason
  `raw_evidence_missing` instead of being credited as canonical-path hits or
  counted as misses. Sequence saved-path coverage is therefore reported over 8
  resolved queries: 1/8 at top-1 and 2/8 at top-5/top-20.
- 6 `saved_sequence_path_invalid` issues cover 3 target-video complete paths
  (`drive_06_sequence`, `drive_08_sequence`, `drive_09_sequence`) in both modes.
  Each path repeated a frame identity that raw evidence also contained but that
  did not match the frozen explicit positive identity, so none counts as a
  sequence success.

These are artifact/provenance findings, not evidence that live retrieval or the
planner behaved incorrectly.

## 10. Limitations

- Only saved result lists are audited. Results beyond the saved cap of 20 per
  mode cannot be reconstructed.
- Raw evidence contains captured scores, not model inputs or intermediate
  retrieval state.
- The audit verifies local filesystem copies, not that the original run read
  exactly these copies.
- It cannot prove live Milvus, planner or VLM correctness.
- Historical model expectations require a captured models-config hash; the
  current checkout configuration is not the historical run configuration.
- Sequence success is auditable only within the saved result rows, and only
  when the path carries a complete, non-fallback temporal status.
- Small-sample stage coverage is descriptive and must not be used for holdout
  retuning.
