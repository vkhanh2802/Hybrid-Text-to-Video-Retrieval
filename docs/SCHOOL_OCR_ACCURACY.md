# Paired OCR accuracy pilot — six public queries

> Follow-up root-cause audit found that the inspected official recognizer
> alphabet lacks many Vietnamese tone-marked characters; the entire supplied
> corpus has zero occurrences of the16tested missing characters. This is more
> specific than generic OCR blur/noise. See
> [OCR_NOTEBOOK_DIAGNOSIS.md](OCR_NOTEBOOK_DIAGNOSIS.md). No source OCR corrections
> or benchmark query changes were made after this finding.

## Purpose and predeclared comparison

Test whether the new school OCR branch improves retrieval or Qwen ranking,
not merely whether OCR hits exist or Qwen scores increase. No production weight,
model, precision, prompt, pixel limit, batch size or corpus changes.

## Completed outcome

**Benefit not established.** All three literal queries had zero exact OCR hits;
A/B retrieval and no-OCR Qwen outputs identical across all six. Known-target
Hit@1/20/100 and MRR unchanged A/B/C. Additional arm/rank-hidden AI visual top1
check found one improved scene result (A/B2/5, C3/5 assessable, one missing-image
query unknown), but the winning OCR is logo/time/noisy tokens, so improvement
cannot be attributed specifically to recognized semantic text without a sham
control.13paired score changes, seven on missing-OCR images.

OCR prompt incurred real cost on identical B/C pools: median across six distinct
query samples15.38→17.42s, mean15.24→31.50s, including two C outliers49.99/70.35s.
No OOM or hidden splitting; invalid2/111 per arm. Not a repeated latency benchmark.
No automatic production config/weight/prompt changes made based on this pilot.

[Full report and per-query results](../benchmark_results/school_ocr_accuracy_20261115_run01/SUMMARY.md).

Experiment: `benchmark_results/school_ocr_accuracy_20261115_run01/`.

- Six anchors are the **first six successful image samples in file order** of
  `benchmarks/drive_observations_20261102.json`. Not selected by OCR or ranks.
- Original anchor plus two neighbors on either side were downloaded from the
  school worker and visually reopened before consulting OCR/caption/rank data.
- Three original scene descriptions retained. Two mixed scene/text questions
  and one literal-only question authored from readable image pixels. No query
  changes after ranking or source OCR audit.
- L16 and L10 anchors have no supplied OCR: two predeclared coverage controls.
- AI-authored/reviewed, **not human verified or held-out**. Sparse positives,
  unjudged frames are unknown, not automatically negative.

| Arm | CPU retrieval | Qwen prompt |
| --- | --- | --- |
| A | Original baseline worker, OCR off | Original caption, no OCR |
| B | New OCR worker, bounded positive weight0.1 | Original caption, no OCR |
| C | **Exact B pool/order/bytes** | Original caption + separate OCR |

B−A isolates OCR retrieval contribution; C−B isolates OCR prompt on the same
pool. Structured query frozen to one original text for all streams; no Groq,
translation or object entities; temporal disabled. Therefore a diagnostic
retrieval/Qwen ablation, not a full DRES/live-planner benchmark.

Pinned production Qwen NF4/revision/SDPA/greedy/max8tokens/left padding, microbatch12,
original top20 only per arm. Missing images are not filled from rank21. One model
loaded on exclusive laptop GPU after explicit user approval to stop/restart web.
Strict OOM: failure is recorded, not hidden subbatch timing. Invalid scores
remain null with ordinary production fallback, not zeros or scalar retries.

## Metrics and inspection

Known-positive first rank, Hit@1/5/20/100 and MRR@100 before and after Qwen.
These measure recovery of specific visually inspected frames, not complete
relevance, precision or universal accuracy. All six queries retained including
ones that fail OCR matching. Final top1 images will be pooled/deduplicated and
shuffled with arm/rank hidden for a small additional full-query visual check.

The frozen CPU comparison found identical rank/score/caption/identity sequences
for A and B on all six queries (zero literal hits), so B−A is also a useful
inference consistency control: a changed Qwen output there is not evidence of
OCR retrieval benefit. Known-positive initial ranks are230,4,48,1,not-in-pool,
not-in-pool respectively. Only two known-positive queries enter the original
Qwen top20; prompt evidence cannot rescue an unseen target from lower ranks.
No missing-image backfill or target injection is allowed to improve this result.

Protocol/query/source/image/model/config hashes frozen in `PACKET.json` before
inference; no identities, targets, labels or rank metadata enter model prompt.
Alternating A/B/C vs C/B/A order, one inference per arm. No significance claim.

## Preliminary source finding (after freeze, not used to rewrite queries)

All three literal questions returned **zero exact OCR hits**. Source audit after
query and retrieval freeze found recognition errors at the known target images:

| Visible words in pixels | Supplied OCR transcription |
| --- | --- |
| `Quảng Ngãi` | `Qung Ngãi` |
| `MÌ TRỨNG` | `MITOONG` |
| `3,2 tấn cá khoai chứa formol` | `3,2 tn cá khoai cha formol` |

Thus the literal retrieval path cannot recover these targets by exact matching.
Preserving diacritics is intentional; queries are not corrupted to match OCR
mistakes and no fuzzy matching or threshold tuning is introduced after seeing
these failures. Whether noisy OCR still helps/hurts Qwen requires the actual
C−B inference results, not inference from these source examples alone.

Evidence: `POST_FREEZE_SOURCE_OCR_AUDIT.json`. Qwen job completed; web restarted
afterward with unchanged OCR/batch12 config. No DRES task/evaluation/hints/submission.
