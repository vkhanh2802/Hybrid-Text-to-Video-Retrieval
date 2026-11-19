"""PUBLIC-query CPU-only checks for staged OCR worker; never load laptop Qwen.

No DRES calls, submissions, Groq calls or mutations of any OCR/caption artifact.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from kis.config import MODEL_MANIFEST, load_runtime_config
from kis.ocr import OCRConfig
from kis.remote_worker import RemoteWorker


def client(path, snapshot=False):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    config = load_runtime_config(environ={})
    if snapshot:
        config = replace(config, ocr_format=settings["ocr_format"], ocr_root=Path(settings["ocr_root"]),
                         ocr_manifest_sha256=settings["ocr_manifest_sha256"],
                         ocr=OCRConfig(mode=settings["ocr_mode"], weight=settings["ocr_weight"]))
    worker = RemoteWorker(settings["url"], settings["token"], config.video_groups)
    worker.connect(config, MODEL_MANIFEST)
    return worker


def run(output, baseline_path, staged_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    baseline, staged = client(baseline_path), client(staged_path, True)
    # Frozen public test inputs, original literal only. No translated guesses.
    # HTV9 was observed verbatim in supplied OCR; a positive smoke test, not an
    # independent relevance/accuracy evaluation. VTV previously gave no_match.
    raw = 'A television news scene with the visible text "HTV9".'
    query = {"raw_query": raw, "query_id": "public_ocr_cpu_check", "query_variants": [raw],
             "needs_ocr": True, "ocr_text": "HTV9", "entities": []}
    runs = []
    pools = {}
    for repetition in range(3):
        for name, worker in (("baseline", baseline), ("ocr", staged)):
            start = perf_counter()
            result = worker.retrieve(query, temporal=False)
            elapsed = perf_counter() - start
            candidates, diag = result["candidates"], result["diagnostics"]
            assert len(candidates) >= 100
            assert len({(r["video_id"], r["frame_id"]) for r in candidates}) == len(candidates)
            if name == "ocr":
                assert diag["ocr"]["status"] not in {"error", "unavailable"}, diag["ocr"]
                assert diag["ocr"]["hits"] > 0
                assert all(r["ocr"]["verified"] is False for r in candidates)
                assert all(r["ocr"]["source_format"] == "public_json" for r in candidates)
            runs.append({"name": name, "repetition": repetition + 1, "rpc_seconds": elapsed,
                         "pool": len(candidates), "stage_seconds": diag["stage_seconds"], "ocr": diag.get("ocr")})
            pools[name] = candidates
    # Relevant post-video-fusion endpoint: identities only, caption untouched.
    rows = deepcopy(pools["baseline"][:100])
    original = deepcopy(rows)
    start = perf_counter()
    evidence = staged.attach_ocr(rows)
    attach_seconds = perf_counter() - start
    assert evidence["status"] == "attached", evidence
    assert [{k: v for k, v in r.items() if k != "ocr"} for r in rows] == original
    assert any(r["ocr"]["status"] == "missing" for r in rows)
    report = {"status": "passed", "query": query, "runs": runs, "ocr_status": staged.info["ocr"],
              "post_fusion_evidence": evidence, "post_fusion_evidence_rpc_seconds": attach_seconds,
              "post_fusion_status_counts": {s: sum(r["ocr"]["status"] == s for r in rows)
                  for s in sorted({r["ocr"]["status"] for r in rows})},
              "warnings": ["CPU-only, no Qwen inference; no quality/accuracy improvement claim",
                           "3 repetitions on one public query; baseline and OCR pools may differ",
                           "Auxiliary positive bonus <=10% of maximum base score, engineering-only, uncalibrated"]}
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Public query only, retain exact pool/evidence for diagnostics without secrets.
    (output / "PUBLIC_POOLS.json").write_text(json.dumps(pools, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="config/school-runtime.local.json")
    parser.add_argument("--staged", default="config/school-ocr-staged.local.json")
    args = parser.parse_args()
    run(args.output, args.baseline, args.staged)
