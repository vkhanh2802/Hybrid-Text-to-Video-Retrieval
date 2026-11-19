"""Explicit public-query live smoke tests, no DRES operations or auto polling."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


QUERIES = [
    {"query": 'A television news scene with the visible text "HTV9".', "ocr_text": "HTV9", "temporal": False},
    {"query": "A street scene with people and traffic.", "temporal": False},
]


def run(output, ocr):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with urlopen("http://127.0.0.1:8000/api/status", timeout=20) as r:
        status = json.loads(r.read())
    assert status["retrieval_backend"] == "school_worker"
    assert status["ocr"]["mode"] == ("retrieve" if ocr else "off"), status["ocr"]
    runs = []
    # On the newly restarted OCR web, first call also warms the local GPU.
    requests = ([QUERIES[0]] if ocr else []) + QUERIES
    for i, query in enumerate(requests):
        request = Request("http://127.0.0.1:8000/api/search", data=json.dumps(query).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        start = perf_counter()
        with urlopen(request, timeout=300) as r:
            response = json.loads(r.read())
            http = r.status
        elapsed = perf_counter() - start
        values = response["results"]
        assert len(values) == len({(r["video_id"], r["frame_id"]) for r in values}) == 100
        batch = response.get("vlm_batch", {})
        assert batch["requested_batch_size"] == 12 and batch["inputs"] <= 20, batch
        if ocr:
            assert response["ocr"]["status"] not in {"error", "unavailable"}, response["ocr"]
            assert all(r["ocr"] and r["ocr"]["verified"] is False for r in values)
            assert all(r["ocr"]["source_format"] == "public_json" for r in values)
        summary = {"http": http, "client_seconds": elapsed, "query": query,
                   "phase": "cold_local_warmup" if ocr and i == 0 else "warm_public_smoke",
                   "frames": 100, "vlm_batch": batch, "stage_seconds": response["stage_seconds"],
                   "vlm_status_counts": dict(Counter(r["vlm_status"] for r in values)),
                   "ocr": response["ocr"],
                   "ocr_status_counts": dict(Counter((r.get("ocr") or {}).get("status", "off") for r in values))}
        (output / f"response_{i:02d}.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        runs.append(summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    report = {"status": "passed", "ocr_enabled": ocr, "runs": runs,
              "warnings": ["Public functional checks, no DRES task/submission",
                           "One sample per query/phase, no median/P95 or accuracy improvement claim",
                           "Planner is live, not a frozen-pool/frozen-prompts A/B quality experiment"]}
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    run(args.output, args.ocr)
