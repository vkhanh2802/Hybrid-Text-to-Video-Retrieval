"""Small paired PUBLIC-query OCR ablation; exploratory, never DRES/production tuning.

Freeze image-authored queries before retrieval, use baseline and OCR workers with
identical structured queries, then GPU production batch12 for A/B/C arms. Qrels
are sparse known positives; unjudged frames are NOT negatives.
"""

from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from time import perf_counter

from tools.benchmark_school_ocr import client


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def anchors(output):
    from PIL import Image, ImageDraw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "anchors").mkdir()
    observations = json.loads((ROOT / "benchmarks/drive_observations_20261102.json").read_text(encoding="utf-8"))
    samples = observations["samples"][:6]
    save(output / "SELECTION.json", {"rule": "First six successful original image-observation samples in file order; no ranking/OCR selection",
         "source_sha256": sha(ROOT / "benchmarks/drive_observations_20261102.json"), "samples": samples,
         "annotation_source": "AI visual inspection, not human verified", "heldout": False})
    worker = client(ROOT / "config/school-runtime.local.json", True)
    sheets = []
    for sample in samples:
        vid, number = sample["video_id"], sample["image"]["relevant_keyframes"][0]
        sheet = Image.new("RGB", (1800, 750), "white")
        draw = ImageDraw.Draw(sheet)
        files = []
        for i, n in enumerate(range(number - 2, number + 3)):
            raw, mime = worker.read(vid, n)
            path = output / "anchors" / f"{vid}_{n:03d}{'.png' if mime == 'image/png' else '.jpg'}"
            with path.open("xb") as f:
                f.write(raw)
            with Image.open(BytesIO(raw)) as im:
                image = im.convert("RGB")
                image.thumbnail((570, 310))
            x, y = (i % 3) * 600, (i // 3) * 370
            draw.text((x + 10, y + 5), f"{sample['sample_id']} {vid}/{n:03d}", fill="black")
            sheet.paste(image, (x + 10, y + 35))
            files.append({"n": n, "path": path.relative_to(output).as_posix(), "sha256": sha(path)})
        sheet.save(output / (sample["sample_id"] + ".jpg"), quality=90)
        sheets.append({"id": sample["sample_id"], "files": files})
    save(output / "ANCHOR_IMAGES.json", sheets)
    print(json.dumps({"phase": "anchors_ready", "samples": len(samples), "images": 30}))


def prepare(output):
    output = Path(output)
    queries = json.loads((output / "QUERIES.json").read_text(encoding="utf-8"))
    if (output / "PACKET.json").exists():
        raise FileExistsError("Frozen packet retained")
    baseline = client(ROOT / "config/school-runtime.ocr-before.local.json")
    ocr = client(ROOT / "config/school-runtime.local.json", True)
    (output / "images").mkdir(exist_ok=False)
    packet = {"schema": "school_ocr_accuracy_v1", "queries_sha256": sha(output / "QUERIES.json"),
              "protocol_sha256": sha(output / "PROTOCOL.json"), "queries": [], "microbatch": 12,
              "selection_sha256": sha(output / "SELECTION.json"), "anchor_manifest_sha256": sha(output / "ANCHOR_IMAGES.json"),
              "sources": {name: sha(ROOT / name) for name in
                  ("kis/vlm_reranker.py", "kis/solve_kis.py", "kis/ranking.py", "config/models.json", "tools/school_ocr_accuracy.py")},
              "models": ocr.info["models"], "baseline_runtime": baseline.info["runtime"],
              "ocr_runtime": ocr.info["runtime"]}
    fetched = {}
    for query in queries:
        raw = query["text"]
        structured = {"query_id": query["id"], "raw_query": raw, "query_variants": [raw],
                      "stream_queries": {k: [raw] for k in ("visual", "bm25", "semantic")},
                      "entities": [], "events": [], "needs_ocr": bool(query.get("literal"))}
        if query.get("literal"):
            structured["ocr_text"] = query["literal"]
        record = {"query": query, "structured_query": structured, "retrieval": {}}
        for name, worker in (("A", baseline), ("B", ocr)):
            start = perf_counter()
            reply = worker.retrieve(structured, temporal=False)
            elapsed = perf_counter() - start
            pool = reply["candidates"]
            assert len(pool) >= 100 and len({(r["video_id"], r["frame_id"]) for r in pool}) == len(pool)
            if name == "B":
                assert reply["diagnostics"]["ocr"]["status"] not in {"error", "unavailable"}
            record["retrieval"][name] = {"pool": pool, "diagnostics": reply["diagnostics"], "rpc_seconds": elapsed}
            for row in pool[:20]:
                identity = (row["video_id"], row["keyframe_n"])
                if identity in fetched:
                    continue
                try:
                    data, mime = worker.read(*identity)
                except FileNotFoundError:
                    fetched[identity] = {"status": "missing"}
                    continue
                path = output / "images" / f"{identity[0]}_{identity[1]:03d}{'.png' if mime == 'image/png' else '.jpg'}"
                with path.open("xb") as f:
                    f.write(data)
                fetched[identity] = {"status": "ready", "path": path.relative_to(output).as_posix(), "sha256": sha(path)}
        record["images"] = {vid + "/" + str(n): value for (vid, n), value in fetched.items()
                            if any(r["video_id"] == vid and r["keyframe_n"] == n
                                   for arm in record["retrieval"].values() for r in arm["pool"][:20])}
        packet["queries"].append(record)
        print(json.dumps({"phase": "retrieved", "query": query["id"],
                          "ocr_hits": record["retrieval"]["B"]["diagnostics"]["ocr"]["hits"]}), flush=True)
    save(output / "PACKET.json", packet)
    print(json.dumps({"phase": "frozen", "queries": len(queries), "distinct_images": len(fetched)}), flush=True)


def metrics(rows, query):
    positives = {(q["video_id"], q["keyframe_n"]) for q in query["positives"]}
    ranks = [i for i, row in enumerate(rows, 1) if (row["video_id"], row["keyframe_n"]) in positives]
    first = min(ranks) if ranks else None
    return {"first_known_positive_rank": first, "known_hit_at_1": int(first is not None and first <= 1),
            "known_hit_at_5": int(first is not None and first <= 5),
            "known_hit_at_20": int(first is not None and first <= 20),
            "known_hit_at_100": int(first is not None and first <= 100),
            "known_mrr_at_100": 1. / first if first is not None and first <= 100 else 0.}


def gpu(output):
    import os
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", KIS_VLM_BATCH_SIZE="12")
    import numpy
    import torch
    from kis.ranking import finalize_vlm_ranking
    from kis.vlm_reranker import VLMReranker
    output = Path(output)
    packet = json.loads((output / "PACKET.json").read_text(encoding="utf-8"))
    if (output / "REPORT.json").exists() or (output / "SCORES.jsonl").exists():
        raise FileExistsError("Prior experiment retained")
    for name, expected in packet["sources"].items():
        assert sha(ROOT / name) == expected, name
    assert sha(output / "QUERIES.json") == packet["queries_sha256"]
    assert sha(output / "PROTOCOL.json") == packet["protocol_sha256"]
    assert sha(output / "SELECTION.json") == packet["selection_sha256"]
    assert sha(output / "ANCHOR_IMAGES.json") == packet["anchor_manifest_sha256"]
    for q in packet["queries"]:
        for im in q["images"].values():
            if im["status"] == "ready":
                assert sha(output / im["path"]) == im["sha256"]
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.manual_seed(20261115)
    start = perf_counter()
    scorer = VLMReranker(batch_size=12)
    load_seconds = perf_counter() - start
    torch.set_num_threads(4)
    records = []
    for qi, q in enumerate(packet["queries"]):
        record = {"query": q["query"], "arms": {}}
        # A=baseline retrieval/no OCR prompt; B=OCR retrieval/no OCR prompt;
        # C=same exact B pool with OCR prompt, separating prompt contribution.
        order = ("A", "B", "C") if qi % 2 == 0 else ("C", "B", "A")
        for arm in order:
            pool = deepcopy(q["retrieval"]["A" if arm == "A" else "B"]["pool"])
            if arm != "C":
                for row in pool:
                    row.pop("ocr", None)
            pending = []
            for i, row in enumerate(pool):
                row["vlm_status"] = "unscored"
                if i >= 20:
                    continue
                im = q["images"][row["video_id"] + "/" + str(row["keyframe_n"])]
                if im["status"] == "missing":
                    row.update(vlm_status="missing_image", vlm_raw_score=None, vlm_error="image_not_found")
                else:
                    pending.append((i, output / im["path"]))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            scores = scorer.score_batch([path for _, path in pending], q["query"]["text"],
                [pool[i].get("retrieval_text", "") for i, _ in pending],
                ocrs=[pool[i].get("ocr") for i, _ in pending] if arm == "C" else None,
                batch_size=12, oom_fallback=False)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            for (i, _), score in zip(pending, scores):
                valid = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 10
                pool[i].update(vlm_raw_score=score if valid else None, vlm_status="reranked" if valid else "invalid_score")
                if not valid:
                    pool[i]["vlm_error"] = "invalid_score"
            ranked, stats = finalize_vlm_ranking(pool, top_k=20, vlm_weight=.5)
            info = {"arm": arm, "seconds": elapsed, "inputs": len(pending),
                    "invalid": sum(s is None for s in scores), "scores": scores,
                    "batch": deepcopy(scorer.last_batch_stats),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "pre_qwen": metrics(q["retrieval"]["A" if arm == "A" else "B"]["pool"], q["query"]),
                    "post_qwen": metrics(ranked, q["query"]), "final_ranking": ranked[:100], "ranking_stats": stats}
            record["arms"][arm] = info
            with (output / "SCORES.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": q["query"]["id"], **info}, ensure_ascii=False) + "\n")
            print(json.dumps({"phase": "scored", "query": q["query"]["id"], "arm": arm,
                  "seconds": elapsed, "inputs": len(pending), "invalid": info["invalid"],
                  "pre": info["pre_qwen"], "post": info["post_qwen"]}), flush=True)
        records.append(record)
    summary = {}
    for arm in ("A", "B", "C"):
        rows = [q["arms"][arm] for q in records]
        summary[arm] = {"queries": len(rows), "invalid_scores": sum(r["invalid"] for r in rows),
                        "inputs": sum(r["inputs"] for r in rows),
                        "mean_seconds": sum(r["seconds"] for r in rows) / len(rows),
                        "metrics": {k: sum(r["post_qwen"][k] for r in rows) / len(rows)
                                    for k in rows[0]["post_qwen"] if k != "first_known_positive_rank"}}
    report = {"status": "completed", "model_load_seconds": load_seconds, "records": records, "summary": summary,
              "limitations": ["Small exploratory image-authored/AI-reviewed, not human-verified or held-out",
                  "Known-positive metrics only, unjudged results not negatives; no precision/accuracy claim",
                  "Six queries share one pinned pipeline and one inference per arm; not a confidence interval",
                  "Same B/C pool/order/bytes, no top20 backfill; invalid scores kept unknown"]}
    save(output / "REPORT.json", report)
    print(json.dumps({"status": "completed", "summary": summary}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("anchors", "prepare", "gpu"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"anchors": anchors, "prepare": prepare, "gpu": gpu}[args.phase](args.output)
