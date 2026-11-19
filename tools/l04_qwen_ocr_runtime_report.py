"""Post-completion, offline runtime-only audit; never runs a model or changes inputs.

Use only after the original worker and report have finished. Hash consistency is
not independent proof of model execution. Synthetic fixtures are test-only.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import l04_qwen_ocr_ablation as schema


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"protocol.json", "config.json", "queries.private.json", "pools.json",
          "input_manifest.json", "coverage.json", "preservation.json", "source_manifest.json"}
SOURCES = {"tools/l04_qwen_ocr_ablation.py", "tools/l04_ocr_ablation.py",
           "kis/vlm_reranker.py", "tools/kaggle_vlm_benchmark.py", "config/models.json"}


def checked_path(value):
    """No traversal, UNC/network paths, links, junctions or cloud placeholders."""
    path = Path(value)
    if ".." in path.parts or str(path).startswith(("\\\\", "//")):
        raise ValueError("Unsafe path")
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x441400:
            raise ValueError("Linked/reparse/cloud path forbidden")
    return path


def scoped(base, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Unsafe manifest path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe manifest path")
    path = checked_path(base / relative)
    if not path.is_relative_to(base):
        raise ValueError("Manifest path outside scope")
    return path


def strict_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer count/identity")
    return value


def seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid elapsed seconds")
    return value


def aggregate(rows):
    elapsed = math.fsum(seconds(row["seconds"]) for row in rows)
    if not math.isfinite(elapsed):
        raise ValueError("Nonfinite aggregate timing")
    return {"calls": len(rows), "seconds_total": elapsed,
            "input_tokens_total": sum(integer(row["input_tokens"], 1) for row in rows),
            "generated_tokens_total": sum(integer(row["generated_tokens"], 1) for row in rows)}


def write_audit(run_dir, output_dir, *, synthetic_test_only=False, repo_root=ROOT):
    """Validate before writing. repo_root is injectable for isolated synthetic tests."""
    run, output, root = map(checked_path, (run_dir, output_dir, repo_root))
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Audit output must be outside the source run")
    if output.exists():
        raise FileExistsError("Audit output must be a new explicit directory")
    if not output.parent.is_dir():
        raise ValueError("Audit output parent must already exist")
    hashes = {}

    def digest(path):
        path = checked_path(path)
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        hashes[str(path)] = value
        return value

    def load(name):
        path = scoped(run, name)
        raw = path.read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        if str(path) in hashes and hashes[str(path)] != value:
            raise ValueError("Input changed during audit: " + str(path))
        hashes[str(path)] = value
        return json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant)

    def match(path, expected):
        if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
            raise ValueError("SHA256 mismatch: " + str(path))

    try:
        frozen = load("freeze.json")
        if frozen["frozen_before_model"] is not True or set(frozen["files"]) != FROZEN:
            raise ValueError("Unsupported/incomplete freeze schema")
        for name, expected in frozen["files"].items():
            match(scoped(run, name), expected)
        artifacts = {name: load(name) for name in sorted(FROZEN)}
        protocol, config = artifacts["protocol.json"], artifacts["config.json"]
        if config != schema.CONFIG or protocol["config"] != config or protocol["query_ids"] != schema.IDS:
            raise ValueError("Unsupported frozen protocol/config schema")
        if frozen["query_source_sha256"] != protocol["query_packet_sha256"]:
            raise ValueError("Query source hash binding mismatch")
        sources = artifacts["source_manifest.json"]
        if len(sources) != len(SOURCES) or {entry["path"] for entry in sources} != SOURCES:
            raise ValueError("Unsupported source manifest schema")
        for entry in sources:
            match(scoped(root, entry["path"]), entry["sha256"])
        for entry in artifacts["preservation.json"]:
            if Path(entry["path"]).parts[0] not in {"benchmarks", "benchmark_results"}:
                raise ValueError("Preservation path outside benchmark scope")
            match(scoped(root, entry["path"]), entry["sha256"])

        context, preflight = load("run_context.json"), load("preflight.json")
        if (context["freeze_sha256"] != hashes[str(run / "freeze.json")]
                or context["preflight_sha256"] != hashes[str(run / "preflight.json")]
                or context["source_manifest"] != sources or context["config"] != config
                or not isinstance(context["cache"], dict) or context["cache"] != preflight["cache"]
                or not isinstance(context["packages"], dict) or context["packages"] != preflight["packages_metadata_only"]):
            raise ValueError("Run context hash/config binding mismatch")
        if preflight["status"] != "PASS" or integer(load("worker_process.json")["returncode"]) != 0:
            raise ValueError("Successful completed worker/preflight required")
        smoke = load("smoke.json")
        if smoke["status"] != "PASS" or smoke["included_in_run"] is not True or smoke["warmup_excluded"] is not False:
            raise ValueError("Successful included smoke required")
        report, rows = load("report.json"), load("checkpoint.json")
        if (report["status"] != "COMPLETE" or report["metrics"] is None
                or report["preflight_status"] != "PASS" or report["blocker"] is not None
                or report["freeze_sha256"] != context["freeze_sha256"]):
            raise ValueError("Complete unblocked report required")
        pools = artifacts["pools.json"]
        queries = artifacts["queries.private.json"]["queries"]
        if [pool["id"] for pool in pools] != schema.IDS or [q["id"] for q in queries] != schema.IDS:
            raise ValueError("Frozen query/pool identity mismatch")
        for pool, query in zip(pools, queries):
            candidates = pool["candidates"]
            if pool["query"] != query["text"] or len(candidates) > config["top_k"]:
                raise ValueError("Invalid frozen pool")
            if [integer(d["baseline_position"], 1) for d in candidates] != list(range(1, len(candidates) + 1)):
                raise ValueError("Invalid baseline positions")
            if len({(d["video_id"], d["keyframe_n"]) for d in candidates}) != len(candidates):
                raise ValueError("Duplicate pool candidate")
        planned = schema.tasks(pools, context)  # Pure hashing/prompt construction only.
        expected_keys = {task[0] for task in planned}
        pairs = len(planned) // 2
        if (len(expected_keys) != len(planned) or not isinstance(rows, dict) or set(rows) != expected_keys
                or integer(frozen["planned_pairs"], 1) != pairs):
            raise ValueError("Incomplete/foreign checkpoint pairs")
        counts = {"planned_queries": len(pools), "planned_pairs": pairs, "planned_arm_scores": len(planned),
                  "completed_arm_scores": len(planned), "complete_pairs": pairs}
        if any(integer(report[key]) != value for key, value in counts.items()):
            raise ValueError("Report completion counts mismatch")
        if [(r["id"], integer(r["pool_size"])) for r in report["per_query"]] != [
                (p["id"], len(p["candidates"])) for p in pools]:
            raise ValueError("Report per-query pool mismatch")
        for key, pool, doc, arm, pair in planned:
            row = rows[key]
            if (row["task_key"] != key or row["query_id"] != pool["id"] or row["arm"] != arm
                    or integer(row["pair_index"]) != pair
                    or integer(row["baseline_position"], 1) != doc["baseline_position"]
                    or row["status"] != "scored" or type(row["score"]) not in (int, float)
                    or schema.parse_score(row["raw"]) != row["score"]):
                raise ValueError("Invalid checkpoint identity/status/score")
            seconds(row["seconds"])
            integer(row["input_tokens"], 1)
            if integer(row["generated_tokens"], 1) > config["max_new_tokens"]:
                raise ValueError("Generated token count exceeds frozen limit")

        totals = {arm: aggregate([r for r in rows.values() if r["arm"] == arm]) for arm in ("A", "B")}
        for arm, total in totals.items():
            recorded = report["runtime"][arm]
            for key in ("calls", "input_tokens_total", "generated_tokens_total"):
                if integer(recorded[key]) != total[key]:
                    raise ValueError("Report runtime count mismatch")
            if not math.isclose(seconds(recorded["seconds_total"]), total["seconds_total"], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("Report runtime timing mismatch")
        per_query = [{"id": pool["id"], **{
            arm: aggregate([r for r in rows.values() if r["query_id"] == pool["id"] and r["arm"] == arm])
            for arm in ("A", "B")}} for pool in pools]
        if not synthetic_test_only and "synthetic" in json.dumps([artifacts, context, report]).lower():
            raise ValueError("Synthetic inputs require --synthetic-test-only")
        result = {"schema": "l04_qwen_ocr_runtime_audit_v1",
                  "status": "SYNTHETIC_TEST_ONLY" if synthetic_test_only else "COMPLETE_RUNTIME_AUDIT",
                  "synthetic_test_only": synthetic_test_only, "source_run": str(run),
                  "counts": counts, "per_query": per_query, "runtime": totals,
                  "source_sha256": dict(hashes), "audit_tool_sha256": schema.sha(Path(__file__)),
                  "limitations": ["Runtime-only derivation; accuracy metrics are neither changed nor recomputed.",
                                  "Hash consistency is not independent proof of genuine model execution.",
                                  "Dataset bytes, cached model weights and hardware are not revalidated.",
                                  "Seconds are recorded per-call elapsed time (including image preparation), not GPU-only or whole-job wall time.",
                                  "Recorded smoke calls are included; model loading is excluded."]}
        # Reject concurrent input changes; no polling and no writes to the source run.
        for name, expected in list(hashes.items()):
            match(Path(name), expected)
        payload = schema.encoded(result)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Unsupported/malformed runtime audit schema: " + str(exc)) from exc

    lines = ["# L04 Qwen OCR runtime-only audit", "", "Status: **" + result["status"] + "**", "",
             *["- " + text for text in result["limitations"]], "",
             "| Query | Arm | Calls | Elapsed seconds | Input tokens | Generated tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for row in per_query + [{"id": "TOTAL", **totals}]:
        for arm in ("A", "B"):
            r = row[arm]
            lines.append(f"| {row['id']} | {arm} | {r['calls']} | {r['seconds_total']:.9f} | "
                         f"{r['input_tokens_total']} | {r['generated_tokens_total']} |")
    output.mkdir()  # Exclusive new directory; never overwrite existing artifacts.
    with (output / "runtime_audit.json").open("xb") as stream:
        stream.write(payload)
    with (output / "runtime_audit.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed original run (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New audit directory outside the run")
    parser.add_argument("--synthetic-test-only", action="store_true", help="Label synthetic fixture output; still require complete state")
    args = parser.parse_args()
    try:
        result = write_audit(args.run_dir, args.output_dir, synthetic_test_only=args.synthetic_test_only)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Runtime audit refused: {exc}\n")
    print(result["status"])


if __name__ == "__main__":
    main()
