"""Export a blind GPU packet and replay returned scores locally against qrels."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import zipfile

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.config import VLM_CONFIG, load_runtime_config
from kis.paths import KIS_ROOT, local_image_settings, video_csv_path
from kis.ranking import finalize_vlm_ranking
from kis.support_calibration import support_metrics
from tools.condition_evidence_evaluation import ArtifactCaptions
from tools.human_holdout import sha
from tools.kaggle_vlm_benchmark import PROMPT_VERSION, task_id, validate_packet
from tools.pipeline_ablation import ARMS, metrics


BUNDLE_FILES = ("kis/__init__.py", "kis/config.py", "kis/paths.py", "kis/temporal.py", "kis/text_reranker.py", "kis/ocr.py",
                "kis/ranking.py", "kis/vlm_reranker.py", "config/models.json",
                "tools/__init__.py", "tools/kaggle_vlm_benchmark.py", "config/requirements-kaggle-vlm.txt")


@lru_cache(maxsize=512)
def mapping(video_id):
    with video_csv_path(video_id).open(encoding="utf-8", newline="") as stream:
        rows = [{"frame_id": int(row["frame_idx"]), "keyframe_n": int(row["n"]), "pts_time": float(row["pts_time"])}
                for row in csv.DictReader(stream)]
    return rows


def image_reference(candidate):
    rows = [row for row in mapping(candidate["video_id"]) if row["frame_id"] == candidate["frame_id"]]
    if len(rows) != 1:
        return {"status": "identity_ambiguous"}
    if candidate.get("keyframe_n") is not None and int(candidate["keyframe_n"]) != rows[0]["keyframe_n"]:
        return {"status": "identity_ambiguous"}
    return {"status": "requested", "video_id": candidate["video_id"], **rows[0]}


def read_cases(source_run):
    source_run = Path(source_run)
    if (source_run / "failure.json").exists() or not (source_run / "case_index.json").is_file():
        raise ValueError("CPU ablation must finish successfully before export")
    index = json.loads((source_run / "case_index.json").read_text(encoding="utf-8"))
    return [json.loads((source_run / "cases" / (item["case_id"] + ".json")).read_text(encoding="utf-8")) for item in index]


def export(source_run, output, source=None):
    source_run, output = Path(source_run), Path(output)
    cases = read_cases(source_run)
    protocol = json.loads((source_run / "protocol.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError("Choose a new packet directory; never overwrite exported tasks")
    output.mkdir(parents=True)
    packet = output / "vlm_packet"
    (packet / "images").mkdir(parents=True)
    captions = ArtifactCaptions(load_runtime_config().caption_root)
    requests, image_requests = [], {}
    bindings = {"schema": "kis_private_vlm_bindings_v1", "source_protocol_sha256": sha(source_run / "protocol.json"),
                "cases": {}, "pairs": []}

    def add_image(candidate, action="", expected_hash=None, with_caption=True):
        ref = image_reference(candidate)
        if ref["status"] == "requested":
            key = f"{ref['video_id']}:{ref['keyframe_n']}"
            existing = image_requests.setdefault(key, {**ref, "expected_hashes": []})
            if expected_hash:
                existing["expected_hashes"].append(expected_hash)
            metadata = captions.get_caption(ref["video_id"], ref["frame_id"])
            if metadata and metadata.get("keyframe_n") is not None and int(metadata["keyframe_n"]) != ref["keyframe_n"]:
                raise ValueError("Caption/frame identity mismatch during packet export")
            ref = {"request_key": key, "pts_time": ref["pts_time"], "action": action,
                   "caption": str(metadata.get("retrieval_text", metadata.get("caption", ""))) if metadata and with_caption else ""}
        else:
            ref = {"status": ref["status"], "pts_time": candidate.get("pts_time", 0), "action": action, "caption": ""}
        return ref

    def add_request(payload, role):
        index = len(requests)
        requests.append({"payload": payload, "priority": 0 if role == "development" else 1})
        return index

    for case in cases:
        unit, case_id = case["unit"], case["case_id"]
        bindings["cases"][case_id] = {"case_sha256": sha(source_run / "cases" / (case_id + ".json")), "modes": {}}
        for caption_mode in protocol["vlm_caption_modes"]:
            selected = {}
            depth = protocol["vlm_top_k_images"] if unit["task"] == "image" else protocol["vlm_top_k_paths"]
            for i, candidate in enumerate(case["results"][:depth]):
                if unit["task"] == "image":
                    images = [add_image(candidate, with_caption=caption_mode == "with_caption")]
                    kind = "image"
                else:
                    temporal = candidate.get("temporal", {})
                    events = temporal.get("events", [])
                    if temporal.get("coverage") != 1 or not events or any(event.get("status") != "retrieved" for event in events):
                        continue  # Never score an incomplete sequence as if a representative image proved it.
                    images = [add_image(event, action=event["action"], with_caption=caption_mode == "with_caption") for event in events]
                    kind = "path"
                selected[str(i)] = add_request({"kind": kind, "caption_mode": caption_mode,
                    "query": unit["raw_query"], "images": images}, unit["role"])
            bindings["cases"][case_id]["modes"][caption_mode] = selected
    seen_units = set()
    for case in cases:
        unit = case["unit"]
        if unit["task"] != "image" or unit["query_id"] in seen_units:
            continue
        seen_units.add(unit["query_id"])
        for judgment in unit["judgments"]:
            pair = {"query_id": unit["query_id"], "role": unit["role"], "split": unit["split"],
                    **judgment, "modes": {}}
            for caption_mode in protocol["vlm_caption_modes"]:
                pair["modes"][caption_mode] = add_request({"kind": "image", "caption_mode": caption_mode,
                    "query": unit["event_text"], "images": [add_image(judgment, expected_hash=judgment["image_sha256"],
                        with_caption=caption_mode == "with_caption")]}, unit["role"])
            bindings["pairs"].append(pair)
    if source is None:
        from kis.local_images import DriveImages
        settings = local_image_settings()
        folder = os.environ.get("KIS_DRIVE_FOLDER_ID") or settings.get("drive_folder_id")
        if not folder:
            raise RuntimeError("Existing read-only Drive image source is not configured")
        source = DriveImages(folder, credentials_file=settings.get("credentials_file"))

    def fetch(item):
        key, ref = item
        try:
            data, mime = source.read(ref["video_id"], ref["keyframe_n"])
            digest = hashlib.sha256(data).hexdigest()
            if any(expected != digest for expected in ref["expected_hashes"]):
                return key, {"status": "image_checksum_mismatch"}
            with Image.open(BytesIO(data)) as image:
                image.verify()
            relative = "images/" + digest + (".png" if mime == "image/png" else ".jpg")
            try:
                with (packet / relative).open("xb") as stream:
                    stream.write(data)
            except FileExistsError:
                pass
            return key, {"status": "ready", "path": relative, "sha256": digest}
        except Exception as exc:
            return key, {"status": "download_failed", "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=4) as pool:
        image_results = dict(pool.map(fetch, image_requests.items()))
    task_ids, public_tasks, priorities = {}, {}, {}
    for index, request in enumerate(requests):
        task = request["payload"]
        for image in task["images"]:
            key = image.pop("request_key", None)
            if key is not None:
                image.update(image_results[key])
        task["task_id"] = task_id(task, VLM_CONFIG)
        task_ids[index] = task["task_id"]
        public_tasks[task["task_id"]] = task
        priorities[task["task_id"]] = min(priorities.get(task["task_id"], 1), request["priority"])
    for case in bindings["cases"].values():
        for mode in case["modes"].values():
            for index, request_index in mode.items():
                mode[index] = task_ids[request_index]
    for pair in bindings["pairs"]:
        pair["modes"] = {mode: task_ids[index] for mode, index in pair["modes"].items()}
    tasks = sorted(public_tasks.values(), key=lambda task: (priorities[task["task_id"]], task["task_id"]))
    manifest = {"schema": "kis_blind_vlm_packet_v1", "model": VLM_CONFIG, "prompt_version": PROMPT_VERSION,
        "benchmark_protocol_sha256": sha(source_run / "protocol.json"), "tasks": tasks,
        "contains_qrels": False, "contains_credentials": False,
        "source_sha256": {name: sha(KIS_ROOT / name) for name in BUNDLE_FILES}}
    atomic_write_json(packet / "manifest.json", manifest)
    validate_packet(packet)
    bindings["manifest_sha256"] = sha(packet / "manifest.json")
    atomic_write_json(output / "bindings_private.json", bindings)
    atomic_write_json(output / "export_summary.json", {"tasks": len(tasks), "cases": len(cases),
        "pair_diagnostics": len(bindings["pairs"]), "unique_image_requests": len(image_requests),
        "download_status": dict(Counter(item["status"] for item in image_results.values())),
        "task_kinds": dict(Counter(task["kind"] for task in tasks)),
        "development_tasks_first": True, "vlm_loaded_locally": False,
        "manifest_sha256": bindings["manifest_sha256"], "bindings_private_sha256": sha(output / "bindings_private.json")})
    print((output / "export_summary.json").read_text(encoding="utf-8"))


def package(packet_root, destination, notebook_path):
    packet_root, destination, notebook_path = map(Path, (packet_root, destination, notebook_path))
    packet = packet_root / "vlm_packet"
    manifest = validate_packet(packet)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or notebook_path.exists():
        raise FileExistsError("Do not overwrite a published benchmark bundle/notebook")
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            if sha(KIS_ROOT / name) != manifest["source_sha256"][name]:
                raise ValueError("Scoring code changed after packet creation")
            archive.write(KIS_ROOT / name, "kis_vlm_benchmark/" + name)
        archive.write(packet / "manifest.json", "kis_vlm_benchmark/vlm_packet/manifest.json")
        paths = sorted({image["path"] for task in manifest["tasks"] for image in task["images"] if image.get("status") == "ready"})
        for relative in paths:
            archive.write(packet / relative, "kis_vlm_benchmark/vlm_packet/" + relative)
    cells = []
    def cell(kind, text):
        value = {"cell_type": kind, "id": f"cell-{len(cells)}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell("markdown", "# KIS — blind VLM benchmark trên Kaggle\nChọn **GPU T4** và bật **Internet**. Thêm ZIP dưới dạng **private Dataset**. Không cần khóa Drive/API.\nKhông chứa ground truth; kết quả được đối chiếu nhãn trên máy sau khi tải về.")
    cell("code", "from pathlib import Path\nimport zipfile, sys, shutil\ninput_root = Path('/kaggle/input')\ninputs = list(input_root.rglob('kis-vlm-benchmark.zip'))\nwork = Path('/kaggle/working/benchmark_code')\nwork.mkdir(exist_ok=True)\nroot = work / 'kis_vlm_benchmark'\nif inputs:\n    assert len(inputs) == 1, 'Add only one benchmark ZIP'\n    with zipfile.ZipFile(inputs[0]) as z:\n        for name in z.namelist():\n            p = Path(name)\n            assert not p.is_absolute() and '..' not in p.parts, 'Unsafe archive path'\n        z.extractall(work)\nelse:\n    # Kaggle may expand uploaded ZIPs automatically. Accept the same allowlisted contents.\n    manifests = [p for p in input_root.rglob('manifest.json') if p.parent.name == 'vlm_packet' and (p.parent.parent / 'tools/kaggle_vlm_benchmark.py').is_file()]\n    assert len(manifests) == 1, 'Add the private benchmark Dataset (ZIP or expanded contents)'\n    shutil.copytree(manifests[0].parent.parent, root, dirs_exist_ok=True)\nsys.path.insert(0, str(root))\nprint(root)\n")
    cell("code", "import subprocess\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root / 'config/requirements-kaggle-vlm.txt')], check=True)\n")
    cell("code", "import torch\nassert torch.cuda.is_available(), 'Enable GPU in Notebook Settings'\nprint(torch.cuda.get_device_name(0))\nfrom tools.kaggle_vlm_benchmark import run\n# Use 100 for an initial smoke run; then None to continue all remaining tasks.\n# Repeating this cell resumes completed task IDs without changing model/prompt/settings.\nMAX_TASKS = None\nOUTPUT = Path('/kaggle/working/vlm-results')\nrun(root / 'vlm_packet', OUTPUT, max_tasks=MAX_TASKS)\n")
    cell("code", "import shutil\nfrom IPython.display import FileLink, display\npath = shutil.make_archive('/kaggle/working/vlm-results', 'zip', OUTPUT)\ndisplay(FileLink(path))\n")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(destination.resolve()), "size_mib": round(destination.stat().st_size/2**20, 2),
                      "notebook": str(notebook_path.resolve()), "tasks": len(manifest["tasks"]),
                      "ground_truth_in_bundle": False}, indent=2))


def replay(source_run, packet_root, result_dir, output):
    source_run, packet_root, result_dir, output = map(Path, (source_run, packet_root, result_dir, output))
    manifest = validate_packet(packet_root / "vlm_packet")
    bindings = json.loads((packet_root / "bindings_private.json").read_text(encoding="utf-8"))
    context = json.loads((result_dir / "run_context.json").read_text(encoding="utf-8"))
    protocol = json.loads((source_run / "protocol.json").read_text(encoding="utf-8"))
    exported = json.loads((packet_root / "export_summary.json").read_text(encoding="utf-8"))
    if (bindings["source_protocol_sha256"] != sha(source_run / "protocol.json")
            or manifest["benchmark_protocol_sha256"] != bindings["source_protocol_sha256"]
            or sha(packet_root / "bindings_private.json") != exported["bindings_private_sha256"]
            or bindings["manifest_sha256"] != sha(packet_root / "vlm_packet/manifest.json")
            or context["manifest_sha256"] != bindings["manifest_sha256"]
            or context.get("execution") != "kaggle_cuda_qwen_nf4" or context.get("model") != VLM_CONFIG
            or context.get("schema") != "kis_blind_vlm_results_v1"
            or context.get("prompt_version") != PROMPT_VERSION or context.get("use_4bit") is not True
            or context.get("min_pixels") != 256*256 or context.get("max_pixels") != 512*512):
        raise ValueError("Returned scoring context does not match the frozen real Kaggle run")
    expected_sources = {"tools/kaggle_vlm_benchmark.py", "kis/vlm_reranker.py", "kis/ranking.py", "config/models.json"}
    if set(context["source_sha256"]) != expected_sources:
        raise ValueError("Missing scoring source fingerprints")
    for name, fingerprint in context["source_sha256"].items():
        if manifest["source_sha256"].get(name) != fingerprint:
            raise ValueError("Returned VLM scoring code differs from packet")
    for name, fingerprint in protocol["source_sha256"].items():
        if sha(KIS_ROOT / name) != fingerprint:
            raise ValueError("A frozen production source changed before replay")
    scores = {}
    for line in (result_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_id"] in scores:
            raise ValueError("Duplicate returned task")
        value = row.get("score")
        if row["status"] == "scored" and (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value != int(value) or not 0 <= value <= 10):
            raise ValueError("Invalid numeric VLM score")
        scores[row["task_id"]] = row
    if set(scores) != {task["task_id"] for task in manifest["tasks"]}:
        raise ValueError("GPU scoring is incomplete or has foreign task IDs; resume before final comparison")
    if output.exists():
        raise FileExistsError("Choose a new replay output")
    output.mkdir(parents=True)
    rows = []
    for case in read_cases(source_run):
        case_id = case["case_id"]
        if sha(source_run / "cases" / (case_id + ".json")) != bindings["cases"][case_id]["case_sha256"]:
            raise ValueError("CPU candidate case changed after export")
        for mode, task_map in bindings["cases"][case_id]["modes"].items():
            candidates = copy.deepcopy(case["results"])
            for candidate in candidates:
                candidate.pop("vlm_raw_score", None)
            for index, task in task_map.items():
                score = scores[task]
                candidates[int(index)]["vlm_raw_score"] = score["score"] if score["status"] == "scored" else None
                candidates[int(index)]["vlm_status"] = score["status"]
            top_k = protocol["vlm_top_k_images"] if case["unit"]["task"] == "image" else protocol["vlm_top_k_paths"]
            ranked, applied = finalize_vlm_ranking(candidates, top_k=top_k, vlm_weight=protocol["vlm_weight"])
            row = {"case_id": case_id, "role": case["unit"]["role"], "task": case["unit"]["task"],
                   "arm": case["arm"], "caption_mode": mode, "vlm_applied": applied,
                   "baseline_metrics": case["metrics"], "vlm_metrics": metrics(ranked, case["unit"]),
                   "tasks_expected": len(task_map), "tasks_scored": sum(scores[t]["status"] == "scored" for t in task_map.values()),
                   "results": ranked[:20], "sequence_verified": False}
            rows.append(row)
            atomic_write_json(output / (case_id + "__" + mode + ".json"), row)
    summary = {}
    for role in ("development", "observed_regression"):
        summary[role] = {}
        for task in ("image", "sequence"):
            summary[role][task] = {}
            fields = (["known_frame_hit_at_1", "known_frame_hit_at_5", "known_frame_hit_at_20", "mrr_at_20"]
                      if task == "image" else ["known_path_hit_at_1", "known_path_hit_at_5", "known_path_hit_at_20", "top_target_video"])
            for arm in ARMS:
                for mode in protocol["vlm_caption_modes"]:
                    group = [row for row in rows if row["role"] == role and row["task"] == task and row["arm"] == arm and row["caption_mode"] == mode]
                    summary[role][task][arm + "__" + mode] = {"queries": len(group),
                        **{field: sum(row["vlm_metrics"][field] for row in group)/len(group) if group else None for field in fields},
                        "tasks_expected": sum(row["tasks_expected"] for row in group), "tasks_scored": sum(row["tasks_scored"] for row in group)}
    pair_reports = {}
    for role in ("development", "observed_regression"):
        group = [pair for pair in bindings["pairs"] if pair["role"] == role]
        paired = [pair for pair in group if all(scores[task]["status"] == "scored" for task in pair["modes"].values())]
        pair_reports[role] = {"total_pairs": len(group), "paired_valid_pairs": len(paired), "threshold": protocol["pair_vlm_positive_threshold"],
            "modes": {mode: support_metrics([{**pair, "visual": scores[pair["modes"][mode]]["score"]/10, "semantic": None}
                for pair in paired], protocol["pair_vlm_positive_threshold"]/10, 1) for mode in protocol["vlm_caption_modes"]}}
    atomic_write_json(output / "summary.json", {"retrieval": summary, "pair_diagnostics": pair_reports,
        "scored_tasks": sum(row["status"] == "scored" for row in scores.values()), "total_tasks": len(scores),
        "automatically_promoted": False, "independent_new_holdout": False,
        "note": "V3 is observed regression. Invalid/missing scores keep production retrieval ranking; never become negative labels."})
    print(json.dumps({"output": str(output.resolve()), "retrieval": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--source-run", type=Path, required=True)
    exp.add_argument("--output-dir", type=Path, required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--packet-root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--notebook", type=Path, required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--source-run", type=Path, required=True)
    rep.add_argument("--packet-root", type=Path, required=True)
    rep.add_argument("--result-dir", type=Path, required=True)
    rep.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.source_run, args.output_dir)
    elif args.command == "package":
        package(args.packet_root, args.output, args.notebook)
    else:
        replay(args.source_run, args.packet_root, args.result_dir, args.output_dir)
