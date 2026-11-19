"""L01 CPU-only retrieval pilot; isolated from production, DRES and VLM.

ZIP transfer/extraction preserves originals. Retrieval uses pinned encoders and
the shared solver, with exact local NPY cosine search instead of Milvus.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import zipfile
from collections import Counter

from tools.transfer_school_data import ROOT, remote_python, sha256_file, ssh_options


QUERIES = (
    "A person cooking food in a kitchen.",
    "A person wearing a helmet riding a motorcycle on a street.",
    "A city skyline at sunset with buildings and water.",
)


def image_destination(name):
    if "\\" in name or ":" in name or "\0" in name:
        raise ValueError("Unsafe ZIP member")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe ZIP path")
    videos = [part for part in path.parts if re.fullmatch(r"L01_V\d+", part)]
    if len(videos) != 1 or not re.fullmatch(r"[0-9]+\.(?:jpg|jpeg|png)", path.name, re.I):
        raise ValueError("ZIP member is not an L01 numbered keyframe")
    if int(Path(path.name).stem) < 1:
        raise ValueError("Invalid image number")
    return PurePosixPath("L01", videos[0], path.name)


def extract_images(archive, destination):
    archive, destination = Path(archive), Path(destination)
    with zipfile.ZipFile(archive) as zipped:
        members = [member for member in zipped.infolist() if not member.is_dir()]
        plan, seen = [], set()
        for member in members:
            if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                raise ValueError("Symlink/encrypted ZIP members are not supported")
            relative = image_destination(member.filename)
            identity = (relative.parts[1], int(Path(relative.name).stem))
            if identity in seen:
                raise ValueError("Duplicate keyframe identity after normalizing ZIP paths")
            if member.file_size <= 0 or member.file_size > 25 * 1024 ** 2:
                raise ValueError("Invalid image size in ZIP")
            seen.add(identity)
            plan.append((member, relative))
        total = sum(member.file_size for member, _ in plan)
        if not plan or total > 30 * 1024 ** 3:
            raise ValueError("Empty ZIP or extraction exceeds pilot budget")
        if shutil.disk_usage(destination.parent).free < total + 1024 ** 3:
            raise RuntimeError("Insufficient disk space for keyframes")
        destination.mkdir(mode=0o700, exist_ok=False)
        digest_records = []
        for member, relative in plan:
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256()
            # Reading to EOF also validates the original ZIP member CRC.
            with zipped.open(member) as source, output.open("xb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
            output.chmod(0o600)
            if output.stat().st_size != member.file_size:
                raise RuntimeError("Extracted image size mismatch")
            digest_records.append({"path": relative.as_posix(), "bytes": member.file_size,
                                   "sha256": digest.hexdigest(), "zip_member": member.filename})
    report = {"images": len(plan), "videos": len({identity[0] for identity in seen}),
              "bytes": total, "crc_checked": True, "sources_deleted": False, "files": digest_records}
    (destination.parent / "KEYFRAMES_MANIFEST.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def run_cpu(artifact_root, image_root, output, *, groups=("L01",), queries=QUERIES, repetitions=2):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      FAISS_OPT_LEVEL="generic", KIS_DATA_ROOT=str(image_root), KIS_VIDEO_GROUPS=",".join(groups))
    import resource
    import numpy as np
    import torch
    import faiss
    from PIL import Image
    from kis.config import MODEL_MANIFEST
    from kis.text import CaptionRetriever, BM25Retriever
    from kis.visual import SiglipEncoder
    from kis.solve_kis import solve_kis
    from kis.ranking import get_search_profile
    from kis.objects import ObjectMetadataLookup
    from kis.temporal import TemporalConfig
    from kis.result_pool import require_frame_count
    from kis.paths import resolve_video_artifact

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(4)
    artifact_root, image_root, output = Path(artifact_root), Path(image_root), Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    rows, matrices, mapped, missing, ambiguous = [], [], 0, [], set()
    mapping_paths = sorted(p for group in groups for p in
        (artifact_root / "mapping" / group).rglob(f"{group}_V*.csv"))
    for path in mapping_paths:
        video_id = path.stem
        with path.open(encoding="utf-8", newline="") as stream:
            entries = list(csv.DictReader(stream))
        vectors = np.load(resolve_video_artifact(artifact_root / "clip-features-32", video_id, ".npy",
                                                video_groups=groups), allow_pickle=False)
        if vectors.shape != (len(entries), 768) or not np.isfinite(vectors).all():
            raise ValueError("Invalid visual vector/mapping shape")
        counts = Counter(int(row["frame_idx"]) for row in entries)
        keep = []
        for position, row in enumerate(entries):
            n, frame_id = int(row["n"]), int(row["frame_idx"])
            image = image_root / video_id.split("_")[0] / video_id / f"{n:03d}.jpg"
            mapped += 1
            if not image.is_file():
                missing.append(f"{video_id}/{n:03d}.jpg")
            if counts[frame_id] > 1:
                ambiguous.add((video_id, frame_id))
                continue
            keep.append(position)
            rows.append({"video_id": video_id, "frame_id": frame_id, "keyframe_n": n})
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Zero stored visual embedding")
        matrices.append(np.asarray((vectors / norms)[keep], dtype=np.float32))
    if missing or len(rows) < 100:
        raise ValueError(f"Image/mapping coverage failed: mapped={mapped}, missing={len(missing)}, sample={missing[:5]}")
    matrix = np.vstack(matrices)

    class LocalVisualRetriever:
        def search(self, vector, top_k=100):
            vector = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if vector.shape != (768,) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid query vector")
            scores = matrix @ (vector / norm)
            indices = np.argsort(-scores, kind="stable")[:top_k]
            return [dict(rows[int(i)], score=float(scores[i])) for i in indices]

    loading = {}
    tick = time.perf_counter()
    captions = CaptionRetriever(video_groups=groups, artifact_root=artifact_root)
    raw_caption_count = len(captions.caption_data)
    if ambiguous:
        original_captions = captions

        class UnambiguousCaptions:
            caption_data = [r for r in original_captions.caption_data
                            if (r["video_id"], r["frame_id"]) not in ambiguous]

            def search(self, query, top_k=100):
                hits = original_captions.search(query, top_k=top_k + 2 * len(ambiguous))
                return [r for r in hits if (r["video_id"], r["frame_id"]) not in ambiguous][:top_k]

            def get_caption(self, video_id, frame_id):
                return original_captions.get_caption(video_id, frame_id)

        captions = UnambiguousCaptions()
    loading["captions_and_minilm"] = time.perf_counter() - tick
    tick = time.perf_counter()
    encoder = SiglipEncoder(device="cpu")
    loading["siglip"] = time.perf_counter() - tick
    tick = time.perf_counter()
    bm25 = BM25Retriever(captions.caption_data, tokenizer_mode="unicode")
    loading["bm25"] = time.perf_counter() - tick
    lookup = ObjectMetadataLookup(mapping_dir=artifact_root / "mapping", objects_dir=artifact_root / "objects")
    visual = LocalVisualRetriever()
    runs = []
    gallery = output / "gallery"
    gallery.mkdir()
    (gallery / "thumbnails").mkdir()
    (gallery / "frames").symlink_to(image_root, target_is_directory=True)
    for query_index, query in enumerate(queries, 1):
        structured = {"raw_query": query, "query_variants": [query], "entities": [],
                      "events": [], "temporal_constraints": [], "needs_ocr": False}
        for repetition in range(repetitions):
            diagnostics = {}
            tick = time.perf_counter()
            results = solve_kis(structured, visual, encoder, bm25, captions, lookup,
                retrieval_top_k=100, result_top_k=100, rerank_top_k=0,
                temporal_config=TemporalConfig(enabled=False),
                policy=get_search_profile("rrf").policy, diagnostics=diagnostics,
                require_full_results=True, temporal_as_auxiliary=True)
            elapsed = time.perf_counter() - tick
            require_frame_count(results)
            if repetition == 0:
                cards = []
                image_started = time.perf_counter()
                for rank, candidate in enumerate(results, 1):
                    relative = PurePosixPath(candidate["video_id"].split("_")[0], candidate["video_id"], f"{int(candidate['keyframe_n']):03d}.jpg")
                    path = image_root.joinpath(*relative.parts)
                    thumb = f"thumbnails/q{query_index}_{rank:03d}.jpg"
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        image.thumbnail((480, 270))
                        image.save(gallery / thumb, quality=80)
                        image.close()
                    candidate["image_relative"] = relative.as_posix()
                    cards.append(f'<article><a href="frames/{relative.as_posix()}" target="_blank"><img loading="lazy" src="{thumb}" alt="{candidate["video_id"]} frame {candidate["frame_id"]}"></a><strong>#{rank} {candidate["video_id"]} / IMG {candidate["keyframe_n"]} / FRAME {candidate["frame_id"]}</strong><p>{html.escape(candidate.get("retrieval_text", ""))}</p></article>')
                image_seconds = time.perf_counter() - image_started
                page = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L01 CPU retrieval pilot</title><style>body{font:16px system-ui;margin:24px;background:#f2f0e8;color:#151515}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}article{background:white;padding:12px;border:1px solid #bbb;overflow-wrap:anywhere}img{width:100%;height:180px;object-fit:contain;background:#111}strong{display:block;margin-top:8px}p{line-height:1.5}nav a{margin-right:16px}</style>'
                nav = " ".join(f'<a href="q{i}.html">Query {i}</a>' for i in range(1, len(queries) + 1))
                page += f'<h1>{html.escape(", ".join(groups))} — Truy hồi thật trên CPU trường</h1><p>Thử nghiệm 100 frame. Không Groq/Qwen; local exact NPY thay Milvus. Chưa đánh giá độ chính xác ground truth.</p><nav>{nav}</nav><h2>{html.escape(query)}</h2><p>Retrieval: {elapsed:.3f}s. Đã đọc và tạo thumbnail đủ 100 ảnh.</p><main>{"".join(cards)}</main></html>'
                (gallery / f"q{query_index}.html").write_text(page, encoding="utf-8")
            else:
                image_seconds = None
            runs.append({"query": query, "repetition": repetition + 1, "retrieval_seconds": elapsed,
                         "images_decode_thumbnail_seconds": image_seconds, "count": len(results),
                         "unique_count": len({(r['video_id'], r['frame_id']) for r in results}),
                         "stage_seconds": diagnostics["stage_seconds"], "results": results})
    (gallery / "index.html").write_text((gallery / "q1.html").read_text(encoding="utf-8"), encoding="utf-8")
    report = {"status": "passed", "scope": ",".join(groups), "cpu_only": True, "threads": 4,
              "mapped_images": mapped, "missing_images": missing, "caption_rows": len(captions.caption_data),
              "raw_caption_rows": raw_caption_count, "ambiguous_frame_ids_excluded": sorted(ambiguous),
              "visual_vectors": len(rows), "loading_seconds": loading, "runs": runs,
              "model_manifest": MODEL_MANIFEST, "torch": torch.__version__, "numpy": np.__version__,
              "faiss": faiss.__version__, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "total_cpu_test_seconds": time.perf_counter() - started,
              "limitations": ["fixed English queries; no Groq planner", "no Qwen/VLM",
                              "exact local NPY cosine instead of Milvus", "selected groups only; not a relevance benchmark"],
              "gallery": str(gallery), "sources_deleted": False}
    (output / "CPU_REPORT.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


BOOTSTRAP = '''
import hashlib,json,pathlib,subprocess,os,zipfile,stat
cfg=CONFIG
base=pathlib.Path(cfg['destination'])
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for c in iter(lambda:s.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
for name in ['L01.zip','pilot-source.zip']:
    part=base/(name+'.partial')
    if digest(part)!=cfg['hashes'][name]: raise RuntimeError('Upload checksum mismatch: '+name)
    part.rename(base/name)
source=base/'source'
source.mkdir(mode=0o700)
with zipfile.ZipFile(base/'pilot-source.zip') as z:
    if set(z.namelist())!=set(cfg['source_members']): raise RuntimeError('Unexpected source bundle members')
    for i in z.infolist():
        p=pathlib.PurePosixPath(i.filename)
        if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16): raise RuntimeError('Unsafe source member')
        out=source.joinpath(*p.parts)
        out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with out.open('xb') as s: s.write(z.read(i))
(source/'dataset').symlink_to(pathlib.Path(cfg['artifact_root']),target_is_directory=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FAISS_OPT_LEVEL='generic',PYTHONPATH=str(source))
log=base/'cpu-pilot.log'
with log.open('x') as stream:
    result=subprocess.run([cfg['python'],'-u','-m','tools.school_l01_pilot','--remote-run','--zip',str(base/'L01.zip'),'--artifact-root',cfg['artifact_root'],'--output-dir',str(base)],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError('Remote pilot failed; source retained. Log tail: '+log.read_text()[-5000:])
report=json.loads((base/'results/CPU_REPORT.json').read_text())
print(json.dumps(dict(destination=str(base),report=report)))
'''


def orchestrate(zip_path, output_dir, target, artifact_root):
    zip_path, output_dir = Path(zip_path).resolve(), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        identities = [image_destination(i.filename).as_posix() for i in infos]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate ZIP destinations")
        unpacked = sum(i.file_size for i in infos)
    source_zip = output_dir / "pilot-source.zip"
    sources = [p for p in (ROOT / "kis").glob("*.py") if p.name != "local_images.py"]
    sources += [ROOT / "tools" / name for name in ["__init__.py", "school_l01_pilot.py", "transfer_school_data.py"]]
    sources += [ROOT / "config" / name for name in ["models.json", "pipeline.json", "object_labels.json"]]
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
    members = [p.relative_to(ROOT).as_posix() for p in sources]
    hashes = {"L01.zip": sha256_file(zip_path), "pilot-source.zip": sha256_file(source_zip)}
    name = "kis-l01-pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {"hashes": hashes, "source_members": members, "artifact_root": artifact_root}
    prepared = remote_python(target, "import pathlib,json,shutil\nh=pathlib.Path.home()\n"
        f"p=h/{name!r}\n"
        f"assert shutil.disk_usage(h).free>{zip_path.stat().st_size + unpacked + 2 * 1024 ** 3}, 'Insufficient disk space'\n"
        "p.mkdir(mode=0o700,exist_ok=False)\n"
        "for n in ['L01.zip.partial','pilot-source.zip.partial']:\n    (p/n).open('xb').close()\n"
        "print(json.dumps(dict(destination=str(p),python=str(h/'KISBranche/.venv/bin/python'))))")
    config.update(prepared)
    (output_dir / "PILOT_STATE.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "uploading_zip", "destination": config["destination"],
                      "zip_bytes": zip_path.stat().st_size, "images": len(infos), "hashes": hashes}), flush=True)
    batch = "\n".join(f'put -a "{p.as_posix()}" "{config["destination"]}/{name}.partial"'
                      for name, p in [("pilot-source.zip", source_zip.resolve()), ("L01.zip", zip_path)]) + "\n"
    subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", target], input=batch, text=True, check=True)
    print(json.dumps({"phase": "extract_and_cpu_retrieval", "destination": config["destination"]}), flush=True)
    result = remote_python(target, BOOTSTRAP.replace("CONFIG", repr(config), 1))
    (output_dir / "REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = result["report"]
    print(json.dumps({"phase": "complete", "destination": result["destination"], "status": report["status"],
                      "mapped_images": report["mapped_images"], "peak_rss_kib": report["peak_rss_kib"],
                      "loading_seconds": report["loading_seconds"],
                      "runs": [{k: r[k] for k in ["query", "repetition", "count", "retrieval_seconds"]} for r in report["runs"]]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--target")
    parser.add_argument("--remote-run", action="store_true")
    args = parser.parse_args()
    if args.remote_run:
        summary = extract_images(args.zip, args.output_dir / "images")
        print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)
        run_cpu(args.artifact_root, args.output_dir / "images", args.output_dir / "results")
    else:
        if not args.target:
            parser.error("--target is required for upload")
        orchestrate(args.zip, args.output_dir, args.target, args.artifact_root)


if __name__ == "__main__":
    main()
