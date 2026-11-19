"""Matched offline retrieval benchmark; never touches DRES or production services.

Uses the shared solver and pinned real models/artifacts. Visual retrieval is
exact NPY cosine on BOTH hosts, not the production Milvus adapter. Planning,
VLM, image download/rendering and video/temporal tasks are outside this test.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time


GROUPS = tuple(f"L{i:02d}" for i in range(1, 21) if i != 18)
QUERIES = (
    "A person cooking food in a kitchen.",
    "A person wearing a helmet riding a motorcycle on a street.",
    "A city skyline at sunset with buildings and water.",
    "Two people sitting on a boat on a river.",
    "A woman in a red shirt speaking into a microphone.",
    "A football match with players running on a green field.",
    "A farmer harvesting vegetables in a field.",
    "A crowded market with people buying fruit.",
    "A train arriving at a railway station.",
    "A person using a computer in an office.",
)


def summary(values):
    values = sorted(values)
    position = (len(values) - 1) * .95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[lower] + (values[upper] - values[lower]) * (position - lower),
            "min": values[0], "max": values[-1], "mean": statistics.mean(values)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark(dataset, output, label, repetitions=3, threads=4):
    dataset, output = Path(dataset).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(threads),
                      MKL_NUM_THREADS=str(threads), FAISS_OPT_LEVEL="generic",
                      KIS_VIDEO_GROUPS=",".join(GROUPS))
    started = time.perf_counter()
    # NumPy first avoids duplicate OpenMP initialization in the existing Windows
    # environment. Do NOT use KMP_DUPLICATE_LIB_OK as a benchmark workaround.
    import numpy as np
    import torch
    import faiss
    from kis.config import MODEL_MANIFEST
    from kis.paths import resolve_video_artifact
    from kis.text import CaptionRetriever, BM25Retriever
    from kis.visual import SiglipEncoder
    from kis.objects import ObjectMetadataLookup
    from kis.ranking import get_search_profile
    from kis.result_pool import require_frame_count
    from kis.solve_kis import solve_kis
    from kis.temporal import TemporalConfig

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(threads)
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "cpu_count": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
                "faiss": faiss.__version__, "cpu_threads": threads}
    hardware["packages"] = {name: importlib.metadata.version(name)
                            for name in ("transformers", "sentence-transformers", "open-clip-torch")}
    try:
        import psutil
        memory = psutil.virtual_memory()
        hardware.update(memory_total=memory.total, memory_available_at_start=memory.available)
    except ImportError:
        pass
    stages = {"imports": time.perf_counter() - started}
    tick = time.perf_counter()
    mapping_paths = sorted(p for group in GROUPS for p in
                           (dataset / "mapping" / group).rglob(f"{group}_V*.csv"))
    if {p.stem.split("_")[0] for p in mapping_paths} != set(GROUPS):
        raise ValueError("Mapping scope incomplete")
    specifications, records, ambiguous = [], [], set()
    for path in mapping_paths:
        with path.open(encoding="utf-8", newline="") as stream:
            entries = list(csv.DictReader(stream))
        count = Counter(int(row["frame_idx"]) for row in entries)
        keep = []
        for i, row in enumerate(entries):
            frame = int(row["frame_idx"])
            if count[frame] > 1:
                ambiguous.add((path.stem, frame))
                continue
            keep.append(i)
            records.append({"video_id": path.stem, "frame_id": frame, "keyframe_n": int(row["n"])})
        feature = resolve_video_artifact(dataset / "clip-features-32", path.stem, ".npy", video_groups=GROUPS)
        specifications.append((feature, len(entries), keep))
    matrix = np.empty((len(records), 768), dtype=np.float32)
    offset = 0
    for feature, count, keep in specifications:
        vectors = np.load(feature, allow_pickle=False)
        if vectors.shape != (count, 768) or not np.isfinite(vectors).all():
            raise ValueError("Invalid visual vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Zero stored embedding")
        matrix[offset:offset + len(keep)] = (vectors / norms)[keep]
        offset += len(keep)
    del vectors, norms
    stages["mapping_and_visual_index"] = time.perf_counter() - tick

    class ExactVisual:
        def search(self, vector, top_k=100):
            vector = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if vector.shape != (768,) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid query embedding")
            scores = matrix @ (vector / norm)
            positions = np.argsort(-scores, kind="stable")[:top_k]
            return [dict(records[int(i)], score=float(scores[i])) for i in positions]

    tick = time.perf_counter()
    captions = CaptionRetriever(video_groups=GROUPS, artifact_root=dataset)
    if ambiguous:
        # No ambiguous identities are present in the imported groups. Fail rather
        # than silently comparing different candidate universes on two hosts.
        raise ValueError("Ambiguous frame identities; explicit benchmark policy needed")
    stages["caption_indexes_and_minilm"] = time.perf_counter() - tick
    tick = time.perf_counter()
    encoder = SiglipEncoder(device="cpu")
    stages["siglip_load"] = time.perf_counter() - tick
    tick = time.perf_counter()
    bm25 = BM25Retriever(captions.caption_data, tokenizer_mode="unicode")
    lookup = ObjectMetadataLookup(mapping_dir=dataset / "mapping", objects_dir=dataset / "objects")
    stages["bm25_load"] = time.perf_counter() - tick
    visual = ExactVisual()

    # Hash all consumed corpus artifacts independently of canonical vs legacy
    # storage paths. This validation is outside the timed query loops.
    tick = time.perf_counter()
    artifacts = []
    for mapping, (feature, _, _) in zip(mapping_paths, specifications):
        artifacts.extend([(f"mapping/{mapping.stem}.csv", sha(mapping)),
                          (f"visual/{mapping.stem}.npy", sha(feature))])
    for component, paths in (("captions", captions.json_path), ("caption_mapping", captions.mapping_path),
                             ("indexes", captions.index_path), ("manifests", captions.manifest_path)):
        for path in paths:
            artifacts.append((f"{component}/{Path(path).name}", sha(path)))
    fingerprint = hashlib.sha256(json.dumps(sorted(artifacts), separators=(",", ":")).encode()).hexdigest()
    stages["corpus_hash_validation"] = time.perf_counter() - tick
    (output / "CORPUS_HASHES.json").write_text(json.dumps(sorted(artifacts)), encoding="utf-8")

    def search(query):
        structured = {"raw_query": query, "query_variants": [query], "entities": [],
                      "events": [], "temporal_constraints": [], "needs_ocr": False}
        diagnostics = {}
        begin = time.perf_counter()
        results = solve_kis(structured, visual, encoder, bm25, captions, lookup,
                            retrieval_top_k=100, result_top_k=100, rerank_top_k=0,
                            temporal_config=TemporalConfig(enabled=False),
                            policy=get_search_profile("rrf").policy, diagnostics=diagnostics,
                            require_full_results=True, temporal_as_auxiliary=True)
        elapsed = time.perf_counter() - begin
        require_frame_count(results)
        return {"seconds": elapsed, "stage_seconds": diagnostics["stage_seconds"],
                "count": len(results), "unique_count": len({(r["video_id"], r["frame_id"]) for r in results}),
                "frames": [[r["video_id"], r["frame_id"], r["keyframe_n"]] for r in results]}

    warmup = search(QUERIES[0])
    runs = []
    for repetition in range(repetitions):
        for query_index, query in enumerate(QUERIES):
            run = dict(query=query, query_index=query_index, repetition=repetition + 1, **search(query))
            runs.append(run)
            print(json.dumps({"label": label, "query": query_index + 1,
                              "repetition": repetition + 1, "seconds": run["seconds"]}), flush=True)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = list((source_root / "kis").glob("*.py"))
    source_paths = [p for p in source_paths if p.name != "local_images.py"]
    source_paths += [Path(__file__).resolve(), source_root / "config/models.json", source_root / "config/pipeline.json",
                     source_root / "config/object_labels.json"]
    source_digest = hashlib.sha256(json.dumps(sorted((p.relative_to(source_root).as_posix(), sha(p))
                                                    for p in source_paths), separators=(",", ":")).encode()).hexdigest()
    report = {"status": "passed", "label": label, "hardware": hardware, "groups": GROUPS,
              "videos": len(mapping_paths), "visual_vectors": len(records),
              "caption_vectors": captions.index.ntotal, "corpus_sha256": fingerprint,
              "source_sha256": source_digest,
              "models": MODEL_MANIFEST, "queries": QUERIES, "repetitions": repetitions,
              "loading_seconds": stages, "warmup": warmup, "runs": runs,
              "warm_seconds": summary([r["seconds"] for r in runs]),
              "warm_stage_seconds": {stage: summary([r["stage_seconds"][stage] for r in runs])
                                     for stage in runs[0]["stage_seconds"]},
              "total_seconds": time.perf_counter() - started,
              "limitations": ["retrieval only, not production end-to-end", "fixed English queries, one variant, no planner",
                              "exact NPY cosine on both hosts, not Milvus", "no VLM, video, temporal, OCR or image IO",
                              "empty entities; no object-evidence IO", "no remote RPC or DRES submission"]}
    try:
        import resource
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    (output / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "label": label, "warm_seconds": report["warm_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    benchmark(args.dataset, args.output, args.label)


if __name__ == "__main__":
    main()
