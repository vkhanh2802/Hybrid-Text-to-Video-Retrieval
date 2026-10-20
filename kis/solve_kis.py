import math
from contextlib import nullcontext
import sys
from time import perf_counter

from kis.objects import apply_object_filter
from kis.paths import is_keyframe_image, resolve_stored_image_path, video_image_dir
from kis.ranking import (
    RetrievalPolicy, queries_for_stream, weighted_rrf, fuse_query_variants,
    finalize_vlm_ranking, static_weight_fusion_3_way,
)


def console_safe_text(value, encoding=None):
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def multi_query_search(query_variants, encoder, retriever, top_k=50):
    """Tìm kiếm vector FAISS (Visual/CLIP)"""
    print(f"[Visual Search] {len(query_variants)} query variants")
    all_results = []
    for query in query_variants:
        vector = encoder.encode_text(query)
        results = retriever.search(vector, top_k=top_k)
        all_results.append(results)
    return all_results

def _resolve_candidate_image_path(candidate, object_lookup):
    stored_image_path = candidate.get("image_path")
    if stored_image_path:
        image_path = resolve_stored_image_path(stored_image_path)
        if is_keyframe_image(image_path, candidate.get("keyframe_n")) and image_path.is_file():
            candidate.setdefault("keyframe_n", int(image_path.stem))
            return image_path

    video_id = candidate["video_id"]
    frame_id = candidate["frame_id"]
    keyframe_n = candidate.get("keyframe_n")
    if keyframe_n is None:
        keyframe_n = object_lookup.get_keyframe_n(video_id, frame_id)

    image_dir = video_image_dir(video_id)
    if keyframe_n is None or not image_dir.is_dir():
        return None

    for image_path in image_dir.iterdir():
        if is_keyframe_image(image_path, keyframe_n) and image_path.is_file():
            candidate["keyframe_n"] = int(keyframe_n)
            candidate["image_path"] = str(image_path)
            return image_path

    return None


def rerank_candidates_with_vlm(
    candidates,
    raw_query,
    vlm_reranker,
    object_lookup,
    top_k=20,
    vlm_weight=0.5,
    verified_first=False,
    image_source=None,
):
    if vlm_reranker is None or top_k <= 0:
        ranked, _ = finalize_vlm_ranking(candidates, top_k=0, vlm_weight=vlm_weight)
        return ranked

    print(f"[VLM] Reranking top {min(top_k, len(candidates))} candidates")
    for candidate in candidates:
        candidate["vlm_status"] = "unscored"
    for candidate in candidates[:top_k]:
        candidate.pop("vlm_error", None)
        try:
            image_context = (image_source.open_candidate(candidate, object_lookup) if image_source is not None
                             else nullcontext(_resolve_candidate_image_path(candidate, object_lookup)))
            with image_context as image_path:
                if image_path is None:
                    raise FileNotFoundError("image_not_found")
                vlm_score = vlm_reranker.score(image_path, raw_query, caption=candidate.get("retrieval_text", ""))
        except FileNotFoundError:
            candidate["vlm_raw_score"] = None
            candidate["vlm_error"] = "image_not_found"
            candidate["vlm_status"] = "missing_image"
            continue
        except Exception as exc:
            candidate["vlm_raw_score"] = None
            candidate["vlm_error"] = str(exc)
            candidate["vlm_status"] = "error"
            continue

        candidate["vlm_raw_score"] = vlm_score
        if not isinstance(vlm_score, (int, float)) or not math.isfinite(vlm_score):
            candidate["vlm_raw_score"] = None
            candidate["vlm_error"] = "invalid_score"
            candidate["vlm_status"] = "invalid_score"
        else:
            candidate["vlm_status"] = "reranked"

    ranked, vlm_applied = finalize_vlm_ranking(
        candidates,
        top_k=top_k,
        vlm_weight=vlm_weight,
    )
    if not vlm_applied:
        print("[VLM] No valid VLM scores; keeping retrieval ranking")
    elif verified_first:
        ranked = [item for item in ranked if item.get("vlm_status") == "reranked"] + [
            item for item in ranked if item.get("vlm_status") != "reranked"
        ]
    return ranked


def solve_kis(
    structured_query,
    visual_retriever,
    visual_encoder,
    text_bm25_retriever,
    text_semantic_retriever,
    object_lookup,
    vlm_reranker=None,
    retrieval_top_k=100,
    rerank_top_k=20,
    result_top_k=100,
    vlm_weight=0.5,
    policy=None,
    candidate_selector=None,
    diagnostics=None,
    image_source=None,
):
    policy = policy or RetrievalPolicy()
    stage_started = perf_counter()
    # 1. Bóc tách dữ liệu
    raw_query = structured_query.get("raw_query", "")
    query_variants = structured_query.get("query_variants") or [raw_query]
    entities = structured_query.get("entities", [])
    vlm_query = raw_query or query_variants[0]

    # 2. Luồng 1: Visual Search (SigLIP/CLIP)
    visual_multi = multi_query_search(
        queries_for_stream(structured_query, "visual", policy),
        visual_encoder,
        visual_retriever,
        top_k=retrieval_top_k,
    )
    visual_results = fuse_query_variants(visual_multi, policy)
    visual_seconds = perf_counter() - stage_started
    stage_started = perf_counter()

    # 3. Luồng 2: Text Search (BM25 - Khớp từ khóa cứng)
    bm25_multi = []
    for q in queries_for_stream(structured_query, "bm25", policy):
        print(f"[BM25 Search] Query: {console_safe_text(q)}")
        bm25_multi.append(
            text_bm25_retriever.search_bm25(q, top_k=retrieval_top_k)
        )
    bm25_results = fuse_query_variants(bm25_multi, policy)
    bm25_seconds = perf_counter() - stage_started
    stage_started = perf_counter()

    # 4. Luồng 3: Semantic Text Search (Tìm kiếm ngữ nghĩa Vector)
    semantic_multi = []
    for q in queries_for_stream(structured_query, "semantic", policy):
        print(f"[Semantic Search] Query: {console_safe_text(q)}")
        semantic_multi.append(
            text_semantic_retriever.search(q, top_k=retrieval_top_k)
        )
    semantic_results = fuse_query_variants(semantic_multi, policy)
    semantic_seconds = perf_counter() - stage_started
    stage_started = perf_counter()
    if diagnostics is not None:
        diagnostics["stream_queries"] = {
            stream: queries_for_stream(structured_query, stream, policy)
            for stream in ("visual", "bm25", "semantic")
        }
        diagnostics["streams"] = {
            "visual": visual_multi, "bm25": bm25_multi, "semantic": semantic_multi,
        }

    # 5. Dung hợp 3 luồng
    if policy.cross_fusion == "rrf":
        candidates = weighted_rrf({"visual": visual_results, "bm25": bm25_results, "semantic": semantic_results}, k=policy.rrf_k)
    else:
        candidates = static_weight_fusion_3_way(visual_results, bm25_results, semantic_results)
    if diagnostics is not None:
        diagnostics["cross_fusion"] = policy.cross_fusion

    # Hydrate visual-only hits from the SAME caption corpus as text retrieval.
    # The same original InternVL caption is used for retrieval and VLM evidence.
    metadata_lookup = getattr(text_semantic_retriever, "get_caption", None)
    if metadata_lookup is not None:
        for candidate in candidates:
            metadata = metadata_lookup(candidate["video_id"], candidate["frame_id"])
            if metadata:
                for field in ("retrieval_text", "caption", "keyframe_n", "image_path", "image_sha256"):
                    if field in metadata:
                        candidate[field] = metadata[field]

    # 6. Bổ sung object evidence
    candidates = apply_object_filter(candidates, entities, object_lookup)
    if diagnostics is not None:
        diagnostics["pre_selection_candidates"] = [dict(item) for item in candidates]
    if candidate_selector is not None:
        candidates = candidate_selector.select(candidates)
    if diagnostics is not None:
        diagnostics["rerank_input"] = [dict(item) for item in candidates[:rerank_top_k]]
    fusion_seconds = perf_counter() - stage_started
    stage_started = perf_counter()

    # 7. VLM reranking và xếp hạng cuối
    candidates = rerank_candidates_with_vlm(
        candidates,
        vlm_query,
        vlm_reranker,
        object_lookup,
        top_k=rerank_top_k,
        vlm_weight=vlm_weight,
        verified_first=policy.verified_first,
        image_source=image_source,
    )
    if diagnostics is not None:
        diagnostics["stage_seconds"] = {
            "visual": visual_seconds, "bm25": bm25_seconds,
            "semantic": semantic_seconds, "fusion_and_selection": fusion_seconds,
            "vlm": perf_counter() - stage_started,
        }

    results = candidates[:result_top_k]
    print(f"[KIS] Complete. Returning {len(results)} results.")
    return results
