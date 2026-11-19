import math
import sqlite3
from copy import deepcopy
from contextlib import nullcontext, ExitStack
import sys
from time import perf_counter

from kis.objects import apply_object_filter
from kis.query_plan import build_event_plan
from kis.temporal import TemporalConfig
from kis.result_pool import complete_frame_pool, unique_frames, require_frame_count
from kis.paths import is_keyframe_image, resolve_stored_image_path, video_image_dir
from kis.ranking import (
    RetrievalPolicy, queries_for_stream, weighted_rrf, fuse_query_variants,
    finalize_vlm_ranking, static_weight_fusion_3_way,
)


def console_safe_text(value, encoding=None):
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def multi_query_search(query_variants, encoder, retriever, top_k=50, vector_cache=None):
    """Tìm kiếm vector FAISS (Visual/CLIP)"""
    print(f"[Visual Search] {len(query_variants)} query variants")
    all_results = []
    for query in query_variants:
        if vector_cache is None:
            vector = encoder.encode_text(query)
        else:
            if query not in vector_cache:
                vector_cache[query] = encoder.encode_text(query)
            vector = vector_cache[query]
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
    base_score_field="score",
    ocr_query=None,
):
    from kis.ocr_routing import select_prompt_ocr
    prompt_query = ocr_query if ocr_query is not None else raw_query
    top_k = min(top_k, 20)
    if vlm_reranker is None or top_k <= 0:
        ranked, _ = finalize_vlm_ranking(candidates, top_k=0, vlm_weight=vlm_weight,
                                       base_score_field=base_score_field)
        return ranked

    print(f"[VLM] Reranking top {min(top_k, len(candidates))} candidates")
    for candidate in candidates:
        candidate["vlm_status"] = "unscored"

    def assign_score(candidate, score, error=None):
        candidate["vlm_raw_score"] = score
        if error:
            candidate.update(vlm_raw_score=None, vlm_error=error, vlm_status="error")
        elif not isinstance(score, (int, float)) or not math.isfinite(score):
            candidate.update(vlm_raw_score=None, vlm_error="invalid_score", vlm_status="invalid_score")
        else:
            candidate["vlm_status"] = "reranked"

    use_batch = getattr(vlm_reranker, "supports_batch", False) is True
    pending = []
    with ExitStack() as stack:
        for candidate in candidates[:top_k]:
            if not use_batch:
                continue
            candidate.pop("vlm_error", None)
            try:
                context = (image_source.open_candidate(candidate, object_lookup) if image_source is not None
                           else nullcontext(_resolve_candidate_image_path(candidate, object_lookup)))
                path = stack.enter_context(context)
                if path is None:
                    raise FileNotFoundError("image_not_found")
                ocr = select_prompt_ocr(candidate.get("ocr"), prompt_query)
                candidate['vlm_ocr_used'] = ocr is not None
                if ocr is not None and getattr(vlm_reranker, "supports_ocr", False) is not True:
                    raise ValueError("VLM scorer does not support separate OCR evidence")
                pending.append((candidate, path, ocr))
            except FileNotFoundError:
                candidate.update(vlm_raw_score=None, vlm_error="image_not_found", vlm_status="missing_image")
            except Exception as exc:
                assign_score(candidate, None, str(exc))
        if pending:
            try:
                kwargs = {"ocrs": [row[2] for row in pending]} if any(row[2] is not None for row in pending) else {}
                values = vlm_reranker.score_batch([row[1] for row in pending], raw_query,
                    [row[0].get("retrieval_text", "") for row in pending], **kwargs)
                if not isinstance(values, (list, tuple)) or len(values) != len(pending):
                    raise RuntimeError("VLM batch returned a different number of scores")
                errors = getattr(vlm_reranker, "last_batch_errors", [])
                for i, ((candidate, _, _), score) in enumerate(zip(pending, values)):
                    error = errors[i] if isinstance(errors, list) and len(errors) == len(pending) else None
                    assign_score(candidate, score, error)
            except Exception as exc:
                for candidate, _, _ in pending:
                    assign_score(candidate, None, str(exc))

    for candidate in ([] if use_batch else candidates[:top_k]):
        candidate.pop("vlm_error", None)
        try:
            image_context = (image_source.open_candidate(candidate, object_lookup) if image_source is not None
                             else nullcontext(_resolve_candidate_image_path(candidate, object_lookup)))
            with image_context as image_path:
                if image_path is None:
                    raise FileNotFoundError("image_not_found")
                kwargs = {"caption": candidate.get("retrieval_text", "")}
                ocr = select_prompt_ocr(candidate.get("ocr"), prompt_query)
                candidate['vlm_ocr_used'] = ocr is not None
                if ocr is not None:
                    if getattr(vlm_reranker, "supports_ocr", False) is not True:
                        raise ValueError("VLM scorer does not support separate OCR evidence")
                    kwargs["ocr"] = ocr
                vlm_score = vlm_reranker.score(image_path, raw_query, **kwargs)
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

        assign_score(candidate, vlm_score)

    ranked, vlm_applied = finalize_vlm_ranking(
        candidates,
        top_k=top_k,
        vlm_weight=vlm_weight,
        base_score_field=base_score_field,
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
    temporal_config=None,
    event_scorer=None,
    text_reranker_config=None,
    text_reranker=None,
    ocr_config=None,
    ocr_retriever=None,
    temporal_as_auxiliary=False,
    require_full_results=False,
    ocr_future=None,
):
    from kis.text_reranker import TextRerankerConfig, rerank_text_candidates
    from kis.ocr import OCRConfig, contains_literal, fuse_ocr, search_ocr

    if require_full_results:
        temporal_as_auxiliary = True
    text_config = text_reranker_config or TextRerankerConfig()
    ocr_config = ocr_config or OCRConfig()
    ocr_hits, ocr_diagnostics = ([], {}) if ocr_future is not None else search_ocr(structured_query, ocr_retriever, ocr_config)
    public_ocr = (ocr_config.mode != "off" and
                  getattr(ocr_retriever, "supports_candidate_ocr", False) is True)
    if public_ocr:
        ocr_diagnostics.update(ocr_retriever.status())
    policy = policy or RetrievalPolicy()
    temporal_config = temporal_config or TemporalConfig()
    plan = build_event_plan(structured_query, max_events=temporal_config.max_events)
    # Temporal OCR seeds are needed before event retrieval; ordinary frame
    # retrieval can overlap OCR CPU I/O/FTS with visual/BM25/semantic work.
    if ocr_future is not None and temporal_config.enabled and plan.active:
        ocr_hits, resolved = ocr_future.result()
        ocr_diagnostics.update(resolved)
        ocr_future = None
    temporal_diagnostics = {"status": "inactive", "reason": plan.reason}
    temporal_seconds = None
    if diagnostics is not None:
        diagnostics["temporal"] = temporal_diagnostics
        diagnostics["ocr"] = ocr_diagnostics
    if temporal_config.enabled and plan.active:
        from kis.event_retrieval import retrieve_event_segments

        event_ocr = {i: [hit for hit in ocr_hits if contains_literal(action, hit["ocr"]["matched_query"])]
                     for i, action in enumerate(plan.actions)} if ocr_config.mode == "retrieve" and ocr_config.weight > 0 else {}
        ocr_diagnostics["temporal_assigned_seed_hits"] = sum(len(hits) for hits in event_ocr.values())
        ocr_diagnostics["temporal_support"] = "candidate_seeds_only_no_gate_change"
        segments = retrieve_event_segments(
            plan, visual_retriever, visual_encoder, text_bm25_retriever,
            text_semantic_retriever, temporal_config, temporal_diagnostics, scorer=event_scorer,
            ocr_hits=event_ocr, ocr_weight=ocr_config.weight,
        )
        if segments:
            segments, text_diagnostics = rerank_text_candidates(
                segments, structured_query, text_semantic_retriever, text_config, text_reranker)
            for segment in segments:
                segment["final_score"] = segment.get("text_ranking_score", segment["score"])
            # A single-image Qwen score cannot verify an ordered sequence.
            # Keep temporal ordering; clip verification is a separate future stage.
            if temporal_as_auxiliary:
                temporal_diagnostics["presentation"] = "auxiliary"
                if diagnostics is not None:
                    diagnostics["temporal_results"] = segments
                    diagnostics["temporal_text_reranker"] = text_diagnostics
            elif diagnostics is not None:
                diagnostics["stage_seconds"] = {"temporal": temporal_diagnostics["seconds"]}
                diagnostics["stage_seconds"]["text_reranker"] = text_diagnostics["seconds"]
                diagnostics["stage_seconds"]["ocr"] = ocr_diagnostics["seconds"]
                diagnostics["text_reranker"] = text_diagnostics
            if not temporal_as_auxiliary:
                return segments[:result_top_k]
        else:
            temporal_diagnostics["reason"] = "no_supported_temporal_path"
        temporal_seconds = temporal_diagnostics.get("seconds")
    elif not temporal_config.enabled:
        temporal_diagnostics.update(status="disabled", reason="disabled_by_config")
    stage_started = perf_counter()
    # 1. Bóc tách dữ liệu
    raw_query = structured_query.get("raw_query", "")
    query_variants = structured_query.get("query_variants") or [raw_query]
    entities = structured_query.get("entities", [])
    vlm_query = raw_query or query_variants[0]

    # 2. Luồng 1: Visual Search (SigLIP/CLIP)
    visual_vectors = {}
    visual_multi = multi_query_search(
        queries_for_stream(structured_query, "visual", policy),
        visual_encoder,
        visual_retriever,
        top_k=retrieval_top_k,
        vector_cache=visual_vectors,
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
    if ocr_future is not None:
        ocr_hits, resolved = ocr_future.result()
        ocr_diagnostics.update(resolved)
        ocr_diagnostics['execution'] = 'overlapped_with_frame_retrieval'
        ocr_diagnostics['join_wait_seconds'] = perf_counter() - stage_started
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
    base_count = len(candidates)
    pre_public_candidates = deepcopy(candidates) if public_ocr and ocr_config.mode == "retrieve" else None
    bounded_ocr = getattr(ocr_retriever, "rank_fusion", None) == "bounded_positive_auxiliary"
    candidates = fuse_ocr(candidates, ocr_hits, ocr_config, bounded_positive=bounded_ocr)
    ocr_diagnostics["candidate_union_added"] = len(candidates) - base_count
    if require_full_results:
        def fetch_expanded(depth):
            nonlocal pre_public_candidates
            expanded_visual = fuse_query_variants(multi_query_search(
                queries_for_stream(structured_query, "visual", policy), visual_encoder,
                visual_retriever, top_k=depth, vector_cache=visual_vectors), policy)
            expanded_bm25 = fuse_query_variants([
                text_bm25_retriever.search_bm25(q, top_k=depth)
                for q in queries_for_stream(structured_query, "bm25", policy)], policy)
            expanded_semantic = fuse_query_variants([
                text_semantic_retriever.search(q, top_k=depth)
                for q in queries_for_stream(structured_query, "semantic", policy)], policy)
            if policy.cross_fusion == "rrf":
                expanded = weighted_rrf({"visual": expanded_visual, "bm25": expanded_bm25,
                                          "semantic": expanded_semantic}, k=policy.rrf_k)
            else:
                expanded = static_weight_fusion_3_way(expanded_visual, expanded_bm25, expanded_semantic)
            if public_ocr and ocr_config.mode == "retrieve":
                pre_public_candidates = deepcopy(expanded)
            combined = fuse_ocr(expanded, ocr_hits, ocr_config, bounded_positive=bounded_ocr)
            ocr_diagnostics["candidate_union_added"] = len(combined) - len(expanded)
            return combined

        candidates, pool_info = complete_frame_pool(candidates, fetch_expanded,
            target=result_top_k, initial_k=retrieval_top_k)
        if diagnostics is not None:
            diagnostics["result_pool"] = pool_info
    if diagnostics is not None:
        diagnostics["cross_fusion"] = policy.cross_fusion
        if ocr_config.mode == "retrieve" and ocr_hits and ocr_config.weight > 0:
            diagnostics["cross_fusion"] += "+ocr_rank"

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

    # Public lookup is independent of literal FTS hits: every potential Qwen
    # candidate gets separate OCR, without changing its original caption.
    if public_ocr and ocr_config.mode == "retrieve":
        attach_started = perf_counter()
        try:
            ocr_retriever.attach_candidates(candidates, ocr_config)
            ocr_diagnostics["candidate_evidence_status"] = "attached"
        except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error) as exc:
            # Invalid joins or an asset change after FTS must not leave a
            # partially applied public OCR boost/union in the retrieval ranking.
            candidates = pre_public_candidates
            ocr_diagnostics.update(status="error", error=str(exc), candidate_union_added=0)
            ocr_diagnostics["candidate_evidence_status"] = "error"
            if diagnostics is not None:
                diagnostics["cross_fusion"] = policy.cross_fusion
            for candidate in candidates:
                metadata = metadata_lookup(candidate["video_id"], candidate["frame_id"]) if metadata_lookup else None
                if metadata:
                    for field in ("retrieval_text", "caption", "keyframe_n", "image_path", "image_sha256"):
                        if field in metadata:
                            candidate[field] = metadata[field]
                candidate["ocr"] = ocr_retriever._evidence("error")
        ocr_diagnostics["attach_seconds"] = perf_counter() - attach_started
        ocr_diagnostics["seconds"] += ocr_diagnostics["attach_seconds"]
        ocr_diagnostics["fusion_policy"] = "bounded_positive_auxiliary" if bounded_ocr else "legacy_ocr_rank"

    # 6. Bổ sung object evidence
    candidates = apply_object_filter(candidates, entities, object_lookup)
    if diagnostics is not None:
        diagnostics["pre_selection_candidates"] = [dict(item) for item in candidates]
    fusion_seconds = max(0., perf_counter() - stage_started - ocr_diagnostics.get("attach_seconds", 0.))
    candidates, text_diagnostics = rerank_text_candidates(
        candidates, structured_query, text_semantic_retriever, text_config, text_reranker)
    if diagnostics is not None:
        diagnostics["text_reranker"] = text_diagnostics
    stage_started = perf_counter()
    if candidate_selector is not None:
        candidates = candidate_selector.select(candidates)
    if require_full_results:
        candidates = unique_frames(candidates)
        if len(candidates) < result_top_k:
            require_frame_count(candidates, result_top_k)
    if diagnostics is not None:
        diagnostics["rerank_input"] = [dict(item) for item in candidates[:rerank_top_k]]
    fusion_seconds += perf_counter() - stage_started
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
        base_score_field="text_ranking_score" if text_config.mode == "rerank" else "score",
        ocr_query=structured_query,
    )
    if diagnostics is not None:
        stage_seconds = {
            "visual": visual_seconds, "bm25": bm25_seconds,
            "semantic": semantic_seconds, "fusion_and_selection": fusion_seconds,
            "text_reranker": text_diagnostics["seconds"],
            "ocr": ocr_diagnostics["seconds"],
            "vlm": perf_counter() - stage_started,
        }
        if temporal_seconds is not None:
            # Event retrieval already ran before the fallback; keep its timing
            # instead of replacing it with baseline-only stage timings.
            stage_seconds["temporal"] = temporal_seconds
        diagnostics["stage_seconds"] = stage_seconds
        batch_stats = getattr(vlm_reranker, "last_batch_stats", {})
        diagnostics["vlm_batch"] = batch_stats if isinstance(batch_stats, dict) else {}

    results = candidates[:result_top_k]
    if require_full_results:
        require_frame_count(results, result_top_k)
    print(f"[KIS] Complete. Returning {len(results)} results.")
    return results
