"""Optional CPU cross-encoder ranking, never pixel/event verification.

No model imports/downloads at module import or construction. Missing evidence,
failed batches and invalid scores retain their retrieval slots and scores.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from numbers import Real
import re
from threading import Lock
from time import perf_counter


@dataclass(frozen=True)
class TextRerankerConfig:
    mode: str = "off"
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    pool_size: int = 100
    batch_size: int = 16
    max_pairs: int = 1200
    chunk_chars: int = 1200
    max_chunks: int = 4
    max_length: int = 512
    query_chars: int = 1600

    def __post_init__(self):
        if self.mode not in {"off", "diagnostics", "rerank"}:
            raise ValueError("Unknown text reranker mode")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Text reranker model must be non-empty")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Text reranker requires a pinned model revision")
        for name, low, high in (("pool_size", 1, 300), ("batch_size", 1, 64),
                                ("max_pairs", 1, 4096), ("chunk_chars", 128, 4000),
                                ("max_chunks", 1, 8), ("max_length", 32, 512),
                                ("query_chars", 128, 8000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"text_reranker.{name} must be an integer in [{low}, {high}]")

    def public(self):
        return dict(asdict(self), device="cpu", local_files_only=True,
                    evidence_source="generated_caption", verified=False)


class LazyCrossEncoder:
    """Pinned safetensors CPU inference; never silently downloads a new model."""

    def __init__(self, config):
        self.config = config
        self.model = self.tokenizer = None
        self.load_error = None
        self._lock = Lock()

    def _load(self):
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        if self.model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            kwargs = dict(revision=self.config.revision, local_files_only=True, trust_remote_code=False)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model, use_safetensors=True, **kwargs)
            if model.config.num_labels != 1:
                raise ValueError("Text reranker needs one relevance logit, not a multi-class head")
            model = model.to(device="cpu").float().eval()
            self.tokenizer, self.model = tokenizer, model
        except Exception as exc:
            self.load_error = "text_model_unavailable: cache the pinned checkpoint locally before enabling"
            raise RuntimeError(self.load_error) from exc

    def score_pairs(self, pairs):
        if not pairs:
            return []
        if len(pairs) > self.config.batch_size:
            raise ValueError("Text inference batch exceeds configured bound")
        with self._lock:
            self._load()
            import torch

            inputs = self.tokenizer([q for q, _ in pairs], [text for _, text in pairs],
                                    padding=True, truncation="longest_first",
                                    max_length=self.config.max_length, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            if tuple(logits.shape) != (len(pairs), 1):
                raise ValueError("Expected one relevance logit per query/text pair")
            return logits[:, 0].float().cpu().tolist()


def text_chunks(text, config):
    """Sample over the whole caption, including the tail when budget >=2."""
    if not isinstance(text, str) or not text.strip():
        return [], 0
    text = text.strip()
    step = config.chunk_chars - min(150, config.chunk_chars // 4)
    starts = [0]
    while starts[-1] + config.chunk_chars < len(text):
        starts.append(starts[-1] + step)
    total = len(starts)
    if total > config.max_chunks:
        indices = ([0] if config.max_chunks == 1 else
                   [round(i * (total - 1) / (config.max_chunks - 1)) for i in range(config.max_chunks)])
        starts = [starts[i] for i in indices]
    return [text[start:start + config.chunk_chars] for start in starts], total - len(starts)


def ranking_query(query):
    """Use the planner's complete English anchor, not a newly invented rewrite."""
    routed = query.get("stream_queries", {}).get("visual")
    options = [(routed[0] if isinstance(routed, list) and routed else None, "stream_queries.visual[0]"),
               (query.get("visual_description"), "visual_description")]
    variants = query.get("query_variants")
    options += [(variants[0] if isinstance(variants, list) and variants else None, "query_variants[0]"),
                (query.get("raw_query"), "raw_query")]
    return next(((text.strip(), source) for text, source in options if isinstance(text, str) and text.strip()),
                ("", "missing"))


def _caption(candidate, captions):
    for field in ("retrieval_text", "caption"):
        text = candidate.get(field)
        if isinstance(text, str) and text.strip():
            return text
    lookup = getattr(captions, "get_caption", None)
    if lookup is not None:
        metadata = lookup(candidate["video_id"], candidate["frame_id"])
        if metadata:
            return metadata.get("retrieval_text") or metadata.get("caption")
    return None


def _evidence_units(candidate, anchor, captions):
    temporal = candidate.get("temporal")
    if temporal is None:
        return [(anchor, _caption(candidate, captions))], None
    # Text cannot fill missing events or establish that a complete sequence exists.
    events = temporal.get("events", [])
    if not events or any(event.get("status") != "retrieved" for event in events):
        return [], "partial_temporal_path"
    if any(not isinstance(event.get("action"), str) or not event["action"].strip() for event in events):
        return [], "missing_event_query"
    return [(event["action"], _caption(event, captions)) for event in events], None


def rerank_text_candidates(candidates, query, captions, config=None, scorer=None):
    """Scored-slot permutation with rank-ladder scores, not logit calibration.

    Only valid text-scored candidates exchange slots within the bounded pool
    and same temporal support count. Unscored candidates keep their exact slot
    and score. Original `score`, fusion evidence, event support and times never
    change. `text_ranking_score` copies the occupied slot's original score so
    downstream VLM can preserve this order without treating logits as probabilities.
    """
    config = config or TextRerankerConfig()
    started = perf_counter()
    diagnostics = {"mode": config.mode, "status": "disabled", "config": config.public(),
                   "verified": False, "scores_are_probabilities": False}
    if config.mode == "off":
        return candidates, dict(diagnostics, seconds=perf_counter() - started)
    values = [dict(candidate) for candidate in candidates]
    original_ranks = {id(candidate): index + 1 for index, candidate in enumerate(values)}
    anchor, source = ranking_query(query)
    diagnostics.update(status="fallback", query_source=source, query_truncated=len(anchor) > config.query_chars,
                       token_truncation="longest_first_bounded_by_max_length", pairs_requested=0,
                       pairs_valid=0, batch_errors=0, omitted_chunks=0)
    pairs, plans = [], {}
    for index, candidate in enumerate(values):
        candidate.pop("text_ranking_score", None)
        candidate.pop("text_logit", None)
        candidate["text_rerank_status"] = "outside_pool" if index >= config.pool_size else "missing_query"
        if index >= config.pool_size or not anchor:
            continue
        try:
            units, reason = _evidence_units(candidate, anchor, captions)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
            candidate["text_rerank_status"] = "evidence_error"
            continue
        if reason:
            candidate["text_rerank_status"] = reason
            continue
        if any(not isinstance(text, str) or not text.strip() for _, text in units):
            candidate["text_rerank_status"] = "missing_text"
            continue
        pending, groups = [], []
        for event_query, text in units:
            chunks, omitted = text_chunks(text, config)
            diagnostics["omitted_chunks"] += omitted
            if len(event_query) > config.query_chars:
                diagnostics["query_truncated"] = True
            groups.append(list(range(len(pairs) + len(pending), len(pairs) + len(pending) + len(chunks))))
            pending.extend((event_query[:config.query_chars], chunk) for chunk in chunks)
        if len(pairs) + len(pending) > config.max_pairs:
            candidate["text_rerank_status"] = "pair_budget_exhausted"
            continue
        pairs.extend(pending)
        plans[index] = groups
    scorer = scorer or (LazyCrossEncoder(config) if pairs else None)
    scores, errors = [None] * len(pairs), {}
    for start in range(0, len(pairs), config.batch_size):
        batch = pairs[start:start + config.batch_size]
        try:
            batch_scores = list(scorer.score_pairs(batch))
            if len(batch_scores) != len(batch):
                raise ValueError("Text scorer output count mismatch")
            for offset, value in enumerate(batch_scores):
                if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
                    scores[start + offset] = float(value)
        except Exception as exc:
            diagnostics["batch_errors"] += 1
            # No traceback/raw evidence exposed through API; distinguish batch
            # failure from invalid numeric score without inventing negatives.
            errors.update({i: type(exc).__name__ for i in range(start, start + len(batch))})
    for index, groups in plans.items():
        indices = [i for group in groups for i in group]
        if any(scores[i] is None for i in indices):
            values[index]["text_rerank_status"] = "inference_error" if any(i in errors for i in indices) else "invalid_score"
            continue
        # Best chunk per event, weakest event across a complete path. This is
        # ranking on generated text, NOT calibrated event/sequence verification.
        values[index]["text_logit"] = min(max(scores[i] for i in group) for group in groups)
        values[index]["text_rerank_status"] = "scored"
    groups = defaultdict(list)
    for index, candidate in enumerate(values[:config.pool_size]):
        if candidate["text_rerank_status"] == "scored":
            temporal = candidate.get("temporal")
            key = len(temporal["events"]) if temporal is not None else None
            groups[key].append(index)
    if config.mode == "rerank":
        for slots in groups.values():
            ranked = sorted((values[i] for i in slots), key=lambda item: -item["text_logit"])
            for slot, candidate in zip(slots, ranked):
                candidate["text_ranking_score"] = float(candidates[slot]["score"])
                values[slot] = candidate
    diagnostics.update(status="applied" if groups and config.mode == "rerank" else "diagnostics" if groups else "fallback",
        pairs_requested=len(pairs), pairs_valid=sum(value is not None for value in scores),
        candidate_status_counts=dict(Counter(item["text_rerank_status"] for item in values)),
        candidates_scored=sum(item["text_rerank_status"] == "scored" for item in values),
        candidate_trace=[{"video_id": item["video_id"], "frame_id": item["frame_id"],
                          "retrieval_rank": original_ranks[id(item)], "text_rank": rank,
                          "status": item["text_rerank_status"], "logit": item.get("text_logit"),
                          "retrieval_score": item["score"], "ranking_score": item.get("text_ranking_score", item["score"])}
                         for rank, item in enumerate(values[:config.pool_size], 1)],
        ranking_policy="scored_slots_same_support_rank_ladder", seconds=perf_counter() - started)
    return values, diagnostics
