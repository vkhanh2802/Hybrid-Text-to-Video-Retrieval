"""Narrow English caption evidence rules, NOT image/sequence verification.

Only explicit adjacent garment attributes are compared. Missing details and
ambiguous subject binding abstain. A contradiction may remove caption-only
support, never independent visual support or a raw similarity score.
"""

import re

import numpy as np


RULE_VERSION = "garment_attributes_v2"
COLORS = {name: name for name in (
    "black", "white", "gray", "blue", "red", "pink", "brown", "yellow",
    "green", "orange", "purple", "beige")}
COLORS["grey"] = "gray"
PATTERNS = {"plain": "plain", "solid": "plain", "checkered": "checkered",
            "checked": "checkered", "chequered": "checkered", "grid-patterned": "checkered",
            "striped": "striped", "dotted": "dotted", "polka-dot": "dotted"}
GARMENTS = {"shirt": "shirt", "tie": "tie", "necktie": "tie", "dress": "dress",
            "jacket": "jacket", "coat": "coat"}
_GARMENT = re.compile(r"\b(" + "|".join(GARMENTS) + r")\b")
_MODIFIERS = set(COLORS) | set(PATTERNS) | {
    "light", "dark", "pale", "bright", "short-sleeved", "long-sleeved", "button-up", "and"}
_AMBIGUOUS = re.compile(
    r"\b(?:no|not|without|neither|nor|maybe|possibly|perhaps|appears|seems|or|"
    r"two|three|several|multiple|another|other|people|men|women|children|presenters|workers|"
    r"guest|colleague|crowd|group|screen|poster|photograph|mannequin|"
    r"multicolored|multicoloured|colorful|holding|carrying)\b")


def _ambiguous(text):
    # Different person classes or explicit plurality: no cross-person binding.
    classes = [bool(re.search(pattern, text)) for pattern in (
        r"\b(?:man|male)\b", r"\b(?:woman|female)\b", r"\b(?:child|boy|girl)\b")]
    person_mentions = re.findall(r"\b(?:man|woman|child|boy|girl|person|presenter|anchor|host|worker|doctor)\b", text)
    mentions = list(_GARMENT.finditer(text))
    # Later OCR/news commentary must not erase an earlier explicit attribute.
    # Keep all preceding context and the complete last garment-bearing sentence;
    # this still blocks "a screen shows ... a man in a white shirt".
    scope = text
    if mentions:
        boundary = re.search(r"[.!?;](?:\s|$)", text[mentions[-1].end():])
        if boundary:
            scope = text[:mentions[-1].end() + boundary.end()]
    global_blocker = re.search(
        r"\b(?:no|not|without|neither|nor|two|three|several|multiple|another|other|"
        r"people|men|women|children|presenters|workers|guest|colleague|crowd|group)\b", text)
    return bool(_AMBIGUOUS.search(scope) or global_blocker) or sum(classes) > 1 or len(person_mentions) > 1


def _attributes(text):
    """Return only single, adjacent attributes of uniquely mentioned garments."""
    mentions = list(_GARMENT.finditer(text))
    output = {}
    for match in mentions:
        garment = GARMENTS[match[0]]
        if sum(GARMENTS[item[0]] == garment for item in mentions) != 1:
            continue
        # Stop at punctuation/nouns/verbs. Do not borrow e.g. a blue screen's color.
        prefix = re.findall(r"[a-z]+(?:-[a-z]+)*|[^\w\s]", text[:match.start()])
        modifiers = []
        for token in reversed(prefix):
            if token not in _MODIFIERS:
                break
            modifiers.append(token)
        # A slash/compound may hide a second color or a negated attribute.
        if prefix and ("/" in prefix[-5:] or prefix[-1].endswith("-colored")):
            continue
        # Post-nominal colors/patterns ("shirt with blue stripes") cannot be
        # safely assigned to the garment's main color by these prefix rules.
        tail = re.split(r"[.,;]", text[match.end():], maxsplit=1)[0]
        if re.match(r"\s+(?:with|featuring)\b", tail) and not _GARMENT.search(tail):
            continue
        colors = {COLORS[token] for token in modifiers if token in COLORS}
        patterns = {PATTERNS[token] for token in modifiers if token in PATTERNS}
        if len(colors) == 1:
            output[(garment, "color")] = next(iter(colors))
        if len(patterns) == 1:
            output[(garment, "pattern")] = next(iter(patterns))
    return output


def check_conditions(action, caption):
    """Three-way evidence over a small subset of conditions, not event truth."""
    action = action.lower() if isinstance(action, str) else ""
    caption = caption.lower() if isinstance(caption, str) else ""
    result = {"rule_version": RULE_VERSION, "source": "generated_caption",
              "pixel_verified": False, "status": "unknown", "conditions": []}
    required = _attributes(action)
    if not required:
        return dict(result, reason="no_supported_condition")
    if _ambiguous(action):
        return dict(result, reason="ambiguous_query_binding_or_language")
    if not caption.strip():
        return dict(result, reason="missing_caption")
    if _ambiguous(caption):
        return dict(result, reason="ambiguous_caption_binding_or_language")
    observed = _attributes(caption)
    for (garment, attribute), expected in sorted(required.items()):
        actual = observed.get((garment, attribute))
        status = "unknown" if actual is None else "consistent" if actual == expected else "contradicted"
        result["conditions"].append({"garment": garment, "attribute": attribute,
                                     "expected": expected, "observed": actual, "status": status})
    statuses = {item["status"] for item in result["conditions"]}
    result["status"] = ("contradicted" if "contradicted" in statuses else
                        "consistent" if statuses == {"consistent"} else "unknown")
    result["reason"] = "explicit_attributes_only"
    return result


def caption_for_frame(captions, frame):
    getter = getattr(captions, "get_caption", None)
    metadata = getter(frame["video_id"], frame["frame_id"]) if getter else None
    if not isinstance(metadata, dict):
        return None
    if (metadata.get("keyframe_n") is not None and frame.get("keyframe_n") is not None
            and int(metadata["keyframe_n"]) != int(frame["keyframe_n"])):
        raise ValueError("Condition caption/local mapping identity mismatch")
    # Use the original caption, not concatenated retrieval keywords/OCR.
    return metadata.get("caption")


def guarded_support(supported, visual, contradictions, visual_threshold):
    supported = np.asarray(supported, dtype=bool)
    contradictions = np.asarray(contradictions, dtype=bool)
    if contradictions.shape != supported.shape:
        raise ValueError("Condition evidence dimensions do not match support")
    visual_support = np.zeros(supported.shape, dtype=bool)
    if visual is not None:
        values = np.asarray(visual, dtype=float)
        if values.shape != supported.shape:
            raise ValueError("Visual evidence dimensions do not match support")
        visual_support = np.isfinite(values) & (values >= visual_threshold)
    return supported & (~contradictions | visual_support)


def apply_condition_evidence(actions, frames, captions, supported, visual, config):
    """Shared runtime/replay path; off is a true no-op with no caption lookup."""
    mode = config.condition_evidence
    stats = {"mode": mode, "rule_version": RULE_VERSION, "source": "generated_caption",
             "pixel_verified": False, "support_removed": 0}
    if mode == "off":
        return supported, stats
    contradictions = np.zeros((len(actions), len(frames)), dtype=bool)
    counts = {status: 0 for status in ("consistent", "contradicted", "unknown")}
    texts = [caption_for_frame(captions, frame) for frame in frames]
    for event, action in enumerate(actions):
        for index, text in enumerate(texts):
            status = check_conditions(action, text)["status"]
            counts[status] += 1
            contradictions[event, index] = status == "contradicted"
    guarded = guarded_support(supported, visual, contradictions, config.min_visual_similarity)
    stats.update(counts, would_remove_caption_only=int(np.count_nonzero(supported & ~guarded)))
    if mode == "semantic_guard":
        stats["support_removed"] = stats["would_remove_caption_only"]
        return guarded, stats
    return supported, stats
