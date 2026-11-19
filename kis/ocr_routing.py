"""Query-grounded OCR routing and compact, optional Qwen evidence.

Only original user spans are searched. No translation, spelling repair, caption
mining, or dataset/ground-truth-specific terms. Missing evidence is unknown.
"""
import re


def raw_text_spans(raw):
    # Explicit text cues work even when the planner forgets a text entity.
    # Stop at a sentence boundary; ':' is allowed inside text (e.g. a headline).
    cue = r'(?:có ghi(?:\s+dòng chữ)?|ghi dòng chữ|dòng chữ|dòng chữ|chữ trên bảng|text reads|words? reading|sign reads)\s*[:：]?\s*'
    spans = []
    for match in re.finditer(cue, raw, flags=re.IGNORECASE):
        tail = re.split(r'[\n.!?]|\.{3}', raw[match.end():], maxsplit=1)[0].strip(' ,:;“”"')
        if not tail:
            continue
        # Paraphrases are not literal quotations. Extract only original proper
        # names/acronyms from that span; never search the entire paraphrase.
        if re.match(r'(?:liên quan|về việc|về nội dung|nói về)\b', tail, re.IGNORECASE):
            for word in re.findall(r'\b[^\W\d_][\w-]*\b', tail, re.UNICODE):
                if len(word) >= 3 and (word.isupper() or word[0].isupper()):
                    spans.append((word, 'raw_text_keyword'))
        elif len(tail) <= 256:
            spans.append((tail, 'raw_text_cue'))
    return spans


def fallback_fragments(text):
    """Bounded contiguous original phrases; a fallback, not fuzzy correction."""
    words = list(re.finditer(r'\S+', text))
    if len(words) < 6:
        return []
    fragments = []
    # Prefix/suffix are deterministic and remain exact original spans.
    for start in (0, len(words) - 4):
        fragment = text[words[start].start():words[start + 3].end()].strip(' ,:;.!?')
        if len(fragment) >= 8:
            fragments.append(fragment)
    return fragments


def select_prompt_ocr(ocr, query, max_chars=384):
    """Return a copied, relevant excerpt or None (the ordinary image prompt).

    Query may be structured (manual literals included) or the original string.
    Full evidence on the result is never changed by prompt selection.
    """
    from kis.ocr import OCRConfig, contains_literal, literal_queries
    if not isinstance(ocr, dict) or ocr.get('source_format') != 'public_json' or ocr.get('status') != 'ok':
        return None
    structured = query if isinstance(query, dict) else {'raw_query': query, 'needs_ocr': True}
    route = literal_queries(structured, OCRConfig())
    if not route:
        return None
    text = ' '.join(str(ocr.get('text', '')).split())
    needles = [r['text'] for r in route]
    needles += [f for r in route for f in fallback_fragments(r['text'])]
    for needle in needles:
        if not contains_literal(text, needle):
            continue
        # Select around a match rather than taking the beginning (often logos).
        index = text.casefold().find(needle.casefold())
        if index < 0:
            continue
        start = max(0, index - 48)
        excerpt = text[start:start + max_chars]
        return {**ocr, 'text': excerpt, 'prompt_excerpt': True}
    return None
