"""Prepare a bilingual approval packet without scoring or changing judgments."""

import argparse
import copy
import json
from pathlib import Path

from kis.artifact_io import atomic_write_json
from tools.human_holdout import sha, validate_annotations


def prepare(packet, annotations, translations, output):
    packet, annotations, translations, output = map(Path, (packet, annotations, translations, output))
    original = json.loads(annotations.read_text(encoding="utf-8"))
    validated = validate_annotations(packet, original)
    proposals = json.loads(translations.read_text(encoding="utf-8"))
    if proposals.get("source_annotations_sha256") != sha(annotations):
        raise ValueError("Translation proposals do not match submitted annotations")
    items = proposals.get("translations", [])
    by_id = {item["query_id"]: item for item in items}
    expected = {query["query_id"] for query in validated["queries"]}
    if len(by_id) != len(items) or set(by_id) != expected:
        raise ValueError("Every reviewed event needs exactly one translation proposal")
    for item in items:
        text = item.get("text_en", "")
        options = item.get("alternatives", [])
        if not isinstance(text, str) or (not text.strip() and not options):
            raise ValueError("Missing translation or explicit clarification alternatives")
        if options and (text.strip() or len(options) < 2 or not item.get("clarification_vi")):
            raise ValueError("Ambiguous translation needs alternatives and no preselected answer")
    if output.exists():
        raise FileExistsError("Translation review already exists; choose a new version")
    draft = copy.deepcopy(original)
    lines = ["# Duyệt bản dịch holdout v3", "",
             "**Chưa chạy retrieval. Chưa đóng băng nhãn. Mọi câu tiếng Anh đều chờ bạn duyệt.**", "",
             "Bản dịch chỉ dựa trên câu tiếng Việt bạn đã viết, không xem ảnh, caption hoặc scores để bổ sung chi tiết.",
             "Giữ nguyên mọi positive/negative/unknown, số frame, thứ tự event và nhóm bỏ qua.", "",
             "## Cách trả lời", "",
             "Đọc các bản dịch bên dưới. Nếu đồng ý, trả lời: **Tôi duyệt các bản dịch không đánh dấu; "
             "[mã event cần xác nhận] = A/B hoặc mô tả khác**. Có thể sửa bất kỳ câu nào bằng cách nêu mã event và ý cần sửa.", "",
             "Bạn không cần tự sửa JSON hoặc tích lại 26 ô trong giao diện.", ""]
    ambiguities = []
    for sample in draft["samples"]:
        if sample["status"] != "reviewed":
            continue
        for index, event in enumerate(sample["events"], 1):
            query_id = f"{sample['sample_id']}_event_{index}"
            item = by_id[query_id]
            # Approval is never inherited from an empty/old English field.
            event["text_en"] = item.get("text_en", "").strip()
            event["english_approved"] = False
            lines.extend([f"## {query_id}", "", f"**Bạn viết:** {event['text_vi'].strip()}", ""])
            if item.get("alternatives"):
                ambiguities.append(query_id)
                lines.extend([f"**Cần xác nhận:** {item['clarification_vi']}", ""])
                for option in item["alternatives"]:
                    lines.extend([f"- **{option['id']} — {option['meaning_vi']}:** {option['text_en']}"])
                lines.append("")
            else:
                lines.extend([f"**Đề xuất tiếng Anh:** {event['text_en']}", ""])
            if item.get("note_vi"):
                lines.extend([f"*Ghi chú:* {item['note_vi']}", ""])
    # Verify that only translation fields changed, not human observations/qrels.
    comparison = copy.deepcopy(draft)
    for old_sample, new_sample in zip(original["samples"], comparison["samples"]):
        for old_event, new_event in zip(old_sample["events"], new_sample["events"]):
            for field in ("text_en", "english_approved"):
                new_event[field] = old_event[field]
    if comparison != original:
        raise ValueError("Translation preparation changed human observations or judgments")
    validate_annotations(packet, draft)
    output.mkdir(parents=True)
    atomic_write_json(output / "human_annotations_translated_draft.json", draft)
    atomic_write_json(output / "translation_proposals.json", proposals)
    (output / "REVIEW_VI_EN.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {"phase": "awaiting_explicit_human_translation_approval", "evaluation_run": False,
             "source_annotations_sha256": sha(annotations), "proposals_sha256": sha(translations),
             "protocol_sha256": original["protocol_sha256"], "manifest_sha256": original["manifest_sha256"],
             "translation_basis": "user_text_vi_only", "images_visually_inspected_for_translation": False,
             "caption_corpus_accessed": False, "retrieval_scores_accessed": False,
             "human_observations_and_judgments_unchanged": True,
             "reviewed_windows": validated["reviewed_windows"], "events": len(expected),
             "clarifications_pending": ambiguities,
             "all_translations_approved": False, "source_tool_sha256": sha(Path(__file__)),
             "draft_sha256": sha(output / "human_annotations_translated_draft.json")}
    atomic_write_json(output / "audit.json", audit)
    print(json.dumps({"output": str(output.resolve()), **audit}, indent=2))


def record_choices(packet, review, confirmations, output):
    """Record explicit choices only; never infer approval of other translations."""
    packet, review, confirmations, output = map(Path, (packet, review, confirmations, output))
    draft_path = review / "human_annotations_translated_draft.json"
    audit = json.loads((review / "audit.json").read_text(encoding="utf-8"))
    receipt = json.loads(confirmations.read_text(encoding="utf-8"))
    if (sha(draft_path) != audit["draft_sha256"] or
            receipt.get("review_draft_sha256") != sha(draft_path) or
            receipt.get("source_annotations_sha256") != audit["source_annotations_sha256"]):
        raise ValueError("Confirmation does not match translation review/source")
    if receipt.get("approval_scope") != "listed_choices_only" or not receipt.get("user_message", "").strip():
        raise ValueError("Explicit user message and limited approval scope required")
    choices = receipt.get("selected_options", {})
    proposals = json.loads((review / "translation_proposals.json").read_text(encoding="utf-8"))
    by_id = {item["query_id"]: item for item in proposals["translations"]}
    if not choices or not set(choices).issubset(audit["clarifications_pending"]):
        raise ValueError("Only pending clarification choices can be recorded")
    original = json.loads(draft_path.read_text(encoding="utf-8"))
    validate_annotations(packet, original)
    draft = copy.deepcopy(original)
    for sample in draft["samples"]:
        if sample["status"] != "reviewed":
            continue
        for index, event in enumerate(sample["events"], 1):
            query_id = f"{sample['sample_id']}_event_{index}"
            if query_id not in choices:
                continue
            matches = [option for option in by_id[query_id].get("alternatives", [])
                       if option["id"] == choices[query_id]]
            if len(matches) != 1 or not matches[0].get("text_en", "").strip():
                raise ValueError("Unknown translation alternative")
            event["text_en"] = matches[0]["text_en"]
            event["english_approved"] = True
    comparison = copy.deepcopy(draft)
    for old_sample, new_sample in zip(original["samples"], comparison["samples"]):
        for index, (old_event, new_event) in enumerate(zip(old_sample["events"], new_sample["events"]), 1):
            if f"{old_sample['sample_id']}_event_{index}" in choices:
                new_event["text_en"] = old_event["text_en"]
                new_event["english_approved"] = old_event["english_approved"]
    if comparison != original:
        raise ValueError("Confirmation changed fields outside its explicit scope")
    validated = validate_annotations(packet, draft)
    if output.exists():
        raise FileExistsError("Confirmation record already exists; choose a new version")
    output.mkdir(parents=True)
    atomic_write_json(output / "human_annotations_translated_draft.json", draft)
    atomic_write_json(output / "confirmation_receipt.json", receipt)
    updated = {**audit, "phase": "awaiting_remaining_translation_approval", "evaluation_run": False,
               "confirmation_sha256": sha(confirmations), "previous_review_audit_sha256": sha(review / "audit.json"),
               "clarifications_pending": [key for key in audit["clarifications_pending"] if key not in choices],
               "approved_query_ids": sorted(choices), "translations_pending": validated["translations_pending"],
               "all_translations_approved": not validated["translations_pending"],
               "draft_sha256": sha(output / "human_annotations_translated_draft.json"),
               "source_tool_sha256": sha(Path(__file__))}
    atomic_write_json(output / "audit.json", updated)
    print(json.dumps({"output": str(output.resolve()), "approved_choices": len(choices),
                      "translations_pending": len(validated["translations_pending"]),
                      "evaluation_run": False}, indent=2))


def approve_remaining(packet, original_annotations, confirmation_review, receipt_path, output):
    """Apply an explicit approval of remaining nonempty translations; no scoring."""
    packet, original_annotations, confirmation_review, receipt_path, output = map(
        Path, (packet, original_annotations, confirmation_review, receipt_path, output))
    draft_path = confirmation_review / "human_annotations_translated_draft.json"
    audit = json.loads((confirmation_review / "audit.json").read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (sha(draft_path) != audit["draft_sha256"] or receipt.get("review_draft_sha256") != sha(draft_path)
            or sha(original_annotations) != audit["source_annotations_sha256"]):
        raise ValueError("Final approval does not match review/original annotations")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    validated = validate_annotations(packet, draft)
    pending = validated["translations_pending"]
    approved = receipt.get("approved_query_ids", [])
    if (receipt.get("approval_scope") != "all_remaining_translations" or not receipt.get("user_message", "").strip()
            or len(approved) != len(set(approved)) or set(approved) != set(pending)
            or audit["clarifications_pending"]):
        raise ValueError("Final approval must explicitly cover every remaining translation")
    approved_draft = copy.deepcopy(draft)
    for sample in approved_draft["samples"]:
        if sample["status"] != "reviewed":
            continue
        for index, event in enumerate(sample["events"], 1):
            if not event["text_en"].strip():
                raise ValueError("An empty English translation cannot be approved")
            if f"{sample['sample_id']}_event_{index}" in approved:
                event["english_approved"] = True
    validate_annotations(packet, approved_draft, require_english=True)
    original = json.loads(original_annotations.read_text(encoding="utf-8"))
    comparison = copy.deepcopy(approved_draft)
    for old_sample, new_sample in zip(original["samples"], comparison["samples"]):
        for old_event, new_event in zip(old_sample["events"], new_sample["events"]):
            for field in ("text_en", "english_approved"):
                new_event[field] = old_event[field]
    if comparison != original:
        raise ValueError("Final approval changed human observations or judgments")
    if output.exists():
        raise FileExistsError("Final approval record already exists")
    approved_draft["translation_approval"] = {
        "source_annotations_sha256": sha(original_annotations), "final_receipt_sha256": sha(receipt_path),
        "prior_confirmation_audit_sha256": sha(confirmation_review / "audit.json"),
        "user_message": receipt["user_message"], "all_translations_approved": True}
    output.mkdir(parents=True)
    atomic_write_json(output / "human_annotations_approved.json", approved_draft)
    atomic_write_json(output / "approval_receipt.json", receipt)
    atomic_write_json(output / "audit.json", {"phase": "all_translations_approved_ready_to_freeze",
        "evaluation_run": False, "source_annotations_sha256": sha(original_annotations),
        "approved_annotations_sha256": sha(output / "human_annotations_approved.json"),
        "approval_receipt_sha256": sha(receipt_path), "source_tool_sha256": sha(Path(__file__)),
        "judgments_unchanged": True, "approved_events": len(validated["queries"])})
    return output / "human_annotations_approved.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.packet, args.annotations, args.translations, args.output_dir)
