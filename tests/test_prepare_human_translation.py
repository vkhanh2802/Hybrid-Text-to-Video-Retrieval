import copy
import json
from pathlib import Path
import tempfile
import unittest

from kis.artifact_io import atomic_write_json
from tests.test_human_holdout import synthetic_annotations, synthetic_packet
from tools.human_holdout import sha, validate_annotations
from tools.prepare_human_translation import approve_remaining, prepare, record_choices


class PrepareHumanTranslationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = synthetic_packet(Path(self.temp.name) / "packet")
        self.annotations = self.root / "human.json"
        atomic_write_json(self.annotations, synthetic_annotations(self.root))
        self.proposals = self.root / "proposals.json"
        self.items = [{"query_id": f"drive_{i:02d}_event_1", "text_en": "A blue rectangle."} for i in (1, 2)]
        self.write_proposals()
        self.output = self.root / "translation_review"

    def write_proposals(self):
        atomic_write_json(self.proposals, {"source_annotations_sha256": sha(self.annotations), "translations": self.items})

    def test_translation_changes_only_english_and_requires_new_approval(self):
        before = self.annotations.read_bytes()
        prepare(self.root, self.annotations, self.proposals, self.output)
        self.assertEqual(self.annotations.read_bytes(), before)
        draft = json.loads((self.output / "human_annotations_translated_draft.json").read_text(encoding="utf-8"))
        self.assertTrue(all(not s["events"][0]["english_approved"] for s in draft["samples"]))
        self.assertEqual(len(validate_annotations(self.root, draft)["translations_pending"]), 2)
        for original, translated in zip(json.loads(before)["samples"], draft["samples"]):
            self.assertEqual(original["events"][0]["text_vi"], translated["events"][0]["text_vi"])
            self.assertEqual(original["events"][0]["judgments"], translated["events"][0]["judgments"])
        with self.assertRaisesRegex(ValueError, "human approval"):
            validate_annotations(self.root, draft, require_english=True)
        with self.assertRaises(FileExistsError):
            prepare(self.root, self.annotations, self.proposals, self.output)

    def test_ambiguity_has_no_default_translation(self):
        self.items[0] = {"query_id": "drive_01_event_1", "text_en": "", "clarification_vi": "Màu nào?",
                         "alternatives": [{"id": "A", "meaning_vi": "Xanh dương", "text_en": "Blue rectangle."},
                                          {"id": "B", "meaning_vi": "Xanh lá", "text_en": "Green rectangle."}]}
        self.write_proposals()
        prepare(self.root, self.annotations, self.proposals, self.output)
        audit = json.loads((self.output / "audit.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["evaluation_run"])
        self.assertEqual(audit["clarifications_pending"], ["drive_01_event_1"])
        draft = json.loads((self.output / "human_annotations_translated_draft.json").read_text(encoding="utf-8"))
        self.assertEqual(draft["samples"][0]["events"][0]["text_en"], "")

    def test_wrong_source_hash_missing_duplicate_or_preselected_alternative_rejected(self):
        variants = [self.items[:1], self.items + [copy.deepcopy(self.items[0])],
                    [{**self.items[0], "text_en": "Blue rectangle.", "clarification_vi": "Which?",
                      "alternatives": [{"id": "A"}, {"id": "B"}]}, self.items[1]]]
        for items in variants:
            self.items = items
            self.write_proposals()
            with self.assertRaises(ValueError):
                prepare(self.root, self.annotations, self.proposals, self.output)
            self.assertFalse(self.output.exists())
        payload = json.loads(self.proposals.read_text(encoding="utf-8"))
        payload["source_annotations_sha256"] = "wrong"
        atomic_write_json(self.proposals, payload)
        with self.assertRaisesRegex(ValueError, "submitted annotations"):
            prepare(self.root, self.annotations, self.proposals, self.output)

    def confirmation_fixture(self):
        self.items[0] = {"query_id": "drive_01_event_1", "text_en": "", "clarification_vi": "Màu nào?",
                         "alternatives": [{"id": "A", "meaning_vi": "Xanh dương", "text_en": "Blue rectangle."},
                                          {"id": "B", "meaning_vi": "Xanh lá", "text_en": "Green rectangle."}]}
        self.write_proposals()
        prepare(self.root, self.annotations, self.proposals, self.output)
        receipt = self.root / "choices.json"
        payload = {"review_draft_sha256": sha(self.output / "human_annotations_translated_draft.json"),
                   "source_annotations_sha256": sha(self.annotations), "approval_scope": "listed_choices_only",
                   "user_message": "Synthetic confirmation: first event is blue", "selected_options": {"drive_01_event_1": "A"}}
        atomic_write_json(receipt, payload)
        return receipt, payload

    def test_only_confirmed_choices_become_approved(self):
        receipt, _ = self.confirmation_fixture()
        before = self.annotations.read_bytes()
        destination = self.root / "confirmed"
        record_choices(self.root, self.output, receipt, destination)
        draft = json.loads((destination / "human_annotations_translated_draft.json").read_text(encoding="utf-8"))
        self.assertTrue(draft["samples"][0]["events"][0]["english_approved"])
        self.assertEqual(draft["samples"][0]["events"][0]["text_en"], "Blue rectangle.")
        self.assertFalse(draft["samples"][1]["events"][0]["english_approved"])
        self.assertEqual(validate_annotations(self.root, draft)["translations_pending"], ["drive_02_event_1"])
        self.assertEqual(self.annotations.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "human approval"):
            validate_annotations(self.root, draft, require_english=True)
        with self.assertRaises(FileExistsError):
            record_choices(self.root, self.output, receipt, destination)

    def test_confirmation_rejects_out_of_scope_choice_or_wrong_review(self):
        receipt, payload = self.confirmation_fixture()
        for changes in ({"selected_options": {"drive_02_event_1": "A"}},
                        {"selected_options": {"drive_01_event_1": "C"}},
                        {"review_draft_sha256": "wrong"}, {"approval_scope": "all_translations"}):
            atomic_write_json(receipt, {**payload, **changes})
            with self.assertRaises(ValueError):
                record_choices(self.root, self.output, receipt, self.root / "invalid")
            self.assertFalse((self.root / "invalid").exists())

    def test_final_approval_preserves_every_label_and_requires_exact_pending_scope(self):
        receipt, _ = self.confirmation_fixture()
        confirmed = self.root / "confirmed"
        record_choices(self.root, self.output, receipt, confirmed)
        final_receipt = self.root / "final_receipt.json"
        payload = {"review_draft_sha256": sha(confirmed / "human_annotations_translated_draft.json"),
                   "approval_scope": "all_remaining_translations", "user_message": "Synthetic final approval",
                   "approved_query_ids": []}
        atomic_write_json(final_receipt, payload)
        with self.assertRaisesRegex(ValueError, "every remaining"):
            approve_remaining(self.root, self.annotations, confirmed, final_receipt, self.root / "final")
        payload["approved_query_ids"] = ["drive_02_event_1"]
        atomic_write_json(final_receipt, payload)
        path = approve_remaining(self.root, self.annotations, confirmed, final_receipt, self.root / "final")
        draft = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_annotations(self.root, draft, require_english=True)["translations_pending"], [])
        for old, new in zip(json.loads(self.annotations.read_text(encoding="utf-8"))["samples"], draft["samples"]):
            self.assertEqual(old["events"][0]["judgments"], new["events"][0]["judgments"])
        with self.assertRaises(FileExistsError):
            approve_remaining(self.root, self.annotations, confirmed, final_receipt, self.root / "final")


if __name__ == "__main__":
    unittest.main()
