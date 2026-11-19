import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from kis.condition_evidence import (apply_condition_evidence, caption_for_frame,
                                    check_conditions, guarded_support)
from kis.temporal import TemporalConfig


class ConditionEvidenceTest(unittest.TestCase):
    def test_explicit_same_garment_color_and_pattern_contradictions(self):
        result = check_conditions("A man wears a plain blue shirt and brown dotted tie.",
                                  "A man wears a checkered white shirt and a gray tie.")
        self.assertEqual(result["status"], "contradicted")
        self.assertFalse(result["pixel_verified"])
        self.assertEqual(result["source"], "generated_caption")
        self.assertEqual(sum(c["status"] == "contradicted" for c in result["conditions"]), 3)
        self.assertEqual(result["conditions"][-1]["status"], "unknown")
        json.dumps(result, allow_nan=False)

    def test_aliases_and_brightness_do_not_invent_conflicts(self):
        result = check_conditions("A man wears a grey checked shirt.", "A man wears a light gray checkered shirt.")
        self.assertEqual(result["status"], "consistent")
        result = check_conditions("A man wears a blue shirt.", "A man wears a dark blue shirt.")
        self.assertEqual(result["status"], "consistent")

    def test_missing_caption_or_attribute_is_unknown(self):
        for caption in (None, "", "A man is in a studio.", "A man wears a shirt and blue tie."):
            with self.subTest(caption=caption):
                self.assertEqual(check_conditions("A man wears a blue shirt.", caption)["status"], "unknown")
        self.assertEqual(check_conditions("A man in a plain blue shirt.", "A man in a blue shirt.")["status"], "unknown")

    def test_no_unsupported_framing_or_object_state_inference(self):
        for query in ("Close-up of a cup on a table.", "Two presenters stand together.", "A person stirs soup."):
            self.assertEqual(check_conditions(query, "A man in a white shirt.")["status"], "unknown")

    def test_subject_ambiguity_and_negation_abstain(self):
        for caption in ("Two men wear white shirts.", "A man in a white shirt and a woman in blue.",
                        "A man and another presenter in a white shirt.", "A man is not wearing a white shirt.",
                        "A man possibly wears a white shirt.", "A man holding a white shirt.",
                        "A presenter in a white shirt beside a presenter in blue.",
                        "A presenter beside a screen showing a white shirt.", "A guest in a white shirt."):
            with self.subTest(caption=caption):
                self.assertEqual(check_conditions("A man in a blue shirt.", caption)["status"], "unknown")
        self.assertEqual(check_conditions("A man in blue and a woman in a red shirt.",
                                         "A woman in a white shirt.")["status"], "unknown")

    def test_conflicting_and_multicolor_attributes_abstain(self):
        for caption in ("A man wears a blue and white shirt.", "A man wears a blue/white shirt.",
                        "A man wears a red shirt over a white shirt.",
                        "A man wears a white shirt with blue stripes.", "A man in a blue-green shirt."):
            with self.subTest(caption=caption):
                self.assertEqual(check_conditions("A man in a red shirt.", caption)["status"], "unknown")

    def test_attributes_not_borrowed_from_background_or_other_garment(self):
        self.assertEqual(check_conditions("A man in a blue shirt.",
                                         "A man beside a blue wall wears a shirt and red tie.")["status"], "unknown")
        result = check_conditions("A man in a blue shirt and red tie.", "A man in a blue shirt and red tie.")
        self.assertEqual(result["status"], "consistent")

    def test_later_unrelated_caption_sentence_does_not_erase_attribute(self):
        query = "A news presenter in a plain blue shirt."
        result = check_conditions(query, "A man in a checked white shirt. The text on the screen reads news.")
        self.assertEqual(result["status"], "contradicted")
        result = check_conditions(query, "A man in a white shirt. He appears to be reporting the news.")
        self.assertEqual(result["status"], "contradicted")
        for caption in ("A screen is visible. A man in a white shirt is shown on it.",
                        "A man in a white shirt. A woman is beside him.",
                        "A man in a white shirt. This is not a blue one."):
            self.assertEqual(check_conditions(query, caption)["status"], "unknown")

    def test_semantic_guard_preserves_visual_support_even_with_contradiction(self):
        support = np.ones((1, 5), dtype=bool)
        visual = np.array([[0.3, 0.25, 0.1, np.nan, np.inf]])
        guarded = guarded_support(support, visual, np.ones((1, 5), dtype=bool), 0.25)
        self.assertEqual(guarded.tolist(), [[True, True, False, False, False]])
        np.testing.assert_array_equal(support, np.ones((1, 5), dtype=bool))
        self.assertFalse(guarded_support(np.array([[False]]), np.array([[1.]]), [[True]], 0.25)[0, 0])

    def test_off_diagnostics_and_guard_share_rules_without_changing_input(self):
        frame = {"video_id": "L01_V001", "frame_id": 10, "keyframe_n": 1}
        captions = SimpleNamespace(get_caption=Mock(return_value={"caption": "A man in a white shirt.", "keyframe_n": 1}))
        support = np.array([[True]])
        result, stats = apply_condition_evidence(["A man in a blue shirt."], [frame], captions,
                                                 support, None, TemporalConfig())
        self.assertIs(result, support)
        captions.get_caption.assert_not_called()
        for mode in ("diagnostics", "semantic_guard"):
            result, stats = apply_condition_evidence(["A man in a blue shirt."], [frame], captions,
                support, None, TemporalConfig(condition_evidence=mode))
            self.assertEqual(bool(result[0, 0]), mode == "diagnostics")
            self.assertEqual(stats["contradicted"], 1)
            self.assertEqual(stats["support_removed"], int(mode == "semantic_guard"))

    def test_identity_mismatch_and_config_errors(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            caption_for_frame(SimpleNamespace(get_caption=lambda *args: {"keyframe_n": 2}),
                              {"video_id": "L01_V001", "frame_id": 10, "keyframe_n": 1})
        with self.assertRaisesRegex(ValueError, "condition evidence"):
            TemporalConfig(condition_evidence="pixel_verified")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            guarded_support([[True]], None, [[True, False]], 0.25)


if __name__ == "__main__":
    unittest.main()
