import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from kis.ranking import get_search_profile
from tools.pipeline_ablation import (CachedEncoder, TemporalVariantText, TemporalVariantVisual,
                                     arm_plan, compact_query, fixed_plan, literal_entities)


class PipelineAblationTest(unittest.TestCase):
    def test_compact_is_retrieval_only_and_preserves_full_verification_text(self):
        text = "A man is wearing a white shirt and a blue tie in front of an orange city backdrop."
        unit = {"query_id": "q", "raw_query": "Mô tả gốc", "event_text": text}
        plan = fixed_plan(unit)
        original = copy.deepcopy(plan)
        compact = arm_plan(plan, "full_compact")
        self.assertEqual(plan, original)
        self.assertEqual(compact["raw_query"], unit["raw_query"])
        self.assertEqual(compact["stream_queries"]["visual"][0], text)
        self.assertIn("white", compact_query(text))
        self.assertLessEqual(len(compact_query(text).split()), 20)

    def test_temporal_verification_actions_not_shortened(self):
        actions = ["A man pours eggs.", "A cook stirs soup."]
        unit = {"query_id": "seq", "raw_query": "First then", "actions": actions}
        plan = arm_plan(fixed_plan(unit), "full_compact")
        self.assertEqual([e["action"] for e in plan["events"]], actions)
        self.assertEqual(plan["temporal_constraints"][0]["order"], actions)

    def test_matched_depth_wrapper_and_static_fallback_not_double_expanded(self):
        policy = get_search_profile("rrf").policy
        base = SimpleNamespace(search=Mock(return_value=[]))
        text = TemporalVariantText(base, "full_deep", policy)
        text.search("Full sentence", top_k=50)
        self.assertEqual(base.search.call_args.kwargs["top_k"], 100)
        text.search("Fallback sentence", top_k=200)
        self.assertEqual(base.search.call_args.kwargs["top_k"], 200)
        encoder = CachedEncoder(SimpleNamespace(encode_text=lambda q: np.array([1., 0.], np.float32)))
        visual = TemporalVariantVisual(base, encoder, ["Full sentence"], "full_deep", policy)
        visual.search(encoder.encode_text("Full sentence"), top_k=50)
        self.assertEqual(base.search.call_args.kwargs["top_k"], 100)

    def test_literal_objects_do_not_invent_synonyms_or_entities(self):
        self.assertEqual(literal_entities("A bus beside a car."), [
            {"type": "object", "value": "car", "attributes": {}},
            {"type": "object", "value": "bus", "attributes": {}}])
        self.assertEqual(literal_entities("An unidentified object."), [])


if __name__ == "__main__":
    unittest.main()
