import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kis.config import milvus_model_properties, RuntimeConfig, load_caption_runtime, load_runtime_config
from kis.ranking import get_search_profile
from kis.visual import SiglipRetriever


class RuntimeConfigTest(unittest.TestCase):
    def test_shared_default_uses_original_new_captions_and_measured_profile(self):
        config = load_runtime_config(environ={})
        self.assertEqual(config.caption_root.name, "caption_generator_v2")
        self.assertEqual(config.video_groups, ("L15", "L16", "L19", "L20", "L21"))
        profile = get_search_profile(config.retrieval_profile)
        self.assertEqual(profile.nprobe, 128)
        self.assertEqual(profile.rerank_top_k, 20)
        self.assertEqual(profile.policy.variant_fusion, "normalized_max")
        self.assertEqual(profile.policy.cross_fusion, "rrf")
        self.assertEqual(profile.policy.rrf_k, 60)

    def test_relative_overrides_are_resolved_and_empty_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"schema_version": 1, "caption_root": "caption_generator_v2", "video_groups": ["L19"], "retrieval_profile": "upgraded"}))
            config = load_runtime_config(path, environ={})
            self.assertTrue(config.caption_root.is_absolute())
            with self.assertRaises(ValueError):
                load_runtime_config(path, environ={"KIS_VIDEO_GROUPS": ""})

    def test_old_caption_or_filtered_view_cannot_enter_active_runtime(self):
        config = RuntimeConfig(Path("new"), ("L19",), "upgraded")
        for record in (
            {"video_id": "L19_V001", "caption_generation": {"backend": "blip"}},
            {"video_id": "L19_V001", "caption_generation": {"backend": "internvl_dense_v3"}, "text_processing": {}},
        ):
            with patch("kis.text.CaptionRetriever", return_value=SimpleNamespace(caption_data=[record])):
                with self.assertRaises(ValueError):
                    load_caption_runtime(config)

    def test_visual_allowlist_is_exact_not_a_whole_group_fallback(self):
        client = Mock()
        client.describe_collection.return_value = {"description": json.dumps(milvus_model_properties())}
        client.search.return_value = [[]]
        with patch("pymilvus.MilvusClient", return_value=client):
            retriever = SiglipRetriever(video_groups=("L19", "L20"), video_ids=["L19_V001"], nprobe=128)
            retriever.search(np.ones(768))
        self.assertEqual(client.search.call_args.kwargs["filter"], 'video_id in ["L19_V001"]')

    def test_scope_cannot_expand_past_l21(self):
        with self.assertRaisesRegex(ValueError, "ends at L21"):
            load_runtime_config(environ={"KIS_VIDEO_GROUPS": "L21,L22"})
        with self.assertRaisesRegex(ValueError, "ends at L21"):
            SiglipRetriever(video_ids=["L22_V001"])


if __name__ == "__main__":
    unittest.main()
