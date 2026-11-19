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
    def test_public_ocr_requires_explicit_format_root_and_l04_opt_in(self):
        default = load_runtime_config(environ={})
        self.assertEqual(default.ocr_format, "strict")
        self.assertIsNone(default.ocr_root)
        config = load_runtime_config(environ={"KIS_VIDEO_GROUPS": "L04", "KIS_OCR_FORMAT": "public_json",
            "KIS_OCR_ROOT": "dataset/ocr/L04", "KIS_OCR_MODE": "retrieve"})
        self.assertEqual(config.video_groups, ("L04",))
        self.assertTrue(config.ocr_root.is_absolute())
        self.assertEqual(config.ocr_root, config.caption_root / "ocr/L04")
        for env in ({"KIS_OCR_FORMAT": "auto"}, {"KIS_OCR_FORMAT": "public_json"}):
            with self.assertRaises(ValueError):
                load_runtime_config(environ=env)

    def test_text_reranker_default_off_and_explicit_mode_validation(self):
        self.assertEqual(load_runtime_config(environ={}).text_reranker.mode, "off")
        config = load_runtime_config(environ={"KIS_TEXT_RERANKER_MODE": "diagnostics"})
        self.assertEqual(config.text_reranker.mode, "diagnostics")
        self.assertEqual(config.text_reranker.pool_size, 100)
        with self.assertRaises(ValueError):
            load_runtime_config(environ={"KIS_TEXT_RERANKER_MODE": "true"})

    def test_temporal_settings_and_environment_disable(self):
        config = load_runtime_config(environ={})
        self.assertTrue(config.temporal.enabled)
        self.assertEqual(config.temporal.max_events, 6)
        self.assertFalse(load_runtime_config(environ={"KIS_DISABLE_TEMPORAL": "true"}).temporal.enabled)

    def test_shared_default_uses_original_new_captions_and_measured_profile(self):
        config = load_runtime_config(environ={})
        self.assertEqual(config.caption_root.name, "dataset")
        self.assertEqual(config.video_groups, (
            "L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08", "L09", "L10", "L11", "L12",
            "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24",
        ))
        profile = get_search_profile(config.retrieval_profile)
        self.assertEqual(profile.nprobe, 128)
        self.assertEqual(profile.rerank_top_k, 20)
        self.assertEqual(profile.policy.variant_fusion, "normalized_max")
        self.assertEqual(profile.policy.cross_fusion, "rrf")
        self.assertEqual(profile.policy.rrf_k, 60)

    def test_relative_overrides_are_resolved_and_empty_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"schema_version": 1, "caption_root": "dataset", "video_groups": ["L19"], "retrieval_profile": "upgraded"}))
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
        with self.assertRaisesRegex(ValueError, "ends at L24"):
            load_runtime_config(environ={"KIS_VIDEO_GROUPS": "L24,L25"})
        with self.assertRaisesRegex(ValueError, "ends at L24"):
            SiglipRetriever(video_ids=["L25_V001"])


if __name__ == "__main__":
    unittest.main()


class LegacyCaptionGroupsTest(unittest.TestCase):
    def run_load(self, legacy):
        rows = [{"video_id": "L01_V001", "caption_generation": {"backend": "internvl_dense_v3"}},
                {"video_id": "L22_V001", "caption": "blip caption"}]
        config = RuntimeConfig(Path("dataset"), ("L01", "L22"), "rrf", legacy_caption_groups=legacy)
        with patch("kis.text.CaptionRetriever", return_value=SimpleNamespace(caption_data=rows)):
            return load_caption_runtime(config)

    def test_untagged_captions_rejected_unless_group_is_listed(self):
        with self.assertRaisesRegex(ValueError, "InternVL"):
            self.run_load(())
        _, videos = self.run_load(("L22",))
        self.assertEqual(videos, ["L01_V001", "L22_V001"])

    def test_pipeline_lists_only_l22_to_l24_as_legacy(self):
        self.assertEqual(load_runtime_config(environ={}).legacy_caption_groups, ("L22", "L23", "L24"))
