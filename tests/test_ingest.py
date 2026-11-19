import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kis.artifact_io import atomic_write_json
from kis.config import RuntimeConfig, load_caption_runtime
from tools.ingest import ingest, index_is_current


class IngestTest(unittest.TestCase):
    def test_v2_import_needs_no_images_preserves_v1_and_reuses_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "caption_generator" / "L15" / "L15_V001.json"
            atomic_write_json(old, [{"caption": "v1 stays unchanged"}])
            old_bytes = old.read_bytes()
            v2 = root / "dataset"
            source = v2 / "L15" / "L15" / "L15_V001.json"
            record = {"video_id": "L15_V001", "keyframe_n": 1, "frame_id": 10,
                      "image_path": "data/L15/L15_V001/001.jpg", "caption": "A person stands beside a desk.",
                      "caption_generation": {"backend": "internvl_dense_v3"}}
            atomic_write_json(source, [record])
            atomic_write_json(v2 / "L25" / "L25_V001.json", [{"unused": "not parsed"}])
            mapping = root / "mapping" / "L15"
            mapping.mkdir(parents=True)
            (mapping / "L15_V001.csv").write_text("n,frame_idx\n1,10\n", encoding="utf-8")
            config = RuntimeConfig(v2, ("L15", "L21"), "rrf")
            with patch("tools.ingest.MAPPING_DIR", root / "mapping"), \
                 patch("sentence_transformers.SentenceTransformer") as model:
                model.return_value.encode.return_value = np.ones((1, 384), dtype=np.float32)
                first = ingest(config=config)
                self.assertEqual((first["videos"], first["records"], first["indexes_built"]), (1, 1, 1))
                second = ingest(config=config)
                self.assertEqual(second["indexes_built"], 0)
                model.assert_called_once()
            self.assertEqual(old.read_bytes(), old_bytes)
            self.assertFalse(source.exists())
            self.assertTrue((v2 / "L25" / "L25_V001.json").exists())
            self.assertFalse((root / "data").exists())
            records = json.loads((v2 / "caption_generator" / "L15" / "L15_V001" / "L15_V001.json").read_text())
            self.assertTrue(index_is_current(v2, "L15_V001", records))
            atomic_write_json(v2 / "caption_generator" / "L25" / "L25_V001.json", [{"unused": True}])
            with patch("kis.text.SentenceTransformer") as model:
                model.return_value.encode.return_value = np.ones((1, 384), dtype=np.float32)
                caption, video_ids = load_caption_runtime(config)
                result = caption.search("A person beside a desk", top_k=5)
            self.assertEqual(video_ids, ["L15_V001"])
            self.assertEqual(caption.index.ntotal, 1)
            self.assertEqual((result[0]["video_id"], result[0]["frame_id"], result[0]["keyframe_n"]), ("L15_V001", 10, 1))
            self.assertEqual(result[0]["retrieval_text"], records[0]["retrieval_text"])

            target = v2 / "caption_generator" / "L15" / "L15_V001" / "L15_V001.json"
            changed = dict(records[0], caption="Changed without rebuilding the index")
            atomic_write_json(target, [changed])
            with patch("kis.text.SentenceTransformer") as model:
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    load_caption_runtime(config)
                model.assert_not_called()

    def test_incomplete_caption_mapping_fails_before_any_v2_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2 = root / "v2"
            atomic_write_json(v2 / "L15" / "L15_V001.json", [])
            (root / "mapping").mkdir()
            (root / "mapping" / "L15_V001.csv").write_text("n,frame_idx\n1,10\n")
            with patch("tools.ingest.MAPPING_DIR", root / "mapping"):
                with self.assertRaisesRegex(ValueError, "Incomplete caption coverage"):
                    ingest(config=RuntimeConfig(v2, ("L15",), "rrf"))
            self.assertFalse((v2 / "caption_generator").exists())
