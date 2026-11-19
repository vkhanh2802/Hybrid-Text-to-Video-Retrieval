import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('faiss'), 'requires FAISS')
class SchoolCaptionAliasTests(unittest.TestCase):
    def test_excludes_both_alias_rows_and_keeps_vector_mapping_order(self):
        import numpy as np
        import faiss
        from kis.school_worker import exclude_ambiguous_captions
        vectors = np.eye(4, dtype=np.float32)
        index = faiss.IndexFlatIP(4)
        index.add(vectors)
        rows = [{'video_id': 'L21_V006', 'frame_id': f, 'keyframe_n': n} for n, f in enumerate([0, 0, 120, 330], 1)]
        captions = SimpleNamespace(caption_data=list(rows), mapping=list(rows), index=index)
        removed = exclude_ambiguous_captions(captions, [('L21_V006', 0)])
        self.assertEqual(removed, 2)
        self.assertEqual([r['keyframe_n'] for r in captions.caption_data], [3, 4])
        self.assertEqual(captions.mapping, rows[2:])
        np.testing.assert_equal(captions.index.reconstruct_n(0, 2), vectors[2:])
        self.assertNotIn(('L21_V006', 0), captions._caption_by_identity)
        self.assertEqual(captions._caption_position['L21_V006', 330], 1)
        self.assertEqual(exclude_ambiguous_captions(captions, [('L21_V006', 0)]), 0)
