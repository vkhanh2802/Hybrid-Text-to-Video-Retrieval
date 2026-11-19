from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from kis.ocr import OCRConfig, fuse_ocr
from kis.school_ocr import SchoolOCRSnapshot, build_snapshot, digest, validate_record
from tools.import_school_ocr import import_data, zip_records


class SchoolOCRTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.rows = []
        for group in ("L01", "L02"):
            vid = group + "_V001"
            path = self.dataset / "mapping" / group / vid / (vid + ".csv")
            path.parent.mkdir(parents=True)
            path.write_text("n,frame_idx,pts_time\n1,101,1.0\n2,202,2.0\n", encoding="utf-8")
            for n, frame in ((1, 101), (2, 202)):
                self.rows.append(dict(video_id=vid, keyframe_n=n, frame_id=frame))
                if group == "L01":
                    path = self.dataset / "ocr" / group / vid / f"{n:03d}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps([self.record(n)], ensure_ascii=False), encoding="utf-8")
        self.snapshot = self.root / "snapshot"
        self.report = build_snapshot(self.dataset, ("L01",), self.snapshot)
        self.config = OCRConfig(mode="retrieve", weight=.1)

    def record(self, n=1):
        text = 'LỚP 6A không phải 6AB' if n == 1 else ''
        return dict(video_id="L01_V001", frame_id=None, keyframe_n=n,
                    image_path=f"data/L01/L01_V001/{n:03d}.jpg", ocr_text=text,
                    retrieval_text=text, status="ok" if text else "empty")

    def reader(self, **kwargs):
        reader = SchoolOCRSnapshot(self.snapshot, self.report["manifest_sha256"], ("L01", "L02"),
                                   self.dataset, kwargs.get("rows", self.rows))
        self.addCleanup(reader.close)
        return reader

    def test_literal_accents_boundaries_and_no_fake_provenance(self):
        reader = self.reader()
        hits = reader.search("lớp 6A", self.config)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["frame_id"], 101)
        self.assertEqual(reader.search("lop 6A", self.config), [])
        self.assertEqual(reader.search("6", self.config), [])
        self.assertEqual(reader.search('"', self.config), [])
        evidence = hits[0]["ocr"]
        self.assertFalse(evidence["verified"])
        self.assertIsNone(evidence["confidence"])
        self.assertIsNone(evidence["image_sha256"])
        self.assertIsNone(evidence["engine"])
        self.assertEqual(reader.status()["missing_groups"], ["L02"])

    def alias_snapshot(self):
        path = self.dataset / 'mapping/L01/L01_V001/L01_V001.csv'
        path.write_text('n,frame_idx,pts_time\n1,0,0\n2,0,0.0333333\n3,100,3.3333\n', encoding='utf-8')
        for n in (1, 2, 3):
            text = 'ALIAS_ONLY' if n < 3 else 'UNIQUE_ONLY'
            row = dict(self.record(n), ocr_text=text, retrieval_text=text, status='ok')
            (self.dataset / f'ocr/L01/L01_V001/{n:03d}.json').write_text(json.dumps([row]), encoding='utf-8')
        snapshot = self.root / 'alias-snapshot'
        report = build_snapshot(self.dataset, ('L01',), snapshot)
        return snapshot, report

    def test_alias_source_kept_but_not_in_fts_or_runtime_evidence(self):
        snapshot, report = self.alias_snapshot()
        reader = SchoolOCRSnapshot(snapshot, report['manifest_sha256'], ('L01',), self.dataset,
                                   [{'video_id': 'L01_V001', 'keyframe_n': 3, 'frame_id': 100}])
        self.addCleanup(reader.close)
        self.assertEqual(reader.status()['records'], 3)
        self.assertEqual(reader.status()['excluded_ambiguous_keyframes'], 2)
        self.assertEqual(reader.status()['bound_records'], 1)
        self.assertEqual(reader.search('ALIAS_ONLY', self.config), [])
        self.assertEqual(reader.search('UNIQUE_ONLY', self.config)[0]['keyframe_n'], 3)
        with self.assertRaises(ValueError):
            reader.attach_candidates([{'video_id': 'L01_V001', 'keyframe_n': 1, 'frame_id': 0}], self.config)

    def test_missing_unique_binding_or_partial_alias_still_rejected(self):
        snapshot, report = self.alias_snapshot()
        for rows in ([], [{'video_id': 'L01_V001', 'keyframe_n': 1, 'frame_id': 0},
                          {'video_id': 'L01_V001', 'keyframe_n': 3, 'frame_id': 100}]):
            with self.assertRaises(ValueError):
                SchoolOCRSnapshot(snapshot, report['manifest_sha256'], ('L01',), self.dataset, rows)

    def test_attach_missing_empty_error_without_changing_caption(self):
        reader = self.reader()
        candidates = [dict(r, retrieval_text="original caption", score=.5) for r in self.rows]
        candidates[0].pop("keyframe_n")
        reader.attach_candidates(candidates, self.config)
        self.assertEqual([r["ocr"]["status"] for r in candidates], ["ok", "empty", "missing", "missing"])
        self.assertTrue(all(r["score"] == .5 and r["retrieval_text"] == "original caption" for r in candidates))
        for field, value in (("frame_id", 1), ("keyframe_n", True), ("image_path", "../bad.jpg")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                reader.attach_candidates([dict(self.rows[0], **{field: value})], self.config)

    def test_same_size_restored_mtime_bundle_manifest_and_mapping_mutation(self):
        for name in ("bundle", "manifest", "mapping"):
            reader = self.reader()
            path = {"bundle": self.snapshot / "L01.json", "manifest": self.snapshot / "MANIFEST.json",
                    "mapping": self.dataset / "mapping/L01/L01_V001/L01_V001.csv"}[name]
            raw, st = path.read_bytes(), path.stat()
            try:
                altered = bytearray(raw)
                altered[-2] ^= 1
                path.write_bytes(altered)
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, "changed"):
                    reader.search("6A", self.config)
                self.assertFalse(reader.status()["available"])
            finally:
                path.write_bytes(raw)

    def test_pin_duplicates_scope_and_mapping_bindings_fail_closed(self):
        with self.assertRaises(ValueError):
            SchoolOCRSnapshot(self.snapshot, "0" * 64, ("L01",), self.dataset, self.rows)
        cases = [self.rows + self.rows[:1], [dict(r, frame_id=999) for r in self.rows],
                 [dict(r, keyframe_n=True) for r in self.rows]]
        for rows in cases:
            with self.assertRaises(ValueError):
                self.reader(rows=rows)
        with self.assertRaises(ValueError):
            self.reader().validate_config(replace(self.config, region="content"))
        with self.assertRaises(FileExistsError):
            build_snapshot(self.dataset, ("L01",), self.snapshot)

    def test_record_contract(self):
        for changes in ({"frame_id": True}, {"frame_id": 1}, {"keyframe_n": True},
                        {"engine": "invented"}, {"retrieval_text": "not OCR"},
                        {"status": "empty"}, {"image_path": "data/L01/L01_V001/002.jpg"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record([dict(self.record(), **changes)], "L01_V001", 1, 101)

    def test_positive_auxiliary_does_not_penalize_missing_or_dominate_rrf_scale(self):
        candidates = [dict(self.rows[0], score=.016), dict(self.rows[2], score=.015)]
        hits = [dict(self.rows[0], score=1., ocr={"verified": False})]
        result = fuse_ocr(candidates, hits, self.config, bounded_positive=True)
        self.assertAlmostEqual(result[0]["score"], .0176)
        self.assertEqual(result[1]["score"], .015)
        self.assertEqual(candidates[0]["score"], .016)
        self.assertEqual(fuse_ocr(candidates, [], self.config, bounded_positive=True), candidates)

    def test_snapshot_config_requires_pin_and_handshake_checks_it(self):
        from kis.config import load_runtime_config
        from kis.remote_worker import runtime_signature
        env = {"KIS_OCR_FORMAT": "school_snapshot", "KIS_OCR_ROOT": str(self.snapshot),
               "KIS_OCR_MODE": "retrieve", "KIS_OCR_WEIGHT": "0.1"}
        with self.assertRaises(ValueError):
            load_runtime_config(environ=env)
        env["KIS_OCR_MANIFEST_SHA256"] = self.report["manifest_sha256"]
        config = load_runtime_config(environ=env)
        self.assertEqual(config.ocr.weight, .1)
        self.assertEqual(runtime_signature(config)["ocr_manifest_sha256"], self.report["manifest_sha256"])

    def test_original_json_mutation_does_not_mutate_frozen_consumed_snapshot(self):
        reader = self.reader()
        (self.dataset / "ocr/L01/L01_V001/001.json").write_text("changed", encoding="utf-8")
        self.assertEqual(len(reader.search("6A", self.config)), 1)
        manifest = json.loads((self.snapshot / "MANIFEST.json").read_bytes())
        self.assertIn("L01/L01_V001/001.json", manifest["groups"]["L01"]["source_json_sha256"])

    def test_zip_validation_and_import_preserve_bytes_and_refuse_overwrite(self):
        base = self.root / "import"
        (base / "archives").mkdir(parents=True)
        archive = base / "archives/L01.zip"
        originals = {}
        with zipfile.ZipFile(archive, "x") as z:
            for path in (self.dataset / "ocr").rglob("*.json"):
                name = path.relative_to(self.dataset / "ocr").as_posix()
                originals[name] = path.read_bytes()
                z.writestr(name, originals[name])
        report = import_data(base, self.dataset, {"L01": digest(archive)}, existing_groups=())
        self.assertEqual(report["snapshot"]["records"], 2)
        for name, raw in originals.items():
            self.assertEqual((base / "dataset/ocr" / name).read_bytes(), raw)
            self.assertEqual((self.dataset / "ocr" / name).read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            import_data(base, self.dataset, {"L01": digest(archive)}, existing_groups=())
        expected = {("L01_V001", 1): 101, ("L01_V001", 2): 202}
        for name in ("../bad.json", "L01/L01_V001/0001.json", "L02/L02_V001/001.json"):
            bad = self.root / (str(len(name)) + ".zip")
            with zipfile.ZipFile(bad, "w") as z:
                z.writestr(name, json.dumps([self.record()]))
            with self.assertRaises(ValueError):
                zip_records(bad, "L01", expected)


if __name__ == "__main__":
    unittest.main()
