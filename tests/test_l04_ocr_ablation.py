import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import l04_ocr_ablation as ab


class L04OCRABTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.vid = "L04_V001"
        self.mapping = {1: 0, 2: 90, 3: 179, 4: 288}
        for kind in ("data", "ocr", "caption_generator", "mapping"):
            (self.root / kind / "L04").mkdir(parents=True)
        (self.root / "data/L04" / self.vid).mkdir()
        (self.root / "ocr/L04" / self.vid).mkdir()
        self.caption_path = self.root / "caption_generator/L04" / (self.vid + ".json")
        self.records = []
        for n in self.mapping:
            image = self.root / "data/L04" / self.vid / f"{n:03d}.jpg"
            image.write_bytes(f"synthetic image bytes {n}".encode())
            self.records.append({"video_id": self.vid, "keyframe_n": n, "frame_id": self.mapping[n],
                                 "image_path": f"data/L04/{self.vid}/{n:03d}.jpg",
                                 "retrieval_text": ("robot carrying board" if n == 1 else "people school courtyard"),
                                 "image_sha256": ab.digest(image)})
        self.write(self.caption_path, self.records)
        (self.root / "mapping/L04" / (self.vid + ".csv")).write_text(
            "n,frame_idx\n1,0\n2,90\n3,179\n4,288\n", encoding="utf-8")
        self.write_ocr(1, "ok", "NGUYỄN 37")
        self.write_ocr(2, "empty", "")
        self.write_ocr(3, "error", "")
        self.packet = {
            "schema_version": 1, "group": "L04", "authoring": {"selection_rule": "synthetic fixture", "language_policy": "fixed"},
            "queries": [{"id": "q1", "kind": "literal", "language": "Vietnamese literal", "text": "NGUYỄN 37",
                         "target": {"video_id": self.vid, "keyframe_n": 1},
                         "source_image": "dataset/" + self.records[0]["image_path"],
                         "image_sha256": self.records[0]["image_sha256"]}]}
        self.queries = self.base / "queries.json"
        self.write(self.queries, self.packet)

    @staticmethod
    def write(path, value):
        path.write_bytes(ab.json_bytes(value))

    def public(self, n=1, status="ok", text="NGUYỄN 37"):
        return [{"video_id": self.vid, "keyframe_n": n, "frame_id": None,
                 "image_path": f"data/L04/{self.vid}/{n:03d}.jpg", "ocr_text": text,
                 "retrieval_text": text, "status": status}]

    def write_ocr(self, n, status, text):
        path = self.root / "ocr/L04" / self.vid / f"{n:03d}.json"
        self.write(path, self.public(n, status, text))

    def test_real_public_shape_and_nonconsecutive_frame_mapping(self):
        key, result = ab.adapt_public(self.public(3), self.vid, self.mapping, 3)
        self.assertEqual(key, (self.vid, 3))
        self.assertEqual(result["text"], "NGUYỄN 37")
        docs, _, _ = ab.load_corpus(self.root)
        self.assertEqual([d["frame_id"] for d in docs], [0, 90, 179, 288])
        payload = self.public(3)
        payload[0]["frame_id"] = 2  # n-1 is not frame_idx.
        with self.assertRaisesRegex(ValueError, "mapped frame_idx"):
            ab.adapt_public(payload, self.vid, self.mapping, 3)

    def test_coverage_and_missing_empty_error_candidates_retained(self):
        docs, coverage, _ = ab.load_corpus(self.root)
        self.assertEqual(len(docs), 4)
        self.assertEqual([doc["ocr_status"] for doc in docs], ["ok", "empty", "error", "missing"])
        for key in ("matched", "public_array_files", "ocr_records"):
            self.assertEqual(coverage[key], 3)
        for key in ("ok", "empty", "error", "missing", "blank"):
            self.assertEqual(coverage[key], 1)
        self.assertEqual(coverage["unchanged_document_text"], 3)
        self.assertEqual(coverage["caption_hash_verified"], 4)
        self.assertEqual(coverage["ocr_image_hash_unverified"], 3)
        for key in ("duplicate", "unknown_status", "technical_object_files", "ocr_without_caption"):
            self.assertEqual(coverage[key], 0)
        for doc in docs[1:]:
            self.assertEqual(doc["ocr_text"], "")

    def test_duplicate_caption_and_ocr_fail_closed(self):
        self.write(self.caption_path, self.records + [self.records[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate caption"):
            ab.load_corpus(self.root)
        self.write(self.caption_path, self.records)
        self.write(self.root / "ocr/L04" / self.vid / "1.json", self.public())
        with self.assertRaisesRegex(ValueError, "Duplicate OCR"):
            ab.load_corpus(self.root)

    def test_malformed_public_types_paths_status_and_missing_text_fail_closed(self):
        variations = [
            {"keyframe_n": True}, {"keyframe_n": "1"}, {"frame_id": False},
            {"status": "unknown"}, {"status": "empty"}, {"status": "error"},
            {"ocr_text": None}, {"ocr_text": "", "retrieval_text": ""},
            {"retrieval_text": "different"}, {"image_path": f"data/{self.vid}/001.jpg"},
            {"image_path": f"data/L04/{self.vid}/002.jpg"},
            {"image_path": f"data/L04/{self.vid}/../001.jpg"}, {"video_id": "L06_V001"},
            {"unexpected": 1},
        ]
        for updates in variations:
            with self.subTest(updates=updates):
                payload = self.public()
                payload[0].update(updates)
                with self.assertRaises(ValueError):
                    ab.adapt_public(payload, self.vid, self.mapping, 1)
        payload = self.public()
        del payload[0]["ocr_text"]
        for bad in (payload, [], self.public() * 2, self.public()[0]):
            with self.assertRaises(ValueError):
                ab.adapt_public(bad, self.vid, self.mapping, 1)
        with self.assertRaises(ValueError):
            ab.adapt_public(self.public(), self.vid, self.mapping, 2)

    def test_hash_and_caption_frame_mismatches_fail_closed(self):
        self.records[0]["image_sha256"] = "0" * 64
        self.write(self.caption_path, self.records)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            ab.load_corpus(self.root)
        del self.records[0]["image_sha256"]
        self.records[0]["frame_id"] = 1
        self.write(self.caption_path, self.records)
        with self.assertRaisesRegex(ValueError, "mapped frame_idx"):
            ab.load_corpus(self.root)

    def test_no_excluded_group_access_and_no_broad_enumeration(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("must reject before filesystem")):
            for group in ("L06", "L05"):
                with self.assertRaisesRegex(ValueError, "Only L04"):
                    ab.checked_path(self.root / "data" / group)
        original_iterdir, original_glob = Path.iterdir, Path.glob
        calls = []

        def iterdir(path):
            calls.append(path)
            self.assertIn(path, (self.root / "ocr/L04", self.root / "ocr/L04" / self.vid))
            return original_iterdir(path)

        def glob(path, pattern):
            calls.append(path)
            self.assertEqual(path, self.root / "caption_generator/L04")
            return original_glob(path, pattern)

        with patch.object(Path, "iterdir", iterdir), patch.object(Path, "glob", glob), patch.object(
                Path, "rglob", side_effect=AssertionError("no recursive discovery")):
            ab.load_corpus(self.root)
        self.assertEqual(len(calls), 3)

    def test_stable_ties_and_no_zero_negative_nonfinite_score_hits(self):
        docs, _, _ = ab.load_corpus(self.root)
        index = ab.LexicalIndex(list(reversed(docs)))
        with patch.object(index.model, "get_scores", return_value=np.array([2.0, 2.0, 0.0, -1.0])):
            self.assertEqual([d["keyframe_n"] for d in index.search("anything")], [3, 4])
        with patch.object(index.model, "get_scores", return_value=np.array([0.0, -1.0, np.nan, np.inf])):
            self.assertEqual(index.search("anything"), [])
        self.assertEqual(index.search("neverpresenttoken"), [])

    def test_metrics_mrr_truncation_and_null_rank(self):
        results = [{"video_id": self.vid, "keyframe_n": n, "score": float(20 - n)} for n in range(1, 13)]
        def measured(n):
            return ab.measure(results, {"video_id": self.vid, "keyframe_n": n})
        self.assertEqual(measured(1)["hit@1"], 1)
        self.assertEqual(measured(5)["hit@5"], 1)
        self.assertEqual(measured(10)["mrr@10"], 0.1)
        self.assertEqual(measured(11)["rank"], 11)
        self.assertEqual(measured(11)["mrr@10"], 0)
        self.assertEqual(len(measured(11)["top10"]), 10)
        self.assertIsNone(measured(13)["rank"])
        self.assertIsNone(measured(13)["score"])

    def test_targets_never_enter_ranking_and_target_change_does_not_change_results(self):
        docs, _, _ = ab.load_corpus(self.root)
        self.assertEqual(ab.LexicalIndex(docs, True).search(self.vid), [])
        first, _, _ = ab.evaluate(docs, self.packet["queries"])
        altered = copy.deepcopy(self.packet["queries"])
        altered[0]["target"]["keyframe_n"] = 4
        second, _, _ = ab.evaluate(docs, altered)
        for arm in ("baseline", "caption_plus_ocr"):
            self.assertEqual(first[0][arm]["top10"], second[0][arm]["top10"])
        self.assertIsNone(first[0]["baseline"]["rank"])
        self.assertEqual(first[0]["caption_plus_ocr"]["rank"], 1)
        self.assertEqual(first[0]["outcome"], "win")

    def test_frozen_queries_config_manifest_hashes_and_no_overwrite(self):
        raw = self.queries.read_bytes()
        output = self.base / "run01"
        report = ab.run(self.root, self.queries, output)
        self.assertEqual(report["queries_sha256"], ab.byte_sha(raw))
        self.assertEqual((output / "queries.frozen.json").read_bytes(), raw)
        self.assertEqual(report["config_sha256"], ab.digest(output / "config.json"))
        self.assertEqual(report["input_manifest_sha256"], ab.digest(output / "input_manifest.json"))
        self.assertEqual(report["implementation_sha256"], ab.digest(Path(ab.__file__)))
        with self.assertRaises(FileExistsError):
            ab.run(self.root, self.queries, output)

    def test_markdown_ocr_evidence_whitespace_clean_json_evidence_unchanged(self):
        text = "NGUYỄN 37 \nmore \t"
        self.write_ocr(1, "ok", text)
        output = self.base / "run01"
        report = ab.run(self.root, self.queries, output)
        self.assertEqual(report["per_query"][0]["target"]["ocr_text"], text)
        markdown = (output / "report.md").read_text(encoding="utf-8")
        self.assertFalse(any(line.rstrip(" \t") != line for line in markdown.splitlines()))

    def test_frozen_image_mismatch_and_query_mutation_fail_closed(self):
        self.packet["queries"][0]["image_sha256"] = "0" * 64
        self.write(self.queries, self.packet)
        with self.assertRaisesRegex(ValueError, "Frozen query image"):
            ab.run(self.root, self.queries, self.base / "run01")
        self.packet["queries"][0]["image_sha256"] = self.records[0]["image_sha256"]
        self.write(self.queries, self.packet)
        original = ab.evaluate

        def mutate(docs, queries):
            self.queries.write_bytes(self.queries.read_bytes() + b"\n")
            return original(docs, queries)

        with patch.object(ab, "evaluate", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during run"):
                ab.run(self.root, self.queries, self.base / "run01")
        self.assertFalse((self.base / "run01").exists())

    def test_frozen_real_packet_sha(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/l04_ocr_exploratory_20261111.json"
        raw, packet = ab.load_queries(path)
        self.assertEqual(ab.byte_sha(raw), "f60b18f82b5b379414c503f08aae8cec7df325df59f7f0f00d7d8ccaa1ead8db")
        self.assertEqual(len(packet["queries"]), 8)
        self.assertEqual(Counter(q["kind"] for q in packet["queries"]), {"literal": 3, "scene": 5})

    def test_unknown_technical_object_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized technical"):
            ab.count_technical({"schema": "unknown"}, self.vid, self.mapping, 1)

    def test_strict_old_technical_object_counted_not_used_as_public_ocr(self):
        model = {"role": "detection", "name": "test_model", "repo": "test/model", "revision": "0" * 40,
                 "files_sha256": {name: "0" * 64 for name in ("inference.json", "inference.pdiparams", "inference.yml")}}
        recognition = dict(model, role="recognition")
        technical = {
            "schema": "ocr-sidecar-v1", "schema_version": 1, "group": "L04", "video_id": self.vid,
            "keyframe_n": 1, "image_name": "001.jpg", "source_path": self.records[0]["image_path"],
            "image_sha256": self.records[0]["image_sha256"], "mapping": {"frame_idx": 0, "timestamp_seconds": 0},
            "engine": {"name": "paddleocr", "versions": {"test": "1"}, "models": [model, recognition],
                       "device": "cpu", "language": "vi", "config_sha256": "0" * 64},
            "status": "unknown", "preprocessing": None, "lines": [], "error": None,
        }
        path = self.root / "ocr/L04" / self.vid / "001.json"
        self.write(path, technical)
        docs, coverage, _ = ab.load_corpus(self.root)
        self.assertEqual(coverage["technical_object_files"], 1)
        self.assertEqual(coverage["technical_count_only"], 1)
        self.assertEqual(coverage["missing"], 2)
        self.assertEqual(docs[0]["ocr_status"], "missing")
        self.assertEqual(docs[0]["ocr_text"], "")
        technical["unexpected"] = "must not silently accept"
        self.write(path, technical)
        from jsonschema.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            ab.load_corpus(self.root)

    def test_duplicate_and_malformed_mapping_fail_closed(self):
        path = self.root / "mapping/L04" / (self.vid + ".csv")
        for text in ("n,frame_idx\n1,0\n1,90\n", "n,frame_idx\n1,0\n2,0\n", "n,frame_idx\n1,x\n", "n,other\n1,0\n"):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                ab.load_corpus(self.root)

    def fusion_query(self, kind="mixed"):
        query = copy.deepcopy(self.packet["queries"][0])
        query["kind"] = kind
        query["judgments"] = [{"video_id": self.vid, "keyframe_n": n, "frame_id": self.mapping[n],
                                "target": query["target"], "label": int(n in (1, 2)),
                                "source_image": "dataset/" + self.records[n - 1]["image_path"],
                                "image_sha256": self.records[n - 1]["image_sha256"], "rationale": "Synthetic visible fixture"}
                               for n in (1, 2, 3, 4)]
        return query

    def test_multiple_positives_earliest_rank_and_cutoff(self):
        results = [{"video_id": self.vid, "keyframe_n": n, "score": 1.0} for n in range(1, 15)]
        result = ab.measure_positives(results, {(self.vid, 11), (self.vid, 3)})
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["mrr@10"], 1 / 3)
        self.assertEqual(result["first_positive_identity"]["keyframe_n"], 3)
        result = ab.measure_positives(results, {(self.vid, 11), (self.vid, 13)})
        self.assertEqual(result["rank"], 11)
        self.assertEqual(result["mrr@10"], 0)
        self.assertIsNone(ab.measure_positives(results, {(self.vid, 99)})["rank"])
        with self.assertRaisesRegex(ValueError, "Empty positives"):
            ab.measure_positives(results, set())

    def test_duplicate_qrels_and_empty_positive_fail_closed(self):
        query = self.fusion_query()
        query["judgments"].append(copy.deepcopy(query["judgments"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate qrel"):
            ab.validate_judgments(query)
        query = self.fusion_query()
        for judgment in query["judgments"]:
            judgment["label"] = 0
        with self.assertRaisesRegex(ValueError, "positive target"):
            ab.validate_judgments(query)
        query["judgments"][0]["label"] = True
        with self.assertRaisesRegex(ValueError, "label/target"):
            ab.validate_judgments(query)

    def test_judgment_image_hash_path_and_mapping_tamper(self):
        docs, _, _ = ab.load_corpus(self.root)
        by_key = {ab.doc_key(doc): doc for doc in docs}
        self.assertEqual(len(ab.validate_judgments(self.fusion_query(), by_key)), 2)
        for updates in ({"image_sha256": "0" * 64}, {"frame_id": 1}, {"source_image": "dataset/data/L04/L04_V001/999.jpg"},
                        {"label": -1}, {"rationale": ""}, {"video_id": "L05_V001"}):
            query = self.fusion_query()
            query["judgments"][0].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                ab.validate_judgments(query, by_key)

    def test_rrf_formula_one_based_missing_and_stable_ties(self):
        def result(n, score=1.0):
            return {"video_id": self.vid, "keyframe_n": n, "score": score}
        rows = ab.reciprocal_rank_fusion([result(2), result(1)], [result(1), result(2), result(3)])
        self.assertEqual([r["keyframe_n"] for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[0]["score"], 1 / 61 + 1 / 62)
        self.assertEqual(rows[0]["component_ranks"], {"caption": 2, "ocr": 1})
        self.assertEqual(rows[2]["component_ranks"], {"caption": None, "ocr": 3})
        self.assertEqual(rows[2]["score"], 1 / 63)
        self.assertEqual(ab.reciprocal_rank_fusion([], []), [])
        self.assertEqual(ab.reciprocal_rank_fusion([result(1)], [])[0]["score"], 1 / 61)

    def test_rrf_zero_negative_nonfinite_duplicate_and_full_rank(self):
        rows = [{"video_id": self.vid, "keyframe_n": n, "score": 1001.0 - n} for n in range(1, 1001)]
        rows[0]["score"], rows[1]["score"], rows[2]["score"], rows[3]["score"] = 0, -1, float("nan"), float("inf")
        fused = ab.reciprocal_rank_fusion(rows, [])
        self.assertEqual(len(fused), 996)
        self.assertEqual(fused[-1]["component_ranks"]["caption"], 996)
        self.assertEqual(fused[-1]["score"], 1 / 1056)
        with self.assertRaisesRegex(ValueError, "Duplicate RRF"):
            ab.reciprocal_rank_fusion(rows[4:5] * 2, [])

    def test_empty_ocr_vocab_missing_error_retain_caption_universe(self):
        docs, _, _ = ab.load_corpus(self.root)
        for doc in docs:
            doc["ocr_text"] = ""
        self.assertEqual(len(ab.LexicalIndex(docs, ocr_only=True).documents), 4)
        self.assertEqual(ab.LexicalIndex(docs, ocr_only=True).search("robot"), [])
        self.assertEqual(ab.LexicalIndex([], ocr_only=True).search("robot"), [])
        docs[0]["ocr_text"] = "___ !!!"
        self.assertEqual(ab.LexicalIndex(docs, ocr_only=True).search("robot"), [])
        before = ab.LexicalIndex(docs).search("robot")
        fused = ab.reciprocal_rank_fusion(before, [])
        self.assertEqual([ab.doc_key(r) for r in before], [ab.doc_key(r) for r in fused])
        self.assertEqual([d["ocr_status"] for d in docs], ["ok", "empty", "error", "missing"])
        self.assertTrue(before)

    @staticmethod
    def selection_fixture():
        names = {f"L04_V{i:03}": [10 * n for n in range(1, 31)] for i in range(1, 31)}
        plan = ab.selection_indices(names, names)
        packet = {"queries": []}
        for i, chosen in enumerate(plan):
            kind = ("literal", "scene", "mixed")[i % 3]
            packet["queries"].append({"id": f"t{i+1:02}", "split": "test", "kind": kind, "assigned_kind": kind,
                                      "target": dict(zip(("video_id", "keyframe_n"), chosen["target"])),
                                      "image_count": chosen["image_count"], "target_position_zero_based": chosen["position"],
                                      "judgments": [{"video_id": v, "keyframe_n": n, "image_list_offset": k - 2}
                                                    for k, (v, n) in enumerate(chosen["neighbors"])]})
        dev = [{"target": {"video_id": v, "keyframe_n": 1}} for v in sorted(ab.EXCLUDED_VIDEOS)]
        return plan, packet, dev

    def test_selection_formula_nonconsecutive_stems_and_disjoint_split(self):
        plan, packet, dev = self.selection_fixture()
        ab.validate_selection(packet, plan, dev)
        self.assertEqual([p["target"][0] for p in plan[::2]],
                         [f"L04_V{n:03}" for n in (2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 29)])
        self.assertEqual([p["target"][1] for p in plan[:2]], [100, 200])
        self.assertEqual(len({p["target"] for p in plan}), 24)
        self.assertEqual(plan[0]["neighbors"], [("L04_V002", n) for n in (80, 90, 100, 110, 120)])
        for field, value in (("split", "dev-reference"), ("assigned_kind", "mixed"), ("target_position_zero_based", 10)):
            bad = copy.deepcopy(packet)
            bad["queries"][0][field] = value
            with self.assertRaisesRegex(ValueError, "Selection"):
                ab.validate_selection(bad, plan, dev)
        with self.assertRaisesRegex(ValueError, "video overlap"):
            ab.validate_selection(packet, plan, [{"target": packet["queries"][0]["target"]}])

    def test_selection_invalid_names_interior_neighbors_and_fallback(self):
        plan, packet, dev = self.selection_fixture()
        query = packet["queries"][0]
        query["kind"], query["fallback"] = "scene", "Text unreadable"
        ab.validate_selection(packet, plan, dev)
        query.pop("fallback")
        with self.assertRaisesRegex(ValueError, "fallback"):
            ab.validate_selection(packet, plan, dev)
        query["kind"] = "literal"
        query["judgments"].pop()
        with self.assertRaisesRegex(ValueError, "neighborhood"):
            ab.validate_selection(packet, plan, dev)
        with self.assertRaises(ValueError):
            ab.selection_indices(["L05_V001"], {})
        names = {f"L04_V{i:03}": list(range(1, 5)) for i in range(1, 31)}
        with self.assertRaisesRegex(ValueError, "interior"):
            ab.selection_indices(names, names)

    def test_four_arm_determinism_text_only_and_mixed_aggregate(self):
        docs, _, _ = ab.load_corpus(self.root)
        queries = [self.fusion_query("mixed"), self.fusion_query("scene")]
        queries[1]["id"], queries[1]["language"] = "q2", "English"
        queries[1]["text"] = "robot"
        first = ab.evaluate_fusion(docs, queries, self.packet["queries"])
        second = ab.evaluate_fusion(docs, queries, self.packet["queries"])
        self.assertEqual(ab.json_bytes(first), ab.json_bytes(second))
        rows, metrics, _ = first
        self.assertEqual(metrics["test/kind/mixed"]["queries"], 1)
        self.assertEqual(metrics["dev-reference/all"]["queries"], 1)
        for arm in ab.ARMS:
            for basis in ("target", "known_positive"):
                for metric in ab.METRICS:
                    expected = (metrics["test/kind/mixed"][arm][basis][metric]
                                + metrics["test/kind/scene"][arm][basis][metric]) / 2
                    self.assertEqual(metrics["test/all"][arm][basis][metric], expected)
            self.assertEqual(rows[0][arm]["top10"], second[0][0][arm]["top10"])
        altered = copy.deepcopy(queries)
        altered[0]["target"]["keyframe_n"] = 2
        for judgment in altered[0]["judgments"]:
            judgment["target"] = altered[0]["target"]
        other, _, _ = ab.evaluate_fusion(docs, altered, [])
        for arm in ab.ARMS:
            self.assertEqual(rows[0][arm]["top10"], other[0][arm]["top10"])

    def test_schema2_packet_protocol_hash_and_real_freeze(self):
        path = Path("benchmarks/l04_ocr_fusion_20261111_v1.json")
        raw, packet = ab.load_queries(path)
        self.assertEqual(ab.byte_sha(raw), "fb5c31f20b5ff25756551f3e8de208c8bc269a953320f596f33a565218d7a5e4")
        self.assertEqual(Counter(q["kind"] for q in packet["queries"]), {"literal": 7, "scene": 9, "mixed": 8})
        self.assertEqual(sum(j["label"] for q in packet["queries"] for j in q["judgments"]), 30)
        self.assertEqual(len({q["target"]["video_id"] for q in packet["queries"]}), 12)
        protocol = Path(packet["protocol_path"])
        self.assertEqual(ab.digest(protocol), packet["protocol_sha256"])
        self.assertEqual(ab.byte_sha(ab.json_bytes(json.loads(protocol.read_bytes())["scoring"])), packet["scoring_config_sha256"])
        bad = copy.deepcopy(packet)
        bad["protocol_sha256"] = "0" * 64
        with patch.object(ab, "load_corpus", side_effect=AssertionError("no corpus before authentication")):
            with self.assertRaisesRegex(ValueError, "protocol/config hash"):
                ab.run_fusion(self.root, path, self.base / "unused", raw, bad)

    def test_legacy_evaluator_two_arm_equivalence_and_prior_frozen_metrics(self):
        docs, _, _ = ab.load_corpus(self.root)
        rows, metrics, _ = ab.evaluate(docs, self.packet["queries"])
        fusion, fm, _ = ab.evaluate_fusion(docs, [], self.packet["queries"])
        for old, new in (("baseline", "caption"), ("caption_plus_ocr", "caption_plus_ocr")):
            self.assertEqual(metrics["all"][old], fm["dev-reference/all"][new]["target"])
            self.assertEqual(rows[0][old]["rank"], fusion[0][new]["target"]["rank"])
        prior = json.loads(Path("benchmark_results/l04_ocr_ab_20261111_run02/report.json").read_bytes())
        self.assertEqual(prior["queries_sha256"], ab.DEV_SHA)
        self.assertEqual(prior["metrics"]["all"]["baseline"]["mrr@10"], 0.09598214285714285)
        self.assertEqual(prior["metrics"]["all"]["caption_plus_ocr"]["mrr@10"], 0.1875)

    def test_declared_18_selection_and_legacy_defaults(self):
        protocol = json.loads(Path("benchmarks/l04_ocr_fusion_new_queries_20261111_v2.protocol.json").read_bytes())
        names = {f"L04_V{i:03}": list(range(1, 31)) for i in range(1, 31)}
        plan = ab.selection_indices(names, names, protocol)
        self.assertEqual(len(plan), 18)
        self.assertEqual([p["target"][0] for p in plan[::2]],
                         [f"L04_V{n:03}" for n in (3, 5, 9, 11, 15, 19, 21, 25, 28)])
        self.assertEqual(len(ab.selection_indices(names, names)), 24)
        bad = copy.deepcopy(protocol)
        bad["selection"]["test_queries"] = 24
        with self.assertRaisesRegex(ValueError, "two queries"):
            ab.selection_indices(names, names, bad)
        bad["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "Legacy protocol"):
            ab.selection_indices(names, names, bad)
        names["L04_V003"] = list(range(1, 11))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            ab.selection_indices(names, names, protocol)
        names["L04_V003"] = [1, 1] + list(range(2, 31))
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            ab.selection_indices(names, names, protocol)

    def test_fresh_packet_no_prior_queries_images_or_numeric_aliases(self):
        path = Path("benchmarks/l04_ocr_fusion_new_queries_20261111_v2.json")
        raw, packet = ab.load_queries(path)
        self.assertEqual(ab.byte_sha(raw), "980151780c72c53d21fe95f964ae37359b7f22e181af9cd2bbe0e9c61b1b3150")
        protocol = json.loads(Path(packet["protocol_path"]).read_bytes())
        self.assertEqual(ab.digest(Path(packet["protocol_path"])), packet["protocol_sha256"])
        self.assertEqual(ab.byte_sha(ab.json_bytes(protocol["scoring"])), packet["scoring_config_sha256"])
        names = {f["video_id"]: list(range(1, f["image_count"] + 1))
                 for f in protocol["selection"]["frozen_plan"]}
        # Include all names without opening images/payloads.
        for vid in protocol["selection"]["eligible_videos"] + protocol["selection"]["excluded_videos"]:
            names.setdefault(vid, list(range(1, 31)))
        plan = ab.selection_indices(names, names, protocol)
        priors = [q for ref in protocol["selection"]["prior_packets"]
                  for q in ab.load_queries(ref["path"])[1]["queries"]]
        dev = ab.load_queries(ab.DEV_PATH)[1]["queries"]
        ab.validate_selection(packet, plan, dev, protocol, priors)
        self.assertEqual(Counter(q["kind"] for q in packet["queries"]), {"literal": 5, "scene": 7, "mixed": 6})
        self.assertEqual(sum(j["label"] for q in packet["queries"] for j in q["judgments"]), 28)
        bad = copy.deepcopy(packet)
        bad["queries"][0]["text"] = priors[0]["text"]
        with self.assertRaisesRegex(ValueError, "query literal reuse"):
            ab.validate_selection(bad, plan, dev, protocol, priors)
        # A differently padded source stem is the SAME prior numeric identity.
        prior = copy.deepcopy(priors[0])
        prior["judgments"] = [dict(packet["queries"][0]["judgments"][0])]
        prior["judgments"][0]["source_image"] = "dataset/data/L04/L04_V003/71.jpg"
        with self.assertRaisesRegex(ValueError, "inspected image reuse"):
            ab.validate_selection(packet, plan, dev, protocol, [prior])
        bad = copy.deepcopy(packet)
        bad["queries"][0]["judgments"].pop()
        with self.assertRaisesRegex(ValueError, "neighborhood"):
            ab.validate_selection(bad, plan, dev, protocol, priors)

    def test_negative_qrel_hash_and_frame_tamper(self):
        docs, _, _ = ab.load_corpus(self.root)
        by_key = {ab.doc_key(doc): doc for doc in docs}
        for updates in ({"image_sha256": "0" * 64}, {"frame_id": 2}):
            query = self.fusion_query()
            self.assertEqual(query["judgments"][2]["label"], 0)
            query["judgments"][2].update(updates)
            with self.assertRaisesRegex(ValueError, "Judgment image/hash/mapping"):
                ab.validate_judgments(query, by_key)

    def test_concat_comparison_and_stable_subset_metrics(self):
        docs, _, _ = ab.load_corpus(self.root)
        queries = [self.fusion_query(kind) for kind in ("literal", "scene", "mixed")]
        for i, query in enumerate(queries):
            query["id"] = f"s{i}"
            query["language"] = "English" if i else "English+Vietnamese"
        rows, metrics, _ = ab.evaluate_fusion(docs, queries, [])
        self.assertEqual(metrics, ab.aggregate_fusion(list(reversed(rows))))
        for comparison in ("caption_plus_ocr_vs_caption", "rrf_vs_caption"):
            for basis in ("target", "known_positive"):
                total = metrics["test/all"][comparison][basis]
                self.assertEqual(sum(total.values()), 3)
                for outcome in ("win", "tie", "loss"):
                    self.assertEqual(total[outcome], sum(metrics["test/kind/" + kind][comparison][basis][outcome]
                                                         for kind in ("literal", "scene", "mixed")))

    @staticmethod
    def review_correction_fixture():
        raw, packet = ab.load_queries("benchmarks/l04_ocr_fusion_new_queries_20261111_v3.json")
        protocol = json.loads(Path(packet["protocol_path"]).read_bytes())
        return raw, packet, protocol

    @staticmethod
    def review_raw(packet):
        # Match the original frozen packets' CRLF serialization for byte-block checks.
        return ab.json_bytes(packet).replace(b"\n", b"\r\n")

    def test_review_correction_freeze_preservation_and_same_selection(self):
        raw, packet, protocol = self.review_correction_fixture()
        self.assertEqual(ab.byte_sha(raw), "af394dbaa95fa1baa39a6721f62617134fb6a3412a44417fe31a9edfa6a707e8")
        self.assertEqual(ab.digest(Path(packet["protocol_path"])), packet["protocol_sha256"])
        self.assertEqual(len(protocol["preservation_manifest"]), 28)
        for entry in protocol["preservation_manifest"]:
            self.assertEqual(ab.digest(Path(entry["path"])), entry["sha256"])
        audit = ab.validate_review_correction(raw, packet, protocol)
        self.assertTrue(audit["unchanged_query_bytes_verified"])
        self.assertIn("NOT a fresh held-out", audit["disclaimer"])
        old_protocol = json.loads(Path(audit["original_protocol"]["path"]).read_bytes())
        names = {f"L04_V{i:03}": list(range(1, 31)) for i in range(1, 31)}
        self.assertEqual(ab.selection_indices(names, names, protocol), ab.selection_indices(names, names, old_protocol))
        self.assertEqual(ab.review_query_counts(packet["queries"]), audit["after_counts"])
        self.assertEqual(audit["after_counts"]["kind"], {"literal": 5, "scene": 8, "mixed": 5})
        for qid in ("t06", "t10"):
            query = next(q for q in packet["queries"] if q["id"] == qid)
            self.assertEqual([j["label"] for j in query["judgments"]], [0, 0, 1, 0, 0])
            self.assertEqual(query["kind"], "scene")
            self.assertEqual(query["judgments"][2]["visible_clock"], "18:44:52" if qid == "t06" else "18:43:23")

    def test_review_correction_rejects_third_query_or_target_qrel_shift(self):
        for field in ("third_query", "target", "judged_identity"):
            _, packet, protocol = self.review_correction_fixture()
            if field == "third_query":
                packet["queries"][0]["text"] += " changed"
                expected = "Only t06/t10"
            elif field == "target":
                packet["queries"][5]["target"]["keyframe_n"] += 1
                expected = "target identity/evidence"
            else:
                packet["queries"][9]["judgments"][3]["keyframe_n"] += 1
                expected = "inspected identity/evidence"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                ab.validate_review_correction(self.review_raw(packet), packet, protocol)

    def test_review_correction_requires_disclaimer_counts_and_exact_subset(self):
        for field in ("disclaimer", "subset", "after_counts"):
            raw, packet, protocol = self.review_correction_fixture()
            if field == "disclaimer":
                protocol["correction"][field] = "fresh held-out"
                expected = "post-result review disclaimer"
            elif field == "subset":
                protocol["correction"][field]["query_ids"][0] = "t06"
                expected = "subset IDs"
            else:
                protocol["correction"][field]["positive"] += 1
                expected = "count mismatch"
            packet["authoring"]["correction"] = copy.deepcopy(protocol["correction"])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                ab.validate_review_correction(raw, packet, protocol)

    def test_originalunchanged16_metrics_reproducible_and_not_pooled(self):
        _, _, protocol = self.review_correction_fixture()
        ids = protocol["correction"]["subset"]["query_ids"]
        rows = json.loads(Path(protocol["correction"]["original_report"]["path"]).read_bytes())["per_query"]
        metrics = ab.original_unchanged_metrics(rows, ids)
        self.assertEqual(metrics["originalunchanged16/all"]["queries"], 16)
        self.assertNotIn("test/all", metrics)
        self.assertFalse(any(name.startswith("dev-reference/") for name in metrics))
        selected = [r for r in rows if r["split"] == "test" and r["id"] in ids]
        for arm in ab.ARMS:
            for basis in ("known_positive", "target"):
                for metric in ab.METRICS:
                    self.assertEqual(metrics["originalunchanged16/all"][arm][basis][metric],
                                     sum(r[arm][basis][metric] for r in selected) / 16)
        for bad_ids in (ids[:-1], ids[:-1] + [ids[0]], ids[:-1] + ["not-a-query"]):
            with self.assertRaises(ValueError):
                ab.original_unchanged_metrics(rows, bad_ids)


if __name__ == "__main__":
    unittest.main()
