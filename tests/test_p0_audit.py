import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kis.temporal import TemporalConfig
from tools.p0_audit import (
    AuditError,
    _aggregate_stages,
    _artifact_detail,
    _artifact_status,
    _captured_model_config,
    _coverage_report,
    _index_root,
    _integer,
    _load_context,
    _normalized_path_map,
    _number,
    _read_manifest,
    _read_mapping,
    _saved_sequence_hits,
    _temporal_feasibility,
    _validate_raw_evidence,
    main,
    run,
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_NO_RECORD = object()


def _config():
    return {
        "status": "ok",
        "visual_embedding": {"dimension": 4},
        "caption_embedding": {
            "model": "caption-model",
            "revision": "caption-revision",
            "dimension": 2,
            "normalized": True,
            "metric": "IP",
        },
        "fast_caption_generation": {
            "model": "caption-model",
            "revision": "caption-revision",
        },
    }


class NumericParsingTest(unittest.TestCase):
    def test_integer_accepts_integer_strings_and_integral_floats(self):
        self.assertEqual(_integer("93", "n"), 93)
        self.assertEqual(_integer(" 93 ", "n"), 93)
        self.assertEqual(_integer("+93", "n"), 93)
        self.assertEqual(_integer(93.0, "n"), 93)

    def test_integer_rejects_bool_fraction_non_finite_and_missing(self):
        values = (True, False, 1.5, "1.5", "1e3", "nan", "inf", float("nan"), None)
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _integer(value, "n")

    def test_number_rejects_bool_and_non_finite(self):
        self.assertEqual(_number(" 31.5 ", "pts_time"), 31.5)
        for value in (True, False, "nan", "inf", float("nan"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _number(value, "pts_time")


class MappingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def mapping(self, text):
        path = self.root / "L01_V001.csv"
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def test_csv_numeric_strings_are_parsed(self):
        detail, frames = _read_mapping(self.mapping(
            'n,pts_time,fps,frame_idx\n"1","0.0","25.0","0"\n"2","4.48","25.0","112"\n'
        ))
        self.assertEqual(detail["status"], "ok")
        self.assertEqual(detail["rows"], 2)
        self.assertTrue(detail["order_valid"])
        self.assertEqual(frames[0], {"frame_id": 0, "keyframe_n": 1, "pts_time": 0.0, "fps": 25.0})
        self.assertEqual(frames[1]["frame_id"], 112)

    def test_fraction_and_non_finite_csv_numbers_are_rejected(self):
        for text in (
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0.5\n",
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,nan\n",
            "n,pts_time,fps,frame_idx\n1,nan,25.0,0\n",
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,True\n",
        ):
            with self.subTest(text=text):
                detail, frames = _read_mapping(self.mapping(text))
                self.assertEqual(detail["status"], "invalid")
                self.assertEqual(frames, [])
                self.assertTrue(detail["issues"])


class ManifestHashTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manifest_mapping_hash_uses_caption_mapping_json(self):
        caption_mapping = self.root / "caption_mapping.json"
        _write_json(caption_mapping, [{"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1}])
        caption_mapping_sha = _sha256(caption_mapping)
        caption_sha, index_sha = "a" * 64, "b" * 64
        manifest = self.root / "L01_V001.meta.json"
        _write_json(manifest, {
            "schema_version": 1,
            "video_id": "L01_V001",
            "count": 1,
            "files": {
                "caption_sha256": caption_sha,
                "mapping_sha256": caption_mapping_sha,
                "index_sha256": index_sha,
            },
            "embedding": {},
        })
        actual_hashes = {
            "caption_sha256": caption_sha,
            "caption_mapping_sha256": caption_mapping_sha,
            "index_sha256": index_sha,
            "mapping_csv_sha256": "not-the-caption-mapping-hash",
        }
        detail = _read_manifest(manifest, "L01_V001", actual_hashes, 1, None)
        self.assertEqual(detail["status"], "ok")
        self.assertEqual(detail["issues"], [])

    def test_manifest_mapping_hash_mismatch_is_invalid(self):
        caption_sha, index_sha = "c" * 64, "d" * 64
        manifest = self.root / "L01_V001.meta.json"
        _write_json(manifest, {
            "schema_version": 1,
            "video_id": "L01_V001",
            "count": 1,
            "files": {
                "caption_sha256": caption_sha,
                "mapping_sha256": "a" * 64,
                "index_sha256": index_sha,
            },
            "embedding": {},
        })
        detail = _read_manifest(
            manifest,
            "L01_V001",
            {
                "caption_sha256": caption_sha,
                "caption_mapping_sha256": "b" * 64,
                "index_sha256": index_sha,
            },
            1,
            None,
        )
        self.assertEqual(detail["status"], "invalid")
        self.assertIn("manifest_hash_mismatch", [issue["code"] for issue in detail["issues"]])

    def test_manifest_requires_all_file_hashes(self):
        manifest = self.root / "L01_V001.meta.json"
        _write_json(manifest, {
            "schema_version": 1,
            "video_id": "L01_V001",
            "count": 1,
            "files": {},
            "embedding": {},
        })
        detail = _read_manifest(manifest, "L01_V001", {}, 1, None)
        self.assertEqual(detail["status"], "invalid")
        missing = [issue["detail"] for issue in detail["issues"]
                   if issue["code"] == "manifest_hash_missing"]
        self.assertEqual(sorted(missing), ["caption_sha256", "index_sha256", "mapping_sha256"])

    def test_manifest_rejects_malformed_hash(self):
        manifest = self.root / "L01_V001.meta.json"
        _write_json(manifest, {
            "schema_version": 1,
            "video_id": "L01_V001",
            "count": 1,
            "files": {
                "caption_sha256": "not-a-hash",
                "mapping_sha256": "a" * 64,
                "index_sha256": "b" * 64,
            },
            "embedding": {},
        })
        detail = _read_manifest(manifest, "L01_V001", {}, 1, None)
        self.assertEqual(detail["status"], "invalid")
        self.assertIn("manifest_hash_invalid", [issue["code"] for issue in detail["issues"]])


class CapturedModelConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.models = {
            "visual_embedding": {"dimension": 4},
            "caption_embedding": {
                "model": "caption-model",
                "revision": "caption-revision",
                "dimension": 2,
                "normalized": True,
                "metric": "IP",
            },
            "fast_caption_generation": {
                "model": "caption-model",
                "revision": "caption-revision",
            },
        }
        self.models_path = self.root / "config" / "models.json"
        _write_json(self.models_path, self.models)

    def tearDown(self):
        self.temp_dir.cleanup()

    def context(self, digest):
        return {"protocol": {"source_sha256": {"config/models.json": digest}}}

    def test_hash_matching_local_config_is_verified(self):
        with patch("tools.p0_audit.KIS_ROOT", self.root):
            result = _captured_model_config(self.context(_sha256(self.models_path)))
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["caption_embedding"], self.models["caption_embedding"])
        self.assertEqual(result["visual_embedding"], self.models["visual_embedding"])

    def test_hash_mismatch_is_unverified_and_has_no_expected_models(self):
        with patch("tools.p0_audit.KIS_ROOT", self.root):
            result = _captured_model_config(self.context("0" * 64))
        self.assertEqual(result["status"], "unverified")
        self.assertIsNone(result["caption_embedding"])
        self.assertIsNone(result["visual_embedding"])

    def test_absent_hash_is_unknown(self):
        with patch("tools.p0_audit.KIS_ROOT", self.root):
            result = _captured_model_config({})
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["caption_embedding"])

    def test_malformed_hash_is_invalid(self):
        with patch("tools.p0_audit.KIS_ROOT", self.root):
            result = _captured_model_config(self.context("not-a-hash"))
        self.assertEqual(result["status"], "invalid")


class PathNormalizationTest(unittest.TestCase):
    def test_saved_caption_backslashes_are_normalized(self):
        self.assertEqual(
            _normalized_path_map({"caption_generator\\L01\\L01_V001.json": "hash"}),
            {"caption_generator/L01/L01_V001.json": "hash"},
        )


class ArtifactStatusTest(unittest.TestCase):
    def test_artifact_status_aggregates_issues_and_unknowns(self):
        self.assertEqual(_artifact_status(["ok"], []), "ok")
        self.assertEqual(_artifact_status(["ok"], [{"code": "x", "detail": "y"}]), "issues")
        self.assertEqual(_artifact_status(["ok", "missing"], []), "issues")
        self.assertEqual(_artifact_status(["ok", "ambiguous"], []), "issues")
        self.assertEqual(_artifact_status(["ok", "unknown"], []), "unknown")
        self.assertEqual(_artifact_status([], []), "unknown")


class ArtifactDetailTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duplicate_mapping_artifacts_are_ambiguous_and_aggregate(self):
        mapping_root = self.root / "mapping"
        for group in ("L01", "L01a"):
            path = mapping_root / group / "L01_V001.csv"
            path.parent.mkdir(parents=True)
            path.write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8")
        caption_root = self.root / "caption"
        roots = {
            "mapping": mapping_root,
            "features": caption_root / "clip-features-32",
            "captions": caption_root / "caption_generator",
            "caption_mapping": caption_root / "caption_mapping",
            "index": caption_root / "index",
        }
        indexes = {name: _index_root(path) for name, path in roots.items()}
        detail, identity = _artifact_detail(
            "L01_V001",
            roots,
            indexes,
            ("L01",),
            {"status": "unknown", "reason": "test"},
            None,
            None,
            None,
            [],
        )
        self.assertEqual(detail["artifacts"]["mapping"]["status"], "ambiguous")
        self.assertEqual(detail["status"], "issues")
        self.assertTrue(any(issue["code"].endswith("duplicate_mapping_artifact")
                            for issue in detail["issues"]))
        self.assertEqual(identity["mapping"], {})

    def test_complete_identity_mismatch_marks_artifact_issue(self):
        mapping_root = self.root / "mapping"
        path = mapping_root / "L01_V001.csv"
        path.parent.mkdir(parents=True)
        path.write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8")
        caption_root = self.root / "caption"
        caption_path = caption_root / "caption_generator" / "L01" / "L01_V001.json"
        _write_json(caption_path, [{
            "video_id": "L01_V001",
            "frame_id": 0,
            "keyframe_n": 2,
            "image_sha256": "a" * 64,
        }])
        caption_mapping_path = caption_root / "caption_mapping" / "L01" / "L01_V001.json"
        _write_json(caption_mapping_path, [{"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1}])
        roots = {
            "mapping": mapping_root,
            "features": caption_root / "clip-features-32",
            "captions": caption_root / "caption_generator",
            "caption_mapping": caption_root / "caption_mapping",
            "index": caption_root / "index",
        }
        indexes = {name: _index_root(root) for name, root in roots.items()}
        detail, _ = _artifact_detail(
            "L01_V001",
            roots,
            indexes,
            ("L01",),
            {"status": "unknown", "reason": "test"},
            None,
            None,
            None,
            [],
        )
        self.assertFalse(detail["identity"]["mapping_vs_caption_generator"])
        self.assertEqual(detail["status"], "issues")
        self.assertIn("caption_generator_identity_sequence_mismatch",
                      [issue["code"] for issue in detail["issues"]])


class SequencePathTest(unittest.TestCase):
    def setUp(self):
        self.sequence = {
            "query_id": "sample_sequence",
            "video_id": "L01_V001",
            "actions": ["first", "second"],
            "event_relevant_keyframes": [[1], [2]],
        }

    def positives(self, event_positives=None):
        event_positives = event_positives or (
            [{"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1,
              "pts_time": 0.0, "event_index": 0}],
            [{"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2,
              "pts_time": 4.0, "event_index": 1}],
        )
        return [positive for group in event_positives for positive in group]

    def result(self, status="retrieved_complete", coverage=1.0, video_id="L01_V001",
               frame_ids=(0, 1), keyframes=(1, 2), times=(0.0, 4.0)):
        events = [{
            "status": "retrieved",
            "video_id": video_id,
            "frame_id": frame_id,
            "keyframe_n": keyframe,
            "pts_time": pts_time,
        } for frame_id, keyframe, pts_time in zip(frame_ids, keyframes, times)]
        return {"video_id": "L01_V001",
                "temporal": {"status": status, "coverage": coverage, "events": events}}

    def raw_record(self):
        return {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.0},
            ],
            "frame_index": {0: 0, 1: 1},
        }

    def hits(self, result, record=_NO_RECORD, positives=None, mode="baseline"):
        if record is _NO_RECORD:
            record = self.raw_record()
        issues = []
        hits, unresolved = _saved_sequence_hits(
            [result], self.sequence, record, TemporalConfig(),
            "sample_sequence", mode, issues,
            self.positives() if positives is None else positives)
        return hits, unresolved, issues

    def test_complete_non_fallback_path_is_a_hit(self):
        for mode in ("baseline", "semantic_guard"):
            with self.subTest(mode=mode):
                hits, unresolved, issues = self.hits(self.result(), mode=mode)
                self.assertEqual(hits, {1: True, 5: True, 20: True})
                self.assertIsNone(unresolved)
                self.assertEqual(issues, [])

    def test_raw_actions_mismatch_is_unresolved_before_path_evaluation(self):
        for actions in (["second", "first"], ["first", "different"]):
            for mode in ("baseline", "semantic_guard"):
                with self.subTest(actions=actions, mode=mode):
                    record = dict(self.raw_record(), actions=actions)
                    with patch("tools.p0_audit._sequence_path_problems",
                               return_value=([], True)) as evaluate:
                        hits, unresolved, issues = self.hits(
                            self.result(), record, mode=mode)
                    self.assertEqual(hits, {1: None, 5: None, 20: None})
                    self.assertEqual(unresolved, "raw_evidence_actions_mismatch")
                    self.assertEqual(issues, [])
                    evaluate.assert_not_called()

    def test_fallback_path_is_not_a_hit(self):
        hits, unresolved, issues = self.hits(self.result(status="fallback"))
        self.assertEqual(hits, {1: False, 5: False, 20: False})
        self.assertIsNone(unresolved)
        self.assertEqual(issues, [])

    def test_zero_coverage_path_is_not_a_hit_and_creates_issue(self):
        hits, unresolved, issues = self.hits(self.result(coverage=0.0))
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])

    def test_coverage_must_be_exactly_one_on_both_sides(self):
        for coverage in (0.9999999995, 1.0000000005):
            for mode in ("baseline", "semantic_guard"):
                with self.subTest(coverage=coverage, mode=mode):
                    hits, unresolved, issues = self.hits(
                        self.result(coverage=coverage), mode=mode)
                    self.assertEqual(hits, {1: False, 5: False, 20: False})
                    self.assertIsNone(unresolved)
                    self.assertEqual(len(issues), 1)
                    self.assertEqual(issues[0]["code"], "saved_sequence_path_invalid")
                    self.assertIn("temporal coverage is not 1", issues[0]["detail"])

    def test_reversed_timestamps_are_invalid(self):
        hits, unresolved, issues = self.hits(self.result(times=(4.0, 0.0)))
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertTrue(any("strictly increasing" in issue["detail"] for issue in issues))

    def test_negative_frame_id_is_invalid(self):
        hits, unresolved, issues = self.hits(self.result(frame_ids=(-1, 1)))
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])

    def test_gap_above_captured_config_is_invalid(self):
        hits, unresolved, issues = self.hits(self.result(times=(0.0, 61.0)))
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertTrue(any("max_gap_seconds" in issue["detail"] for issue in issues))

    def test_keyframe_outside_explicit_positive_is_not_a_hit(self):
        record = self.raw_record()
        record["frames"][1]["keyframe_n"] = 3
        hits, unresolved, issues = self.hits(self.result(keyframes=(1, 3)), record)
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])
        self.assertTrue(any("does not match an explicit positive" in issue["detail"]
                            for issue in issues))

    def test_local_evidence_identity_mismatch_is_invalid(self):
        record = {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 2, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.0},
            ],
            "frame_index": {0: 0, 1: 1},
        }
        hits, unresolved, issues = self.hits(self.result(), record)
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertTrue(any("local evidence" in issue["detail"] for issue in issues))

    def test_raw_and_saved_same_wrong_frame_identity_is_invalid(self):
        record = {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 5, "keyframe_n": 1, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.0},
            ],
            "frame_index": {5: 0, 1: 1},
        }
        hits, unresolved, issues = self.hits(self.result(frame_ids=(5, 1)), record)
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])
        self.assertTrue(any("does not match an explicit positive" in issue["detail"]
                            for issue in issues))

    def test_raw_and_saved_same_wrong_timestamp_is_invalid_even_with_positive_keyframe(self):
        record = {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.5},
            ],
            "frame_index": {0: 0, 1: 1},
        }
        hits, unresolved, issues = self.hits(self.result(times=(0.0, 4.5)), record)
        self.assertIs(hits[20], False)
        self.assertIsNone(unresolved)
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])
        self.assertTrue(any("does not match an explicit positive" in issue["detail"]
                            for issue in issues))

    def test_missing_raw_evidence_is_unresolved_even_for_wrong_identity(self):
        hits, unresolved, issues = self.hits(self.result(frame_ids=(5, 1)), None)
        self.assertEqual(hits, {1: None, 5: None, 20: None})
        self.assertEqual(unresolved, "raw_evidence_missing")
        self.assertEqual(issues, [])

    def test_missing_raw_evidence_keeps_canonical_path_unresolved(self):
        hits, unresolved, issues = self.hits(self.result(), None)
        self.assertEqual(hits, {1: None, 5: None, 20: None})
        self.assertEqual(unresolved, "raw_evidence_missing")
        self.assertEqual(issues, [])

    def test_invalid_raw_evidence_is_unresolved_not_a_hit(self):
        record = self.raw_record()
        record["status"] = "invalid"
        hits, unresolved, issues = self.hits(self.result(), record)
        self.assertEqual(hits, {1: None, 5: None, 20: None})
        self.assertEqual(unresolved, "raw_evidence_invalid")
        self.assertEqual(issues, [])

    def test_missing_alternative_positive_does_not_hide_a_valid_path(self):
        positives = self.positives((
            [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1,
                 "pts_time": 0.0, "event_index": 0},
                {"video_id": "L01_V001", "frame_id": 99, "keyframe_n": 3,
                 "pts_time": 2.0, "event_index": 0},
            ],
            [{"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2,
              "pts_time": 4.0, "event_index": 1}],
        ))
        record = {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.0},
            ],
            "frame_index": {0: 0, 1: 1},
        }
        hits, unresolved, issues = self.hits(self.result(), record, positives)
        self.assertEqual(hits, {1: True, 5: True, 20: True})
        self.assertIsNone(unresolved)
        self.assertEqual(issues, [])


class SequenceCoverageIdentityTest(unittest.TestCase):
    def setUp(self):
        self.sequence = {
            "query_id": "sample_sequence",
            "video_id": "L01_V001",
            "event_query_ids": ["sample_event_1", "sample_event_2"],
            "actions": ["first", "second"],
            "event_relevant_keyframes": [[1], [2]],
        }
        self.event_by_id = {
            "sample_event_1": {
                "event_text": "first",
                "_positives": [{"video_id": "L01_V001", "frame_id": 0,
                                "keyframe_n": 1, "pts_time": 0.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
            "sample_event_2": {
                "event_text": "second",
                "_positives": [{"video_id": "L01_V001", "frame_id": 1,
                                "keyframe_n": 2, "pts_time": 4.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
        }

    def row(self):
        events = [
            {"video_id": "L01_V001", "frame_id": 5, "keyframe_n": 1,
             "pts_time": 0.0, "status": "retrieved"},
            {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2,
             "pts_time": 4.0, "status": "retrieved"},
        ]
        metrics = {
            "known_path_hit_at_1": 0.0,
            "known_path_hit_at_5": 0.0,
            "known_path_hit_at_20": 0.0,
            "top_target_video": True,
        }
        return {
            "query_id": "sample_sequence",
            "task": "sequence",
            "query": {
                "query_id": "sample_sequence",
                "video_id": "L01_V001",
                "event_query_ids": ["sample_event_1", "sample_event_2"],
                "actions": ["first", "second"],
                "event_relevant_keyframes": [[1], [2]],
            },
            "events": [{"action": "first"}, {"action": "second"}],
            "candidate_pool": {
                "strategy": "round_robin",
                "rrf_k": 60,
                "max_videos": 12,
                "ranked_videos": [{"video_id": "L01_V001", "selected": True}],
            },
            "candidate_videos": ["L01_V001"],
            "target_video_in_pool": True,
            "modes": {
                mode: {
                    "results": [{
                        "video_id": "L01_V001",
                        "frame_id": 5,
                        "keyframe_n": 1,
                        "pts_time": 0.0,
                        "temporal": {
                            "status": "retrieved_complete",
                            "coverage": 1.0,
                            "events": events,
                        },
                    }],
                    "metrics": metrics,
                }
                for mode in ("baseline", "semantic_guard")
            },
        }

    def raw(self):
        frames = [
            {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1, "pts_time": 0.0},
            {"video_id": "L01_V001", "frame_id": 5, "keyframe_n": 1, "pts_time": 0.0},
            {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 4.0},
        ]
        return {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": frames,
            "frame_index": {frame["frame_id"]: index for index, frame in enumerate(frames)},
            "visual": [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]],
            "semantic": None,
        }

    def canonical_row(self):
        row = self.row()
        for event in row["events"]:
            event["streams"] = {stream: [] for stream in ("visual", "bm25", "semantic")}
        for mode in ("baseline", "semantic_guard"):
            result = row["modes"][mode]["results"][0]
            result["frame_id"] = 0
            result["temporal"]["events"][0]["frame_id"] = 0
            for cutoff in (1, 5, 20):
                row["modes"][mode]["metrics"][f"known_path_hit_at_{cutoff}"] = 1.0
        return row

    def test_canonical_path_with_matching_raw_actions_is_a_hit_in_both_modes(self):
        issues = []
        report = _coverage_report(
            "sample_sequence", "sequence", None, self.sequence, self.canonical_row(),
            self.event_by_id, {("sample_sequence", "L01_V001"): self.raw()},
            TemporalConfig(), {"L01"}, issues, {})
        self.assertEqual(issues, [])
        for mode in ("baseline", "semantic_guard"):
            self.assertIs(report["modes"][mode]["saved_metrics_match"], True)
            for stage_name in ("saved_top_1", "saved_top_5", "saved_top_20"):
                saved = report["modes"][mode]["stages"][stage_name]
                self.assertEqual(saved["status"], "known")
                self.assertEqual(saved["covered"], 1)
                self.assertEqual(saved["total"], 1)

    def test_raw_actions_mismatch_keeps_saved_cutoffs_and_aggregate_unresolved(self):
        for actions in (["second", "first"], ["first", "different"]):
            with self.subTest(actions=actions):
                record = dict(self.raw(), actions=actions)
                issues = []
                report = _coverage_report(
                    "sample_sequence", "sequence", None, self.sequence, self.canonical_row(),
                    self.event_by_id, {("sample_sequence", "L01_V001"): record},
                    TemporalConfig(), {"L01"}, issues, {})
                self.assertEqual(report["status"], "unresolved")
                self.assertEqual(report["reason"], "raw_evidence_actions_mismatch")
                self.assertEqual(issues, [{
                    "code": "raw_evidence_actions_mismatch",
                    "detail": "sample_sequence/L01_V001",
                }])
                self.assert_saved_unresolved(report, "raw_evidence_actions_mismatch")

    def test_coverage_uses_frozen_full_identity_even_when_raw_agrees(self):
        issues = []
        report = _coverage_report(
            "sample_sequence", "sequence", None, self.sequence, self.row(),
            self.event_by_id, {("sample_sequence", "L01_V001"): self.raw()},
            TemporalConfig(), {"L01"}, issues, {})
        self.assertIn("saved_sequence_path_invalid",
                      [issue["code"] for issue in issues])
        for mode in ("baseline", "semantic_guard"):
            saved = report["modes"][mode]["stages"]["saved_top_1"]
            self.assertEqual(saved["status"], "known")
            self.assertEqual(saved["covered"], 0)
            self.assertEqual(saved["total"], 1)

    def assert_saved_unresolved(self, report, reason):
        aggregate = _aggregate_stages([report])
        for mode in ("baseline", "semantic_guard"):
            self.assertIsNone(report["modes"][mode]["saved_metrics_match"])
            for stage_name in ("saved_top_1", "saved_top_5", "saved_top_20"):
                stage = report["modes"][mode]["stages"][stage_name]
                self.assertEqual(stage["status"], "unresolved", (mode, stage_name))
                self.assertEqual(stage["covered"], 0, (mode, stage_name))
                self.assertEqual(stage["resolved"], 0, (mode, stage_name))
                self.assertEqual(stage["total"], 1, (mode, stage_name))
                self.assertEqual(stage["unresolved"], 1, (mode, stage_name))
                self.assertEqual(stage["reasons"], {reason: 1}, (mode, stage_name))
                saved = aggregate["sequence"][mode][stage_name]
                self.assertEqual(saved, {
                    "status": "unknown", "unit": "queries", "covered": 0,
                    "resolved": 0, "total": 1, "unresolved": 1, "queries": 1,
                    "queries_with_unresolved": 1, "not_applicable_queries": 0,
                    "coverage": None, "reasons": {reason: 1},
                }, (mode, stage_name))

    def test_missing_raw_evidence_leaves_saved_cutoffs_unresolved(self):
        report = _coverage_report(
            "sample_sequence", "sequence", None, self.sequence, self.row(),
            self.event_by_id, {}, TemporalConfig(), {"L01"}, [], {})
        self.assert_saved_unresolved(report, "raw_evidence_missing")

    def test_invalid_raw_evidence_leaves_saved_cutoffs_unresolved(self):
        record = self.raw()
        record["status"] = "invalid"
        report = _coverage_report(
            "sample_sequence", "sequence", None, self.sequence, self.row(),
            self.event_by_id, {("sample_sequence", "L01_V001"): record},
            TemporalConfig(), {"L01"}, [], {})
        self.assert_saved_unresolved(report, "raw_evidence_invalid")

    def test_unknown_raw_evidence_leaves_saved_cutoffs_unresolved(self):
        record = self.raw()
        record["status"] = "unknown"
        report = _coverage_report(
            "sample_sequence", "sequence", None, self.sequence, self.row(),
            self.event_by_id, {("sample_sequence", "L01_V001"): record},
            TemporalConfig(), {"L01"}, [], {})
        self.assert_saved_unresolved(report, "raw_evidence_invalid")

    def test_mixed_saved_sequence_outcomes_keep_aggregate_denominators(self):
        record = self.raw()
        record["frames"] = [record["frames"][0], record["frames"][2]]
        record["frame_index"] = {0: 0, 1: 1}
        record["visual"] = [[0.9, 0.9], [0.9, 0.9]]
        reports = []
        cases = (
            ("hit", record, True),
            ("miss", record, False),
            ("missing", None, True),
            ("invalid", dict(record, status="invalid"), True),
        )
        for name, raw, canonical in cases:
            query_id = f"{name}_sequence"
            sequence = dict(self.sequence, query_id=query_id)
            row = self.row()
            row["query_id"] = query_id
            row["query"]["query_id"] = query_id
            if canonical:
                for mode in ("baseline", "semantic_guard"):
                    result = row["modes"][mode]["results"][0]
                    result["frame_id"] = 0
                    result["temporal"]["events"][0]["frame_id"] = 0
                    for cutoff in (1, 5, 20):
                        row["modes"][mode]["metrics"][f"known_path_hit_at_{cutoff}"] = 1.0
            raw_evidence = {} if raw is None else {(query_id, "L01_V001"): raw}
            reports.append(_coverage_report(
                query_id, "sequence", None, sequence, row, self.event_by_id,
                raw_evidence, TemporalConfig(), {"L01"}, [], {}))

        aggregate = _aggregate_stages(reports)
        for mode in ("baseline", "semantic_guard"):
            for stage_name in ("saved_top_1", "saved_top_5", "saved_top_20"):
                saved = aggregate["sequence"][mode][stage_name]
                self.assertEqual(saved, {
                    "status": "partial", "unit": "queries", "covered": 1,
                    "resolved": 2, "total": 4, "unresolved": 2, "queries": 4,
                    "queries_with_unresolved": 2, "not_applicable_queries": 0,
                    "coverage": 0.5,
                    "reasons": {"raw_evidence_invalid": 1, "raw_evidence_missing": 1},
                }, (mode, stage_name))
            local = aggregate["sequence"][mode]["local_frames"]
            self.assertEqual(local["queries"], 4)
            self.assertEqual(local["total"], 8)
            self.assertEqual(local["resolved"], 4)
            self.assertEqual(local["unresolved"], 4)


class TemporalAlternativeTest(unittest.TestCase):
    def setUp(self):
        self.sequence = {
            "query_id": "sample_sequence",
            "video_id": "L01_V001",
            "actions": ["first", "second"],
            "event_relevant_keyframes": [[1, 2], [3]],
        }
        self.record = {
            "status": "ok",
            "actions": ["first", "second"],
            "frames": [
                {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1, "pts_time": 0.0},
                {"video_id": "L01_V001", "frame_id": 1, "keyframe_n": 2, "pts_time": 2.0},
                {"video_id": "L01_V001", "frame_id": 2, "keyframe_n": 3, "pts_time": 5.0},
            ],
            "frame_index": {0: 0, 1: 1, 2: 2},
            "visual": [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]],
            "semantic": None,
        }
        self.positives = [
            {"video_id": "L01_V001", "frame_id": 99, "keyframe_n": 2,
             "pts_time": 2.0, "event_index": 0},
            {"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1,
             "pts_time": 0.0, "event_index": 0},
            {"video_id": "L01_V001", "frame_id": 2, "keyframe_n": 3,
             "pts_time": 5.0, "event_index": 1},
        ]

    def test_missing_alternative_does_not_hide_a_complete_path(self):
        issues = []
        feasible, reason, missing = _temporal_feasibility(
            self.record, self.sequence, self.positives, TemporalConfig(), "sample_sequence", issues)
        self.assertTrue(feasible)
        self.assertIsNone(reason)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["frame_id"], 99)
        self.assertIn("positive_local_frame_missing", [issue["code"] for issue in issues])

    def test_reordered_raw_actions_are_unresolved(self):
        record = dict(self.record, actions=["second", "first"])
        issues = []
        feasible, reason, missing = _temporal_feasibility(
            record, self.sequence, self.positives, TemporalConfig(), "sample_sequence", issues)
        self.assertIsNone(feasible)
        self.assertEqual(reason, "raw_evidence_actions_mismatch")
        self.assertEqual(missing, [])
        self.assertIn("raw_evidence_actions_mismatch", [issue["code"] for issue in issues])


class MissingRowStageTest(unittest.TestCase):
    def label(self):
        return {
            "query_id": "sample_event",
            "video_id": "L01_V001",
            "event_text": "first",
            "judgments": [],
            "_positives": [{"video_id": "L01_V001", "frame_id": 0,
                            "keyframe_n": 1, "pts_time": 0.0}],
            "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
        }

    def event_row(self, events):
        return {
            "query_id": "sample_event",
            "task": "event_frame",
            "query": {
                "query_id": "sample_event",
                "video_id": "L01_V001",
                "event_text": "first",
                "judgments": [],
            },
            "events": events,
        }

    def test_missing_row_keeps_unresolved_stage_units(self):
        issues = []
        report = _coverage_report(
            "sample_event", "event_frame", self.label(), None, None, {}, {},
            TemporalConfig(), {"L01"}, issues, {})
        self.assertEqual(report["status"], "missing_row")
        for mode in ("baseline", "semantic_guard"):
            stage = report["modes"][mode]["stages"]["saved_top_1"]
            self.assertEqual(stage["status"], "unresolved")
            self.assertEqual(stage["total"], 1)
            self.assertEqual(stage["unresolved"], 1)
            self.assertEqual(stage["reasons"], {"saved_row_missing": 1})
        stages = _aggregate_stages([report])
        saved = stages["event_frame"]["baseline"]["saved_top_1"]
        self.assertEqual(saved["queries"], 1)
        self.assertEqual(saved["total"], 1)
        self.assertEqual(saved["unresolved"], 1)
        self.assertEqual(saved["queries_with_unresolved"], 1)
        self.assertEqual(saved["reasons"], {"saved_row_missing": 1})

    def test_missing_sequence_row_uses_query_and_frame_denominators(self):
        sequence = {
            "query_id": "sample_sequence",
            "video_id": "L01_V001",
            "event_query_ids": ["sample_event_1", "sample_event_2"],
            "actions": ["first", "second"],
            "event_relevant_keyframes": [[1], [2]],
        }
        event_by_id = {
            "sample_event_1": {
                "_positives": [{"video_id": "L01_V001", "frame_id": 0,
                                "keyframe_n": 1, "pts_time": 0.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
            "sample_event_2": {
                "_positives": [{"video_id": "L01_V001", "frame_id": 1,
                                "keyframe_n": 2, "pts_time": 4.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
        }
        report = _coverage_report(
            "sample_sequence", "sequence", None, sequence, None, event_by_id, {},
            TemporalConfig(), {"L01"}, [], {})
        temporal = report["modes"]["baseline"]["stages"]["temporal_feasibility"]
        self.assertEqual(temporal["total"], 1)
        self.assertEqual(temporal["unresolved"], 1)
        self.assertEqual(temporal["reasons"], {"saved_row_missing": 1})
        local = report["modes"]["baseline"]["stages"]["local_frames"]
        self.assertEqual(local["total"], 2)
        self.assertEqual(local["unresolved"], 2)

    def assert_mismatch_stages(self, report, reason):
        self.assertEqual(report["status"], "label_mismatch")
        self.assertEqual(sorted(report["modes"]), ["baseline", "semantic_guard"])
        for mode in ("baseline", "semantic_guard"):
            stages = report["modes"][mode]["stages"]
            self.assertEqual(stages["temporal_feasibility"]["status"], "not_applicable")
            for stage in ("visual_hits", "bm25_hits", "semantic_hits", "union_hits",
                          "candidate_pool", "local_frames", "support",
                          "saved_top_1", "saved_top_5", "saved_top_20"):
                self.assertEqual(stages[stage]["status"], "unresolved", stage)
                self.assertEqual(stages[stage]["covered"], 0, stage)
                self.assertEqual(stages[stage]["resolved"], 0, stage)
                self.assertEqual(stages[stage]["total"], 1, stage)
                self.assertEqual(stages[stage]["unresolved"], 1, stage)
                self.assertEqual(stages[stage]["reasons"], {reason: 1}, stage)
        aggregate = _aggregate_stages([report])
        for mode in ("baseline", "semantic_guard"):
            saved = aggregate["event_frame"][mode]["saved_top_1"]
            self.assertEqual(saved["queries"], 1)
            self.assertEqual(saved["total"], 1)
            self.assertEqual(saved["resolved"], 0)
            self.assertEqual(saved["unresolved"], 1)
            self.assertEqual(saved["queries_with_unresolved"], 1)
            self.assertEqual(saved["reasons"], {reason: 1})
            frames = aggregate["event_frame"][mode]["local_frames"]
            self.assertEqual(frames["total"], 1)
            self.assertEqual(frames["unresolved"], 1)

    def test_task_mismatch_keeps_per_query_and_aggregate_denominators(self):
        row = self.event_row([{"action": "first"}])
        row["task"] = "sequence"
        issues = []
        report = _coverage_report(
            "sample_event", "event_frame", self.label(), None, row, {}, {},
            TemporalConfig(), {"L01"}, issues, {})
        self.assert_mismatch_stages(report, "saved_row_task_mismatch")
        self.assertIn("saved_row_task_mismatch",
                      [issue["code"] for issue in issues])

    def test_query_mismatch_keeps_per_query_and_aggregate_denominators(self):
        row = self.event_row([{"action": "first"}])
        row["query"]["event_text"] = "a different action"
        issues = []
        report = _coverage_report(
            "sample_event", "event_frame", self.label(), None, row, {}, {},
            TemporalConfig(), {"L01"}, issues, {})
        self.assert_mismatch_stages(report, "saved_row_query_mismatch")
        self.assertIn("saved_row_query_mismatch",
                      [issue["code"] for issue in issues])

    def test_event_action_mismatch_is_label_mismatch_without_stream_or_result_credit(self):
        for events in ("wrong", "missing", "malformed"):
            with self.subTest(events=events):
                if events == "wrong":
                    row = self.event_row([{"action": "a different action"}])
                elif events == "missing":
                    row = self.event_row([])
                else:
                    row = self.event_row([None])
                row["modes"] = {
                    mode: {
                        "results": [{"video_id": "L01_V001", "frame_id": 0,
                                     "keyframe_n": 1, "pts_time": 0.0}],
                        "metrics": {"known_frame_hit_at_1": 1.0},
                    }
                    for mode in ("baseline", "semantic_guard")
                }
                if events == "wrong":
                    row["events"][0]["streams"] = {
                        stream: [{"video_id": "L01_V001", "frame_id": 0}]
                        for stream in ("visual", "bm25", "semantic")
                    }
                issues = []
                report = _coverage_report(
                    "sample_event", "event_frame", self.label(), None, row, {}, {},
                    TemporalConfig(), {"L01"}, issues, {})
                self.assert_mismatch_stages(report, "saved_row_query_mismatch")
                self.assertEqual([issue["code"] for issue in issues],
                                 ["saved_row_query_mismatch"])


class AuditFixture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_run = self.root / "source"
        self.source_run.mkdir()
        self.raw_root = self.source_run / "raw_evidence"
        self.raw_root.mkdir()
        self.labels_path = self.root / "labels.json"
        self.mapping_root = self.root / "mapping"
        self.caption_root = self.root / "caption"
        self.output = self.root / "report.json"
        self.event_text = "A person holds a red object."
        self.positive_hash = "a" * 64
        self.judgments = [
            {
                "video_id": "L01_V001",
                "frame_id": 0,
                "keyframe_n": 1,
                "pts_time": 0.0,
                "image_sha256": self.positive_hash,
                "judgment": "positive",
                "sample_id": "drive_01",
            },
            {
                "video_id": "L01_V001",
                "frame_id": 1,
                "keyframe_n": 2,
                "pts_time": 4.0,
                "image_sha256": "b" * 64,
                "judgment": "negative",
                "sample_id": "drive_01",
            },
            {
                "video_id": "L01_V001",
                "frame_id": 2,
                "keyframe_n": 3,
                "pts_time": 8.0,
                "image_sha256": "c" * 64,
                "judgment": "unknown",
                "sample_id": "drive_01",
            },
        ]
        self.label_query = {
            "query_id": "drive_01_event_1",
            "split": "holdout",
            "video_id": "L01_V001",
            "event_text": self.event_text,
            "judgments": copy.deepcopy(self.judgments),
        }
        self.labels = {
            "schema": "kis_human_holdout_labels_v1",
            "annotation_source": "human_visual_observation",
            "human_verified": True,
            "caption_used_for_annotation": False,
            "queries": [self.label_query],
            "human_annotations": {"samples": []},
            "skipped_samples": [],
        }
        self._write_labels()
        self.context = {
            "mode": "frozen_human_holdout",
            "label_sha256": _sha256(self.labels_path),
            "annotation_source": "human_visual_observation",
            "human_verified": True,
            "temporal_config": TemporalConfig().public(),
            "video_groups": ["L01"],
            "caption_artifact_sha256": {},
        }
        self.mapping_sha = self._write_artifacts()
        self.context["caption_artifact_sha256"] = {
            "caption_generator/L01/L01_V001.json":
                _sha256(self.caption_root / "caption_generator" / "L01" / "L01_V001.json"),
        }
        self._write_context()
        self._write_row_and_evidence()
        self._write_local_artifact_hashes()
        self.current_config_patch = patch("tools.p0_audit._current_config", return_value=_config())
        self.faiss_patch = patch(
            "tools.p0_audit._faiss_status",
            return_value={"status": "unknown", "reason": "test environment"},
        )
        self.current_config_patch.start()
        self.faiss_patch.start()

    def tearDown(self):
        self.faiss_patch.stop()
        self.current_config_patch.stop()
        self.temp_dir.cleanup()

    def _write_labels(self):
        _write_json(self.labels_path, self.labels)

    def _write_context(self):
        _write_json(self.source_run / "context.json", self.context)

    def _write_artifacts(self):
        mapping_path = self.mapping_root / "L01" / "L01_V001.csv"
        mapping_path.parent.mkdir(parents=True)
        mapping_path.write_text(
            'n,pts_time,fps,frame_idx\n"1","0.0","25.0","0"\n',
            encoding="utf-8",
            newline="\n",
        )
        features_path = self.caption_root / "clip-features-32" / "L01" / "L01_V001.npy"
        features_path.parent.mkdir(parents=True)
        np.save(features_path, np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32))
        caption_path = self.caption_root / "caption_generator" / "L01" / "L01_V001.json"
        _write_json(caption_path, [{
            "video_id": "L01_V001",
            "frame_id": 0,
            "keyframe_n": 1,
            "image_path": "data/L01/L01_V001/001.jpg",
            "caption": "A person holds a red object.",
            "retrieval_text": "A person holds a red object.",
            "image_sha256": self.positive_hash,
            "caption_generation": {
                "backend": "test",
                "model": "caption-model",
                "revision": "caption-revision",
            },
        }])
        caption_mapping_path = self.caption_root / "caption_mapping" / "L01" / "L01_V001.json"
        _write_json(caption_mapping_path, [{
            "video_id": "L01_V001",
            "frame_id": 0,
            "keyframe_n": 1,
            "image_path": "data/L01/L01_V001/001.jpg",
        }])
        index_path = self.caption_root / "index" / "L01" / "L01_V001.index"
        index_path.parent.mkdir(parents=True)
        index_path.write_bytes(b"local-index-placeholder")
        _write_json(self.caption_root / "index" / "L01" / "L01_V001.meta.json", {
            "schema_version": 1,
            "video_id": "L01_V001",
            "count": 1,
            "files": {
                "caption_sha256": _sha256(caption_path),
                "mapping_sha256": _sha256(caption_mapping_path),
                "index_sha256": _sha256(index_path),
            },
            "embedding": {
                "model": "caption-model",
                "revision": "caption-revision",
                "dimension": 2,
                "normalized": True,
                "metric": "IP",
            },
        })
        return _sha256(mapping_path)

    def _write_row_and_evidence(self):
        query = copy.deepcopy(self.label_query)
        row = {
            "query_id": query["query_id"],
            "task": "event_frame",
            "split": "holdout",
            "same_pool_and_raw_scores": True,
            "query": query,
            "candidate_pool": {
                "strategy": "round_robin",
                "rrf_k": 60,
                "max_videos": 12,
                "ranked_videos": [{"video_id": "L01_V001", "selected": True}],
            },
            "candidate_videos": ["L01_V001"],
            "target_video_in_pool": True,
            "events": [{
                "action": self.event_text,
                "streams": {
                    "visual": [{"video_id": "L01_V001", "frame_id": 0, "score": 0.9}],
                    "bm25": [{"video_id": "L01_V001", "frame_id": 0, "score": 0.8}],
                    "semantic": [{"video_id": "L01_V001", "frame_id": 0, "score": 0.7}],
                },
            }],
            "modes": {},
            "failures": {},
        }
        result = [{"video_id": "L01_V001", "frame_id": 0, "keyframe_n": 1,
                   "pts_time": 0.0, "score": 0.9}]
        for mode in ("baseline", "semantic_guard"):
            row["modes"][mode] = {
                "results": copy.deepcopy(result),
                "metrics": {
                    "known_frame_hit_at_1": 1.0,
                    "known_frame_hit_at_5": 1.0,
                    "known_frame_hit_at_20": 1.0,
                    "top_target_video": True,
                },
            }
        _write_json(self.source_run / "drive_01_event_1.json", row)
        _write_json(self.source_run / "summary.json", {
            "label_sha256": _sha256(self.labels_path),
        })
        _write_json(self.raw_root / "drive_01_event_1_L01_V001.json", {
            "unit_id": "drive_01_event_1",
            "video_id": "L01_V001",
            "actions": [self.event_text],
            "frames": [{"video_id": "L01_V001", "frame_id": 0,
                        "keyframe_n": 1, "pts_time": 0.0}],
            "visual": [[0.9]],
            "semantic": [[0.9]],
            "stats": {},
        })

    def _write_local_artifact_hashes(self):
        features_path = self.caption_root / "clip-features-32" / "L01" / "L01_V001.npy"
        _write_json(self.source_run / "local_artifact_sha256.json", {
            "L01_V001": {
                "mapping_sha256": self.mapping_sha,
                "features_sha256": _sha256(features_path),
            },
        })

    def build_report(self, output=None):
        return run(self.source_run, self.labels_path, self.mapping_root,
                   self.caption_root, self.output if output is None else output)


class AuditRunTest(AuditFixture):
    def test_report_stages_provenance_and_guard_unknown(self):
        report = self.build_report()
        self.assertEqual(report["schema"], "kis_p0_audit_v1")
        self.assertEqual(report["summary"]["status"], "unknown")
        self.assertEqual(report["summary"]["query_status"], {"ok": 1})
        self.assertEqual(report["summary"]["issues"]["total"], 0)
        self.assertEqual(report["summary"]["counts"]["explicit_positive_frames"], 1)
        self.assertEqual(report["summary"]["counts"]["unknown_judgments"], 1)
        self.assertEqual(report["provenance"]["tool"]["models_loaded"], False)
        self.assertEqual(report["provenance"]["tool"]["network_used"], False)
        self.assertEqual(report["provenance"]["labels"]["sha256"], _sha256(self.labels_path))
        self.assertFalse(report["provenance"]["current_config_snapshot"]["historical"])
        for service in ("live_milvus", "planner", "vlm"):
            self.assertEqual(report["provenance"]["unknown_state"][service]["status"], "unknown")
        baseline = report["summary"]["stages"]["event_frame"]["baseline"]
        self.assertEqual(baseline["visual_hits"]["status"], "known")
        self.assertEqual(baseline["visual_hits"]["coverage"], 1.0)
        self.assertEqual(baseline["candidate_pool"]["coverage"], 1.0)
        self.assertEqual(baseline["local_frames"]["coverage"], 1.0)
        self.assertEqual(baseline["support"]["coverage"], 1.0)
        self.assertEqual(baseline["saved_top_1"]["coverage"], 1.0)
        per_query_guard = report["queries"][0]["modes"]["semantic_guard"]["stages"]["support"]
        self.assertEqual(per_query_guard["status"], "unresolved")
        guard = report["summary"]["stages"]["event_frame"]["semantic_guard"]
        self.assertEqual(guard["support"]["status"], "unknown")
        self.assertEqual(guard["support"]["resolved"], 0)
        self.assertIsNone(guard["support"]["coverage"])
        self.assertEqual(guard["support"]["reasons"], {"guard_mask_absent": 1})

    def test_unknown_current_config_does_not_invalidate_captured_artifacts(self):
        with patch("tools.p0_audit._current_config",
                   return_value={"status": "unknown", "reason": "test"}):
            report = self.build_report()
        self.assertEqual(report["summary"]["issues"]["total"], 0)
        artifact = report["artifacts"]["L01_V001"]
        self.assertEqual(artifact["artifacts"]["index_manifest"]["status"], "ok")
        self.assertEqual(artifact["artifacts"]["visual_features"]["status"], "ok")

    def test_output_is_deterministic_and_never_overwritten(self):
        first = self.root / "first.json"
        second = self.root / "second.json"
        run(self.source_run, self.labels_path, self.mapping_root, self.caption_root, first)
        run(self.source_run, self.labels_path, self.mapping_root, self.caption_root, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        before = first.read_bytes()
        with self.assertRaises(AuditError):
            run(self.source_run, self.labels_path, self.mapping_root, self.caption_root, first)
        self.assertEqual(first.read_bytes(), before)

    def test_main_returns_zero_for_report_with_audit_issues(self):
        (self.raw_root / "drive_01_event_1_L01_V001.json").unlink()
        output = self.root / "main.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main([
                "--source-run", str(self.source_run),
                "--labels", str(self.labels_path),
                "--mapping-root", str(self.mapping_root),
                "--caption-root", str(self.caption_root),
                "--output", str(output),
            ])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("raw_evidence_missing", report["summary"]["issues"]["by_code"])

    def test_main_returns_two_for_fatal_input(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = main([
                "--source-run", str(self.root / "missing"),
                "--labels", str(self.labels_path),
                "--mapping-root", str(self.mapping_root),
                "--caption-root", str(self.caption_root),
                "--output", str(self.root / "missing.json"),
            ])
        self.assertEqual(code, 2)
        self.assertIn("source_run_missing", stderr.getvalue())

    def test_missing_raw_evidence_is_unresolved_not_zero_denominator(self):
        (self.raw_root / "drive_01_event_1_L01_V001.json").unlink()
        report = self.build_report()
        stage = report["summary"]["stages"]["event_frame"]["baseline"]["local_frames"]
        self.assertEqual(stage["unit"], "positive_frames")
        self.assertEqual(stage["total"], 1)
        self.assertEqual(stage["resolved"], 0)
        self.assertEqual(stage["unresolved"], 1)
        self.assertIsNone(stage["coverage"])
        self.assertEqual(stage["reasons"], {"raw_evidence_missing": 1})
        self.assertIn("raw_evidence_missing", report["summary"]["issues"]["by_code"])

    def test_unknown_and_negative_judgments_do_not_form_a_denominator(self):
        for judgment in self.label_query["judgments"]:
            if judgment["judgment"] == "positive":
                judgment["judgment"] = "unknown"
        self._write_labels()
        self.context["label_sha256"] = _sha256(self.labels_path)
        self._write_context()
        _write_json(self.source_run / "summary.json", {
            "label_sha256": _sha256(self.labels_path),
        })
        report = self.build_report()
        self.assertEqual(report["summary"]["query_status"], {"no_explicit_positive": 1})
        self.assertEqual(report["summary"]["counts"]["explicit_positive_frames"], 0)
        stage = report["summary"]["stages"]["event_frame"]["baseline"]["visual_hits"]
        self.assertEqual(stage["total"], 0)
        self.assertEqual(stage["resolved"], 0)
        self.assertIsNone(stage["coverage"])
        self.assertEqual(stage["status"], "not_applicable")

    def test_out_of_scope_query_is_not_a_denominator(self):
        self.context["video_groups"] = ["L02"]
        self._write_context()
        report = self.build_report()
        self.assertEqual(report["summary"]["query_status"], {"out_of_scope": 1})
        self.assertEqual(report["summary"]["counts"]["explicit_positive_frames"], 0)
        stage = report["summary"]["stages"]["event_frame"]["baseline"]["visual_hits"]
        self.assertEqual(stage["total"], 0)
        self.assertIsNone(stage["coverage"])
        self.assertIn("query_out_of_scope", report["summary"]["issues"]["by_code"])

    def test_label_hash_mismatch_is_fatal(self):
        self.context["label_sha256"] = "0" * 64
        self._write_context()
        output = self.root / "mismatch.json"
        with self.assertRaisesRegex(AuditError, "label_hash_mismatch"):
            run(self.source_run, self.labels_path, self.mapping_root, self.caption_root, output)
        self.assertFalse(output.exists())

    def test_sequence_event_order_mismatch_is_label_mismatch(self):
        event_by_id = {
            "sample_event_1": {
                "_positives": [{"video_id": "L01_V001", "frame_id": 0,
                                "keyframe_n": 1, "pts_time": 0.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
            "sample_event_2": {
                "_positives": [{"video_id": "L01_V001", "frame_id": 1,
                                "keyframe_n": 2, "pts_time": 4.0}],
                "_judgment_counts": {"positive": 1, "negative": 0, "unknown": 0},
            },
        }
        sequence = {
            "query_id": "sample_sequence",
            "video_id": "L01_V001",
            "event_query_ids": ["sample_event_1", "sample_event_2"],
            "actions": ["first", "second"],
            "event_relevant_keyframes": [[1], [2]],
        }
        row = {
            "query_id": "sample_sequence",
            "task": "sequence",
            "query": {
                "query_id": "sample_sequence",
                "video_id": "L01_V001",
                "event_query_ids": ["sample_event_1", "sample_event_2"],
                "actions": ["second", "first"],
                "event_relevant_keyframes": [[1], [2]],
            },
            "events": [{"action": "second"}, {"action": "first"}],
        }
        issues = []
        report = _coverage_report(
            "sample_sequence", "sequence", None, sequence, row, event_by_id, {},
            TemporalConfig(), {"L01"}, issues, {},
        )
        self.assertEqual(report["status"], "label_mismatch")
        self.assertIn("sequence_event_order_mismatch",
                      [issue["code"] for issue in report["issues"]])
        for mode in ("baseline", "semantic_guard"):
            stages = report["modes"][mode]["stages"]
            self.assertEqual(stages["temporal_feasibility"]["total"], 1)
            self.assertEqual(stages["temporal_feasibility"]["unresolved"], 1)
            self.assertEqual(stages["temporal_feasibility"]["reasons"],
                             {"sequence_event_order_mismatch": 1})
            self.assertEqual(stages["local_frames"]["total"], 2)
            self.assertEqual(stages["local_frames"]["unresolved"], 2)
        aggregate = _aggregate_stages([report])
        self.assertEqual(aggregate["sequence"]["baseline"]["saved_top_1"]["queries"], 1)
        self.assertEqual(aggregate["sequence"]["baseline"]["saved_top_1"]["total"], 1)
        self.assertEqual(aggregate["sequence"]["baseline"]["local_frames"]["total"], 2)

    def test_raw_evidence_actions_mismatch_is_unresolved(self):
        path = self.raw_root / "drive_01_event_1_L01_V001.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["actions"] = ["A different action."]
        _write_json(path, data)
        report = self.build_report()
        self.assertEqual(report["queries"][0]["status"], "unresolved")
        self.assertEqual(report["summary"]["query_status"], {"unresolved": 1})
        support = report["queries"][0]["modes"]["baseline"]["stages"]["support"]
        self.assertEqual(support["status"], "unresolved")
        self.assertEqual(support["reasons"], {"raw_evidence_actions_mismatch": 1})
        self.assertIn("raw_evidence_actions_mismatch",
                      report["summary"]["issues"]["by_code"])

    def test_event_frame_action_mismatch_is_label_mismatch_in_full_run(self):
        path = self.source_run / "drive_01_event_1.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        variants = {
            "wrong": [{"action": "A different action."}],
            "missing": [],
            "malformed": [None],
        }
        for name, events in variants.items():
            with self.subTest(variant=name):
                row = copy.deepcopy(original)
                row["events"] = copy.deepcopy(events)
                _write_json(path, row)
                report = self.build_report(self.root / f"action_{name}.json")
                query = report["queries"][0]
                self.assertEqual(query["status"], "label_mismatch")
                self.assertEqual(report["summary"]["counts"]["audited_events"], 0)
                for mode in ("baseline", "semantic_guard"):
                    stage = query["modes"][mode]["stages"]["saved_top_1"]
                    self.assertEqual(stage["status"], "unresolved")
                    self.assertEqual(stage["total"], 1)
                    self.assertEqual(stage["unresolved"], 1)
                    self.assertEqual(stage["reasons"], {"saved_row_query_mismatch": 1})
                aggregate = report["summary"]["stages"]["event_frame"]["baseline"]["saved_top_1"]
                self.assertEqual(aggregate["queries"], 1)
                self.assertEqual(aggregate["total"], 1)
                self.assertEqual(aggregate["unresolved"], 1)
                self.assertIn("saved_row_query_mismatch",
                              report["summary"]["issues"]["by_code"])

    def test_issue_histogram_and_total_use_deduplicated_list(self):
        from tools import p0_audit as p0_audit_module

        original = p0_audit_module._load_saved_rows

        def load_with_duplicates(source_run):
            rows, files, issues = original(source_run)
            issues.extend([
                {"code": "duplicate_probe", "detail": "same detail"},
                {"code": "duplicate_probe", "detail": "same detail"},
            ])
            return rows, files, issues

        with patch("tools.p0_audit._load_saved_rows", side_effect=load_with_duplicates):
            report = self.build_report()
        expected = [{"code": "duplicate_probe", "detail": "same detail"}]
        self.assertEqual(report["issues"], expected)
        self.assertEqual(report["summary"]["issues"]["total"], 1)
        self.assertEqual(report["summary"]["issues"]["by_code"], {"duplicate_probe": 1})
        self.assertEqual(sum(report["summary"]["issues"]["by_code"].values()),
                         report["summary"]["issues"]["total"])
        self.assertEqual(report["summary"]["status"], "issues")

    def test_current_config_is_not_used_as_historical_expectation(self):
        expected = _config()
        captured = {
            "status": "verified",
            "source": "config/models.json",
            "path": "config/models.json",
            "sha256": "a" * 64,
            "expected_sha256": "a" * 64,
            "reason": None,
            "visual_embedding": expected["visual_embedding"],
            "caption_embedding": expected["caption_embedding"],
            "fast_caption_generation": expected["fast_caption_generation"],
        }
        wrong = {
            "status": "ok",
            "visual_embedding": {"dimension": 999},
            "caption_embedding": {
                "model": "other-model", "revision": "other-revision",
                "dimension": 999, "normalized": False, "metric": "L2",
            },
            "fast_caption_generation": {"model": "other-model", "revision": "other-revision"},
        }
        with patch("tools.p0_audit._current_config", return_value=wrong), \
                patch("tools.p0_audit._captured_model_config", return_value=captured):
            report = self.build_report()
        self.assertEqual(report["summary"]["issues"]["total"], 0)
        artifact = report["artifacts"]["L01_V001"]
        self.assertEqual(artifact["artifacts"]["index_manifest"]["status"], "ok")
        self.assertEqual(artifact["artifacts"]["visual_features"]["status"], "ok")
        self.assertEqual(report["provenance"]["captured_model_config"]["status"], "verified")
        self.assertEqual(
            report["provenance"]["current_config_snapshot"]["visual_embedding"]["dimension"], 999)

    def test_conflicting_current_config_without_captured_config_stays_unknown(self):
        wrong = {
            "status": "ok",
            "visual_embedding": {"dimension": 999},
            "caption_embedding": {
                "model": "other-model", "revision": "other-revision",
                "dimension": 999, "normalized": False, "metric": "L2",
            },
            "fast_caption_generation": {"model": "other-model", "revision": "other-revision"},
        }
        with patch("tools.p0_audit._current_config", return_value=wrong), \
                patch("tools.p0_audit._captured_model_config",
                      return_value={"status": "unknown", "reason": "test"}):
            report = self.build_report()
        self.assertEqual(report["summary"]["issues"]["total"], 0)
        artifact = report["artifacts"]["L01_V001"]
        self.assertEqual(artifact["artifacts"]["index_manifest"]["status"], "ok")
        self.assertEqual(artifact["artifacts"]["visual_features"]["status"], "ok")

    def test_incomplete_captured_temporal_config_exits_two(self):
        self.context["temporal_config"].pop("max_gap_seconds")
        self._write_context()
        output = self.root / "incomplete.json"
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = main([
                "--source-run", str(self.source_run),
                "--labels", str(self.labels_path),
                "--mapping-root", str(self.mapping_root),
                "--caption-root", str(self.caption_root),
                "--output", str(output),
            ])
        self.assertEqual(code, 2)
        self.assertIn("context_temporal_config_missing_fields", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_malformed_raw_evidence_dimensions_are_invalid(self):
        record = {"status": "ok", "issues": [], "path": "raw.json", "sha256": "x"}
        data = {
            "unit_id": "sample",
            "video_id": "L01_V001",
            "actions": ["first"],
            "frames": [{"video_id": "L01_V001", "frame_id": 0,
                        "keyframe_n": 1, "pts_time": 0.0}],
            "visual": [[0.9, 0.8]],
            "semantic": None,
        }
        result = _validate_raw_evidence(data, "sample_L01_V001.json", record)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["issues"][0]["code"], "raw_evidence_invalid")
        self.assertIn("one value per frame", result["issues"][0]["detail"])


class ContextValidationTest(unittest.TestCase):
    def test_context_requires_captured_video_groups(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        context_path = root / "context.json"
        _write_json(context_path, {
            "mode": "frozen_human_holdout",
            "label_sha256": "0" * 64,
            "temporal_config": TemporalConfig().public(),
            "video_groups": [],
        })
        with self.assertRaisesRegex(AuditError, "video_groups_invalid"):
            _load_context(context_path, "0" * 64)

    def test_context_requires_complete_temporal_config_fields(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        context_path = root / "context.json"
        temporal = TemporalConfig().public()
        temporal.pop("max_gap_seconds")
        _write_json(context_path, {
            "mode": "frozen_human_holdout",
            "label_sha256": "0" * 64,
            "temporal_config": temporal,
            "video_groups": ["L01"],
        })
        with self.assertRaisesRegex(AuditError, "context_temporal_config_missing_fields"):
            _load_context(context_path, "0" * 64)

    def test_label_hash_must_match_context(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        context_path = root / "context.json"
        _write_json(context_path, {
            "mode": "frozen_human_holdout",
            "label_sha256": "a" * 64,
            "temporal_config": TemporalConfig().public(),
            "video_groups": ["L01"],
        })
        with self.assertRaisesRegex(AuditError, "label_hash_mismatch"):
            _load_context(context_path, "b" * 64)


if __name__ == "__main__":
    unittest.main()
