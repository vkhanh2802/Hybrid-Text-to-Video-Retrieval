import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.paths import KIS_ROOT
from tools.human_holdout import (FROZEN_SOURCES, build_packet, freeze, prior_exposure,
                                 sha, validate_annotations)


def synthetic_packet(root):
    """Synthetic colored rectangles only; never reads real holdout images."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir()
    atomic_write_json(root / "exclusions.json", {"excluded_video_ids": ["L01_V099"]})
    atomic_write_json(root / "protocol.json", {"schema": "kis_human_holdout_protocol_v1",
        "source_sha256": {name: sha(KIS_ROOT / name) for name in FROZEN_SOURCES},
        "exclusions_sha256": sha(root / "exclusions.json")})
    samples = []
    for i in (1, 2):
        sample_id, video_id = f"drive_{i:02d}", f"L01_V{i:03d}"
        image = Image.new("RGB", (480, 180), "navy" if i == 1 else "orange")
        image.save(root / "images" / f"{sample_id}.jpg")
        image.close()
        samples.append({"sample_id": sample_id, "video_id": video_id, "split": "holdout",
            "contact_sheet": f"{sample_id}.jpg", "frames": [{"video_id": video_id, "frame_id": n*30,
                "keyframe_n": n, "pts_time": float(n), "image_sha256": f"synthetic-{i}-{n}",
                "status": "downloaded"} for n in (1, 2, 3)]})
    atomic_write_json(root / "images" / "manifest.json", {"samples": samples})
    build_packet(root)
    return root


def synthetic_annotations(root):
    data = json.loads((Path(root) / "annotation_template.json").read_text(encoding="utf-8"))
    data.update(reviewer="Synthetic test fixture (not a human label)", image_only_confirmed=True)
    for sample in data["samples"]:
        sample["status"] = "reviewed"
        sample["events"] = [{"text_vi": "Một hình chữ nhật màu xanh.", "text_en": "A blue rectangle.",
            "english_approved": True, "judgments": [{"sample_id": sample["sample_id"], "keyframe_n": n,
                "judgment": "positive" if n == 1 else "negative" if n == 2 else "unknown"} for n in (1, 2, 3)]}]
    return data


class HumanHoldoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = synthetic_packet(Path(self.temp.name) / "packet")

    def test_exposure_targets_and_local_pools_not_seed_or_preflight(self):
        root = Path(self.temp.name) / "history"
        root.mkdir()
        atomic_write_json(root / "history.json", {
            "samples": [{"video_id": "L01_V001"}], "query": {"video_id": "L01_V002"},
            "candidate_videos": ["L01_V003"], "cases": [{"selected_videos": ["L01_V004"]}],
            "condition_stats": {"L01_V005": {}}, "judgments": [{"video_id": "L01_V006"}],
            "streams": {"visual": [{"video_id": "L01_V007", "score": 1}]},
            "preflight_rows": [{"video_id": "L01_V008", "ready": True}],
            "excluded_video_ids": ["L01_V009"], "source_snapshot": "L01_V010"})
        videos, files = prior_exposure([root])
        self.assertEqual(videos, [f"L01_V{i:03d}" for i in range(1, 7)])
        self.assertEqual(len(files), 1)
        (root / "broken.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            prior_exposure([root])

    def test_template_has_no_automatic_judgments_or_external_dependencies(self):
        data = json.loads((self.root / "annotation_template.json").read_text(encoding="utf-8"))
        self.assertFalse(data["image_only_confirmed"])
        self.assertTrue(all(sample["events"] == [] and sample["status"] == "pending" for sample in data["samples"]))
        html = (self.root / "review.html").read_text(encoding="utf-8")
        self.assertNotIn("__PACKET_JSON__", html)
        self.assertNotIn('src="http', html)
        self.assertIn("unreviewed", html)
        with self.assertRaises(FileExistsError):
            build_packet(self.root)

    def test_validate_and_freeze_preserve_unknown_and_human_provenance(self):
        data = synthetic_annotations(self.root)
        result = validate_annotations(self.root, data, require_english=True)
        self.assertEqual(result["eligible_pair_queries"], 2)
        self.assertEqual(result["queries"][0]["judgments"][-1]["judgment"], "unknown")
        path, output = self.root / "submitted.json", self.root / "labels.json"
        atomic_write_json(path, data)
        freeze(self.root, path, output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["annotation_source"], "human_visual_observation")
        with self.assertRaises(FileExistsError):
            freeze(self.root, path, output)

    def test_translation_can_wait_but_cannot_freeze_without_approval(self):
        data = synthetic_annotations(self.root)
        event = data["samples"][0]["events"][0]
        event.update(text_en="", english_approved=False)
        self.assertEqual(validate_annotations(self.root, data)["translations_pending"], ["drive_01_event_1"])
        with self.assertRaisesRegex(ValueError, "human approval"):
            validate_annotations(self.root, data, require_english=True)

    def test_unreviewed_missing_duplicate_or_unavailable_judgment_rejected(self):
        for mutation in ("unreviewed", "missing", "duplicate", "unavailable"):
            with self.subTest(mutation=mutation):
                data = synthetic_annotations(self.root)
                rows = data["samples"][0]["events"][0]["judgments"]
                if mutation == "unreviewed":
                    rows[0]["judgment"] = "unreviewed"
                elif mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                else:
                    rows[0]["keyframe_n"] = 999
                with self.assertRaises(ValueError):
                    validate_annotations(self.root, data)

    def test_all_positive_event_allowed_but_not_eligible_for_fpr(self):
        data = synthetic_annotations(self.root)
        for sample in data["samples"]:
            for row in sample["events"][0]["judgments"]:
                row["judgment"] = "positive"
        result = validate_annotations(self.root, data)
        self.assertEqual(result["eligible_pair_queries"], 0)

    def test_skipping_requires_reason_and_every_sample_needs_status(self):
        data = synthetic_annotations(self.root)
        data["samples"][1]["status"] = "skipped"
        with self.assertRaisesRegex(ValueError, "reason"):
            validate_annotations(self.root, data)
        data["samples"][1]["skip_reason"] = "Synthetic unavailable image"
        self.assertEqual(validate_annotations(self.root, data)["reviewed_windows"], 1)
        data["samples"].pop()
        with self.assertRaisesRegex(ValueError, "Every sampled"):
            validate_annotations(self.root, data)

    def test_sequence_positive_times_must_be_strictly_ordered(self):
        data = synthetic_annotations(self.root)
        sample = data["samples"][0]
        event = copy.deepcopy(sample["events"][0])
        sample["events"].append(event)
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            validate_annotations(self.root, data)
        for row in event["judgments"]:
            row["judgment"] = "positive" if row["keyframe_n"] == 2 else "negative"
        self.assertEqual(len(validate_annotations(self.root, data)["queries"]), 3)

    def test_integrity_and_attestation_block_invalid_freeze(self):
        data = synthetic_annotations(self.root)
        data["image_only_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "human confirmation"):
            validate_annotations(self.root, data)
        data["image_only_confirmed"] = True
        (self.root / "images" / "drive_01.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Contact sheet"):
            validate_annotations(self.root, data)

    def test_frozen_rule_change_rejected_without_touching_real_code(self):
        data = synthetic_annotations(self.root)
        protocol = json.loads((self.root / "protocol.json").read_text(encoding="utf-8"))
        protocol["source_sha256"]["kis/condition_evidence.py"] = "not-the-frozen-hash"
        atomic_write_json(self.root / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "Frozen implementation changed"):
            validate_annotations(self.root, data)


if __name__ == "__main__":
    unittest.main()
