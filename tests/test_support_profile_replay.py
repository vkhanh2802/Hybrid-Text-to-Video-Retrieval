import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.support_profile_replay import run


class SupportProfileReplayTest(unittest.TestCase):
    def test_replay_rejects_non_dev_or_unselected_profile_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "profile.json"
            for profile in ({"split": "holdout", "selected": {"visual_threshold": 0.1}},
                            {"split": "dev", "selected": None}):
                calibration.write_text(json.dumps(profile), encoding="utf-8")
                with patch("tools.support_profile_replay.load_dev_records", return_value=([], {})):
                    with self.assertRaisesRegex(ValueError, "development-selected"):
                        run(root / "labels.json", root, calibration, root / "output")
                self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
