import unittest

from tools.check_visual_consistency import select_dev_images


class VisualConsistencyTest(unittest.TestCase):
    def test_diagnostic_excludes_holdout_and_sequence_queries(self):
        image = {"split": "dev", "task": "image", "query_id": "dev_image"}
        labels = {"queries": [image, {"split": "holdout", "task": "image"}, {"split": "dev", "task": "sequence"}]}
        self.assertEqual(select_dev_images(labels), [image])

    def test_excess_budget_or_no_dev_images_stop_before_model_load(self):
        for queries in ([], [{"split": "dev", "task": "image"}] * 5):
            with self.assertRaises(ValueError):
                select_dev_images({"queries": queries})


if __name__ == "__main__":
    unittest.main()
