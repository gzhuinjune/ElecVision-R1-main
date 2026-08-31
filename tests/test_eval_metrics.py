import unittest
from pathlib import Path

from elecvision_r1.eval_metrics import coco_style_map, load_detection_records


class EvalMetricTests(unittest.TestCase):
    def test_coco_style_map_for_verification_example(self):
        root = Path(__file__).resolve().parents[1]
        pred = load_detection_records(root / "examples" / "verification_predictions.json", prediction=True)
        gt = load_detection_records(root / "examples" / "verification_annotations.json", prediction=False)
        metrics = coco_style_map(pred, gt)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["AP50"], 1.0)


if __name__ == "__main__":
    unittest.main()
