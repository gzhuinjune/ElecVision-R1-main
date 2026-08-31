import unittest

from elecvision_r1.reward import FormatReward, GroundingReward, bbox_iou, mean_average_precision, parse_grounding_answer


class RewardTests(unittest.TestCase):
    def test_parser_accepts_answer_tag_json(self):
        records = parse_grounding_answer('<answer>[{"bbox_2d":[1,2,5,6],"label":"oil_leakage"}]</answer>')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "oil_leakage")

    def test_iou(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_map_score_for_complete_match(self):
        pred = [{"bbox_2d": [10, 20, 50, 50], "label": "oil_leakage", "score": 1.0}]
        gt = [{"bbox_2d": [10, 20, 50, 50], "label": "oil_leakage"}]
        self.assertAlmostEqual(mean_average_precision(pred, gt, [0.5]), 1.0)

    def test_grounding_reward_uses_length_penalty(self):
        solution = '<answer>[{"bbox_2d":[10,20,50,50],"label":"oil_leakage"}]</answer>'
        duplicated = '<answer>[{"bbox_2d":[10,20,50,50],"label":"oil_leakage"},{"bbox_2d":[10,20,50,50],"label":"oil_leakage"}]</answer>'
        reward = GroundingReward().score_text(duplicated, solution)
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 1.0)

    def test_grounding_reward_keeps_equipment_and_defect_sets_separate(self):
        solution = (
            '<answer>{"equipment":[{"bbox_2d":[0,0,10,10],"label":"shared"}],'
            '"defects":[{"bbox_2d":[20,20,30,30],"label":"shared"}]}</answer>'
        )
        wrong_channel = '<answer>{"defects":[{"bbox_2d":[0,0,10,10],"label":"shared"}]}</answer>'
        correct_channel = '<answer>{"equipment":[{"bbox_2d":[0,0,10,10],"label":"shared"}]}</answer>'
        scorer = GroundingReward(length_penalty=False)
        self.assertEqual(scorer.score_text(wrong_channel, solution), 0.0)
        self.assertGreater(scorer.score_text(correct_channel, solution), 0.0)

    def test_format_reward_requires_answer_tags(self):
        scorer = FormatReward()
        self.assertEqual(scorer.score_text('[{"bbox_2d":[1,2,5,6],"label":"oil_leakage"}]'), 0.0)
        self.assertEqual(scorer.score_text('<answer>[{"bbox_2d":[1,2,5,6],"label":"oil_leakage"}]</answer>'), 1.0)
        self.assertEqual(scorer.score_text('<answer>[]</answer>'), 1.0)
        self.assertEqual(scorer.score_text('<answer>{"equipment":[],"defects":[]}</answer>'), 1.0)


if __name__ == "__main__":
    unittest.main()
