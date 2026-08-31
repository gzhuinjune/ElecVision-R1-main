import unittest
from pathlib import Path

from elecvision_r1.grpo import ElecVisionGRPOScorer, group_relative_advantages, load_jsonl


class GRPOTests(unittest.TestCase):
    def test_advantages_are_group_normalized(self):
        values = group_relative_advantages([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(values), 0.0, places=5)
        self.assertGreater(values[-1], values[0])

    def test_group_scoring(self):
        root = Path(__file__).resolve().parents[1]
        rows = load_jsonl(root / "examples" / "verification_candidates.jsonl")
        result = ElecVisionGRPOScorer().score_records(rows)[0]
        self.assertEqual(result.record_id, "verification_record_001")
        self.assertEqual(len(result.scores), 3)
        self.assertGreater(result.scores[0].total_reward, result.scores[-1].total_reward)


if __name__ == "__main__":
    unittest.main()
