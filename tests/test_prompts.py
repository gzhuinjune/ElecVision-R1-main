import unittest

from elecvision_r1.prompts import build_grounding_prompt


class PromptTests(unittest.TestCase):
    def test_grounding_prompt_contains_required_schema(self):
        prompt = build_grounding_prompt(
            equipment_labels={"cysb_cyg": "Oil Conservator"},
            defect_labels={"sly_bjbmyw": "Oil Leakage from Oil Storage Cabinet"},
            rag_context="oil conservator --may_exhibit--> oil leakage",
        )
        self.assertIn("<answer>", prompt)
        self.assertIn("bbox_2d", prompt)
        self.assertIn("label", prompt)
        self.assertIn("cysb_cyg", prompt)
        self.assertIn("sly_bjbmyw", prompt)
        self.assertIn("Retrieved path evidence", prompt)


if __name__ == "__main__":
    unittest.main()
