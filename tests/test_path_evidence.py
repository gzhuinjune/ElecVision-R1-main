import unittest
from pathlib import Path

from elecvision_r1.path_evidence import PathEvidenceRetriever


class PathEvidenceTests(unittest.TestCase):
    def test_node_retrieval_finds_relevant_seed(self):
        root = Path(__file__).resolve().parents[1]
        retriever = PathEvidenceRetriever.from_json(root / "examples" / "verification_path_graph.json")
        hits = retriever.retrieve_nodes("pressure gauge damaged meter panel", top_k=3)
        self.assertTrue(any(hit.node_id == "pressure_gauge" for hit in hits))

    def test_query_embedding_can_drive_node_retrieval(self):
        graph = {
            "pressure_gauge": {"description": "Gauge", "embedding": [1.0, 0.0]},
            "oil_conservator": {"description": "Conservator", "embedding": [0.0, 1.0]},
        }
        retriever = PathEvidenceRetriever(graph, [])
        hits = retriever.retrieve_nodes("external embedding query", query_embedding=[1.0, 0.0], top_k=1)
        self.assertEqual(hits[0].node_id, "pressure_gauge")

    def test_path_context_contains_defect_and_action(self):
        root = Path(__file__).resolve().parents[1]
        retriever = PathEvidenceRetriever.from_json(root / "examples" / "verification_path_graph.json")
        evidence = retriever.retrieve_for_query("oil conservator oil leakage maintenance sly_bjbmyw", top_k=3)
        context = retriever.to_prompt_context(evidence)
        self.assertIn("sly_bjbmyw", context)
        self.assertIn("maintenance_action", context)


if __name__ == "__main__":
    unittest.main()
