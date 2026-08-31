import importlib.util
import unittest
from pathlib import Path


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_grpo_torch", ROOT / "scripts" / "train_grpo_torch.py")
train_grpo_torch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_grpo_torch)


class DummyTokenizer:
    def batch_decode(self, rows, skip_special_tokens=True):
        return [" ".join(str(int(value)) for value in row.tolist()) for row in rows]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainGRPOTorchScriptTests(unittest.TestCase):
    def test_decode_slices_decoder_only_outputs(self):
        input_ids = torch.tensor([[1, 2]])
        generated = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
        decoded = train_grpo_torch._decode_generated_completions(torch, DummyTokenizer(), generated, input_ids)
        self.assertEqual(decoded, ["3 4", "5 6"])

    def test_decode_keeps_encoder_decoder_outputs(self):
        input_ids = torch.tensor([[1, 2]])
        generated = torch.tensor([[3, 4], [5, 6]])
        decoded = train_grpo_torch._decode_generated_completions(torch, DummyTokenizer(), generated, input_ids)
        self.assertEqual(decoded, ["3 4", "5 6"])

    def test_refresh_reference_model_copies_policy_parameters(self):
        policy = torch.nn.Linear(2, 2)
        reference = torch.nn.Linear(2, 2)
        with torch.no_grad():
            policy.weight.fill_(2.0)
            policy.bias.fill_(3.0)
            reference.weight.zero_()
            reference.bias.zero_()
        train_grpo_torch._refresh_reference_model(torch, policy, reference, torch.device("cpu"))
        self.assertTrue(torch.equal(reference.weight, policy.weight))
        self.assertTrue(torch.equal(reference.bias, policy.bias))
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))


if __name__ == "__main__":
    unittest.main()
