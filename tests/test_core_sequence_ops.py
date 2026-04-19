import unittest

import torch
import torch.nn.functional as F

from core.sequence_ops import (
    gather_token_log_probs,
    masked_next_token_sequence_log_probs,
    next_token_cross_entropy,
    shift_for_next_token,
)


class SequenceOpsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.logits = torch.randn(2, 6, 13)
        self.token_ids = torch.tensor(
            [
                [1, 5, 3, 4, 2, 0],
                [1, 7, 8, 6, 0, 0],
            ],
            dtype=torch.long,
        )
        self.mask = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_shift_for_next_token_alignment(self):
        shift_logits, shift_labels = shift_for_next_token(
            self.logits,
            self.token_ids,
        )
        self.assertEqual(tuple(shift_logits.shape), (2, 5, 13))
        self.assertEqual(tuple(shift_labels.shape), (2, 5))
        self.assertTrue(torch.equal(shift_labels, self.token_ids[:, 1:]))

    def test_gather_token_log_probs_matches_manual(self):
        shift_logits, shift_labels = shift_for_next_token(
            self.logits,
            self.token_ids,
        )
        manual = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            2,
            shift_labels.unsqueeze(2),
        ).squeeze(2)
        actual = gather_token_log_probs(shift_logits, shift_labels)
        self.assertTrue(torch.allclose(actual, manual, atol=1e-6))

    def test_next_token_cross_entropy_matches_pytorch(self):
        shift_logits, shift_labels = shift_for_next_token(
            self.logits,
            self.token_ids,
        )
        expected = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=0,
        )
        actual = next_token_cross_entropy(
            self.logits,
            self.token_ids,
            ignore_index=0,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_masked_next_token_sequence_log_probs(self):
        shift_logits, shift_labels = shift_for_next_token(
            self.logits,
            self.token_ids,
        )
        shift_mask = self.mask[:, 1:]

        manual_token_lp = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            2,
            shift_labels.unsqueeze(2),
        ).squeeze(2)
        expected = (manual_token_lp * shift_mask).sum(dim=-1)

        actual = masked_next_token_sequence_log_probs(
            self.logits,
            self.token_ids,
            self.mask,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
