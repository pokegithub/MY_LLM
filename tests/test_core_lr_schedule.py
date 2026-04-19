import unittest

from core.lr_schedule import cosine_warmup_lr, cosine_warmup_lr_from_tokens


class CosineWarmupScheduleTests(unittest.TestCase):
    def test_step_schedule_hits_expected_boundaries(self):
        max_lr = 1e-3
        min_lr = 1e-4
        total_steps = 100
        warmup_steps = 10

        self.assertAlmostEqual(
            cosine_warmup_lr(
                step=0,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                max_lr=max_lr,
                min_lr=min_lr,
            ),
            0.0,
            places=10,
        )
        self.assertAlmostEqual(
            cosine_warmup_lr(
                step=warmup_steps,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                max_lr=max_lr,
                min_lr=min_lr,
            ),
            max_lr,
            places=10,
        )
        self.assertAlmostEqual(
            cosine_warmup_lr(
                step=total_steps,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                max_lr=max_lr,
                min_lr=min_lr,
            ),
            min_lr,
            places=10,
        )

    def test_token_schedule_matches_step_schedule(self):
        step = 32
        tokens_per_step = 8192
        total_tokens = 2_000_000
        warmup_tokens = 200_000
        max_lr = 2e-4
        min_lr = 2e-5

        expected = cosine_warmup_lr(
            step=step,
            total_steps=total_tokens // tokens_per_step,
            warmup_steps=warmup_tokens // tokens_per_step,
            max_lr=max_lr,
            min_lr=min_lr,
        )
        actual = cosine_warmup_lr_from_tokens(
            step=step,
            tokens_per_step=tokens_per_step,
            total_tokens=total_tokens,
            warmup_tokens=warmup_tokens,
            max_lr=max_lr,
            min_lr=min_lr,
        )
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
