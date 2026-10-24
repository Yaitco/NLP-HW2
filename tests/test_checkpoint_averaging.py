import tempfile
from pathlib import Path
import unittest

import torch

from average_checkpoints import average_checkpoints, exponential_weights


class AveragingTests(unittest.TestCase):
    def test_weights_endpoints_and_validation(self):
        self.assertEqual(exponential_weights(3, 0), [0, 0, 1])
        self.assertEqual(exponential_weights(3, 1), [1 / 3] * 3)
        weights = exponential_weights(3, 0.5)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(weights[2] / weights[1], 2)
        for beta in [-1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                exponential_weights(3, beta)

    def test_average_uses_latest_buffers_and_does_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"{i}.pth" for i in range(3)]
            for i, path in enumerate(paths):
                torch.save(
                    {
                        "state_dict": {
                            "weight": torch.tensor([i * 7.0], dtype=torch.float16),
                            "counter": torch.tensor(i),
                        }
                    },
                    path,
                )
            averaged = average_checkpoints(paths, 0.5)
            self.assertEqual(averaged["weight"].dtype, torch.float32)
            torch.testing.assert_close(averaged["weight"], torch.tensor([10.0]))
            self.assertEqual(averaged["counter"].item(), 2)
            torch.testing.assert_close(
                average_checkpoints(paths, 0)["weight"], torch.tensor([14.0])
            )
            torch.testing.assert_close(
                average_checkpoints(paths, 1)["weight"], torch.tensor([7.0])
            )
            original = torch.load(paths[0], weights_only=False)
            self.assertEqual(original["state_dict"]["weight"].item(), 0)
            anchor = Path(temporary) / "anchor.pth"
            torch.save(
                {
                    "state_dict": {
                        "weight": torch.tensor([28.0], dtype=torch.float16),
                        "counter": torch.tensor(-1),
                    }
                },
                anchor,
            )
            anchored = average_checkpoints(paths, 0.5, anchor=anchor)
            torch.testing.assert_close(anchored["weight"], torch.tensor([12.25]))
            self.assertEqual(anchored["counter"].item(), 2)
            torch.save({"state_dict": {"wrong_key": torch.ones(2)}}, paths[-1])
            with self.assertRaises(ValueError):
                average_checkpoints(paths, 0.5)


if __name__ == "__main__":
    unittest.main()
