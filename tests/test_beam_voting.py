"""Numeric consensus, invalid candidates and the existing execution sandbox."""

import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from src.code_preprocessing import restore_code
from src.utils.beam_voting import BeamVoter
from src.utils.code_executor import ExecutionResult


class BeamVotingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.codes = [
            "def solve(): return 100",
            "def solve(): return 2 + 2",
            "def solve(): return 8 / 2",
            "def solve(): return 5 - 1",
            "def solve(): return 1 / 0",
            "def solve(): return float('nan')",
            "def solve(): return 4.0000001",
            "def solve(): return 4.1",
            "    result = 2 + 2\n",
        ]
        vocabulary = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        vocabulary.update({code: index + 4 for index, code in enumerate(self.codes)})
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
        self.tokenizer_path = Path(self.directory.name) / "tokenizer.json"
        tokenizer.save(str(self.tokenizer_path))
        self.results = {
            self.codes[0]: ExecutionResult("ok", Fraction(100)),
            **{code: ExecutionResult("ok", Fraction(4)) for code in self.codes[1:4]},
            self.codes[4]: ExecutionResult("runtime_error"),
            self.codes[5]: ExecutionResult("non_finite"),
            self.codes[6]: ExecutionResult("ok", Fraction("4.0000001")),
            self.codes[7]: ExecutionResult("ok", Fraction("4.1")),
            restore_code(self.codes[8]): ExecutionResult("ok", Fraction(4)),
        }
        self.executor = Mock()
        self.executor.run.side_effect = self.results.__getitem__

    def voter(self, **kwargs):
        with patch(
            "src.utils.beam_voting.SafeCodeExecutor", return_value=self.executor
        ):
            return BeamVoter(self.tokenizer_path, **kwargs)

    @staticmethod
    def beams(rows):
        return torch.tensor([[[index + 4, 3, 0] for index in row] for row in rows])

    def test_majority_can_override_best_beam_and_duplicates_get_one_vote(self):
        voter = self.voter()
        beams = self.beams([[0, 1, 2, 3], [0, 0, 1, 2]])
        scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]] * 2)
        actual = voter.select(beams, scores)
        self.assertTrue(torch.equal(actual, torch.stack([beams[0, 1], beams[1, 2]])))
        # Identical code across inputs is executed once per selection call.
        self.assertEqual(self.executor.run.call_count, 4)
        voter.select(beams, scores)
        self.assertEqual(self.executor.run.call_count, 8)

    def test_ties_use_best_score_and_all_failures_fall_back_to_best_beam(self):
        voter = self.voter()
        beams = self.beams([[0, 1, 4], [4, 5, 4]])
        scores = torch.tensor([[-0.3, -0.2, -0.1], [-0.3, -0.1, -0.2]])
        actual = voter.select(beams, scores)
        self.assertTrue(torch.equal(actual, torch.stack([beams[0, 1], beams[1, 1]])))

    def test_unfinished_and_nonfinite_scored_candidates_are_never_executed(self):
        voter = self.voter()
        beams = self.beams([[1, 0, 2, 3]])
        beams[0, 0, 1] = 0  # A valid-looking but unfinished program has no vote.
        actual = voter.select(
            beams, torch.tensor([[-0.1, -0.4, -torch.inf, torch.nan]])
        )
        self.assertTrue(torch.equal(actual, beams[:, 1]))
        self.executor.run.assert_called_once_with(self.codes[0])

    def test_numeric_tolerance_and_exact_rationals(self):
        beams = self.beams([[0, 1, 6]])
        scores = torch.tensor([[-0.1, -0.2, -0.3]])
        self.assertTrue(torch.equal(self.voter().select(beams, scores), beams[:, 1]))
        self.assertTrue(
            torch.equal(self.voter(rtol=0, atol=0).select(beams, scores), beams[:, 0])
        )
        self.results[self.codes[0]] = ExecutionResult("ok", Fraction(2**100))
        self.results[self.codes[1]] = ExecutionResult("ok", Fraction(2**100 + 1))
        self.results[self.codes[6]] = self.results[self.codes[1]]
        self.assertTrue(
            torch.equal(self.voter(rtol=0, atol=0).select(beams, scores), beams[:, 1])
        )

    def test_compact_body_is_wrapped_before_execution(self):
        beams = self.beams([[8]])
        actual = self.voter(code_format="solve_body_result").select(
            beams, torch.tensor([[-0.1]])
        )
        self.assertTrue(torch.equal(actual, beams[:, 0]))
        self.executor.run.assert_called_once_with(restore_code(self.codes[8]))

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is required")
    def test_real_sandbox_selects_numeric_majority(self):
        voter = BeamVoter(self.tokenizer_path, workers=2)
        beams = self.beams([[0, 1, 2]])
        self.assertTrue(
            torch.equal(
                voter.select(beams, torch.tensor([[-0.1, -0.2, -0.3]])), beams[:, 1]
            )
        )

    def test_sandbox_failure_has_no_unsafe_fallback(self):
        with patch(
            "src.utils.beam_voting.SafeCodeExecutor",
            side_effect=RuntimeError("sandbox unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "sandbox unavailable"):
                BeamVoter(self.tokenizer_path)

    def test_explicit_process_mode_works_without_bubblewrap(self):
        with patch("src.utils.code_executor.shutil.which", return_value=None) as which:
            voter = BeamVoter(self.tokenizer_path, workers=2, sandbox=False)
            beams = self.beams([[0, 1, 2]])
            self.assertTrue(
                torch.equal(
                    voter.select(beams, torch.tensor([[-0.1, -0.2, -0.3]])), beams[:, 1]
                )
            )
            which.assert_not_called()

    def test_batched_and_individual_execution_select_the_same_beams(self):
        beams = self.beams([[0, 1, 2, 3], [0, 0, 1, 2], [4, 5, 0, 7]])
        scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]] * 3)
        single = BeamVoter(
            self.tokenizer_path, workers=2, sandbox=False, execution_batch_size=1
        )
        batch = BeamVoter(
            self.tokenizer_path, workers=2, sandbox=False, execution_batch_size=16
        )
        self.assertTrue(
            torch.equal(single.select(beams, scores), batch.select(beams, scores))
        )


if __name__ == "__main__":
    unittest.main()
