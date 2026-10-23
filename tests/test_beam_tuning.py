"""Cached validation tuning and exact beam score rescaling."""

from contextlib import nullcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from datasets import Dataset
import nbformat
import torch

from src.code_preprocessing import restore_code
from src.utils.beam_voting import rescore_lengths, same_result, weighted_consensus_index
from src.utils.code_executor import ExecutionResult
from src.utils.io_utils import ROOT_PATH


class BeamTuningTests(unittest.TestCase):
    def test_rescaling_recovers_raw_log_probabilities(self):
        raw, lengths = [-10.0, -30.0, -math.inf], [2, 10, 4]
        scores = [s / length for s, length in zip(raw, lengths)]
        self.assertEqual(rescore_lengths(scores, lengths, 1, 0), raw)
        self.assertEqual(rescore_lengths(scores, lengths, 1, 1), scores)
        self.assertTrue(same_result(Fraction(4), Fraction("4.0000001")))
        self.assertFalse(same_result(None, None))
        values = [None, Fraction(4), Fraction(4), Fraction(100)]
        self.assertEqual(weighted_consensus_index(values, [0, -1, -1, -0.5], 1), 1)

    def test_optional_notebook_caches_executions_and_applies_tuned_settings(self):
        notebook = nbformat.read(ROOT_PATH / "notebooks/inference.ipynb", as_version=4)
        sources = {cell.id: cell.source for cell in notebook.cells}
        validation = Dataset.from_dict(
            {
                "question": ["q"] * 5,
                "code": ["    result = 42\n"] * 5,
            }
        )
        evaluator = Mock(workers=1)
        evaluator.evaluate.side_effect = lambda beams, scores: dict(
            codes=[["correct", "wrong"]] * len(beams),
            results=[[Fraction(42), Fraction(0)]] * len(beams),
            lengths=[[2, 10]] * len(beams),
            scores=[[-1.0, -0.9]] * len(beams),
        )
        evaluator.executor.run.side_effect = [
            ExecutionResult("ok", Fraction(42))
        ] * 4 + [ExecutionResult("runtime_error")]
        tokenizer = Mock()
        tokenizer.encode_batch.return_value = [SimpleNamespace(ids=[4])] * 5
        model = Mock()
        model.generate.side_effect = lambda ids, **kw: (
            torch.zeros(len(ids), 2, 10),
            torch.zeros(len(ids), 2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            scope = dict(
                RUN_TUNING=True,
                VALIDATION_PATH="validation",
                TUNING_CACHE=str(Path(tmp) / "beams.json"),
                folder=Path(tmp),
                config={"code_format": "solve_body_result", "add_imports": True},
                NUM_BEAMS=2,
                MAX_NEW_TOKENS=128,
                LENGTH_PENALTY=1.0,
                SANDBOX=False,
                EXECUTION_BATCH_SIZE=16,
                BATCH_SIZE=2,
                DEVICE="cpu",
                pad_id=0,
                torch=torch,
                Path=Path,
                json=json,
                model=model,
                tokenizer=tokenizer,
                pad_sequence=torch.nn.utils.rnn.pad_sequence,
                restore_code=restore_code,
                voter=Mock(),
                BeamVoter=lambda *args, **kwargs: evaluator,
            )
            with patch("datasets.load_from_disk", return_value=validation), patch(
                "src.utils.beam_voting.BeamVoter", return_value=evaluator
            ), patch("torch.autocast", return_value=nullcontext()):
                exec(sources["tuning-generate"], scope)
                self.assertEqual(model.generate.call_count, 3)
                model.generate.reset_mock()
                evaluator.evaluate.reset_mock()
                evaluator.executor.run.reset_mock()
                exec(sources["tuning-generate"], scope)
                model.generate.assert_not_called()
                evaluator.evaluate.assert_not_called()
                evaluator.executor.run.assert_not_called()
                exec(sources["tuning-search"], scope)
            self.assertEqual(scope["LENGTH_PENALTY"], 0.0)
            self.assertEqual(scope["voter"].temperature, scope["best_temperature"])
            self.assertEqual(scope["tune"], [0, 1])
            self.assertEqual(scope["check"], [2, 3])
            # Mismatching beam-generation settings must not silently reuse stale scores.
            scope["NUM_BEAMS"] = 3
            with patch("datasets.load_from_disk", return_value=validation), patch(
                "src.utils.beam_voting.BeamVoter", return_value=evaluator
            ), self.assertRaisesRegex(ValueError, "Cache settings differ"):
                exec(sources["tuning-generate"], scope)


if __name__ == "__main__":
    unittest.main()
