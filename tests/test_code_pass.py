import logging
import math
import os
import shutil
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import torch

from src.metrics.code_pass import CodePassRate
from src.utils.code_executor import ExecutionResult, SafeCodeExecutor


class AvailabilityTests(unittest.TestCase):
    def test_no_unsandboxed_fallback(self):
        with patch("src.utils.code_executor.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No unsafe fallback"):
                SafeCodeExecutor()


class ProcessModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("src.utils.code_executor.shutil.which", return_value=None) as which:
            cls.executor = SafeCodeExecutor(sandbox=False)
            which.assert_not_called()

    def test_numeric_results_imports_and_invalid_code(self):
        for code, expected in (
            ("def solve(): return 42", ExecutionResult("ok", Fraction(42))),
            (
                "def solve():\n    import numpy as np\n    return np.int64(7)",
                ExecutionResult("ok", Fraction(7)),
            ),
            (
                "def solve():\n    import sympy as s\n    return s.Rational(2, 3)",
                ExecutionResult("ok", Fraction(2, 3)),
            ),
            ("def solve(): return 1/0", ExecutionResult("runtime_error")),
            ("def solve(): return float('nan')", ExecutionResult("non_finite")),
            ("def solve(:", ExecutionResult("compile_error")),
        ):
            with self.subTest(code=code):
                self.assertEqual(self.executor.run(code), expected)

    def test_fresh_process_with_clean_environment_and_temporary_directory(self):
        project_dir = str(Path.cwd())
        with patch.dict(os.environ, {"PROCESS_TEST_SECRET": "hidden"}):
            code = f"""import os
def solve():
    assert os.environ.get('PROCESS_TEST_SECRET') is None
    assert os.getcwd() != {project_dir!r}
    import math
    math.sqrt = lambda value: 999
    return 0
"""
            self.assertEqual(self.executor.run(code).value, 0)
        self.assertEqual(
            self.executor.run("import math\ndef solve(): return math.sqrt(4)").value, 2
        )

    def test_mode_has_host_filesystem_access(self):
        # This mode is intentionally process isolation, not a filesystem sandbox.
        with tempfile.NamedTemporaryFile() as file:
            file.write(b"123")
            file.flush()
            self.assertEqual(
                self.executor.run(
                    f"def solve(): return int(open({file.name!r}).read())"
                ).value,
                123,
            )

    def test_timeout_terminates_worker_and_next_program_still_runs(self):
        executor = SafeCodeExecutor(timeout=0.5, sandbox=False)
        started = time.monotonic()
        self.assertEqual(
            executor.run("def solve():\n    while True: pass").status, "timeout"
        )
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(executor.run("def solve(): return 7").value, 7)



@unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is required")
class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = SafeCodeExecutor(timeout=2)

    def test_number_types(self):
        cases = [
            ("return 2**100 + 1", Fraction(2**100 + 1)),
            ("return 0.5", Fraction(1, 2)),
            ("from fractions import Fraction; return Fraction(2, 3)", Fraction(2, 3)),
            ("from decimal import Decimal; return Decimal('0.1')", Fraction(1, 10)),
            ("import numpy as np; return np.int64(42)", Fraction(42)),
            ("import sympy as s; return s.Rational(2, 3)", Fraction(2, 3)),
            ("from scipy.special import factorial; return factorial(5)", Fraction(120)),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(self.executor.run("def solve(): " + body), ExecutionResult("ok", expected))

    def test_invalid_solutions(self):
        cases = [
            ("def solve(:", "compile_error"),
            ("def other(): return 1", "missing_solve"),
            ("def solve(): return 1/0", "runtime_error"),
            ("def solve(): return True", "non_numeric"),
            ("def solve(): return '42'", "non_numeric"),
            ("def solve(): return [42]", "non_numeric"),
            ("def solve(): return complex(1, 2)", "non_numeric"),
            ("def solve(): return float('nan')", "non_finite"),
            ("def solve(): return float('inf')", "non_finite"),
        ]
        for code, expected in cases:
            with self.subTest(code=code):
                self.assertEqual(self.executor.run(code).status, expected)

    def test_host_files_environment_and_fds_are_hidden(self):
        with tempfile.NamedTemporaryFile() as secret, patch.dict(os.environ, {"CODE_EXECUTOR_SECRET": "hidden"}):
            secret.write(b"private"); secret.flush()
            code = f'''import os
from pathlib import Path
def solve():
    assert not Path({secret.name!r}).exists()
    assert not Path({str(Path.cwd())!r}).exists()
    assert not Path('/home/carlos').exists()
    assert os.environ.get('CODE_EXECUTOR_SECRET') is None
    assert not Path('/dev/nvidia0').exists()
    return 1
'''
            self.assertEqual(self.executor.run(code), ExecutionResult("ok", Fraction(1)))

    def test_filesystem_read_only(self):
        for path in ("/tmp/forbidden", "/work/forbidden", "/dev/forbidden", "/opt/python/forbidden"):
            with self.subTest(path=path):
                code = f"def solve():\n    try: open({path!r}, 'w')\n    except OSError: return 1\n    return 0"
                self.assertEqual(self.executor.run(code).value, 1)

    def test_network_and_fork_are_isolated(self):
        import socket
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0)); server.listen()
            port = server.getsockname()[1]
            code = f'''import socket, os
def solve():
    with socket.socket() as s:
        s.settimeout(0.2)
        try: s.connect(('127.0.0.1', {port}))
        except OSError: pass
        else: return 0
    try: pid = os.fork()
    except OSError: return 1
    if pid == 0: os._exit(0)
    os.waitpid(pid, 0)
    return 0
'''
            self.assertEqual(self.executor.run(code).value, 1)

    def test_deadline_cannot_be_cancelled(self):
        start = time.monotonic()
        result = self.executor.run("import signal,time\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\ndef solve():\n    while True: time.sleep(1)")
        self.assertEqual(result.status, "timeout")
        self.assertLess(time.monotonic() - start, 4)

    def test_crash_memory_and_output_flood_do_not_break_parent(self):
        cases = [
            "import os\nos._exit(42)",
            "def solve(): return bytearray(2*1024**3)",
            "import os\nwhile True: os.write(1, b'x' * 4096)",
            "import os\nos.write(1, b'[' * 2000 + b']' * 2000)\nos._exit(0)",
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertNotEqual(self.executor.run(code).status, "ok")
        self.assertEqual(self.executor.run("def solve(): return 7").value, 7)

    def test_fresh_process_between_programs(self):
        self.executor.run("import math\nmath.sqrt = lambda x: 999\ndef solve(): return 0")
        self.assertEqual(self.executor.run("import math\ndef solve(): return math.sqrt(4)").value, 2)


@unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is required")
class PassRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metric = CodePassRate("artifacts/tokenizer.json", workers=2)

    def setUp(self):
        self.metric.reset()

    def encode(self, codes, eos=True):
        tok = self.metric.tokenizer
        rows = [tok.encode(c, add_special_tokens=False) + ([tok.eos_token_id] if eos else []) for c in codes]
        size = max(map(len, rows))
        return torch.tensor([r + [tok.pad_token_id] * (size - len(r)) for r in rows])

    def test_weighted_batches_invalid_reference_and_reset(self):
        self.metric.update(self.encode(["def solve(): return 3"]), self.encode(["def solve(): return 3"]))
        self.metric.update(self.encode(["def solve(): return 9"] * 3), self.encode([
            "def solve(): return 1", "def solve(): return 2", "def solve(): return None",
        ]))
        result = self.metric.compute()
        self.assertEqual(result["pass_at_1"], 1/3)
        self.assertEqual(result["reference_errors"], 1)
        self.assertEqual(result["pass_total"], 3)
        self.metric.reset()
        self.assertTrue(math.isnan(self.metric.compute()["pass_at_1"]))

    def test_tolerance_exact_large_integers_and_truncation(self):
        self.metric.update(self.encode([
            "def solve(): return 1.000001", "def solve(): return 2**100 + 1", "def solve(): return 2**100 + 1",
        ]), self.encode([
            "def solve(): return 1.0", "def solve(): return 2**100 + 1", "def solve(): return 0",
        ]))
        self.metric.update(self.encode(["def solve(): return 1"], eos=False), self.encode(["def solve(): return 1"]))
        result = self.metric.compute()
        self.assertEqual(result["pass_at_1"], .5)
        self.assertEqual(result["generation_truncated"], 1)

    def test_candidate_exception_counts_as_failure(self):
        self.metric.update(self.encode(["def solve(): return '2'"]), self.encode(["def solve(): return 2"]))
        self.assertEqual(self.metric.compute()["pass_at_1"], 0)
        self.assertEqual(self.metric.compute()["prediction_non_numeric"], 1)

    def test_body_targets_are_wrapped_inside_existing_sandbox(self):
        metric = CodePassRate("artifacts/tokenizer.json", workers=2,
                              code_format="solve_body_result")
        metric.update(self.encode(["    result = 3\n", "    result = 9\n"]),
                      self.encode(["    result = 3\n", "    result = 2\n"]))
        self.assertEqual(metric.compute()["pass_at_1"], .5)
        self.assertEqual(metric.compute()["reference_errors"], 0)

    def test_generated_final_variable_is_returned_but_references_stay_strict(self):
        metric = CodePassRate('artifacts/tokenizer.json', workers=2, code_format='solve_body_result')
        metric.update(self.encode(['    result = 1\n    answer = 3\n', '    answer = 3\n']),
                      self.encode(['    result = 3\n', '    answer = 3\n']))
        result = metric.compute()
        self.assertEqual(result['pass_at_1'], 1)
        self.assertEqual(result['reference_errors'], 1)

    def test_execution_batches_span_validation_updates_and_flush_the_tail(self):
        predicted = [f"def solve(): return {i % 3}" for i in range(67)]
        references = list(predicted)
        predicted[3] = 'def solve(): return 99'
        predicted[4] = 'def solve(): return 1 / 0'
        references[5] = 'def solve(): return None'
        generated, labels = self.encode(predicted), self.encode(references)
        generated[-1][generated[-1] == self.metric.tokenizer.eos_token_id] = self.metric.tokenizer.pad_token_id
        self.metric.update(generated, labels)
        expected = self.metric.compute()
        metric = CodePassRate('artifacts/tokenizer.json', workers=2, execution_batch_size=64)
        with patch.object(metric.executor, 'run_batch', wraps=metric.executor.run_batch) as run:
            metric.update(generated[:32], labels[:32])
            run.assert_not_called()
            metric.update(generated[32:64], labels[32:64])
            self.assertEqual(len(metric.pending), 0)
            # One invalid reference is excluded from the first 64 examples.
            self.assertIn(63, [len(c.args[0]) for c in run.call_args_list])
            metric.update(generated[64:], labels[64:])
            self.assertEqual(len(metric.pending), 3)
            actual = metric.compute()
            self.assertTrue(all(len(c.args[0]) <= 64 for c in run.call_args_list))
            calls = run.call_count
            self.assertEqual(actual, metric.compute())
            self.assertEqual(run.call_count, calls)
        self.assertEqual(actual, expected)
        metric.update(generated[:32], labels[:32])
        metric.reset()
        self.assertEqual(metric.pending, [])
        self.assertTrue(metric.reference_cache)


@unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is required")
class TrainerIntegrationTests(unittest.TestCase):
    def test_generation_logging_epoch_reset_and_training(self):
        from omegaconf import OmegaConf
        from src.loss import Seq2SeqCrossEntropyLoss
        from src.trainer import Trainer

        metric = CodePassRate("artifacts/tokenizer.json", workers=2)
        tokenizer = metric.tokenizer

        def encode(codes):
            rows = [torch.tensor(tokenizer.encode(c, add_special_tokens=False)
                                 + [tokenizer.eos_token_id]) for c in codes]
            return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                                   padding_value=tokenizer.pad_token_id)

        predictions = encode(["def solve(): return 1", "def solve(): return 0", "def solve(): return 3"])
        labels = encode(["def solve(): return 1", "def solve(): return 2", "def solve(): return 3"])
        batches = [{"input_ids": torch.tensor(ids).reshape(-1, 1),
                    "attention_mask": torch.ones(len(ids), 1, dtype=torch.long),
                    "labels": labels[ids]} for ids in ([0, 1], [2])]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))
                self.generate_calls = 0

            def forward(self, labels, **batch):
                # Perfect teacher-forced predictions must NOT give a perfect pass rate.
                logits = torch.nn.functional.one_hot(labels, len(tokenizer)).float() * 20
                return {"logits": logits + self.bias}

            def generate(self, input_ids, attention_mask, max_new_tokens):
                assert not self.training
                assert max_new_tokens == 128
                self.generate_calls += 1
                return predictions[input_ids[:, 0]]

        class Writer:
            def __init__(self):
                self.scalars = {}

            def set_step(self, *args):
                pass

            def add_scalar(self, key, value):
                self.scalars[key] = value

        config = OmegaConf.create({
            "trainer": {"device_tensors": ["input_ids", "attention_mask", "labels"],
                        "n_epochs": 1, "save_period": 1, "save_dir": "saved",
                        "monitor": "off", "generation": {"max_new_tokens": 128}},
            "writer": {"loss_names": ["loss"], "run_name": "pass_test"},
        })
        model, writer = Model(), Writer()
        trainer = Trainer(
            model=model, criterion=Seq2SeqCrossEntropyLoss(pad_id=tokenizer.pad_token_id),
            metrics={"train": [], "inference": [], "generation": [metric]},
            optimizer=torch.optim.SGD(model.parameters(), lr=.01), lr_scheduler=None,
            config=config, device="cpu", dataloaders={"train": batches, "val": batches},
            logger=logging.getLogger("test"), writer=writer,
            batch_transforms={"train": None, "inference": None},
        )
        for epoch in (1, 2):
            result = trainer._evaluation_epoch(epoch, "val", batches)
            self.assertEqual(result["pass_at_1"], 2/3)
            self.assertEqual(result["pass_total"], 3)
            self.assertEqual(writer.scalars["pass_at_1"], 2/3)
            self.assertLess(result["loss"], 1e-3)
        self.assertEqual(model.generate_calls, 4)
        trainer.is_train = True
        model.train()
        trainer.process_batch(dict(batches[0]), trainer.train_metrics)
        self.assertEqual(model.generate_calls, 4)


if __name__ == "__main__":
    unittest.main()
