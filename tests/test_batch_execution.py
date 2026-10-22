"""Batch AST execution preserves numeric results, deadlines and candidate order."""

import subprocess
import time
import unittest
from fractions import Fraction
from unittest.mock import patch

from src.utils.code_executor import ExecutionResult, SafeCodeExecutor


class BatchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = SafeCodeExecutor(sandbox=False)

    def test_arithmetic_errors_and_numeric_types_match_separate_processes(self):
        codes = [
            "def solve():\n    a = 12\n    b = 5\n    return a / b",
            "def solve(): return round(sum(range(10)) / 7, 3)",
            "def solve(): return 2**100 + 1",
            "def solve(): return [2, 3, 4][1]",
            "def solve():\n    result = 0\n    for x in range(5): result += x\n    return result",
            "def solve():\n    if 3 > 2: return 7\n    return 9",
            "def solve(): return 1 / 0",
            "def solve(): return True",
            "def solve(): return float('nan')",
            "def solve(): return float('inf')",
            "def solve(): return '4'",
            "def solve(:",
        ]
        expected = [self.executor.run(code) for code in codes]
        original = subprocess.Popen
        with patch("src.utils.code_executor.subprocess.Popen", wraps=original) as popen:
            actual = self.executor.run_batch(codes)
        self.assertEqual(actual, expected)
        self.assertEqual(popen.call_count, 1)

    def test_candidate_globals_are_separate(self):
        codes = [
            "def solve():\n    value = 10\n    return value",
            "def solve(): return value",
            "def solve(): return _candidate_0",
            "def solve(): return solve",
        ]
        self.assertEqual(
            self.executor.run_batch(codes), [self.executor.run(code) for code in codes]
        )

    def test_timeout_is_per_candidate_and_later_candidates_still_run(self):
        executor = SafeCodeExecutor(timeout=0.5, sandbox=False)
        codes = [
            "def solve(): return 1",
            "def solve():\n    while True: pass",
            "def solve(): return 3",
        ]
        original = subprocess.Popen
        started = time.monotonic()
        with patch("src.utils.code_executor.subprocess.Popen", wraps=original) as popen:
            actual = executor.run_batch(codes)
        self.assertEqual(
            actual,
            [
                ExecutionResult("ok", Fraction(1)),
                ExecutionResult("timeout"),
                ExecutionResult("ok", Fraction(3)),
            ],
        )
        self.assertEqual(popen.call_count, 1)
        self.assertLess(time.monotonic() - started, 2)

    def test_imports_and_module_side_effects_keep_fresh_process_semantics(self):
        codes = [
            "def solve():\n    import math\n    math.sqrt = lambda x: 999\n    return 0",
            "def solve():\n    import math\n    return math.sqrt(4)",
            "value = 17\ndef solve(): return value",
            "def helper(): return 5\ndef solve(): return helper()",
        ]
        self.assertEqual(
            self.executor.run_batch(codes), [self.executor.run(code) for code in codes]
        )

    def test_compiler_semantic_error_does_not_drop_other_candidates(self):
        codes = ["def solve(): return 7", "def solve(): break", "def solve(): return 9"]
        self.assertEqual(
            self.executor.run_batch(codes), [self.executor.run(code) for code in codes]
        )

    def test_empty_invalid_and_oversized_inputs(self):
        self.assertEqual(self.executor.run_batch([]), [])
        codes = [
            None,
            "",
            "a" * (self.executor.max_code_bytes + 1),
            "def solve(): return 1",
        ]
        self.assertEqual(
            self.executor.run_batch(codes), [self.executor.run(code) for code in codes]
        )

    def test_partial_worker_output_recovers_remaining_candidates(self):
        codes = [
            "def solve(): return 1",
            "def solve(): return 2",
            "def solve(): return 3",
        ]
        original = self.executor._run_process

        def partial(payload, *args):
            if "codes" in payload:
                return (
                    b'{"status": "ok", "numerator": "1", "denominator": "1"}\n{"status":',
                    "worker_failure",
                )
            return original(payload, *args)

        with patch.object(
            self.executor, "_run_process", side_effect=partial
        ), patch.object(self.executor, "run", wraps=self.executor.run) as run:
            actual = self.executor.run_batch(codes)
        self.assertEqual(
            actual, [ExecutionResult("ok", Fraction(i)) for i in (1, 2, 3)]
        )
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
