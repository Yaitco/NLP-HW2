"""Functional Pass@1 on autoregressive generations (not teacher-forced logits)."""

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from src.code_preprocessing import restore_code
from src.tokenizer import load_tokenizer
from src.utils.code_executor import SafeCodeExecutor


class CodePassRate:
    def __init__(self, tokenizer_path, timeout=2.0, memory_mb=512, workers=4,
                 rtol=1e-5, atol=1e-6, code_format="full", sandbox=True, add_imports=False):
        if code_format not in ("full", "solve_body_result"):
            raise ValueError("Unknown code_format")
        self.code_format = code_format
        self.add_imports = add_imports
        if workers < 1 or not isinstance(workers, int):
            raise ValueError("workers must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
            raise ValueError("rtol and atol must be finite and non-negative")
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.executor = SafeCodeExecutor(timeout=timeout, memory_mb=memory_mb, sandbox=sandbox)
        self.workers = workers
        self.rtol, self.atol = Fraction(str(rtol)), Fraction(str(atol))
        self.reference_cache = {}
        self.reset()

    def reset(self):
        # References are constant across epochs; never reuse generated outputs.
        self.counts = Counter()

    def _decode(self, rows):
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        clean = []
        for row in rows:
            if eos in row:
                row = row[:row.index(eos)]
            clean.append([token for token in row if token not in (-100, pad)])
        codes = self.tokenizer.batch_decode(
            clean, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return [restore_code(code, add_imports=self.add_imports) for code in codes] if self.code_format == "solve_body_result" else codes

    def update(self, generated_ids, labels):
        if labels is None:
            raise ValueError("Pass@1 requires reference labels; public test has none")
        predicted_rows = generated_ids.detach().cpu().tolist()
        label_rows = labels.detach().cpu().tolist()
        if len(predicted_rows) != len(label_rows):
            raise ValueError("Generated and reference batch sizes differ")
        predictions, references = self._decode(predicted_rows), self._decode(label_rows)
        # Each solution runs in a fresh subprocess.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            missing = list(dict.fromkeys(c for c in references if c not in self.reference_cache))
            self.reference_cache.update(zip(missing, pool.map(self.executor.run, missing)))
            candidates = []
            for row, code, reference in zip(predicted_rows, predictions, references):
                expected = self.reference_cache[reference]
                if expected.status != "ok":
                    self.counts["reference_errors"] += 1
                    self.counts[f"reference_{expected.status}"] += 1
                    continue
                self.counts["pass_total"] += 1
                if self.tokenizer.eos_token_id not in row:
                    self.counts["generation_truncated"] += 1
                    self.counts["prediction_errors"] += 1
                    continue
                candidates.append((code, expected.value))
            results = pool.map(self.executor.run, (code for code, _ in candidates))
            for (_, expected), actual in zip(candidates, results):
                if actual.status != "ok":
                    self.counts["prediction_errors"] += 1
                    self.counts[f"prediction_{actual.status}"] += 1
                elif abs(actual.value - expected) <= max(self.atol, self.rtol * abs(expected)):
                    self.counts["pass_correct"] += 1
                else:
                    self.counts["wrong_answer"] += 1

    def compute(self):
        keys = ("pass_correct", "pass_total", "reference_errors", "prediction_errors",
                "generation_truncated", "wrong_answer")
        result = {key: self.counts[key] for key in keys}
        result.update(self.counts)
        total = result["pass_total"]
        result["pass_at_1"] = result["pass_correct"] / total if total else float("nan")
        return result
