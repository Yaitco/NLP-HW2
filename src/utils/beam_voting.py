"""Choose one beam by voting over numeric results from the code executor."""

import math
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import torch
from tokenizers import Tokenizer

from src.code_preprocessing import restore_code
from src.utils.code_executor import SafeCodeExecutor
from src.utils.io_utils import ROOT_PATH


class BeamVoter:
    def __init__(
        self,
        tokenizer_path,
        code_format="full",
        timeout=2.0,
        memory_mb=512,
        workers=4,
        rtol=1e-5,
        atol=1e-6,
        sandbox=True,
        execution_batch_size=16,
        add_imports=False,
        temperature=1.0,
    ):
        if code_format not in ("full", "solve_body_result"):
            raise ValueError("Unknown code_format")
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(execution_batch_size, int) or execution_batch_size < 1:
            raise ValueError("execution_batch_size must be a positive integer")
        if any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
            raise ValueError("rtol and atol must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature
        self.tokenizer = Tokenizer.from_file(str(ROOT_PATH / tokenizer_path))
        self.eos = self.tokenizer.token_to_id("<eos>")
        self.pad = self.tokenizer.token_to_id("<pad>")
        if self.eos is None or self.pad is None:
            raise ValueError("Tokenizer must define <eos> and <pad>")
        self.code_format, self.workers = code_format, workers
        self.add_imports = add_imports
        self.execution_batch_size = 1 if sandbox else execution_batch_size
        self.rtol, self.atol = Fraction(str(rtol)), Fraction(str(atol))
        self.executor = SafeCodeExecutor(
            timeout=timeout, memory_mb=memory_mb, sandbox=sandbox
        )

    def select(self, beams, scores):
        """IDs [B, K, T] + length-normalized scores [B, K] -> selected IDs [B, T].

        Each valid candidate contributes exp((score - max_score) / temperature).
        Identical code is executed once, but each candidate contributes its weight.
        Ties use the best beam score; all failures fall back to the original best.
        """
        if (
            beams.ndim != 3
            or scores.ndim != 2
            or beams.shape[:2] != scores.shape
            or not beams.shape[1]
        ):
            raise ValueError("Expected beams [B, K, T] and scores [B, K]")
        rows = beams.detach().cpu().flatten(0, 1).tolist()
        score_rows = scores.detach().cpu().tolist()
        flat_scores = [score for row in score_rows for score in row]
        complete = [self.eos in row for row in rows]
        clean = [
            (
                [token for token in row[: row.index(self.eos)] if token != self.pad]
                if done
                else []
            )
            for row, done in zip(rows, complete)
        ]
        codes = self.tokenizer.decode_batch(clean, skip_special_tokens=True)
        if self.code_format == "solve_body_result":
            codes = [restore_code(code, add_imports=self.add_imports) for code in codes]
        runnable = list(
            dict.fromkeys(
                code
                for code, done, score in zip(codes, complete, flat_scores)
                if done and math.isfinite(score)
            )
        )
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            if self.execution_batch_size == 1:
                values = pool.map(self.executor.run, runnable)
            else:
                chunks = [
                    runnable[start : start + self.execution_batch_size]
                    for start in range(0, len(runnable), self.execution_batch_size)
                ]
                values = (
                    value
                    for chunk in pool.map(self.executor.run_batch, chunks)
                    for value in chunk
                )
            results = dict(zip(runnable, values))

        selected, k = [], beams.shape[1]
        for batch_index, row_scores in enumerate(score_rows):
            order = sorted(
                range(k),
                key=lambda i: (
                    row_scores[i] if math.isfinite(row_scores[i]) else -math.inf
                ),
                reverse=True,
            )
            valid = [
                index
                for index in order
                if complete[batch_index * k + index]
                and math.isfinite(row_scores[index])
                and results[codes[batch_index * k + index]].status == "ok"
            ]
            max_score = row_scores[valid[0]] if valid else 0.0
            groups = []
            for index in valid:
                flat = batch_index * k + index
                code = codes[flat]
                result = results.get(code)
                weight = math.exp((row_scores[index] - max_score) / self.temperature)
                for group in groups:
                    value = group["value"]
                    if abs(result.value - value) <= max(
                        self.atol, self.rtol * max(abs(result.value), abs(value))
                    ):
                        group["members"].append(index)
                        group["weights"].append(weight)
                        break
                else:
                    groups.append(
                        {"value": result.value, "members": [index], "weights": [weight]}
                    )
            winner = (
                max(
                    groups,
                    key=lambda group: (
                        math.fsum(group["weights"]),
                        row_scores[group["members"][0]],
                    ),
                )
                if groups
                else None
            )
            selected.append(winner["members"][0] if winner else order[0])
        chosen = torch.tensor(selected, dtype=torch.long, device=beams.device)
        return beams[torch.arange(len(selected), device=beams.device), chosen]
