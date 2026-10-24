"""Choose one beam by voting over numeric results from the code executor."""

import math
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import torch
from tokenizers import Tokenizer

from src.utils.code_postprocessing import restore_generated_code
from src.utils.code_executor import SafeCodeExecutor
from src.utils.io_utils import ROOT_PATH


def rescore_lengths(scores, lengths, old_penalty, new_penalty):
    """Re-rank a fixed beam pool using its original generated token lengths."""
    return [
        score * max(1, length) ** (old_penalty - new_penalty)
        for score, length in zip(scores, lengths)
    ]


def same_result(left, right, rtol=1e-5, atol=1e-6):
    if left is None or right is None:
        return False
    return abs(left - right) <= max(
        Fraction(str(atol)), Fraction(str(rtol)) * max(abs(left), abs(right))
    )


def weighted_consensus_index(values, scores, temperature=1.0, rtol=1e-5, atol=1e-6):
    """Select with cached execution results; no code execution here."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    order = sorted(
        range(len(scores)),
        key=lambda j: scores[j] if math.isfinite(scores[j]) else -math.inf,
        reverse=True,
    )
    valid = [j for j in order if values[j] is not None and math.isfinite(scores[j])]
    if not valid:
        return order[0]
    groups = []
    for j in valid:
        for group in groups:
            if same_result(values[j], values[group[0]], rtol, atol):
                group.append(j)
                break
        else:
            groups.append([j])
    maximum = scores[valid[0]]
    best = max(
        groups,
        key=lambda group: (
            math.fsum(math.exp((scores[j] - maximum) / temperature) for j in group),
            scores[group[0]],
        ),
    )
    return best[0]


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

    def evaluate(self, beams, scores):
        """Decode and execute once; return CPU codes, results, lengths and scores."""
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
            codes = [restore_generated_code(code, add_imports=self.add_imports) for code in codes]
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

        k = beams.shape[1]
        values = [
            (
                results[code].value
                if done and math.isfinite(score) and results[code].status == "ok"
                else None
            )
            for code, done, score in zip(codes, complete, flat_scores)
        ]
        lengths = [
            row.index(self.eos) + 1 if done else len(row)
            for row, done in zip(rows, complete)
        ]
        return {
            "codes": [codes[i : i + k] for i in range(0, len(codes), k)],
            "results": [values[i : i + k] for i in range(0, len(values), k)],
            "lengths": [lengths[i : i + k] for i in range(0, len(lengths), k)],
            "scores": score_rows,
        }

    def select(self, beams, scores):
        """Execute candidates, then select the best code by weighted consensus."""
        details = self.evaluate(beams, scores)
        selected = [
            weighted_consensus_index(
                values, row_scores, self.temperature, self.rtol, self.atol
            )
            for values, row_scores in zip(details["results"], details["scores"])
        ]
        chosen = torch.tensor(selected, dtype=torch.long, device=beams.device)
        return beams[torch.arange(len(selected), device=beams.device), chosen]
