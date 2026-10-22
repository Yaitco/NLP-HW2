"""Prepare a numeric-only training selection from the existing reference audit.

No repairs, deduplication, length filtering, or model training. Full token-level
alignment is checked before using raw-dataset audit row indices.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from datasets import DatasetDict, DownloadConfig, load_dataset, load_from_disk
from src.analysis.code_audit import STATUS
from src.tokenizer import load_tokenizer
from src.utils.code_executor import SafeCodeExecutor
from src.utils.io_utils import ROOT_PATH


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_PATH / path


def verify_alignment(data, raw, tokenizer, batch_size=4096):
    """Every saved row must match the corresponding audited raw row."""
    eos = tokenizer.token_to_id("<eos>")
    for start in tqdm(range(0, len(data), batch_size), desc="Verify row alignment"):
        stop = min(start + batch_size, len(data))
        saved, original = data[start:stop], raw[start:stop]
        inputs = tokenizer.encode_batch(original["question"], add_special_tokens=False)
        targets = tokenizer.encode_batch(original["code"], add_special_tokens=False)
        for offset, (source, target) in enumerate(zip(inputs, targets)):
            labels = saved["labels"][offset]
            if saved["input_ids"][offset] != source.ids or labels not in (
                target.ids,
                target.ids + [eos],
            ):
                raise ValueError(
                    f"Audit/source alignment mismatch at row {start + offset}"
                )


def prepare(source, tokenizer_path, audit_dir, output, local_files_only=True):
    source, tokenizer_path, audit_dir, output = map(
        resolve, (source, tokenizer_path, audit_dir, output)
    )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output")
    audit = json.loads((audit_dir / "manifest.json").read_text())
    signature = audit["signature"]
    if not audit.get("complete"):
        raise ValueError("Reference audit is incomplete")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if tokenizer_sha != signature["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the one used in the audit")
    data = load_from_disk(str(source))
    if isinstance(data, DatasetDict):
        data = data["train"]
    raw = load_dataset(
        signature["dataset"],
        split="train",
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    if raw._fingerprint != signature["fingerprint"] or len(raw) != signature["rows"]:
        raise ValueError("Raw dataset differs from the audited revision")
    if len(data) > len(raw):
        raise ValueError("Saved training dataset is longer than audited raw train")
    tokenizer = load_tokenizer(tokenizer_path).backend_tokenizer
    verify_alignment(data, raw, tokenizer)
    statuses = np.load(audit_dir / "status.npy", mmap_mode="r", allow_pickle=False)
    if len(statuses) != len(raw):
        raise ValueError("Audit status file length mismatch")
    statuses = np.asarray(statuses[: len(data)])
    keep = statuses == STATUS["numeric_finite"]
    # Environment-sensitive and rare failures are rechecked in the current
    # per-program sandbox; common non-numeric outputs reuse the existing audit.
    rare = np.flatnonzero(np.isin(statuses, [2, 5, 6, 7, 8, 9, 10]))
    executor = SafeCodeExecutor(timeout=2.0, memory_mb=512)
    codes = [raw[int(index)]["code"] for index in rare]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            tqdm(
                pool.map(executor.run, codes),
                total=len(codes),
                desc="Recheck rare failures",
            )
        )
    rechecks = {}
    for index, result in zip(rare, results):
        keep[index] = result.status == "ok"
        rechecks[int(index)] = result.status
    indices = np.flatnonzero(keep).astype(np.int64)
    if not len(indices):
        raise ValueError("Cleaning would remove every row")
    manifest = {
        "policy": "finite_real_solve_only",
        "source_fingerprint": data._fingerprint,
        "source_rows": len(data),
        "source_path": str(source),
        "raw_fingerprint": raw._fingerprint,
        "kept_rows": len(indices),
        "removed_rows": int((~keep).sum()),
        "tokenizer_sha256": tokenizer_sha,
        "audit_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(statuses, return_counts=True))
        },
        "rare_rechecked": len(rare),
        "rare_results": dict(Counter(rechecks.values())),
        "recovered_rows": [int(i) for i in rare if keep[i]],
        "alignment": "all saved question/code token sequences checked against raw prefix",
    }
    output.mkdir(parents=True)
    np.save(output / "train_indices.npy", indices, allow_pickle=False)
    (output / "train_indices.json").write_text(json.dumps(manifest, indent=2) + "\n")
    names = {value: name for name, value in STATUS.items()}
    with (output / "excluded.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_index", "audit_status", "rechecked_status"])
        for index in np.flatnonzero(~keep):
            writer.writerow(
                [int(index), names[int(statuses[index])], rechecks.get(int(index), "")]
            )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="../hw2/artifacts/tokenized_dataset")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--audit-dir", default="artifacts/dataset_audit/train")
    parser.add_argument("--output", default="artifacts/clean_numeric_1m")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    prepare(
        args.source,
        args.tokenizer,
        args.audit_dir,
        args.output,
        local_files_only=not args.allow_download,
    )
