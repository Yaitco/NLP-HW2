"""Save length-filter indices without copying an already tokenized dataset."""

import argparse
import json

import numpy as np
import pyarrow.compute as pc
import yaml

from datasets import load_from_disk
from src.utils.io_utils import ROOT_PATH


def filter_lengths(dataset_path, output, max_length=128, splits=("train",)):
    """Limit source and target separately; labels must already include EOS."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not splits or any(split not in ("train", "validation") for split in splits):
        raise ValueError(
            "Choose train and/or validation; keep the public test complete"
        )
    source, output = ROOT_PATH / dataset_path, ROOT_PATH / output
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    data = load_from_disk(str(source))
    reports = {}
    for split in splits:
        dataset = data[split]
        inputs = pc.list_value_length(dataset.data.column("input_ids")).to_numpy()
        targets = pc.list_value_length(dataset.data.column("labels")).to_numpy()
        if dataset._indices is not None:
            mapping = dataset._indices.column(0).to_numpy()
            inputs, targets = inputs[mapping], targets[mapping]
        keep = (inputs <= max_length) & (targets <= max_length)
        indices = np.flatnonzero(keep).astype(np.int64)
        if not len(indices):
            raise ValueError(f"Length filter would remove every {split} row")
        report = {
            "source_path": str(source),
            "source_split": split,
            "source_fingerprint": dataset._fingerprint,
            "source_rows": len(dataset),
            "kept_rows": len(indices),
            "removed_rows": int((~keep).sum()),
            "max_input_tokens": max_length,
            "max_target_tokens_with_eos": max_length,
            "mean_target_tokens_with_eos": float(targets[keep].mean()),
        }
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / f"{split}_indices.npy", indices, allow_pickle=False)
        (output / f"{split}_indices.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reports[split] = report
    print(json.dumps(reports, indent=2), flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/filter_length_128_16k.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT_PATH / args.config).read_text())
    filter_lengths(**config)
