import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from torch.utils.data import Dataset

from datasets import DatasetDict, DownloadConfig, load_dataset, load_from_disk
from src.utils.io_utils import ROOT_PATH


class TinySMSDataset(Dataset):
    """Read the existing Arrow dataset without copying all rows to Python."""

    def __init__(self, path, split="train", limit=None, indices_path=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT_PATH / path
        data = load_from_disk(str(path))
        if isinstance(data, DatasetDict):
            data = data[split]
        if indices_path is not None:
            indices_path = Path(indices_path)
            if not indices_path.is_absolute():
                indices_path = ROOT_PATH / indices_path
            manifest = json.loads(indices_path.with_suffix(".json").read_text())
            if manifest["source_fingerprint"] != data._fingerprint or manifest[
                "source_rows"
            ] != len(data):
                raise ValueError("Clean indices belong to a different source dataset")
            indices = np.load(indices_path, allow_pickle=False)
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or len(indices) == 0
                or indices[0] < 0
                or indices[-1] >= len(data)
                or np.any(indices[1:] <= indices[:-1])
            ):
                raise ValueError(
                    "Clean indices must be nonempty, sorted, unique, and in range"
                )
            if len(indices) != manifest["kept_rows"]:
                raise ValueError("Clean indices do not match the manifest")
            data = data.select(indices)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive or null")
            data = data.select(range(min(limit, len(data))))
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_lengths(self):
        """Read Arrow offsets, without materializing every token list in Python."""
        table = self.data.data
        source = pc.list_value_length(table.column("input_ids")).to_numpy()
        if "labels" in table.column_names and table.column("labels").null_count == 0:
            # +1 accounts for EOS appended by the collator to old targets.
            target = pc.list_value_length(table.column("labels")).to_numpy() + 1
            source = np.maximum(source, target)
        # Dataset.select can retain the original storage with an index mapping.
        if self.data._indices is not None:
            source = source[self.data._indices.column(0).to_numpy()]
        return source

    def __getitem__(self, index):
        row = self.data[index]
        result = {"input_ids": row["input_ids"]}
        result["attention_mask"] = row.get(
            "attention_mask", [1] * len(row["input_ids"])
        )
        if row.get("labels") is not None:
            result["labels"] = row["labels"]
        return result


class TinySMSRawDataset(TinySMSDataset):
    """Tokenize a small held-out slice without rewriting the saved training set."""

    def __init__(
        self, path, tokenizer_path, start, limit, split="train", local_files_only=False
    ):
        from src.tokenizer import load_tokenizer

        if start < 0 or limit <= 0:
            raise ValueError("start must be non-negative and limit positive")
        data = load_dataset(
            path,
            split=split,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
        if start + limit > len(data):
            raise ValueError("Requested validation range exceeds dataset length")
        data = data.select(range(start, start + limit))
        if any(code is None for code in data["code"]):
            raise ValueError("Validation needs reference code; test labels are missing")
        tokenizer = load_tokenizer(tokenizer_path).backend_tokenizer

        def tokenize(batch):
            inputs = tokenizer.encode_batch(batch["question"], add_special_tokens=False)
            labels = tokenizer.encode_batch(batch["code"], add_special_tokens=False)
            return {
                "input_ids": [e.ids for e in inputs],
                "attention_mask": [e.attention_mask for e in inputs],
                "labels": [e.ids for e in labels],
            }

        self.data = data.map(tokenize, batched=True, remove_columns=data.column_names)
