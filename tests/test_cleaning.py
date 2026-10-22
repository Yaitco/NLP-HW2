"""Numeric-only filtering and dataset alignment on synthetic rows."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from datasets import Dataset, DatasetDict
from filter_lengths import filter_lengths
from prepare_clean_dataset import prepare, verify_alignment
from src.datasets.tinysms import TinySMSDataset
from src.tokenizer import select_numeric_train, tokenize_dataset, train_tokenizer
from src.utils.code_executor import ExecutionResult


class FakeTokenizer:
    def token_to_id(self, token):
        return 3

    def train_from_iterator(self, batches, trainer, length):
        self.training_texts = [text for batch in batches for text in batch]
        self.requested_vocab_size = trainer.vocab_size
        assert len(self.training_texts) == length

    def save(self, path):
        Path(path).write_text("new 16k tokenizer")

    def encode_batch(self, texts, **kwargs):
        return [
            SimpleNamespace(ids=[ord(c) for c in text], attention_mask=[1] * len(text))
            for text in texts
        ]


class CleaningTests(unittest.TestCase):
    def test_length_filter_keeps_128_including_eos_without_truncation(self):
        raw = Dataset.from_dict(
            {
                "input_ids": [[10] * 128, [11] * 129, [12], [13] * 127],
                "labels": [
                    [20] * 127 + [3],
                    [21] * 127 + [3],
                    [22] * 128 + [3],
                    [23, 23, 3],
                ],
            }
        )
        test = Dataset.from_dict({"input_ids": [[10] * 200], "labels": [None]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "indices"
            DatasetDict(
                train=raw, validation=raw.select([0, 2]), test=test
            ).save_to_disk(source)
            report = filter_lengths(source, output)
            selected = TinySMSDataset(source, indices_path=output / "train_indices.npy")
            self.assertEqual(len(selected), 2)
            self.assertEqual(selected[0]["labels"], raw[0]["labels"])
            self.assertEqual(len(selected[0]["labels"]), 128)
            self.assertEqual(selected[1]["labels"], [23, 23, 3])
            self.assertEqual(report["train"]["removed_rows"], 2)
            self.assertEqual(len(TinySMSDataset(source)), 4)
            self.assertEqual(len(TinySMSDataset(source, split="validation")), 2)
            self.assertEqual(
                len(TinySMSDataset(source, split="test")[0]["input_ids"]), 200
            )
            self.assertFalse((output / "validation_indices.npy").exists())
            both = root / "both"
            filter_lengths(source, both, splits=("train", "validation"))
            self.assertEqual(
                len(
                    TinySMSDataset(
                        source,
                        split="validation",
                        indices_path=both / "validation_indices.npy",
                    )
                ),
                1,
            )

    def test_full_tokenization_excludes_validation_and_retains_public_test(self):
        raw = Dataset.from_dict(
            {
                "question": [f"q{i}" for i in range(6)],
                "code": [f"c{i}" for i in range(6)],
            }
        )
        public_test = Dataset.from_dict({"question": ["test"], "code": [None]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer, audit_dir, output = (
                root / "tokenizer.json",
                root / "audit",
                root / "full",
            )
            audit_dir.mkdir()
            signature = {
                "dataset": "fake",
                "fingerprint": raw._fingerprint,
                "rows": 6,
                "tokenizer_sha256": hashlib.sha256(
                    b"old 8k audit tokenizer"
                ).hexdigest(),
            }
            (audit_dir / "manifest.json").write_text(
                json.dumps({"complete": True, "signature": signature})
            )
            np.save(
                audit_dir / "status.npy", np.array([0, 0, 3, 0, 2, 0], dtype=np.uint8)
            )
            rechecked = root / "rechecked.json"
            rechecked.write_text(
                json.dumps(
                    {
                        "raw_fingerprint": raw._fingerprint,
                        "tokenizer_sha256": signature["tokenizer_sha256"],
                        "recovered_rows": [4],
                    }
                )
            )
            config = {
                "dataset": {
                    "path": "fake",
                    "question_column": "question",
                    "code_column": "code",
                    "local_files_only": True,
                },
                "tokenizer": {
                    "output_path": str(tokenizer),
                    "vocab_size": 16384,
                    "min_frequency": 2,
                    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                    "training_examples": 100000,
                },
                "preprocessing": {
                    "output_path": str(output),
                    "splits": {"train": {"max_examples": None}},
                    "num_proc": None,
                    "batch_size": 2,
                    "cleaning": {
                        "audit_dir": str(audit_dir),
                        "rechecked_manifest": str(rechecked),
                        "validation_start": 1,
                        "validation_size": 2,
                    },
                },
            }
            backend = FakeTokenizer()
            with patch(
                "src.tokenizer.datasets.load_dataset",
                side_effect=lambda *args, **kwargs: DatasetDict(
                    train=raw, test=public_test
                ),
            ), patch("src.tokenizer.Tokenizer", return_value=backend), patch(
                "src.tokenizer.build_tokenizer",
                return_value=SimpleNamespace(backend_tokenizer=backend),
            ):
                train_tokenizer(config)
                self.assertEqual(backend.requested_vocab_size, 16384)
                self.assertEqual(
                    set(backend.training_texts),
                    {f"{prefix}{i}" for prefix in ("q", "c") for i in (0, 3, 4, 5)},
                )
                tokenize_dataset(config)
            train = TinySMSDataset(output)
            val = TinySMSDataset(output, split="validation")
            test = TinySMSDataset(output, split="test")
            self.assertEqual(
                [row["input_ids"] for row in train],
                [[ord(c) for c in f"q{i}"] for i in (0, 3, 4, 5)],
            )
            self.assertEqual(
                [row["input_ids"] for row in val],
                [[ord(c) for c in f"q{i}"] for i in (1, 2)],
            )
            self.assertTrue(all(row["labels"][-1] == 3 for row in train))
            self.assertNotIn("labels", test[0])
            report = json.loads((output / "preparation.json").read_text())
            self.assertEqual(report["train_rows"], 4)
            self.assertEqual(
                report["audit_tokenizer_sha256"], signature["tokenizer_sha256"]
            )
            self.assertEqual(
                report["tokenizer_sha256"],
                hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            )
            self.assertNotEqual(
                report["tokenizer_sha256"], report["audit_tokenizer_sha256"]
            )
            self.assertEqual(report["code_tokens_with_eos"]["mean"], 3.0)
            signature["fingerprint"] = "different raw dataset"
            (audit_dir / "manifest.json").write_text(
                json.dumps({"complete": True, "signature": signature})
            )
            with self.assertRaisesRegex(ValueError, "raw dataset must match"):
                select_numeric_train(DatasetDict(train=raw, test=public_test), config)

    def test_alignment_checks_every_row(self):
        raw = Dataset.from_dict({"question": ["a", "b", "c"], "code": ["x", "y", "z"]})
        data = Dataset.from_dict(
            {"input_ids": [[97], [98], [99]], "labels": [[120], [121, 3], [122]]}
        )
        verify_alignment(data, raw, FakeTokenizer(), batch_size=2)
        swapped = data.select([0, 2, 1])
        with self.assertRaisesRegex(ValueError, "row 1"):
            verify_alignment(swapped, raw, FakeTokenizer(), batch_size=2)

    def test_numeric_filter_keeps_duplicates_long_rows_and_original_tokens(self):
        raw = Dataset.from_dict(
            {
                "question": ["same", "tuple", "same", "long" * 300, "complex"],
                "code": ["one", "tuple", "one", "recover", "complex"],
            }
        )
        data = Dataset.from_dict(
            {
                "input_ids": [[ord(c) for c in x] for x in raw["question"]],
                "labels": [[ord(c) for c in x] for x in raw["code"]],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audit, output = root / "source", root / "audit", root / "clean"
            DatasetDict(train=data).save_to_disk(source)
            audit.mkdir()
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            signature = {
                "dataset": "fake",
                "fingerprint": raw._fingerprint,
                "rows": len(raw),
                "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            }
            (audit / "manifest.json").write_text(
                json.dumps({"complete": True, "signature": signature})
            )
            np.save(audit / "status.npy", np.array([0, 3, 0, 2, 7], dtype=np.uint8))
            executor = SimpleNamespace(
                run=lambda code: (
                    ExecutionResult("ok")
                    if code == "recover"
                    else ExecutionResult("non_numeric")
                )
            )
            with patch("prepare_clean_dataset.load_dataset", return_value=raw), patch(
                "prepare_clean_dataset.load_tokenizer",
                return_value=SimpleNamespace(backend_tokenizer=FakeTokenizer()),
            ), patch("prepare_clean_dataset.SafeCodeExecutor", return_value=executor):
                manifest = prepare(source, tokenizer, audit, output)
            selected = TinySMSDataset(source, indices_path=output / "train_indices.npy")
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[0], selected[1])
            self.assertEqual(selected[2]["input_ids"], data[3]["input_ids"])
            self.assertEqual(len(selected[2]["input_ids"]), 1200)
            np.testing.assert_array_equal(selected.get_lengths(), [4, 4, 1200])
            self.assertEqual(manifest["recovered_rows"], [3])
            self.assertEqual(len(TinySMSDataset(source)), 5)
            metadata = output / "train_indices.json"
            wrong = json.loads(metadata.read_text())
            wrong["source_fingerprint"] = "other"
            metadata.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "different source"):
                TinySMSDataset(source, indices_path=output / "train_indices.npy")


if __name__ == "__main__":
    unittest.main()
