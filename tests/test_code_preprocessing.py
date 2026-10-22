"""Normalization semantics on synthetic solutions, never dataset execution."""

import io
import json
import tempfile
import tokenize
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.code_preprocessing import normalize_code, restore_code
from src.tokenizer import prepare_code_dataset, tokenize_dataset, train_tokenizer


class CodeNormalizationTests(unittest.TestCase):
    def assert_equivalent(self, source):
        # These snippets are handwritten unit-test fixtures, not dataset programs.
        original, restored = {}, {}
        body = normalize_code(source)
        exec(source, original)
        exec(restore_code(body), restored)
        self.assertEqual(original["solve"](), restored["solve"]())
        self.assertFalse(
            any(
                t.type == tokenize.COMMENT
                for t in tokenize.generate_tokens(
                    io.StringIO(restore_code(body)).readline
                )
            )
        )
        return body

    def test_final_expression_and_comments(self):
        body = self.assert_equivalent(
            "# header\ndef solve() -> int:\n a=10 # miles\n return a / 5.5 # answer\n"
        )
        self.assertIn("    result = a / 5.5\n", body)
        self.assertNotIn("def solve", body)
        self.assertNotIn("return", body)

    def test_existing_result_and_multiline_final_expression(self):
        body = self.assert_equivalent("def solve():\n result=3\n return result\n")
        self.assertEqual(body, "    result = 3\n")
        self.assert_equivalent("def solve():\n x=3\n return (\n x +\n 4\n )\n")

    def test_string_hashes_and_multiline_whitespace_are_preserved(self):
        body = self.assert_equivalent(
            'def solve():\n s = """# literal\n    keep these spaces\n      and these\n"""\n return s # real comment\n'
        )
        self.assertIn("# literal", body)
        self.assertNotIn("real comment", body)

    def test_imports_and_early_returns(self):
        self.assert_equivalent("def solve():\n import math\n return math.sqrt(9)\n")
        self.assert_equivalent("def solve():\n if True:\n  return 7\n return 4\n")
        self.assert_equivalent(
            "def solve():\n if True:\n  return 7\n else:\n  return 4\n"
        )
        self.assert_equivalent("def solve():\n pass\n")

    def test_missing_code_and_unsupported_structure(self):
        self.assertIsNone(normalize_code(None))
        for source in [
            "def broken(:",
            "import math\ndef solve(): return 1",
            "def solve(x=1): return x",
            "@decorator\ndef solve(): return 1",
        ]:
            with self.subTest(source=source), self.assertRaises(
                (SyntaxError, ValueError)
            ):
                normalize_code(source)

    def test_unrelated_functions_before_and_after_solve_are_removed(self):
        for source in [
            "def unrelated() -> int:\n return 99\ndef solve() -> int:\n result=21\n return result\n",
            "def solve() -> int:\n result=21\n return result\ndef unrelated() -> int:\n return 99\n",
        ]:
            with self.subTest(source=source):
                body = self.assert_equivalent(source)
                self.assertEqual(body, "    result = 21\n")

    def test_required_functions_and_executable_definitions_are_not_dropped(self):
        for source in [
            "def helper(): return 7\ndef solve(): return helper()",
            "def helper(): return 7\ndef solve(): return globals()['helper']()",
            "def solve(): return 7\ndef extra(x=print('side effect')): return x",
            "def solve(): return 7\n@decorator\ndef extra(): return 9",
            "def solve(): return 7\ndef extra() -> annotation_factory(): return 9",
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize_code(source)

    def test_preparation_tokenizer_corpus_and_tokenized_labels(self):
        train = Dataset.from_dict(
            {
                "question": ["q0", "q1"],
                "code": [
                    "def solve():\n x=2 # units\n return x\n",
                    "def solve(): return 3\ndef unrelated(): return 99\n",
                ],
            }
        )
        validation = Dataset.from_dict(
            {"question": ["val"], "code": ["def solve(): return 'not numeric'\n"]}
        )
        test = Dataset.from_dict({"question": ["public"], "code": [None]})
        data = DatasetDict(train=train, validation=validation, test=test)

        class FakeTokenizer:
            def train_from_iterator(self, batches, trainer, length):
                self.texts = [text for batch in batches for text in batch]

            def save(self, path):
                Path(path).write_text("fake tokenizer for tests")

            def token_to_id(self, token):
                return 3

            def encode_batch(self, texts, **kwargs):
                return [
                    SimpleNamespace(
                        ids=[ord(c) for c in text], attention_mask=[1] * len(text)
                    )
                    for text in texts
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "dataset": {
                    "path": "fake",
                    "question_column": "question",
                    "code_column": "code",
                },
                "tokenizer": {
                    "output_path": str(root / "tokenizer.json"),
                    "vocab_size": 512,
                    "min_frequency": 2,
                    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                    "training_examples": 100000,
                },
                "preprocessing": {
                    "code_normalization": {
                        "output_path": str(root / "normalized"),
                    },
                    "batch_size": 2,
                    "num_proc": None,
                    "output_path": str(root / "tokenized"),
                    "splits": {"train": {"max_examples": None}},
                },
            }
            with patch("src.tokenizer.datasets.load_dataset", return_value=data):
                prepare_code_dataset(config)
            from datasets import load_from_disk

            normalized = load_from_disk(str(root / "normalized"))
            self.assertEqual(normalized["train"][0]["original_code"], train[0]["code"])
            self.assertEqual(
                normalized["train"][0]["code"], "    x = 2\n    result = x\n"
            )
            self.assertEqual(normalized["test"][0]["code"], None)
            self.assertEqual(len(normalized["validation"]), 1)
            with patch(
                "src.code_preprocessing.normalize_batch",
                side_effect=AssertionError("No repeated formatting"),
            ):
                prepare_code_dataset(config)
            fake = FakeTokenizer()
            with patch("src.tokenizer.Tokenizer", return_value=fake), patch(
                "src.tokenizer.build_tokenizer",
                return_value=SimpleNamespace(backend_tokenizer=fake),
            ):
                train_tokenizer(config)
                self.assertEqual(
                    set(fake.texts),
                    {"q0", "q1", "    x = 2\n    result = x\n", "    result = 3\n"},
                )
                tokenize_dataset(config)
            encoded = load_from_disk(str(root / "tokenized"))
            self.assertEqual(
                encoded["train"][0]["labels"],
                [ord(c) for c in normalized["train"][0]["code"]] + [3],
            )
            self.assertIsNone(encoded["test"][0]["labels"])
            metadata = json.loads((root / "tokenized/preparation.json").read_text())
            self.assertEqual(metadata["code_format"], "solve_body_result")
            self.assertEqual(metadata["normalizer"], "ast_unparse")
            self.assertEqual(metadata["rows"], {"train": 2, "validation": 1, "test": 1})
            manifest_path = root / "normalized/preparation.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("normalizer")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different configuration"):
                prepare_code_dataset(config)


if __name__ == "__main__":
    unittest.main()
