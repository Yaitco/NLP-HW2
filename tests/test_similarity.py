"""Check pooling and blockwise search against tiny independent CPU examples."""

from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch
from torch.nn import functional as F

from analyze_similarity import encode, encoder_dtype, main, masked_mean, merge_nearest
from src.datasets.tinysms import TinySMSDataset
from src.model import Seq2SeqTransformer, Seq2SeqTransformerConfig


class SimilarityTests(unittest.TestCase):
    def test_encoder_uses_fp16_when_cuda_has_no_native_bf16(self):
        self.assertEqual(encoder_dtype(torch.device("cpu")), torch.float32)
        with patch("torch.cuda.device", return_value=nullcontext()), patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ):
            self.assertEqual(encoder_dtype(torch.device("cuda")), torch.float16)
        with patch("torch.cuda.device", return_value=nullcontext()), patch(
            "torch.cuda.is_bf16_supported", return_value=True
        ):
            self.assertEqual(encoder_dtype(torch.device("cuda")), torch.bfloat16)

    def test_validation_neighbors_build_a_loadable_subset_and_frozen_checkpoint(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source_rows = [[4, 5], [4, 5], [6, 7], [4, 8], [8, 9, 10], [7, 8], [9, 10]]
            query_rows = [[4, 5], [6, 7]]
            train = Dataset.from_dict(
                {"input_ids": source_rows, "labels": [[4]] * len(source_rows)}
            )
            DatasetDict(
                train=train,
                validation=Dataset.from_dict(
                    {"input_ids": query_rows, "labels": [[4]] * 2}
                ),
            ).save_to_disk(str(folder / "data"))
            np.save(folder / "eligible.npy", np.array([0, 2, 3, 5, 6]))
            (folder / "eligible.json").write_text(
                json.dumps(
                    dict(
                        source_fingerprint=load_from_disk(
                            str(folder / "data/train")
                        )._fingerprint,
                        source_rows=len(train),
                        kept_rows=5,
                    )
                )
            )
            tokenizer = Tokenizer(
                WordLevel({str(i): i for i in range(17)}, unk_token="1")
            )
            tokenizer.save(str(folder / "tokenizer.json"))
            model = Seq2SeqTransformer(
                Seq2SeqTransformerConfig(
                    vocab_size=17,
                    max_src_len=12,
                    max_tgt_len=8,
                    d_model=16,
                    n_heads=2,
                    encoder_layers=1,
                    decoder_layers=1,
                    d_ff=32,
                    dropout=0.0,
                    pad_id=0,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                )
            ).eval()
            config = OmegaConf.create(
                dict(
                    model={
                        "_target_": "src.model.Seq2SeqTransformer",
                        "config": {
                            "_target_": "src.model.Seq2SeqTransformerConfig",
                            **asdict(model.config),
                        },
                    },
                    data={
                        "dataset_path": str(folder / "data"),
                        "tokenizer_path": str(folder / "tokenizer.json"),
                    },
                    datasets={"train": {"indices_path": str(folder / "eligible.npy")}},
                )
            )
            checkpoint = folder / "model.pth"
            torch.save(
                dict(epoch=7, config=config, state_dict=model.state_dict()), checkpoint
            )
            output = folder / "retrieval"
            with patch(
                "sys.argv",
                [
                    "analyze_similarity.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--chunk-size",
                    "2",
                    "--query-split",
                    "validation",
                    "--top-k",
                    "2",
                    "--exclude-identical-questions",
                    "--save-train-subset",
                ],
            ):
                main()
            actual = np.load(output / "nearest.npz")["train_indices"]
            candidates = torch.tensor([3, 5, 6])
            expected = (
                (
                    encode(model, query_rows, torch.device("cpu"))
                    @ encode(
                        model, [source_rows[i] for i in candidates], torch.device("cpu")
                    ).T
                )
                .topk(2, dim=1)
                .indices
            )
            np.testing.assert_array_equal(actual, candidates[expected].numpy())
            subset = TinySMSDataset(
                folder / "data", indices_path=output / "train_indices.npy"
            )
            self.assertEqual(len(subset), len(np.unique(actual)))
            self.assertEqual(
                [subset[i]["input_ids"] for i in range(len(subset))],
                [source_rows[i] for i in np.unique(actual)],
            )
            frozen = torch.load(output / "initial_model.pth", weights_only=False)
            self.assertEqual(frozen["epoch"], 7)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, frozen["state_dict"][key])
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["excluded_identical_train_rows"], 2)
            self.assertEqual(report["identical_question_in_top_k"], 0)

    def test_pooling_ignores_padding_and_returns_a_unit_vector(self):
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [100.0, -100.0]]])
        actual = masked_mean(hidden, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(actual, torch.full((1, 2), 2**-0.5))

    def test_block_search_matches_full_matrix_with_original_row_ids(self):
        generator = torch.Generator().manual_seed(42)
        queries = F.normalize(torch.randn(7, 9, generator=generator), dim=1)
        train = F.normalize(torch.randn(17, 9, generator=generator), dim=1)
        row_ids = torch.arange(17) * 3 + 7
        expected_scores, expected_ids = (queries @ train.T).topk(5, dim=1)
        scores = torch.full((7, 5), -torch.inf)
        ids = torch.full((7, 5), -1, dtype=torch.long)
        for start in range(0, len(train), 3):
            scores, ids = merge_nearest(
                queries,
                train[start : start + 3],
                row_ids[start : start + 3],
                scores,
                ids,
            )
        torch.testing.assert_close(scores, expected_scores)
        self.assertTrue(torch.equal(ids, row_ids[expected_ids]))

    def test_encoder_embedding_does_not_depend_on_other_rows_padding(self):
        torch.manual_seed(42)
        model = Seq2SeqTransformer(
            Seq2SeqTransformerConfig(
                vocab_size=17,
                max_src_len=12,
                max_tgt_len=8,
                pad_id=0,
                decoder_start_token_id=2,
                eos_token_id=3,
                d_model=16,
                n_heads=2,
                encoder_layers=2,
                decoder_layers=2,
                d_ff=32,
                dropout=0.0,
            )
        ).eval()
        single = encode(model, [[4, 5]], torch.device("cpu"))
        batched = encode(model, [[4, 5], [6, 7, 8, 9, 10]], torch.device("cpu"))
        torch.testing.assert_close(single[0], batched[0])


if __name__ == "__main__":
    unittest.main()
