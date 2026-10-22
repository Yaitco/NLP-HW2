"""Check pooling and blockwise search against tiny independent CPU examples."""

import unittest

import torch
from torch.nn import functional as F

from analyze_similarity import encode, masked_mean, merge_nearest
from src.model import Seq2SeqTransformer, Seq2SeqTransformerConfig


class SimilarityTests(unittest.TestCase):
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
