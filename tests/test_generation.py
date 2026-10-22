"""Cached decoding and beam selection on tiny CPU models; no datasets/checkpoints."""

import math
import unittest
from unittest.mock import patch

import torch

from src.model.transformer import Seq2SeqTransformer, Seq2SeqTransformerConfig


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Seq2SeqTransformer(
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
        self.inputs = torch.tensor([[4, 5, 0, 0], [6, 7, 8, 0]])
        self.mask = self.inputs != 0

    @torch.no_grad()
    def test_each_cached_step_matches_full_forward_even_with_padding(self):
        model = self.model
        mask = model.make_source_allowed_mask(self.mask)
        memory = model.encoder(model.src_pos_embedding(self.inputs), mask)
        targets = torch.tensor([[2, 4, 0, 7, 8], [2, 0, 5, 9, 6]])
        cache = None
        cross_keys = None
        for size in range(1, targets.shape[1] + 1):
            prefix = targets[:, :size]
            actual, cache = model._decode_next(prefix, memory, mask, cache, True)
            expected = model(self.inputs, self.mask, decoder_input_ids=prefix)[
                "logits"
            ][:, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(cache[0][0][0].shape[2], size)
            if cross_keys is not None:
                self.assertIs(cache[0][1][0], cross_keys)
            cross_keys = cache[0][1][0]

    @torch.no_grad()
    def full_prefix_greedy(self, steps):
        # Independent reference using the training forward and the entire prefix.
        tokens = torch.full((len(self.inputs), 1), 2, dtype=torch.long)
        finished = torch.zeros(len(self.inputs), dtype=torch.bool)
        for _ in range(min(steps, self.model.config.max_tgt_len)):
            logits = self.model(self.inputs, self.mask, decoder_input_ids=tokens)[
                "logits"
            ]
            next_token = torch.where(finished, 0, logits[:, -1].argmax(-1))
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            finished |= next_token == 3
            if finished.all():
                break
        return tokens[:, 1:]

    def test_greedy_matches_original_algorithm_and_respects_context_limit(self):
        for ties in (False, True):
            if ties:
                self.model.lm_head.weight.data.zero_()
            expected = self.full_prefix_greedy(12)
            for use_cache in (False, True):
                actual = self.model.generate_greedy(
                    self.inputs, self.mask, 12, use_cache=use_cache
                )
                self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(expected.shape, (2, 8))
        self.assertEqual(
            self.model.generate(self.inputs, max_new_tokens=0).shape, (2, 0)
        )

    def test_beam_cache_reordering_matches_full_prefix_at_every_step(self):
        original = self.model._decode_next
        calls = []

        def checked(tokens, memory, mask, cache, use_cache):
            actual, next_cache = original(tokens, memory, mask, cache, use_cache)
            expected, _ = original(tokens, memory, mask, None, False)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            calls.append(tokens.clone())
            return actual, next_cache

        with patch.object(self.model, "_decode_next", side_effect=checked):
            cached = self.model.generate(
                self.inputs, self.mask, 6, num_beams=4, length_penalty=0.7
            )
        uncached = self.model.generate(
            self.inputs, self.mask, 6, num_beams=4, length_penalty=0.7, use_cache=False
        )
        self.assertTrue(torch.equal(cached, uncached))
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[-1].shape[0], 8)

    def test_cross_projections_run_once_per_layer_per_request(self):
        calls = []
        hooks = [
            block.ca.k_proj.register_forward_hook(lambda *args: calls.append(1))
            for block in self.model.decoder.blocks
        ]
        try:
            for _ in range(2):
                self.model.generate(self.inputs, self.mask, 4, num_beams=3)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(calls), 4)  # 2 layers * 2 independent requests.

    def test_beam_search_can_beat_greedy(self):
        def logits(tokens, *args):
            scores = torch.full((len(tokens), 17), -100.0)
            if tokens.shape[1] == 1:
                scores[:, 4], scores[:, 5] = math.log(0.6), math.log(0.4)
            else:
                scores[:] = 0
                scores[tokens[:, -1] == 5] = -100
                scores[tokens[:, -1] == 5, 3] = 0
            return scores, None

        with patch.object(self.model, "_decode_next", side_effect=logits):
            greedy = self.model.generate(self.inputs, max_new_tokens=2)
            beam = self.model.generate(self.inputs, max_new_tokens=2, num_beams=2)
        self.assertTrue(torch.equal(greedy, torch.tensor([[4, 0], [4, 0]])))
        self.assertTrue(torch.equal(beam, torch.tensor([[5, 3], [5, 3]])))

    def test_finished_beams_pad_and_length_penalty_changes_ranking(self):
        def logits(tokens, *args):
            scores = torch.full((len(tokens), 17), -100.0)
            if tokens.shape[1] == 1:
                scores[:, 4], scores[:, 3] = math.log(0.6), math.log(0.4)
            else:
                scores[:] = math.log(0.5 / 16)
                scores[:, 3] = math.log(0.5)
            return scores, None

        with patch.object(self.model, "_decode_next", side_effect=logits):
            short = self.model.generate(
                self.inputs, max_new_tokens=4, num_beams=2, length_penalty=0
            )
            long = self.model.generate(
                self.inputs, max_new_tokens=4, num_beams=2, length_penalty=2
            )
        self.assertTrue(torch.equal(short, torch.tensor([[3, 0], [3, 0]])))
        self.assertTrue(torch.equal(long, torch.tensor([[4, 3], [4, 3]])))

    def test_completed_candidate_is_not_lost_when_pruned(self):
        def logits(tokens, *args):
            scores = torch.full((len(tokens), 17), -100.0)
            if tokens.shape[1] == 1:
                scores[:, 4], scores[:, 3] = math.log(0.9), math.log(0.1)
            elif tokens.shape[1] == 2:
                scores[:, 5], scores[:, 6] = math.log(0.6), math.log(0.4)
            else:
                scores[:] = 0
            return scores, None

        with patch.object(self.model, "_decode_next", side_effect=logits):
            result = self.model.generate(
                self.inputs, max_new_tokens=3, num_beams=2, length_penalty=0
            )
        self.assertTrue(torch.equal(result, torch.tensor([[3, 0, 0], [3, 0, 0]])))

    def test_bfloat16_cache_matches_full_prefix(self):
        model = self.model
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            mask = model.make_source_allowed_mask(self.mask)
            memory = model.encoder(model.src_pos_embedding(self.inputs), mask)
            targets = torch.tensor([[2, 4, 5], [2, 7, 8]])
            cache = None
            for size in range(1, 4):
                actual, cache = model._decode_next(
                    targets[:, :size], memory, mask, cache, True
                )
                expected, _ = model._decode_next(
                    targets[:, :size], memory, mask, None, False
                )
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
            self.assertEqual(cache[0][0][0].dtype, torch.bfloat16)

    def test_invalid_generation_parameters(self):
        for kwargs in (
            {"num_beams": 0},
            {"num_beams": 18},
            {"max_new_tokens": -1},
            {"length_penalty": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.generate(self.inputs, **kwargs)


if __name__ == "__main__":
    unittest.main()
