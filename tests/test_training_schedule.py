"""CPU tensor checks only: no dataset downloads, GPU, or training experiment."""

import copy
import logging
import math
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.loss import Seq2SeqCrossEntropyLoss
from src.trainer import Trainer
from src.utils.io_utils import ROOT_PATH
from src.utils.lr_scheduler import (
    build_lr_scheduler,
    inverse_sqrt,
    training_budget,
    warmup_cosine,
)


class Writer:
    def __init__(self):
        self.values = {}

    def set_step(self, *args):
        pass

    def add_scalar(self, name, value):
        self.values[name] = value


class TokenClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 3)
        self.calls = 0
        self.fail_on = None

    def forward(self, input_ids, **batch):
        self.calls += 1
        if self.calls == self.fail_on:
            raise torch.cuda.OutOfMemoryError("Simulated OOM on CPU")
        return {"logits": self.proj(input_ids)}


class AccumulationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.batches = []
        for length in (4, 2, 5, 3, 1, 4):
            labels = torch.randint(1, 3, (1, length), generator=generator)
            if length > 2:
                labels[0, -1] = 0
            self.batches.append(
                {
                    "input_ids": torch.randn(1, length, 3, generator=generator),
                    "labels": labels,
                }
            )

    def make_trainer(
        self, model, accumulation=4, skip_oom=False, epoch_len=None, **trainer_config
    ):
        config = OmegaConf.create(
            {
                "trainer": {
                    "device_tensors": ["input_ids", "labels"],
                    "n_epochs": 2,
                    "save_period": 1,
                    "save_dir": "saved",
                    "monitor": "off",
                    "log_step": 2,
                    "max_grad_norm": 0.3,
                    "accumulate_grad_batches": accumulation,
                },
                "writer": {"loss_names": ["loss"], "run_name": "unit_test"},
            }
        )
        config.trainer.update(trainer_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return Trainer(
            model=model,
            criterion=Seq2SeqCrossEntropyLoss(pad_id=0),
            metrics={"train": [], "inference": []},
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=config,
            device="cpu",
            dataloaders={"train": self.batches},
            logger=logging.getLogger("unit_test"),
            writer=Writer(),
            batch_transforms={"train": None, "inference": None},
            skip_oom=skip_oom,
            epoch_len=epoch_len,
        )

    def reference_update(self, trainer, group):
        # Merge all supervised tokens into one physical batch. This checks
        # normalization independently of the implementation's microbatch losses.
        inputs, labels = [], []
        for batch in group:
            valid = batch["labels"] != 0
            inputs.append(batch["input_ids"][valid])
            labels.append(batch["labels"][valid])
        trainer.optimizer.zero_grad(set_to_none=True)
        logits = trainer.model.proj(torch.cat(inputs))
        loss = torch.nn.functional.cross_entropy(logits, torch.cat(labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 0.3)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()

    def assert_same_parameters(self, left, right):
        for actual, expected in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_unequal_token_counts_and_partial_group_match_full_batches(self):
        model = TokenClassifier()
        reference = self.make_trainer(copy.deepcopy(model))
        trainer = self.make_trainer(model)
        with patch.object(
            trainer, "_clip_grad_norm", wraps=trainer._clip_grad_norm
        ) as clip:
            for epoch in (1, 2):
                trainer._train_epoch(epoch)
                self.reference_update(reference, self.batches[:4])
                self.reference_update(reference, self.batches[4:])
                self.assert_same_parameters(model, reference.model)
            self.assertEqual(clip.call_count, 4)
        self.assertEqual(trainer.optimizer_steps, 4)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 4)
        self.assertTrue(all("logits" not in b for b in self.batches))

    def test_accumulation_one_preserves_per_batch_updates(self):
        model = TokenClassifier()
        reference = self.make_trainer(copy.deepcopy(model), accumulation=1)
        trainer = self.make_trainer(model, accumulation=1)
        trainer._train_epoch(1)
        for batch in self.batches:
            self.reference_update(reference, [batch])
        self.assert_same_parameters(model, reference.model)
        self.assertEqual(trainer.optimizer_steps, len(self.batches))

    def test_iteration_limit_flushes_partial_group(self):
        model = TokenClassifier()
        reference = self.make_trainer(copy.deepcopy(model))
        trainer = self.make_trainer(model, epoch_len=5)
        trainer._train_epoch(1)
        self.reference_update(reference, self.batches[:4])
        self.reference_update(reference, self.batches[4:5])
        self.assert_same_parameters(model, reference.model)
        self.assertEqual(model.calls, 5)
        self.assertEqual(trainer.optimizer_steps, 2)

    def test_oom_discards_all_partial_gradients(self):
        model = TokenClassifier()
        reference = self.make_trainer(copy.deepcopy(model))
        model.fail_on = 3
        trainer = self.make_trainer(model, skip_oom=True)
        with patch("torch.cuda.empty_cache"):
            trainer._train_epoch(1)
        self.reference_update(reference, self.batches[4:])
        self.assert_same_parameters(model, reference.model)
        self.assertEqual(trainer.optimizer_steps, 1)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 1)

    def test_scheduler_does_not_advance_on_skipped_amp_update(self):
        trainer = self.make_trainer(TokenClassifier())

        class OverflowScaler:
            def __init__(self):
                self.value = 16
                self.updates = 0
                self.unscales = 0

            def scale(self, loss):
                return loss

            def get_scale(self):
                return self.value

            def unscale_(self, optimizer):
                self.unscales += 1

            def step(self, optimizer):
                if self.updates > 0:
                    optimizer.step()

            def update(self):
                self.updates += 1
                self.value = 8

        trainer.scaler = OverflowScaler()
        trainer._train_epoch(1)
        self.assertEqual(trainer.scaler.unscales, 2)
        self.assertEqual(trainer.scaler.updates, 2)
        self.assertEqual(trainer.optimizer_steps, 1)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 1)

    def test_invalid_accumulation_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.make_trainer(TokenClassifier(), accumulation=count)


class ScheduleTests(unittest.TestCase):
    def make_optimizer(self):
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.1)

    def test_warmup_cosine_boundaries(self):
        optimizer = self.make_optimizer()
        scheduler = warmup_cosine(
            optimizer, total_steps=10, warmup_steps=2, min_lr=0.01, start_factor=0.1
        )
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(rates[0], 0.01)
        self.assertAlmostEqual(rates[1], 0.055)
        self.assertAlmostEqual(rates[2], 0.1)
        self.assertAlmostEqual(rates[-1], 0.01)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))

    def test_scheduler_state_roundtrip(self):
        optimizer = self.make_optimizer()
        scheduler = warmup_cosine(optimizer, 10, 2, 0.01)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        restored_opt = self.make_optimizer()
        restored = warmup_cosine(restored_opt, 10, 2, 0.01)
        restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        for _ in range(7):
            optimizer.step()
            scheduler.step()
            restored_opt.step()
            restored.step()
            self.assertEqual(
                optimizer.param_groups[0]["lr"], restored_opt.param_groups[0]["lr"]
            )

    def test_auto_length_uses_optimizer_updates_including_partial_groups(self):
        config = OmegaConf.create(
            {
                "_target_": "src.utils.lr_scheduler.warmup_cosine",
                "total_steps": None,
                "warmup_steps": 1,
                "min_lr": 0.01,
            }
        )
        optimizer = self.make_optimizer()
        scheduler = build_lr_scheduler(
            config,
            optimizer,
            microbatches_per_epoch=6,
            accumulate_grad_batches=4,
            n_epochs=2,
        )
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        self.assertGreater(optimizer.param_groups[0]["lr"], 0.01)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_invalid_schedule_rejected(self):
        for total, warmup in ((10, 10), (10, -1), (0, 0), (3, 1.5)):
            with self.subTest(total=total, warmup=warmup), self.assertRaises(
                ValueError
            ):
                warmup_cosine(self.make_optimizer(), total, warmup)

    def test_inverse_sqrt_warmup_decay_and_resume(self):
        optimizer = self.make_optimizer()
        scheduler = inverse_sqrt(
            optimizer, epoch_len=8, accumulate_grad_batches=4, warmup_epochs=1
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        restored_optimizer = self.make_optimizer()
        restored = inverse_sqrt(
            restored_optimizer, epoch_len=8, accumulate_grad_batches=4, warmup_epochs=1
        )
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        for update in range(3, 9):
            optimizer.step()
            scheduler.step()
            restored_optimizer.step()
            restored.step()
            self.assertAlmostEqual(
                optimizer.param_groups[0]["lr"], 0.1 * math.sqrt(2 / update)
            )
            self.assertEqual(
                optimizer.param_groups[0]["lr"],
                restored_optimizer.param_groups[0]["lr"],
            )

    def test_long_profile_starts_from_scratch_on_full_clean_data(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(ROOT_PATH / "src/configs")
        ):
            cfg = compose(config_name="tinysms_strong_long")
        self.assertIsNone(cfg.trainer.from_pretrained)
        self.assertIsNone(cfg.trainer.resume_from)

        self.assertEqual(cfg.data.dataset_path, "artifacts/tokenized_full_clean")
        self.assertEqual(cfg.datasets.val._target_, "src.datasets.TinySMSDataset")
        self.assertEqual(cfg.datasets.val.split, "validation")
        self.assertNotIn("start", cfg.datasets.val)
        self.assertNotIn("indices_path", cfg.datasets.train)
        self.assertEqual(
            cfg.lr_scheduler._target_, "src.utils.lr_scheduler.inverse_sqrt"
        )
        self.assertEqual(
            training_budget(cfg.trainer.epoch_len, 4, cfg.trainer.n_epochs),
            (530000, 132500),
        )

    def test_16k_profile_uses_one_tokenizer_for_data_collation_and_pass(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(ROOT_PATH / "src/configs")
        ):
            cfg = compose(config_name="tinysms_strong_long_16k")
        self.assertEqual(cfg.model.config.vocab_size, 16384)
        self.assertEqual(cfg.trainer.max_time_hours, 10)
        self.assertEqual(cfg.trainer.n_epochs, 2000)
        self.assertEqual(cfg.trainer.amp.dtype, "bfloat16")
        self.assertEqual(cfg.data.dataset_path, "artifacts/tokenized_full_clean_16k")
        self.assertEqual(cfg.data.tokenizer_path, "artifacts/tokenizer_16k.json")
        self.assertEqual(
            cfg.datasets.train.indices_path, "artifacts/length128_16k/train_indices.npy"
        )
        self.assertNotIn("indices_path", cfg.datasets.val)
        self.assertEqual(cfg.collate_fn.tokenizer_path, cfg.data.tokenizer_path)
        self.assertEqual(
            cfg.metrics.generation[0].tokenizer_path, cfg.data.tokenizer_path
        )
        self.assertIsNone(cfg.trainer.from_pretrained)
        self.assertIsNone(cfg.trainer.resume_from)

    def test_configs_remain_compatible_and_dirty_profile_keeps_all_rows(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(ROOT_PATH / "src/configs")
        ):
            for name in (
                "tinysms_one_batch",
                "tinysms_baseline",
                "tinysms_train",
                "tinysms_fast",
                "tinysms_amp",
                "tinysms_compile",
            ):
                cfg = compose(config_name=name)
                self.assertEqual(
                    cfg.lr_scheduler._target_, "torch.optim.lr_scheduler.StepLR"
                )
            cfg = compose(config_name="tinysms_strong_dirty")
        self.assertNotIn("step_size", cfg.lr_scheduler)
        self.assertNotIn("gamma", cfg.lr_scheduler)
        self.assertIsNone(cfg.datasets.train.limit)
        self.assertEqual(cfg.datasets.train._target_, "src.datasets.TinySMSDataset")
        self.assertEqual(
            cfg.dataloader.batch_size * cfg.trainer.accumulate_grad_batches, 128
        )
        self.assertEqual(cfg.datasets.val.start, 1000000)
        self.assertEqual(cfg.datasets.val.limit, 2048)
        self.assertGreaterEqual(cfg.model.config.max_src_len, 744)
        self.assertGreaterEqual(cfg.model.config.max_tgt_len, 1146)
        updates = (
            math.ceil(31250 / cfg.trainer.accumulate_grad_batches)
            * cfg.trainer.n_epochs
        )
        self.assertEqual(updates, 23439)


if __name__ == "__main__":
    unittest.main()
