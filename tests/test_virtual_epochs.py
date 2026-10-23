"""Virtual-epoch boundaries and microbatch logging on synthetic CPU tensors."""

import copy
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import test_training_schedule as schedule_tests
import torch
from torch.utils.data import DataLoader

from src.datasets.samplers import LengthGroupedBatchSampler
from src.utils.lr_scheduler import training_budget, warmup_cosine_epochs


class Writer:
    def __init__(self):
        self.step = 0
        self.mode = "train"
        self.values = []

    def set_step(self, step, mode="train"):
        self.step, self.mode = step, mode

    def add_scalar(self, name, value):
        self.values.append((self.step, self.mode, name, value))


class GenerationMetric:
    def reset(self):
        self.count = 0

    def update(self, generated, labels):
        self.count += len(labels)

    def compute(self):
        return {"pass_at_1": 0.5, "pass_total": self.count}


class Model(schedule_tests.TokenClassifier):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.generation_steps = []

    def forward(self, input_ids, **batch):
        return {"logits": self.proj(self.dropout(input_ids))}

    def generate(self, input_ids, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        self.generation_steps.append(self.step())
        return torch.ones(len(input_ids), 1, dtype=torch.long)


class VirtualEpochTests(unittest.TestCase):
    def setUp(self):
        self.fixture = schedule_tests.AccumulationTests()
        self.fixture.setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def make_trainer(self, model=None, epoch_len=8, n_epochs=2.5, pass_interval=2):
        trainer = self.fixture.make_trainer(
            Model() if model is None else model,
            epoch_len=epoch_len,
            n_epochs=n_epochs,
            pass_every_epochs=pass_interval,
            monitor="max val_pass_at_1",
            early_stop=0,
            seed=42,
            generation={"max_new_tokens": 8},
        )
        trainer.writer = Writer()
        trainer.checkpoint_dir = Path(self.temporary.name)
        trainer.evaluation_dataloaders = {"val": self.fixture.batches[:2]}
        trainer.metrics["generation"] = [GenerationMetric()]
        trainer.model.step = lambda: trainer.microbatches_seen
        sampler = LengthGroupedBatchSampler([4, 2, 5, 3, 1, 4], batch_size=1, seed=42)
        trainer.train_dataloader = DataLoader(
            self.fixture.batches, batch_sampler=sampler, collate_fn=lambda rows: rows[0]
        )
        return trainer

    def test_microbatch_axis_validation_cadence_and_fractional_final_epoch(self):
        trainer = self.make_trainer()
        with patch.object(
            trainer, "_evaluation_epoch", wraps=trainer._evaluation_epoch
        ) as evaluate:
            trainer._train_process()
        self.assertEqual(
            [
                (c.kwargs["step"], c.kwargs["compute_generation"])
                for c in evaluate.call_args_list
            ],
            [(8, False), (16, True), (20, True)],
        )
        self.assertEqual(trainer.model.generation_steps, [16, 16, 20, 20])
        self.assertEqual(trainer.microbatches_seen, 20)
        self.assertEqual(trainer.optimizer_steps, 5)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 5)
        self.assertEqual(trainer.mnt_mode, "max")
        self.assertTrue(trainer.is_train and trainer.model.training)
        loss_positions = [
            step
            for step, mode, name, value in trainer.writer.values
            if mode == "val" and name == "loss"
        ]
        self.assertEqual(loss_positions, [8, 16, 20])
        self.assertIn((20, "val", "epoch", 2.5), trainer.writer.values)
        self.assertIn((20, "train", "optimizer_steps", 5), trainer.writer.values)
        saved = torch.load(
            trainer.checkpoint_dir / "checkpoint-last.pth", weights_only=False
        )
        self.assertEqual(saved["epoch_len"], 8)
        self.assertEqual(saved["microbatches_seen"], 20)
        self.assertFalse(list(trainer.checkpoint_dir.glob("checkpoint-epoch*.pth")))
        self.assertTrue((trainer.checkpoint_dir / "model_best.pth").exists())

    def test_retains_last_ten_snapshots_without_changing_older_weights(self):
        trainer = self.make_trainer()
        trainer.keep_last_checkpoints = 10
        for epoch in range(1, 13):
            with torch.no_grad():
                next(trainer.model.parameters()).fill_(epoch)
            trainer._save_checkpoint(epoch, save_best=(epoch == 2))
        snapshots = sorted(trainer.checkpoint_dir.glob("checkpoint-epoch*.pth"))
        self.assertEqual([p.name for p in snapshots], [
            f"checkpoint-epoch{epoch:06d}.pth" for epoch in range(3, 13)
        ])
        for path, epoch in [(snapshots[0], 3), (snapshots[-1], 12),
                            (trainer.checkpoint_dir / "checkpoint-last.pth", 12),
                            (trainer.checkpoint_dir / "model_best.pth", 2)]:
            state = torch.load(path, weights_only=False)
            self.assertEqual(state["epoch"], epoch)
            self.assertTrue(torch.all(next(iter(state["state_dict"].values())) == epoch))
            self.assertIn("optimizer", state)
        # Reopening the same run must prune its existing history as well.
        reopened = self.make_trainer()
        reopened.keep_last_checkpoints = 10
        reopened._save_checkpoint(13)
        self.assertFalse(snapshots[0].exists())
        self.assertEqual(len(list(trainer.checkpoint_dir.glob("checkpoint-epoch*.pth"))), 10)

    def test_optional_baseline_evaluates_before_updates_and_is_not_repeated_on_resume(self):
        trainer = self.make_trainer(n_epochs=1)
        trainer.cfg_trainer.validate_before_training = True
        with patch.object(trainer, "_evaluation_epoch", wraps=trainer._evaluation_epoch) as evaluate:
            trainer._train_process()
        self.assertEqual([c.kwargs['step'] for c in evaluate.call_args_list], [0, 8])
        self.assertTrue(evaluate.call_args_list[0].kwargs['compute_generation'])
        self.assertEqual(trainer.model.generation_steps[:2], [0, 0])
        trainer.cfg_trainer.resume_from = 'checkpoint-last.pth'
        trainer._resume_checkpoint(trainer.checkpoint_dir / 'checkpoint-last.pth')
        with patch.object(trainer, '_evaluation_epoch') as evaluate:
            trainer._train_process()
        evaluate.assert_not_called()

    def test_beam_and_greedy_validation_have_separate_counters_and_names(self):
        trainer = self.make_trainer()
        trainer.cfg_trainer.compare_greedy = True
        trainer.cfg_trainer.generation.num_beams = 16
        with patch.object(trainer.model, 'generate', return_value=torch.ones(1, 1, dtype=torch.long)) as generate:
            values = trainer._evaluation_epoch(0, 'val', self.fixture.batches[:2])
        self.assertEqual([c.kwargs['num_beams'] for c in generate.call_args_list], [16, 1, 16, 1])
        self.assertEqual(values['beam_pass_total'], 2)
        self.assertEqual(values['greedy_pass_total'], 2)
        self.assertIn('beam_pass_at_1', values)
        self.assertIn('greedy_pass_at_1', values)
        self.assertIn('loss', values)
        trainer._evaluation_epoch(1, 'val', self.fixture.batches[:2])
        self.assertEqual(trainer.greedy_metrics[0].count, 2)
        self.assertEqual(trainer.metrics['generation'][0].count, 2)

    def test_resume_preserves_stream_and_rng_across_virtual_epochs(self):
        model = Model()
        full = self.make_trainer(copy.deepcopy(model))
        partial = self.make_trainer(copy.deepcopy(model))
        torch.manual_seed(123)
        full._train_process()

        class Pause(Exception):
            pass

        save = partial._save_checkpoint

        def pause_after_save(*args, **kwargs):
            save(*args, **kwargs)
            if partial.microbatches_seen == 8:
                raise Pause()

        torch.manual_seed(123)
        with patch.object(
            partial, "_save_checkpoint", side_effect=pause_after_save
        ), self.assertRaises(Pause):
            partial._train_process()
        # The previous virtual-epoch implementation stored the length in config.
        path = partial.checkpoint_dir / "checkpoint-last.pth"
        state = torch.load(path, weights_only=False)
        state.pop("epoch_len")
        state["config"].trainer.epoch_len = 8
        torch.save(state, path)
        resumed = self.make_trainer(copy.deepcopy(model))
        resumed._resume_checkpoint(resumed.checkpoint_dir / "checkpoint-last.pth")
        self.assertEqual(resumed.start_epoch, 2)
        self.assertEqual(resumed.microbatches_seen, 8)
        resumed._train_process()
        self.fixture.assert_same_parameters(full.model, resumed.model)
        self.assertEqual(resumed.microbatches_seen, 20)
        self.assertEqual(resumed.optimizer_steps, 5)
        # A completed fractional epoch must not replay its tail on resume.
        completed = self.make_trainer(copy.deepcopy(model))
        completed._resume_checkpoint(completed.checkpoint_dir / "checkpoint-last.pth")
        with patch.object(completed, "_evaluation_epoch") as evaluate:
            completed._train_process()
            evaluate.assert_not_called()

    def test_nondivisible_epoch_and_tail_flush_partial_accumulation(self):
        trainer = self.make_trainer(epoch_len=6, n_epochs=2.5)
        trainer._train_process()
        self.assertEqual(trainer.microbatches_seen, 15)
        self.assertEqual(trainer.optimizer_steps, 5)
        self.assertEqual(trainer.model.generation_steps, [12, 12, 15, 15])

    def test_resume_inside_virtual_epoch_completes_only_its_remainder(self):
        model = Model()
        full = self.make_trainer(copy.deepcopy(model))
        partial = self.make_trainer(copy.deepcopy(model))
        torch.manual_seed(123)
        full._train_process()

        class Pause(Exception):
            pass

        log_scalar = partial.writer.add_scalar

        def pause_inside_epoch(name, value):
            log_scalar(name, value)
            if name == "optimizer_steps" and partial.microbatches_seen == 12:
                partial._save_checkpoint(partial._last_epoch)
                raise Pause()

        torch.manual_seed(123)
        with patch.object(
            partial.writer, "add_scalar", side_effect=pause_inside_epoch
        ), self.assertRaises(Pause):
            partial._train_process()
        resumed = self.make_trainer(copy.deepcopy(model))
        resumed._resume_checkpoint(resumed.checkpoint_dir / "checkpoint-last.pth")
        self.assertEqual(resumed.start_epoch, 2)
        self.assertEqual(resumed.microbatches_seen, 12)
        with patch.object(
            resumed, "_evaluation_epoch", wraps=resumed._evaluation_epoch
        ) as evaluate:
            resumed._train_process()
        self.assertEqual([c.kwargs["step"] for c in evaluate.call_args_list], [16, 20])
        self.fixture.assert_same_parameters(full.model, resumed.model)
        self.assertEqual(resumed.optimizer_steps, 5)

    def test_validation_frequency_does_not_change_training_dropout(self):
        model = Model()
        first = self.make_trainer(copy.deepcopy(model))
        second = self.make_trainer(copy.deepcopy(model), pass_interval=1)
        torch.manual_seed(123)
        first._train_process()
        torch.manual_seed(123)
        second._train_process()
        self.fixture.assert_same_parameters(first.model, second.model)

    def test_val_loss_is_weighted_by_nonpadding_tokens(self):
        trainer = self.make_trainer()
        inputs, labels = [], []
        for batch in self.fixture.batches[:2]:
            valid = batch["labels"] != 0
            inputs.append(batch["input_ids"][valid])
            labels.append(batch["labels"][valid])
        expected = torch.nn.functional.cross_entropy(
            trainer.model.proj(torch.cat(inputs)), torch.cat(labels)
        )
        result = trainer._evaluation_epoch(
            0, "val", self.fixture.batches[:2], compute_generation=False, step=1
        )
        self.assertAlmostEqual(result["loss"], float(expected.detach()), places=6)
        self.assertNotIn("pass_at_1", result)
        self.assertEqual(trainer.model.generation_steps, [])
        self.assertTrue(trainer.model.training)

    def test_validation_releases_logits_before_generation_and_temporary_batches(self):
        references = []

        class MemoryModel(Model):
            def forward(self, **batch):
                output = super().forward(**batch)
                self.logits_ref = weakref.ref(output["logits"])
                output["temporary"] = torch.ones(10)
                references.extend([self.logits_ref, weakref.ref(output["temporary"])])
                return output

            def generate(self, **kwargs):
                assert self.logits_ref() is None
                generated = super().generate(**kwargs)
                references.append(weakref.ref(generated))
                return generated

        trainer = self.make_trainer(MemoryModel())
        with patch("src.trainer.base_trainer.gc.collect") as collect:
            result = trainer._evaluation_epoch(1, "val", self.fixture.batches[:2])
            collect.assert_called_once()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(result["pass_at_1"], 0.5)
        self.assertTrue(trainer.model.training)

    def test_validation_passes_beam_settings_to_generation(self):
        trainer = self.make_trainer()
        trainer.cfg_trainer.generation.update(
            {"num_beams": 3, "use_cache": True, "length_penalty": 0.7}
        )
        with patch.object(
            trainer.model, "generate", wraps=trainer.model.generate
        ) as generate:
            trainer._evaluation_epoch(1, "val", self.fixture.batches[:2])
        self.assertEqual(generate.call_args.kwargs["num_beams"], 3)
        self.assertTrue(generate.call_args.kwargs["use_cache"])
        self.assertEqual(generate.call_args.kwargs["length_penalty"], 0.7)

    def test_resume_rejects_changed_schedule(self):
        trainer = self.make_trainer()
        trainer._train_process()
        incompatible = self.make_trainer(n_epochs=3)
        with self.assertRaisesRegex(ValueError, "changed n_epochs"):
            incompatible._resume_checkpoint(
                trainer.checkpoint_dir / "checkpoint-last.pth"
            )

    def test_validation_voting_selects_one_candidate_and_reuses_the_voter(self):
        trainer = self.make_trainer()
        trainer.cfg_trainer.beam_voting = True
        trainer.cfg_trainer.generation.num_beams = 3
        trainer.config.data = {
            "tokenizer_path": "unused.json",
            "code_format": "solve_body_result",
        }
        beams = torch.ones(1, 3, 2, dtype=torch.long)
        scores = torch.zeros(1, 3)
        with patch.object(
            trainer.model, "generate", return_value=(beams, scores)
        ) as generate, patch("src.trainer.trainer.BeamVoter") as voter:
            voter.return_value.select.side_effect = lambda ids, scores: ids[:, 0]
            for epoch in (1, 2):
                result = trainer._evaluation_epoch(
                    epoch, "val", self.fixture.batches[:2]
                )
                self.assertEqual(result["pass_total"], 2)
        self.assertTrue(generate.call_args.kwargs["return_beams"])
        self.assertEqual(generate.call_args.kwargs["num_beams"], 3)
        voter.assert_called_once_with(
            "unused.json",
            code_format="solve_body_result",
            sandbox=True,
            execution_batch_size=16,
            temperature=1.0,
        )
        self.assertEqual(voter.return_value.select.call_count, 4)

    def test_resume_allows_changed_eval_batch_size_and_preserves_training_state(self):
        trainer = self.make_trainer()
        trainer.config.dataloader = {"batch_size": 128, "num_workers": 0}
        trainer._train_epoch(1)
        trainer._save_checkpoint(1)
        path = trainer.checkpoint_dir / "checkpoint-last.pth"
        state = torch.load(path, weights_only=False)

        for saved_eval in ("missing", None, 128):
            with self.subTest(saved_eval=saved_eval):
                saved = copy.deepcopy(state)
                if saved_eval != "missing":
                    saved["config"].dataloader.eval_batch_size = saved_eval
                torch.save(saved, path)

                resumed = self.make_trainer()
                resumed.config.dataloader = {
                    "batch_size": 128,
                    "num_workers": 0,
                    "eval_batch_size": 32,
                }
                resumed._resume_checkpoint(path)
                self.fixture.assert_same_parameters(trainer.model, resumed.model)
                self.assertEqual(resumed.config.dataloader.eval_batch_size, 32)
                self.assertEqual(resumed.start_epoch, 2)
                self.assertEqual(resumed.microbatches_seen, 8)
                self.assertEqual(resumed.optimizer_steps, 2)
                self.assertEqual(
                    resumed.lr_scheduler.state_dict(), trainer.lr_scheduler.state_dict()
                )
                self.assertEqual(
                    resumed.optimizer.param_groups[0]["lr"],
                    trainer.optimizer.param_groups[0]["lr"],
                )
                for original, restored in zip(
                    trainer.optimizer.state.values(), resumed.optimizer.state.values()
                ):
                    torch.testing.assert_close(
                        original["momentum_buffer"], restored["momentum_buffer"]
                    )
                self.assertTrue(
                    torch.equal(torch.get_rng_state(), state["rng_state"]["torch"])
                )

        for changed in ({"batch_size": 32}, {"num_workers": 2}):
            with self.subTest(changed=changed):
                incompatible = self.make_trainer()
                incompatible.config.dataloader = {
                    "batch_size": 128,
                    "num_workers": 0,
                    "eval_batch_size": 32,
                    **changed,
                }
                with self.assertRaisesRegex(ValueError, "changed dataloader"):
                    incompatible._resume_checkpoint(path)

    def test_wall_time_limit_saves_progress_and_counts_time_on_resume(self):
        trainer = self.make_trainer()
        trainer.time_limit_seconds = 0.3
        clock = [0.0]
        process = trainer._process_train_group

        def timed_group(*args):
            result = process(*args)
            clock[0] += 0.2
            return result

        with patch(
            "src.trainer.base_trainer.time.monotonic", side_effect=lambda: clock[0]
        ), patch.object(trainer, "_process_train_group", side_effect=timed_group):
            trainer.train()
        self.assertEqual(trainer.microbatches_seen, 8)
        self.assertEqual(trainer.optimizer_steps, 2)
        self.assertEqual(trainer.model.generation_steps, [8, 8])
        self.assertAlmostEqual(trainer.training_seconds, 0.4)
        path = trainer.checkpoint_dir / "checkpoint-last.pth"
        self.assertTrue(path.exists())
        resumed = self.make_trainer()
        resumed.time_limit_seconds = 0.3
        resumed._resume_checkpoint(path)
        self.assertAlmostEqual(resumed.training_seconds, 0.4)
        with patch.object(resumed, "_train_epoch") as train_epoch:
            resumed.train()
            train_epoch.assert_not_called()

    def test_budget_conversion_preserves_original_update_count(self):
        self.assertEqual(training_budget(31250, 4, 3), (93750, 23439))
        self.assertEqual(training_budget(1000, 4, 93.756), (93756, 23439))
        self.assertEqual(training_budget(6, 4, 2.5), (15, 5))
        self.assertEqual(training_budget(1000, 8, 93.756), (93756, 11720))
        with self.assertRaises(ValueError):
            training_budget(8, 4, 2.123)

    def test_scheduler_is_configured_in_virtual_epochs(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.1)
        scheduler = warmup_cosine_epochs(
            optimizer,
            epoch_len=8,
            n_epochs=2.5,
            accumulate_grad_batches=4,
            warmup_epochs=0.5,
            min_lr=0.01,
            start_factor=0.1,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
        for update in range(5):
            optimizer.step()
            scheduler.step()
            if update == 0:
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
