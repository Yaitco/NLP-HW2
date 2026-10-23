import copy
import gc
import math
import random
import time
from itertools import islice

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.utils.lr_scheduler import training_budget


class BaseTrainer:
    """One epoch is `epoch_len` microbatches, or a full dataloader by default."""

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        self.config = config
        self.cfg_trainer = cfg = config.trainer
        self.model, self.criterion, self.metrics = model, criterion, metrics
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.device, self.logger, self.writer = device, logger, writer
        self.batch_transforms = batch_transforms or {}
        self.skip_oom = skip_oom
        self.is_train = self.compute_generation = True
        self.optimizer_steps = self.microbatches_seen = self.not_improved_count = 0
        self.start_epoch = 1
        self._iterator = None
        self.training_seconds = 0.0
        self._training_started_at = None
        hours = cfg.get("max_time_hours")
        if hours is not None and (not math.isfinite(hours) or hours <= 0):
            raise ValueError("max_time_hours must be positive or null")
        self.time_limit_seconds = hours * 3600 if hours is not None else math.inf

        self.train_dataloader = dataloaders["train"]
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.epoch_len = len(self.train_dataloader) if epoch_len is None else epoch_len
        self.accumulate_grad_batches = cfg.get("accumulate_grad_batches", 1)
        self.total_microbatches, _ = training_budget(
            self.epoch_len, self.accumulate_grad_batches, cfg.n_epochs
        )
        self.epochs = math.ceil(cfg.n_epochs)
        self.log_step = cfg.get("log_step", 50)
        self.pass_every_epochs = cfg.get("pass_every_epochs", 1)
        self.save_period = cfg.get("save_period", 1)
        self.keep_last_checkpoints = cfg.get("keep_last_checkpoints", 0)
        if type(self.keep_last_checkpoints) is not int or self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be a non-negative integer")
        for value in (self.log_step, self.pass_every_epochs, self.save_period):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Logging, Pass@1 and checkpoint intervals must be positive integers"
                )
        if self.accumulate_grad_batches > 1 and not hasattr(
            criterion, "normalization_weight"
        ):
            raise ValueError(
                "Accumulation requires criterion.normalization_weight(batch)"
            )

        amp = cfg.get("amp", {})
        self.amp_enabled = (
            amp.get("enabled", False) and torch.device(device).type == "cuda"
        )
        dtype = amp.get("dtype", "auto")
        if dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("AMP dtype must be auto, float16, or bfloat16")
        self.amp_dtype = torch.float16
        if self.amp_enabled:
            with torch.cuda.device(device):
                bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            if dtype == "bfloat16" and not bf16:
                raise ValueError("This GPU does not support native BF16; use float16")
            if dtype == "bfloat16" or (dtype == "auto" and bf16):
                self.amp_dtype = torch.bfloat16
            logger.info(f"AMP: {self.amp_dtype}")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.mnt_mode, self.mnt_metric = "off", None
        monitor = cfg.get("monitor", "off")
        if monitor != "off":
            self.mnt_mode, self.mnt_metric = monitor.split()
            if self.mnt_mode not in ("min", "max"):
                raise ValueError("monitor must be 'off', 'min metric', or 'max metric'")
        self.mnt_best = math.inf if self.mnt_mode == "min" else -math.inf
        self.early_stop = cfg.get("early_stop", 0)
        if self.early_stop <= 0:
            self.early_stop = math.inf
        self.train_metrics = MetricTracker(
            *config.writer.loss_names, "grad_norm", *[m.name for m in metrics["train"]]
        )
        self.evaluation_metrics = MetricTracker(
            *config.writer.loss_names, *[m.name for m in metrics["inference"]]
        )
        self.checkpoint_dir = ROOT_PATH / cfg.save_dir / config.writer.run_name
        if cfg.get("resume_from"):
            self._resume_checkpoint(self.checkpoint_dir / cfg.resume_from)
        elif cfg.get("from_pretrained"):
            self._from_pretrained(cfg.from_pretrained)

    def train(self):
        # On interruption, resume the last completed epoch's checkpoint.
        self._training_started_at = time.monotonic()
        try:
            self._train_process()
        finally:
            self.training_seconds = self._elapsed_training_seconds()
            self._training_started_at = None

    def _elapsed_training_seconds(self):
        elapsed = (
            time.monotonic() - self._training_started_at
            if self._training_started_at is not None
            else 0
        )
        return self.training_seconds + elapsed

    def _time_limit_reached(self):
        return self._elapsed_training_seconds() >= self.time_limit_seconds

    def _train_process(self):
        if (
            self.cfg_trainer.get("validate_before_training", False)
            and self.microbatches_seen == 0
        ):
            logs = {}
            for part, loader in self.evaluation_dataloaders.items():
                values = self._evaluation_epoch(
                    0, part, loader, compute_generation=True, step=0
                )
                logs.update({f"{part}_{name}": value for name, value in values.items()})
            self.logger.info(f"Baseline before training: {logs}")
            if self._monitor_performance(logs):
                self._save_checkpoint(0, save_best=True)
        for epoch in range(self.start_epoch, self.epochs + 1):
            if (
                self.microbatches_seen >= self.total_microbatches
                or self._time_limit_reached()
            ):
                break
            self._last_epoch = epoch
            logs = self._train_epoch(epoch)
            self.logger.info(
                f"Epoch {self.microbatches_seen / self.epoch_len:g}: {logs}"
            )
            best = self._monitor_performance(logs)
            final = (
                self.microbatches_seen == self.total_microbatches
                or self._time_limit_reached()
            )
            if best or final or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            if self._time_limit_reached():
                self.logger.info("Time budget reached; checkpoint saved")
                break
            if self.not_improved_count >= self.early_stop:
                self.logger.info("Early stopping")
                break

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        length = min(
            self.epoch_len - self.microbatches_seen % self.epoch_len,
            self.total_microbatches - self.microbatches_seen,
        )
        if self._iterator is None:
            self._iterator = iter(self._iteration_batches())
        iterator = islice(self._iterator, length)
        logs = {}
        processed = 0
        with tqdm(total=length, desc=f"train epoch {epoch}") as progress:
            for group, weights in self._train_batch_groups(iterator):
                previous = self.microbatches_seen
                result = self._process_train_group(group, weights)
                self.microbatches_seen += len(group)
                processed += len(group)
                progress.update(len(group))
                if result is None:
                    continue
                batch, grad_norm = result
                self.train_metrics.update_tensor("grad_norm", grad_norm)
                if (
                    previous // self.log_step < self.microbatches_seen // self.log_step
                    or processed == length
                ):
                    self.writer.set_step(self.microbatches_seen)
                    self.writer.add_scalar(
                        "epoch", self.microbatches_seen / self.epoch_len
                    )
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.param_groups[0]["lr"]
                    )
                    self.writer.add_scalar("optimizer_steps", self.optimizer_steps)
                    logs = self._log_scalars(self.train_metrics)
                    self._log_batch(processed - 1, batch)
                    self.train_metrics.reset()
                del batch, result  # Release training logits before validation.

        run_pass = (
            epoch % self.pass_every_epochs == 0
            or self.microbatches_seen == self.total_microbatches
            or self._time_limit_reached()
        )
        self.optimizer.zero_grad(set_to_none=True)
        for part, loader in self.evaluation_dataloaders.items():
            values = self._evaluation_epoch(
                self.microbatches_seen / self.epoch_len,
                part,
                loader,
                compute_generation=run_pass,
                step=self.microbatches_seen,
            )
            logs.update({f"{part}_{name}": value for name, value in values.items()})
        return logs

    def _train_batch_groups(self, iterator):
        """Buffer one accumulation group on CPU; normalize by supervised tokens."""
        weight = getattr(self.criterion, "normalization_weight", lambda **batch: 1)
        while group := list(islice(iterator, self.accumulate_grad_batches)):
            weights = [weight(**batch) for batch in group]
            if min(weights) <= 0:
                raise ValueError("Every microbatch must contain supervised targets")
            yield group, weights

    def _process_train_group(self, group, weights):
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for raw_batch, weight in zip(group, weights):
                batch = self.process_batch(
                    dict(raw_batch),
                    self.train_metrics,
                    loss_scale=weight / sum(weights),
                )
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            if not self.skip_oom:
                raise
            self.logger.warning("OOM: discarding the entire accumulation group")
            self.train_metrics.reset()
            torch.cuda.empty_cache()
            return None
        # Do not catch optimizer OOM: some parameters may already have changed.
        return batch, self._optimizer_step()

    def _iteration_batches(self):
        """Continue across epoch boundaries and reconstruct the cursor on resume."""
        loader = self.train_dataloader
        pass_index, offset = divmod(self.microbatches_seen, len(loader))
        while True:
            seed = self.cfg_trainer.get("seed", 42) + pass_index
            if hasattr(loader, "generator"):
                loader.generator = torch.Generator().manual_seed(seed)
            sampler = getattr(loader, "batch_sampler", None)
            if hasattr(sampler, "pass_index"):
                sampler.pass_index = pass_index
            elif hasattr(getattr(loader, "sampler", None), "generator"):
                loader.sampler.generator = torch.Generator().manual_seed(seed)
            yield from islice(iter(loader), offset, None)
            pass_index, offset = pass_index + 1, 0

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grad_norm = self._clip_grad_norm()
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= previous_scale:
            self.optimizer_steps += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return grad_norm

    def _clip_grad_norm(self):
        maximum = self.cfg_trainer.get("max_grad_norm")
        return clip_grad_norm_(
            self.model.parameters(), math.inf if maximum is None else maximum
        )

    def _evaluation_epoch(
        self, epoch, part, dataloader, *, compute_generation=True, step=None
    ):
        previous = self.is_train, self.model.training, self.compute_generation
        self.is_train, self.compute_generation = False, compute_generation
        self.model.eval()
        self.evaluation_metrics.reset()
        generation_metrics = (
            self.metrics.get("generation", []) if compute_generation else []
        )
        compare_greedy = compute_generation and self.cfg_trainer.get(
            "compare_greedy", False
        )
        if compare_greedy and not hasattr(self, "greedy_metrics"):
            # Share immutable setup/reference caches, but keep separate counters.
            self.greedy_metrics = [copy.copy(metric) for metric in generation_metrics]
        greedy_metrics = self.greedy_metrics if compare_greedy else []
        for metric in [*generation_metrics, *greedy_metrics]:
            metric.reset()
        device = torch.device(self.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        try:
            # Validation must not consume the training dropout RNG stream.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for batch in tqdm(dataloader, desc=part):
                    batch = self.process_batch(dict(batch), self.evaluation_metrics)
                    del batch
                self.writer.set_step(
                    self.microbatches_seen if step is None else step, part
                )
                self.writer.add_scalar("epoch", epoch)
                result = self._log_scalars(self.evaluation_metrics)
                groups = [
                    ("beam_" if compare_greedy else "", generation_metrics),
                    ("greedy_", greedy_metrics),
                ]
                for prefix, group in groups:
                    for metric in group:
                        values = {
                            prefix + name: value
                            for name, value in metric.compute().items()
                        }
                        for name, value in values.items():
                            self.writer.add_scalar(name, value)
                        result.update(values)
            return result
        finally:
            self.evaluation_metrics.reset()
            gc.collect()
            if device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            self.is_train, training, self.compute_generation = previous
            self.model.train(training)

    def _monitor_performance(self, logs):
        if self.mnt_mode == "off" or self.mnt_metric not in logs:
            return False
        value = logs[self.mnt_metric]
        improved = (
            value <= self.mnt_best if self.mnt_mode == "min" else value >= self.mnt_best
        )
        self.not_improved_count = 0 if improved else self.not_improved_count + 1
        if improved:
            self.mnt_best = value
        return improved

    def move_batch_to_device(self, batch):
        for name in self.cfg_trainer.device_tensors:
            batch[name] = batch[name].to(
                self.device, non_blocking=self.cfg_trainer.get("non_blocking", False)
            )
        return batch

    def transform_batch(self, batch):
        mode = "train" if self.is_train else "inference"
        for name, transform in ((self.batch_transforms or {}).get(mode) or {}).items():
            batch[name] = transform(batch[name])
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        pass

    def _log_scalars(self, tracker):
        values = tracker.result()
        for name, value in values.items():
            self.writer.add_scalar(name, value)
        return values

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "epoch_len": self.epoch_len,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict() if self.lr_scheduler else None
            ),
            "grad_scaler": self.scaler.state_dict(),
            "config": self.config,
            "optimizer_steps": self.optimizer_steps,
            "microbatches_seen": self.microbatches_seen,
            "training_seconds": self._elapsed_training_seconds(),
            "monitor_best": self.mnt_best,
            "not_improved_count": self.not_improved_count,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.device(self.device).type == "cuda"
                    else []
                ),
            },
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in ["checkpoint-last.pth"] + (["model_best.pth"] if save_best else []):
            path = self.checkpoint_dir / name
            temporary = path.with_suffix(".pth.tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            if self.config.writer.get("log_checkpoints", False):
                self.writer.add_checkpoint(str(path), str(self.checkpoint_dir.parent))
            self.logger.info(f"Saved {path}")

        if self.keep_last_checkpoints:
            path = self.checkpoint_dir / f"checkpoint-epoch{epoch:06d}.pth"
            temporary = path.with_suffix(".pth.tmp")
            temporary.unlink(missing_ok=True)
            # Same immutable file as last; later saves replace last atomically.
            temporary.hardlink_to(self.checkpoint_dir / "checkpoint-last.pth")
            temporary.replace(path)
            snapshots = sorted(
                (p for p in self.checkpoint_dir.glob("checkpoint-epoch*.pth")
                 if p.stem.removeprefix("checkpoint-epoch").isdigit()),
                key=lambda p: int(p.stem.removeprefix("checkpoint-epoch")),
            )
            for old in snapshots[:-self.keep_last_checkpoints]:
                old.unlink()

    def _resume_checkpoint(self, resume_path):
        # Training checkpoints contain optimizer/config/RNG: load only trusted files.
        checkpoint = torch.load(
            resume_path, map_location=self.device, weights_only=False
        )
        saved = checkpoint["config"]
        saved_epoch_len = checkpoint.get("epoch_len", saved.trainer.get("epoch_len"))
        if saved_epoch_len != self.epoch_len:
            raise ValueError(
                "Cannot resume with changed or unknown epoch_len; use from_pretrained to load weights"
            )
        for name in (
            "model",
            "optimizer",
            "lr_scheduler",
            "datasets",
            "dataloader",
            "batching",
        ):
            saved_value, current_value = saved.get(name), self.config.get(name)
            if name == "dataloader":
                # Evaluation batches do not affect the training stream or schedule.
                saved_value, current_value = dict(saved_value or {}), dict(
                    current_value or {}
                )
                saved_value.pop("eval_batch_size", None)
                current_value.pop("eval_batch_size", None)
            if saved_value != current_value:
                raise ValueError(
                    f"Cannot resume with changed {name}; use from_pretrained for a new experiment"
                )
        for name in ("accumulate_grad_batches", "seed", "n_epochs"):
            if saved.trainer.get(name) != self.cfg_trainer.get(name):
                raise ValueError(f"Cannot resume with changed {name}")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if self.scaler.is_enabled() and checkpoint["grad_scaler"]:
            self.scaler.load_state_dict(checkpoint["grad_scaler"])
        self.optimizer_steps = checkpoint["optimizer_steps"]
        self.microbatches_seen = checkpoint["microbatches_seen"]
        self.training_seconds = checkpoint.get("training_seconds", 0.0)
        self.mnt_best = checkpoint["monitor_best"]
        self.not_improved_count = checkpoint["not_improved_count"]
        self.start_epoch = self.microbatches_seen // self.epoch_len + 1
        self._iterator = None
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] and torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        self.logger.info(f"Resuming at microbatch {self.microbatches_seen}")

    def _from_pretrained(self, pretrained_path):
        checkpoint = torch.load(
            pretrained_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint))
