"""Learning-rate schedules measured in successful optimizer updates."""

import math

from hydra.utils import instantiate
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def training_budget(microbatches_per_epoch, accumulate_grad_batches, n_epochs):
    """Translate full/fractional epochs to microbatches and optimizer updates."""
    if any(
        type(value) is not int or value < 1
        for value in (microbatches_per_epoch, accumulate_grad_batches)
    ):
        raise ValueError("Epoch length and accumulation must be positive integers")
    if (
        type(n_epochs) not in (int, float)
        or not math.isfinite(n_epochs)
        or n_epochs <= 0
    ):
        raise ValueError("n_epochs must be finite and positive")
    exact = microbatches_per_epoch * n_epochs
    microbatches = round(exact)
    if microbatches < 1 or not math.isclose(
        exact, microbatches, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("n_epochs * epoch_len must be a whole number of microbatches")
    complete, tail = divmod(microbatches, microbatches_per_epoch)
    updates = complete * math.ceil(microbatches_per_epoch / accumulate_grad_batches)
    updates += math.ceil(tail / accumulate_grad_batches)
    return microbatches, updates


def warmup_cosine_epochs(
    optimizer,
    epoch_len,
    n_epochs,
    accumulate_grad_batches,
    warmup_epochs=4,
    min_lr=3e-5,
    start_factor=1e-3,
):
    """Configure the schedule in virtual epochs; advance it on weight updates."""
    _, total_steps = training_budget(epoch_len, accumulate_grad_batches, n_epochs)
    warmup_steps = 0
    if warmup_epochs:
        _, warmup_steps = training_budget(
            epoch_len, accumulate_grad_batches, warmup_epochs
        )
    return warmup_cosine(optimizer, total_steps, warmup_steps, min_lr, start_factor)


def warmup_cosine(
    optimizer,
    total_steps,
    warmup_steps=1000,
    min_lr=3e-5,
    start_factor=1e-3,
):
    """Linear warmup followed by cosine decay, with serializable PyTorch state."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be an integer in [0, total_steps)")
    if not math.isfinite(start_factor) or not 0 < start_factor <= 1:
        raise ValueError("start_factor must be in (0, 1]")
    if not math.isfinite(min_lr) or not 0 <= min_lr <= min(
        group["lr"] for group in optimizer.param_groups
    ):
        raise ValueError("min_lr must be finite and between zero and the peak LR")

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def inverse_sqrt(optimizer, epoch_len, accumulate_grad_batches, warmup_epochs=16):
    """Linear warmup to optimizer LR, then LR * sqrt(warmup_updates / update)."""
    _, warmup_updates = training_budget(
        epoch_len, accumulate_grad_batches, warmup_epochs
    )

    def multiplier(update):
        update = max(1, update)
        return min(update / warmup_updates, math.sqrt(warmup_updates / update))

    return LambdaLR(optimizer, multiplier)


def build_lr_scheduler(
    config,
    optimizer,
    *,
    microbatches_per_epoch,
    accumulate_grad_batches,
    n_epochs,
):
    """Resolve an automatic schedule length from the actual dataloader length."""
    if config is None:
        return None
    kwargs = {}
    if "total_steps" in config and config.total_steps is None:
        _, kwargs["total_steps"] = training_budget(
            microbatches_per_epoch,
            accumulate_grad_batches,
            n_epochs,
        )
    return instantiate(config, optimizer=optimizer, **kwargs)
