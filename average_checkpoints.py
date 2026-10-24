"""Select a finite-window exponential checkpoint average by validation Pass@1."""

import argparse
import json
import logging
import math
from pathlib import Path
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.trainer import Trainer
from src.utils.io_utils import ROOT_PATH


def exponential_weights(count, beta):
    if count < 1 or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("Need checkpoints and finite beta in [0, 1]")
    weights = [beta ** (count - 1 - i) for i in range(count)]
    total = math.fsum(weights)
    return [w / total for w in weights]


def average_checkpoints(paths, beta, anchor=None):
    """Oldest -> newest; FP32 accumulation, latest non-floating buffers."""
    weights = exponential_weights(len(paths), beta)
    if anchor is not None:
        count = len(paths)
        weights = [beta**count] + [
            (1 - beta) * beta ** (count - 1 - i) for i in range(count)
        ]
        paths = [anchor, *paths]
    averaged = {}
    spec = None
    for path, weight in zip(paths, weights):
        if weight == 0:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        current = checkpoint["state_dict"]
        current_spec = {
            key: (value.shape, value.dtype) for key, value in current.items()
        }
        if spec is None:
            spec = current_spec
        elif spec != current_spec:
            raise ValueError(f"Incompatible state dict: {path}")
        for key, value in current.items():
            if value.is_floating_point():
                if key not in averaged:
                    averaged[key] = value.float().clone().mul_(weight)
                else:
                    averaged[key].add_(value.float(), alpha=weight)
            else:
                averaged[key] = value.clone()
        del current, checkpoint
    return averaged


class NullWriter:
    def set_step(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default="artifacts/ema_validation")
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0, 0.5, 0.8, 0.9, 0.95, 0.99, 1]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--anchor-best",
        action="store_true",
        help="Initialize recursive EMA from model_best, then apply the last snapshots",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    run, output = ROOT_PATH / args.run, ROOT_PATH / args.output
    paths = sorted(
        (
            p
            for p in run.glob("checkpoint-epoch*.pth")
            if p.stem.removeprefix("checkpoint-epoch").isdigit()
        ),
        key=lambda p: int(p.stem.removeprefix("checkpoint-epoch")),
    )[-10:]
    if not paths or args.batch_size < 1:
        parser.error("Need numbered checkpoints and a positive batch size")
    for beta in args.betas:
        exponential_weights(len(paths), beta)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Use a new output directory for another search")
    latest = torch.load(paths[-1], map_location="cpu", weights_only=False, mmap=True)
    original_config = latest["config"]
    epoch = latest["epoch"]
    config = OmegaConf.create(OmegaConf.to_container(original_config, resolve=True))
    config.trainer.resume_from = None
    config.trainer.from_pretrained = None
    config.trainer.compare_greedy = False
    config.trainer.beam_voting = False
    config.trainer.amp.dtype = "auto"
    config.trainer.generation.num_beams = 1
    del latest
    device = torch.device(args.device)
    model = instantiate(config.model).to(device)
    validation = instantiate(config.datasets.val)
    loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=instantiate(config.collate_fn),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logger = logging.getLogger("ema_validation")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    trainer = Trainer(
        model=model,
        criterion=instantiate(config.loss_function).to(device),
        metrics=instantiate(config.metrics),
        optimizer=None,
        lr_scheduler=None,
        config=config,
        device=device,
        dataloaders={"train": [None], "val": loader},
        epoch_len=config.trainer.epoch_len,
        logger=logger,
        writer=NullWriter(),
    )
    results, best, best_ema = [], None, None

    def rank(record):
        metrics = record["validation"]
        if not math.isfinite(metrics["pass_at_1"]) or not math.isfinite(
            metrics["loss"]
        ):
            raise ValueError("Validation produced non-finite metrics")
        return metrics["pass_at_1"], -metrics["loss"]

    # Baseline and every averaged model share the exact same reference cache.
    for index, beta in enumerate([None] + args.betas):
        started = time.perf_counter()
        if beta is None:
            baseline = torch.load(
                run / "model_best.pth",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            state = baseline["state_dict"]
            candidate_epoch = baseline["epoch"]
        else:
            state = average_checkpoints(
                paths, beta, anchor=run / "model_best.pth" if args.anchor_best else None
            )
            candidate_epoch = epoch
        model.load_state_dict(state, strict=True)
        print(f"Evaluating beta={beta}, epoch={candidate_epoch}", flush=True)
        metrics = trainer._evaluation_epoch(
            index, "val", loader, compute_generation=True
        )
        record = dict(
            beta=beta,
            epoch=candidate_epoch,
            validation=metrics,
            seconds=time.perf_counter() - started,
        )
        payload = dict(
            state_dict=state,
            config=original_config,
            epoch=candidate_epoch,
            averaging=dict(
                beta=beta,
                checkpoints=[str(p) for p in paths],
                anchor=str(run / "model_best.pth") if args.anchor_best else None,
                rule=(
                    "start from anchor, then EMA = beta * EMA + (1-beta) * snapshot"
                    if args.anchor_best
                    else "normalized weights beta**age, age=0 for newest"
                ),
            ),
            validation=metrics,
        )
        if beta is not None and (best_ema is None or rank(record) > rank(best_ema)):
            torch.save(payload, output / "best_ema.pth")
            best_ema = record
        if best is None or rank(record) > rank(best):
            torch.save(payload, output / "model_best.pth")
            best = record
        results.append(record)
        (output / "summary.json").write_text(
            json.dumps(
                dict(
                    checkpoints=[str(p) for p in paths],
                    anchor_best=args.anchor_best,
                    criterion="max greedy Pass@1, then min val_loss",
                    generation=OmegaConf.to_container(
                        config.trainer.generation, resolve=True
                    ),
                    results=results,
                    best=best,
                    best_ema=best_ema,
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(record), flush=True)
        del state, payload
        if beta is None:
            del baseline
    print("Winner:", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
