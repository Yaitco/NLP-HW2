import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.lr_scheduler import build_lr_scheduler, training_budget

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(
        config.trainer.seed,
        deterministic=config.trainer.get("deterministic", True),
    )
    torch.set_float32_matmul_precision(
        config.trainer.get("matmul_precision", "highest")
    )

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    compile_config = config.trainer.get("compile", {})
    if compile_config.get("enabled", False):
        # Compile in place: state_dict names and custom generation methods stay intact.
        model.compile(
            mode=compile_config.get("mode", "default"),
            dynamic=compile_config.get("dynamic", True),
            fullgraph=compile_config.get("fullgraph", False),
        )
        logger.info("torch.compile enabled; the first steps include compilation.")

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_kwargs = {"params": trainable_params}
    if config.optimizer.get("fused", False) and torch.device(device).type != "cuda":
        optimizer_kwargs["fused"] = False
    optimizer = instantiate(config.optimizer, **optimizer_kwargs)
    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    microbatches_per_epoch = (
        len(dataloaders["train"]) if epoch_len is None else epoch_len
    )
    accumulation = config.trainer.get("accumulate_grad_batches", 1)
    lr_scheduler = build_lr_scheduler(
        config.lr_scheduler,
        optimizer,
        microbatches_per_epoch=microbatches_per_epoch,
        accumulate_grad_batches=accumulation,
        n_epochs=config.trainer.n_epochs,
    )
    microbatches, updates = training_budget(
        microbatches_per_epoch, accumulation, config.trainer.n_epochs
    )
    logger.info(
        f"Epoch: {microbatches_per_epoch} microbatches; accumulation: {accumulation}; "
        f"budget: {microbatches} microbatches / {updates} optimizer updates"
    )

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
