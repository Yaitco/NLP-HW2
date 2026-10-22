from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn as default_collate_fn
from src.datasets.samplers import LengthGroupedBatchSampler
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    collate_fn = (
        instantiate(config.collate_fn)
        if config.get("collate_fn") is not None
        else default_collate_fn
    )

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        loader_config = dict(config.dataloader)
        eval_batch_size = loader_config.pop("eval_batch_size", None)
        if dataset_partition != "train" and eval_batch_size is not None:
            loader_config["batch_size"] = eval_batch_size
        if loader_config.get("num_workers", 0) == 0:
            loader_config.pop("prefetch_factor", None)
            loader_config["persistent_workers"] = False

        batching = config.get("batching", {})
        if dataset_partition == "train" and batching.get("group_by_length", False):
            sampler = LengthGroupedBatchSampler(
                dataset.get_lengths(),
                batch_size=loader_config.pop("batch_size"),
                bucket_size_multiplier=batching.get("bucket_size_multiplier", 50),
                seed=config.trainer.seed,
            )
            partition_dataloader = instantiate(
                loader_config,
                dataset=dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                worker_init_fn=set_worker_seed,
            )
        else:
            partition_dataloader = instantiate(
                loader_config,
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=(dataset_partition == "train"),
                worker_init_fn=set_worker_seed,
            )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
