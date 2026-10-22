import math

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """Shuffle each pass, then form similarly sized batches within random pools."""

    def __init__(self, lengths, batch_size, bucket_size_multiplier=50, seed=42):
        if batch_size <= 0 or bucket_size_multiplier <= 0:
            raise ValueError("Batch size and bucket multiplier must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.pass_index = 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.pass_index)
        self.pass_index += 1
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        batches = []
        for start in range(0, len(order), self.pool_size):
            pool = order[start : start + self.pool_size]
            pool.sort(key=lambda index: self.lengths[index], reverse=True)
            batches.extend(
                pool[offset : offset + self.batch_size]
                for offset in range(0, len(pool), self.batch_size)
            )
        for index in torch.randperm(len(batches), generator=generator).tolist():
            yield batches[index]
