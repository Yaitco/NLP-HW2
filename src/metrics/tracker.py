import torch


class MetricTracker:
    """Weighted scalar means; CUDA sums stay on device until logging."""

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._totals = dict.fromkeys(keys, 0.0)
        self.reset()

    def reset(self):
        self._totals = dict.fromkeys(self._totals, 0.0)
        self._counts = dict.fromkeys(self._totals, 0)

    def update(self, key, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            if not value.is_cuda:
                value = value.item()
        self._totals[key] += value * n
        self._counts[key] += n

    update_tensor = update

    def flush(self):
        keys = [k for k, v in self._totals.items() if isinstance(v, torch.Tensor)]
        if keys:
            values = torch.stack([self._totals[k] for k in keys]).cpu().tolist()
            self._totals.update(zip(keys, values))

    def avg(self, key):
        self.flush()
        return self._totals[key] / self._counts[key] if self._counts[key] else 0.0

    def result(self):
        self.flush()
        return {k: self._totals[k] / n if n else 0.0 for k, n in self._counts.items()}

    def keys(self):
        return self._totals.keys()
