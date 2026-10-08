"""Latency encoding.

Stronger input → earlier spike, single spike per neuron. Implementation:
- Map x in [0, 1] to a spike time t = floor((1 - x) * (T - 1)).
- Inputs below `threshold` never spike.
"""
import torch


def latency_encode(x: torch.Tensor, num_steps: int, threshold: float = 0.01) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    times = torch.floor((1.0 - x) * (num_steps - 1)).long()  # [B, F]
    times = times.clamp(0, num_steps - 1)
    spikes = torch.zeros(num_steps, *x.shape, device=x.device, dtype=x.dtype)
    t_index = times.unsqueeze(0)
    arange = torch.arange(num_steps, device=x.device).view(num_steps, *([1] * x.ndim))
    mask = (arange == t_index) & (x.unsqueeze(0) > threshold)
    spikes = mask.to(x.dtype)
    return spikes
