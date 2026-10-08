"""Rate (Poisson) encoding.

Input  : x of shape [B, F], values in [0, 1] interpreted as spike probabilities.
Output : spikes of shape [T, B, F] with Bernoulli(x) draws at each timestep.
"""
import torch


def rate_encode(x: torch.Tensor, num_steps: int) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    spikes = torch.bernoulli(x.unsqueeze(0).expand(num_steps, *x.shape))
    return spikes
