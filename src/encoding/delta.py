"""Delta modulation encoding.

For tabular (non-sequential) data we synthesise a temporal signal by holding
the input value constant for the first half of the window then dropping it to
zero, and emit spikes on the temporal contrast. The result is a transient
spike burst whose magnitude is proportional to the input value — close in
spirit to snntorch.spikegen.delta.
"""
import torch


def delta_encode(x: torch.Tensor, num_steps: int, threshold: float = 0.05) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    half = max(1, num_steps // 2)
    signal = torch.zeros(num_steps, *x.shape, device=x.device, dtype=x.dtype)
    signal[:half] = x
    # Temporal differences with a leading zero
    prev = torch.zeros_like(signal[0])
    out = torch.zeros_like(signal)
    for t in range(num_steps):
        diff = signal[t] - prev
        out[t] = (diff.abs() > threshold).to(x.dtype) * diff.sign()
        prev = signal[t]
    # Treat negative spikes as no-op for downstream (binary spike trains)
    return (out > 0).to(x.dtype)
