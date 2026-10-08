"""Permuted-time control (arm A2).

Identical to latency encoding in every respect except one: which feature
receives which spike time. Per sample it preserves exactly

  * the set of features that spike        (identical to latency)
  * the total input spike count           (identical)
  * the multiset of spike times           (identical, so the per-timestep input
                                           current profile is unchanged)

and destroys only the amplitude-to-time assignment. That makes
``latency - permuted`` the value of the amplitude-to-time *mapping*, and
``permuted - delta@theta_lat`` the value of temporal *spreading* as such.

Uniform random times over {0..T-1} would be a weaker control: latency's time
distribution is skewed by the amplitude distribution, so uniform sampling would
also change the temporal envelope and the two effects would be confounded.

Determinism
-----------
The permutation must be fixed per sample and not redrawn each epoch --- a fresh
draw per epoch turns the control into data augmentation and confounds it with a
regularisation effect.

The permutation is therefore derived from a counter-based hash of the sample's
own feature vector and the run seed, rather than from a stored table keyed on a
row index. Three reasons: the encoder API receives only ``x`` and has no row
identifier to key on; content-derivation is invariant to shuffling and to batch
composition, which a positional key is not; and it needs no artifact, so it
cannot fall out of sync with a split manifest. Identical rows receive identical
permutations, which is the correct behaviour for a deterministic encoder --- it
makes the arm a function of the input, as the other two encoders are.

The run seed enters through ``HISNN_SEED``, which the experiment runners already
set per subprocess, so different seeds draw different permutations.
"""
import os

import torch

#: Kept below 2**32 so products with 30-bit state stay inside int64 and cannot
#: overflow. Values are the usual odd mixing constants.
_C1, _C2, _C3 = 2654435761, 2246822519, 3266489917
_MASK = 0x3FFFFFFF          # 30 bits of state

_SEED = int(os.environ.get("HISNN_SEED", "0"))


def _mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Counter-based hash of two int64 tensors into [0, 2**30)."""
    h = (a * _C1 + b * 40503 + _SEED * 2971215073 + 12345) & _MASK
    h = ((h ^ (h >> 15)) * _C2) & _MASK
    h = ((h ^ (h >> 13)) * _C3) & _MASK
    return h ^ (h >> 16)


def permuted_latency_encode(x: torch.Tensor, num_steps: int,
                            threshold: float = 0.01) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    B, d = x.shape
    active = x > threshold
    times = torch.floor((1.0 - x) * (num_steps - 1)).long().clamp(0, num_steps - 1)

    # Row hash: a weighted reduction over quantised features. Collisions are
    # harmless --- this seeds a permutation, it does not need to be injective.
    q = (x * 65535.0).round().long()
    fidx = torch.arange(d, device=x.device, dtype=torch.long)
    w = _mix(fidx, torch.zeros_like(fidx)) | 1
    row = (q * w).sum(dim=1) & _MASK                                    # [B]

    # Active features' own times, ascending; inactive pushed past the tail so
    # they never occupy a rank below an active one.
    sentinel = torch.full_like(times, num_steps + 1)
    sorted_times, _ = torch.sort(torch.where(active, times, sentinel), dim=1)

    # A random rank per active feature. The 2.0 sentinel guarantees active
    # features occupy ranks 0..n_b-1 in random order, so the gather below is a
    # bijection from the active set onto its own time multiset.
    key = (_mix(row.unsqueeze(1).expand(B, d), fidx.unsqueeze(0).expand(B, d))
           .to(x.dtype) / float(_MASK + 1))
    key = torch.where(active, key, torch.full_like(key, 2.0))
    rank = torch.argsort(torch.argsort(key, dim=1), dim=1)

    new_t = torch.gather(sorted_times, 1, rank)

    spikes = torch.zeros(num_steps, B, d, device=x.device, dtype=x.dtype)
    b, f = torch.nonzero(active, as_tuple=True)
    spikes[new_t[b, f], b, f] = 1.0
    return spikes
