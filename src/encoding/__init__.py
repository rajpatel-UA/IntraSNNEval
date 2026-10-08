from src.encoding.rate import rate_encode
from src.encoding.latency import latency_encode
from src.encoding.delta import delta_encode
from src.encoding.permuted_latency import permuted_latency_encode

ENCODERS = {
    "rate": rate_encode,
    "latency": latency_encode,
    "delta": delta_encode,
    # Arm A2: latency with the amplitude-to-time assignment permuted. Gated
    # like latency, so it accepts a threshold.
    "permuted_latency": permuted_latency_encode,
}


def get_encoder(name: str, threshold: float = None):
    """Return the encoder, optionally overriding its amplitude gate.

    The gate is a first-class experimental factor, not an implementation
    detail: latency gates at 0.01 and delta at 0.05 by default, while rate has
    no gate at all, so the three-way encoding comparison confounds the code with
    the presence and value of a threshold. `threshold` exists so that confound
    can be isolated (see the threshold-matched ablation). Rate takes no
    threshold and rejects one rather than silently ignoring it.
    """
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoding {name!r}. Available: {list(ENCODERS)}")
    fn = ENCODERS[name]
    if threshold is None:
        return fn
    if name == "rate":
        raise ValueError("rate encoding has no amplitude gate; passing a "
                         "threshold would misrepresent the comparison")
    from functools import partial
    return partial(fn, threshold=threshold)
