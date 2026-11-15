"""Byzantine-robust aggregation and watermark-defense primitives.

All functions operate on plain NumPy arrays (``dict[str, np.ndarray]`` keyed by
parameter name) so they can be unit-tested without Flower or PyTorch.  The
strategy layer is responsible for converting ``ArrayRecord`` <-> NumPy dicts.

Terminology
-----------
``global_model`` : current global model weights (``dict[str, np.ndarray]``).
``models``       : per-client model weights (one ``dict[str, np.ndarray]`` each).
``num_examples`` : per-client training set size (the FedAvg weighting factor).
``ber_values``   : per-client watermark BER (for watermark-based defenses).

For the model-update defenses (FLTrust, Krum, Trimmed Mean, Median) the client
"update" is defined as ``delta_i = model_i - global_model``, which is the
standard gradient/update proxy in Byzantine-robust FL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DefenseOutcome:
    """Result of applying a defense to a single aggregation round."""

    aggregated: dict[str, np.ndarray]  # final global model weights
    kept: list[int]  # indices (into the input list) of clients kept
    weights: list[float]  # per-client weight, aligned to all clients (0 if excluded)
    excluded: list[bool]  # per-client excluded flag, aligned to all clients
    extra: dict = field(default_factory=dict)  # diagnostics (thresholds, scores, ...)


def _weighted_average_models(models, weights):
    """``sum_i weights[i] * models[i]`` — mirrors Flower's aggregate_arrayrecords.

    Matches the reference FedAvg exactly (including NumPy's float promotion of
    integer tensors such as BatchNorm ``num_batches_tracked``).
    """
    out: dict[str, np.ndarray] = {}
    for weight, model in zip(weights, models):
        for key, value in model.items():
            term = value * weight
            if key not in out:
                out[key] = term
            else:
                out[key] += term
    return out


def _flatten(model: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([a.reshape(-1) for a in model.values()])


def _add(global_model, delta):
    return {k: global_model[k] + delta[k] for k in global_model}


def _deltas(models, global_model):
    return [{k: m[k] - global_model[k] for k in global_model} for m in models]


# --------------------------------------------------------------------------- #
# FedAvg (no defense) and watermark-based defenses
# --------------------------------------------------------------------------- #


def fedavg(global_model, models, num_examples, ber_values=None, **kwargs):
    """Plain federated averaging (no defense)."""
    n = len(models)
    total = float(sum(num_examples))
    weights = [w / total for w in num_examples]
    return DefenseOutcome(
        aggregated=_weighted_average_models(models, weights),
        kept=list(range(n)),
        weights=weights,
        excluded=[False] * n,
        extra={"method": "fedavg", "threshold": float("inf")},
    )


def _watermark_threshold_defense(
    method, threshold_fn, global_model, models, num_examples, ber_values, **kwargs
):
    """Shared logic for MAD / fixed-threshold watermark defenses."""
    n = len(models)
    bers = np.asarray(ber_values, dtype=float)
    threshold = threshold_fn(bers)

    excluded = [bool(ber > threshold) for ber in ber_values]
    kept = [i for i in range(n) if not excluded[i]]

    fallback = False
    if not kept:
        # No client below threshold -> keep everyone rather than null-aggregating.
        kept = list(range(n))
        excluded = [False] * n
        fallback = True

    total = float(sum(num_examples[i] for i in kept))
    weights = [0.0] * n
    for i in kept:
        weights[i] = num_examples[i] / total

    agg = _weighted_average_models([models[i] for i in kept], [weights[i] for i in kept])

    med = float(np.median(bers)) if len(bers) else None
    mad = float(np.median(np.abs(bers - med))) if len(bers) >= 2 else None
    return DefenseOutcome(
        aggregated=agg,
        kept=kept,
        weights=weights,
        excluded=excluded,
        extra={
            "method": method,
            "threshold": threshold,
            "median": med,
            "mad": mad,
            "fallback": fallback,
        },
    )


def mad_defense(global_model, models, num_examples, ber_values=None, mad_alpha=4.0, **kwargs):
    """Dynamic per-round threshold ``median(BER) + mad_alpha * MAD(BER)``."""
    ber_values = ber_values or []

    def threshold_fn(bers):
        if len(bers) < 2:
            return float("inf")
        med = float(np.median(bers))
        mad = float(np.median(np.abs(bers - med)))
        return med + mad_alpha * mad

    return _watermark_threshold_defense(
        "mad", threshold_fn, global_model, models, num_examples, ber_values, **kwargs
    )


def fixed_defense(global_model, models, num_examples, ber_values=None, max_ber=1.0, **kwargs):
    """Fixed watermark threshold: exclude any client with BER > ``max_ber``."""
    ber_values = ber_values or []

    def threshold_fn(_bers):
        return float(max_ber)

    return _watermark_threshold_defense(
        "fixed", threshold_fn, global_model, models, num_examples, ber_values, **kwargs
    )


# --------------------------------------------------------------------------- #
# Distance / statistical Byzantine-robust defenses
# --------------------------------------------------------------------------- #


def krum(global_model, models, num_examples, n_attackers=0, m=None, **kwargs):
    """Multi-Krum.

    Scores each update by the sum of squared L2 distances to its ``n - f - 2``
    nearest neighbours, then averages the ``m`` lowest-scoring updates.
    """
    n = len(models)
    deltas = _deltas(models, global_model)
    flat = [_flatten(d).astype(np.float32) for d in deltas]

    if m is None:
        m = max(1, n - n_attackers - 2)
    m = min(m, n)
    k_nearest = min(max(1, n - n_attackers - 2), n - 1)

    # Squared pairwise distances via Gram matrix (memory-safe: O(n^2), not O(n*d)).
    norms = np.array([float(np.dot(f, f)) for f in flat], dtype=np.float64)
    gram = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            gram[i, j] = gram[j, i] = float(np.dot(flat[i], flat[j]))
    dist_sq = norms[:, None] + norms[None, :] - 2.0 * gram
    np.fill_diagonal(dist_sq, np.inf)

    scores = np.sum(np.sort(dist_sq, axis=1)[:, :k_nearest], axis=1)
    selected = np.argsort(scores)[:m].tolist()

    kept = [int(i) for i in selected]
    weights = [0.0] * n
    for i in kept:
        weights[i] = 1.0 / m
    agg = _add(global_model, _weighted_average_models([deltas[i] for i in kept], [1.0 / m] * m))

    return DefenseOutcome(
        aggregated=agg,
        kept=kept,
        weights=weights,
        excluded=[i not in kept for i in range(n)],
        extra={"method": "krum", "m": m, "n_attackers": n_attackers, "scores": scores.tolist()},
    )


def trimmed_mean(global_model, models, num_examples, trim_count=0, **kwargs):
    """Coordinate-wise trimmed mean of client updates.

    ``trim_count`` lowest and highest values are dropped per coordinate.
    """
    n = len(models)
    deltas = _deltas(models, global_model)
    lo = max(0, int(trim_count))
    hi = n - lo
    if hi <= lo:
        hi = lo + 1

    out = {}
    for k in global_model:
        stacked = np.stack([d[k] for d in deltas], axis=0)
        if lo > 0:
            stacked = np.sort(stacked, axis=0)[lo:hi]
        out[k] = stacked.mean(axis=0)

    weights = [1.0 / n] * n
    return DefenseOutcome(
        aggregated=_add(global_model, out),
        kept=list(range(n)),
        weights=weights,
        excluded=[False] * n,
        extra={"method": "trimmed-mean", "trim_count": lo},
    )


def coordinate_median(global_model, models, num_examples, **kwargs):
    """Coordinate-wise median of client updates."""
    n = len(models)
    deltas = _deltas(models, global_model)
    out = {}
    for k in global_model:
        stacked = np.stack([d[k] for d in deltas], axis=0)
        out[k] = np.median(stacked, axis=0)

    weights = [1.0 / n] * n
    return DefenseOutcome(
        aggregated=_add(global_model, out),
        kept=list(range(n)),
        weights=weights,
        excluded=[False] * n,
        extra={"method": "median"},
    )


def fltrust(global_model, models, num_examples, server_update=None, **kwargs):
    """FLTrust (Cao et al., 2021).

    Weights each client update by ``ReLU(cos(delta_i, server_update))`` after
    re-scaling it to the server update's magnitude.
    """
    n = len(models)
    if server_update is None:
        raise ValueError("FLTrust requires `server_update` (the server's own update).")

    deltas = _deltas(models, global_model)
    g0 = _flatten(server_update).astype(np.float32)
    g0_norm = float(np.linalg.norm(g0)) + 1e-12

    trust_scores = []
    normalized_deltas = []
    for d in deltas:
        df = _flatten(d).astype(np.float32)
        d_norm = float(np.linalg.norm(df)) + 1e-12
        cos = float(np.dot(df, g0) / (d_norm * g0_norm))
        ts = max(0.0, cos)
        trust_scores.append(ts)
        normalized_deltas.append({k: d[k] * (g0_norm / d_norm) for k in d})

    ts_sum = float(sum(trust_scores))
    if ts_sum <= 1e-12:
        # All trust scores zero -> fall back to the server update alone.
        agg = _add(global_model, server_update)
        weights = [0.0] * n
    else:
        weights = [t / ts_sum for t in trust_scores]
        agg = _add(global_model, _weighted_average_models(normalized_deltas, weights))

    return DefenseOutcome(
        aggregated=agg,
        kept=list(range(n)),
        weights=weights,
        excluded=[False] * n,
        extra={
            "method": "fltrust",
            "trust_scores": trust_scores,
            "ts_sum": ts_sum,
            "server_norm": g0_norm,
        },
    )


# --------------------------------------------------------------------------- #
# Registry / dispatch
# --------------------------------------------------------------------------- #

_DEFENSE_REGISTRY = {
    "none": fedavg,
    "fedavg": fedavg,
    "mad": mad_defense,
    "fixed": fixed_defense,
    "fltrust": fltrust,
    "krum": krum,
    "trimmed-mean": trimmed_mean,
    "median": coordinate_median,
}


def run_defense(method, global_model, models, num_examples, ber_values=None, **cfg):
    """Apply a defense by name and return its :class:`DefenseOutcome`."""
    fn = _DEFENSE_REGISTRY.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown defense-method '{method}'. Available: {sorted(_DEFENSE_REGISTRY)}"
        )
    return fn(
        global_model,
        models,
        num_examples,
        ber_values=ber_values or [],
        **cfg,
    )
