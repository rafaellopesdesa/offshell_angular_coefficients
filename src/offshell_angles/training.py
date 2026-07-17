"""Adapters between angular soft targets and ``nsbi-common-utils``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassNormalizations:
    """Raw normalizations of the ``t`` and ``1-t`` weighted measures."""

    z_t: float
    z_one_minus_t: float

    @property
    def odds_correction(self) -> float:
        """Factor converting the normalized ratio into ``A(x)/B(x)``."""

        return self.z_t / self.z_one_minus_t


def prepare_weighted_classification(
    events: pd.DataFrame,
    target,
    *,
    weight_column: str = "weight",
    shuffle_seed: int = 42,
) -> tuple[pd.DataFrame, ClassNormalizations]:
    r"""Duplicate events into the ``t`` and ``1-t`` hypotheses.

    The returned class weights are independently normalized, as required by
    ``density_ratio_trainer``.  Keeping the raw normalization constants is
    essential because the fitted ratio is a ratio of normalized densities.
    """

    target = np.asarray(target, dtype=np.float64)
    if target.shape != (len(events),):
        raise ValueError("target must have exactly one value per event")
    if np.any(~np.isfinite(target)) or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target values must be finite and lie in [0, 1]")

    weights = events[weight_column].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(weights)):
        raise ValueError("MC weights must be finite")

    numerator = events.copy()
    denominator = events.copy()
    numerator["weights"] = weights * target
    denominator["weights"] = weights * (1.0 - target)

    z_t = float(numerator["weights"].sum())
    z_one_minus_t = float(denominator["weights"].sum())
    if z_t <= 0.0 or z_one_minus_t <= 0.0:
        raise ValueError("Both reweighted hypotheses must have positive normalization")

    numerator["weights_normed"] = numerator["weights"] / z_t
    denominator["weights_normed"] = denominator["weights"] / z_one_minus_t
    numerator["train_labels"] = np.float32(1.0)
    denominator["train_labels"] = np.float32(0.0)
    numerator["source_event_id"] = np.arange(len(events), dtype=np.int64)
    denominator["source_event_id"] = np.arange(len(events), dtype=np.int64)

    combined = pd.concat([numerator, denominator], ignore_index=True)
    combined = combined.sample(
        frac=1.0, random_state=shuffle_seed, ignore_index=True
    )
    return combined, ClassNormalizations(z_t, z_one_minus_t)


def recover_conditional_moment(
    normalized_ratio,
    normalizations: ClassNormalizations,
    bound: float,
):
    r"""Recover ``E[t|x]`` and ``E[h|x]`` from the toolkit ratio.

    If ``r`` is the ratio of independently normalized class densities, then

    .. math::

       E[t|x] = \frac{(Z_t/Z_{1-t})r}{1 + (Z_t/Z_{1-t})r}.
    """

    ratio = np.asarray(normalized_ratio, dtype=np.float64)
    if np.any(ratio < 0.0) or np.any(~np.isfinite(ratio)):
        raise ValueError("Predicted density ratios must be finite and non-negative")

    corrected_odds = normalizations.odds_correction * ratio
    eta = corrected_odds / (1.0 + corrected_odds)
    moment = bound * (2.0 * eta - 1.0)
    return eta, moment

