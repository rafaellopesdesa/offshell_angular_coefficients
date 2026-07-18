"""Adapters between angular soft targets and ``nsbi-common-utils``."""

from __future__ import annotations

from collections.abc import Iterable
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


def as_float32_features(
    events: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Return a copy with neural-network feature columns stored as float32.

    PyTorch exports the density-ratio network to ONNX with float32 inputs.
    Scikit-learn scalers preserve the floating dtype of their inputs, so
    float64 feature columns would otherwise reach ONNX Runtime as doubles.
    Non-feature columns, including MC weights, retain their original dtype.
    """

    columns = list(feature_columns)
    missing = [column for column in columns if column not in events.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    return events.astype({column: np.float32 for column in columns}, copy=True)


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


def class_balanced_validation_bce(
    scores,
    target,
    weights,
    *,
    score_clip: float = 1.0e-7,
) -> float:
    r"""Evaluate a classifier on untouched source events.

    Each source event represents both hypotheses, with signed weights ``w*t``
    and ``w*(1-t)``.  The two hypotheses are normalized independently by their
    signed sums before the binary cross entropy is evaluated, exactly matching
    the weights passed to ``density_ratio_trainer`` without duplicating the
    validation rows.  With negative MC weights this is a signed objective, not
    a non-negative proper scoring rule, but it remains directly comparable
    among ensemble members trained with the same prescription.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (scores.shape == target.shape == weights.shape):
        raise ValueError("scores, target, and weights must have the same shape")
    if scores.size == 0:
        raise ValueError("validation arrays must not be empty")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(target)) or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target values must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(weights)):
        raise ValueError("validation weights must be finite")
    if not 0.0 < score_clip < 0.5:
        raise ValueError("score_clip must lie strictly between 0 and 0.5")

    positive_weights = weights * target
    negative_weights = weights * (1.0 - target)
    z_positive = float(positive_weights.sum())
    z_negative = float(negative_weights.sum())
    if z_positive <= 0.0 or z_negative <= 0.0:
        raise ValueError("both validation hypotheses must have positive weight")

    clipped_scores = np.clip(scores, score_clip, 1.0 - score_clip)
    positive_loss = -np.dot(positive_weights / z_positive, np.log(clipped_scores))
    negative_loss = -np.dot(
        negative_weights / z_negative, np.log1p(-clipped_scores)
    )
    return float(0.5 * (positive_loss + negative_loss))


def validation_loss_outlier_mask(
    losses,
    *,
    mad_scale: float = 5.0,
    min_relative_excess: float = 0.05,
    absolute_margin: float = 1.0e-4,
) -> tuple[np.ndarray, float]:
    """Flag ensemble losses above a robust, one-sided consensus threshold.

    Non-finite losses are always rejected.  Among finite losses, the threshold
    is the median plus the largest of a scaled median absolute deviation, a
    relative margin, and an absolute numerical floor.  Requiring a meaningful
    relative excess prevents harmless per-seed fluctuations from triggering
    expensive retraining.  Finite negative values are permitted because a BCE
    evaluated with signed MC weights is not bounded below.  If every attempt
    failed, the returned threshold is infinite and every member is marked for
    retry.
    """

    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array")
    if mad_scale <= 0.0:
        raise ValueError("mad_scale must be positive")
    if min_relative_excess < 0.0 or absolute_margin < 0.0:
        raise ValueError("loss margins must be non-negative")

    valid = np.isfinite(losses)
    if not np.any(valid):
        # There is no ensemble consensus yet. Treat every failed/non-finite
        # attempt as retryable; a later pass can establish a finite threshold.
        return np.ones(losses.shape, dtype=bool), float("inf")

    finite_losses = losses[valid]
    median = float(np.median(finite_losses))
    mad = float(np.median(np.abs(finite_losses - median)))
    robust_margin = mad_scale * 1.4826 * mad
    relative_margin = min_relative_excess * max(abs(median), np.finfo(float).eps)
    threshold = median + max(robust_margin, relative_margin, absolute_margin)
    return (~valid) | (losses > threshold), float(threshold)


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
