import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from offshell_angles import (
    as_float32_features,
    class_balanced_validation_bce,
    prepare_weighted_classification,
    recover_conditional_moment,
    validation_loss_outlier_mask,
)


def test_reweighted_classes_and_normalization_correction():
    events = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "weight": [1.0, 2.0, 1.5, 0.5],
        }
    )
    target = np.array([0.2, 0.7, 0.4, 0.9])
    training, normalizations = prepare_weighted_classification(events, target)

    class_sums = training.groupby("train_labels")["weights_normed"].sum()
    np.testing.assert_allclose(class_sums.to_numpy(), [1.0, 1.0])
    assert len(training) == 2 * len(events)
    assert training["source_event_id"].value_counts().eq(2).all()

    for x_value in (0.0, 1.0):
        mask = events["x"].to_numpy() == x_value
        weights = events.loc[mask, "weight"].to_numpy()
        local_target = target[mask]
        a = np.sum(weights * local_target)
        b = np.sum(weights * (1.0 - local_target))
        normalized_ratio = (a / normalizations.z_t) / (
            b / normalizations.z_one_minus_t
        )
        eta, moment = recover_conditional_moment(
            [normalized_ratio], normalizations, bound=3.0
        )
        np.testing.assert_allclose(eta, [a / (a + b)])
        np.testing.assert_allclose(moment, [3.0 * (2.0 * eta[0] - 1.0)])


def test_reweighted_classification_preserves_negative_event_weights():
    events = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "weight": [2.0, -0.25, 1.0],
        }
    )
    target = np.array([0.2, 0.7, 0.5])
    training, normalizations = prepare_weighted_classification(events, target)

    numerator = training[training["train_labels"] == 1.0].sort_values(
        "source_event_id"
    )
    denominator = training[training["train_labels"] == 0.0].sort_values(
        "source_event_id"
    )
    expected_numerator = events["weight"].to_numpy() * target
    expected_denominator = events["weight"].to_numpy() * (1.0 - target)
    np.testing.assert_allclose(numerator["weights"], expected_numerator)
    np.testing.assert_allclose(denominator["weights"], expected_denominator)
    assert numerator.loc[numerator["source_event_id"] == 1, "weights"].item() < 0.0
    assert denominator.loc[denominator["source_event_id"] == 1, "weights"].item() < 0.0
    np.testing.assert_allclose(normalizations.z_t, expected_numerator.sum())
    np.testing.assert_allclose(
        normalizations.z_one_minus_t, expected_denominator.sum()
    )



def test_float32_features_match_onnx_input_dtype():
    events = pd.DataFrame(
        {
            "x0": np.array([1.0, 2.0], dtype=np.float64),
            "x1": np.array([3.0, 5.0], dtype=np.float64),
            "weight": np.array([0.25, 0.75], dtype=np.float64),
        }
    )

    converted = as_float32_features(events, ["x0", "x1"])

    assert converted[["x0", "x1"]].dtypes.eq(np.dtype("float32")).all()
    assert converted["weight"].dtype == np.dtype("float64")
    assert events["x0"].dtype == np.dtype("float64")

    scaled = MinMaxScaler().fit_transform(converted[["x0", "x1"]])
    assert scaled.dtype == np.dtype("float32")


def test_float32_features_reject_missing_columns():
    with pytest.raises(KeyError, match="missing"):
        as_float32_features(pd.DataFrame({"x": [1.0]}), ["missing"])


def test_class_balanced_validation_bce_prefers_discriminating_scores():
    target = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.array([1.0, 3.0, 2.0, 4.0])

    good_loss = class_balanced_validation_bce(
        [0.9, 0.8, 0.2, 0.1], target, weights
    )
    uninformative_loss = class_balanced_validation_bce(
        np.full(4, 0.5), target, weights
    )

    assert good_loss < uninformative_loss
    assert uninformative_loss == pytest.approx(np.log(2.0))


def test_class_balanced_validation_bce_matches_independent_normalization():
    scores = np.array([0.8, 0.3])
    target = np.array([0.25, 0.75])
    weights = np.array([2.0, 1.0])
    positive = weights * target
    negative = weights * (1.0 - target)
    expected = -0.5 * (
        np.dot(positive / positive.sum(), np.log(scores))
        + np.dot(negative / negative.sum(), np.log1p(-scores))
    )

    assert class_balanced_validation_bce(scores, target, weights) == pytest.approx(
        expected
    )


def test_class_balanced_validation_bce_preserves_signed_mc_weights():
    scores = np.array([0.8, 0.4, 0.3])
    target = np.array([0.2, 0.7, 0.5])
    weights = np.array([2.0, -0.25, 1.0])
    positive = weights * target
    negative = weights * (1.0 - target)
    expected = -0.5 * (
        np.dot(positive / positive.sum(), np.log(scores))
        + np.dot(negative / negative.sum(), np.log1p(-scores))
    )

    assert class_balanced_validation_bce(scores, target, weights) == pytest.approx(
        expected
    )


def test_validation_loss_outlier_mask_rejects_only_large_high_loss():
    rejected, threshold = validation_loss_outlier_mask(
        [0.691, 0.694, 0.696, 1.8]
    )

    np.testing.assert_array_equal(rejected, [False, False, False, True])
    assert 0.696 < threshold < 1.8


def test_validation_loss_outlier_mask_tolerates_seed_fluctuations():
    rejected, threshold = validation_loss_outlier_mask(
        [0.691, 0.694, 0.696, 0.700]
    )

    assert not rejected.any()
    assert threshold > 0.700


def test_validation_loss_outlier_mask_retries_nonfinite_losses():
    rejected, threshold = validation_loss_outlier_mask(
        [0.691, np.inf, np.nan, 0.696]
    )

    np.testing.assert_array_equal(rejected, [False, True, True, False])
    assert np.isfinite(threshold)


def test_validation_loss_outlier_mask_retries_when_every_attempt_failed():
    rejected, threshold = validation_loss_outlier_mask([np.inf, np.nan, np.inf])

    assert rejected.all()
    assert np.isinf(threshold)


def test_validation_loss_outlier_mask_accepts_finite_signed_bce():
    rejected, _ = validation_loss_outlier_mask([-0.20, -0.19, -0.18, np.inf])

    np.testing.assert_array_equal(rejected, [False, False, False, True])
