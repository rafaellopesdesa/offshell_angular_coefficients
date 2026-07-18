import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from offshell_angles import (
    as_float32_features,
    prepare_weighted_classification,
    recover_conditional_moment,
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
