import warnings

import numpy as np

from offshell_angles import (
    angular_modes,
    angular_target,
    binned_angular_coefficient,
    inclusive_angular_coefficients,
    inclusive_angular_statistics,
    symmetric_angular_bound,
    symmetric_angular_harmonic,
)


def test_constant_mode_matches_four_pi_normalization():
    angles = np.linspace(0.1, 2.9, 7)
    values = symmetric_angular_harmonic(
        angles, 0.3, angles[::-1], -0.7, 0, 0, 0, 0
    )
    np.testing.assert_allclose(values, 1.0 / (4.0 * np.pi), rtol=1.0e-14)
    np.testing.assert_allclose(4.0 * np.pi * values, 1.0, rtol=1.0e-14)


def test_target_respects_analytic_bound():
    rng = np.random.default_rng(7)
    size = 10_000
    theta1 = np.arccos(rng.uniform(-1.0, 1.0, size))
    theta2 = np.arccos(rng.uniform(-1.0, 1.0, size))
    phi1 = rng.uniform(-np.pi, np.pi, size)
    phi2 = rng.uniform(-np.pi, np.pi, size)

    h, target, bound = angular_target(
        theta1, phi1, theta2, phi2, 2, 1, 3, -2, component="imag"
    )
    assert bound == symmetric_angular_bound(2, 1, 3, -2)
    assert np.max(np.abs(h)) <= bound
    assert np.all((0.0 <= target) & (target <= 1.0))
    np.testing.assert_allclose(h, bound * (2.0 * target - 1.0))



def test_canonical_mode_order_through_l3():
    modes = angular_modes(3)

    assert len(modes) == 16
    assert modes[:4] == ((0, 0), (1, -1), (1, 0), (1, 1))
    assert modes[-1] == (3, 3)

    pairs = [
        (alpha, beta)
        for alpha_index, alpha in enumerate(modes)
        for beta in modes[alpha_index:]
    ]
    assert len(pairs) == 16 * 17 // 2
    assert all(
        l1 < l2 or (l1 == l2 and m1 <= m2)
        for ((l1, m1), (l2, m2)) in pairs
    )


def test_inclusive_projection_normalization_and_triangle():
    rng = np.random.default_rng(19)
    size = 200
    theta1 = np.arccos(rng.uniform(-1.0, 1.0, size))
    theta2 = np.arccos(rng.uniform(-1.0, 1.0, size))
    phi1 = rng.uniform(-np.pi, np.pi, size)
    phi2 = rng.uniform(-np.pi, np.pi, size)
    weights = rng.uniform(0.1, 2.0, size)

    modes, coefficients = inclusive_angular_coefficients(
        theta1, phi1, theta2, phi2, weights, l_max=3
    )

    upper = np.triu_indices(len(modes))
    lower = np.tril_indices(len(modes), k=-1)
    assert np.isfinite(coefficients[upper]).all()
    assert np.isnan(coefficients[lower].real).all()
    assert np.isnan(coefficients[lower].imag).all()
    np.testing.assert_allclose(
        coefficients[0, 0],
        weights.sum(),
        rtol=2.0e-14,
        atol=2.0e-14 * weights.sum(),
    )

    alpha_index = modes.index((1, 0))
    beta_index = modes.index((3, -2))
    basis = symmetric_angular_harmonic(
        theta1, phi1, theta2, phi2, 1, 0, 3, -2
    )
    expected = 4.0 * np.pi * np.sum(weights * np.conjugate(basis))
    np.testing.assert_allclose(
        coefficients[alpha_index, beta_index],
        expected,
        rtol=2.0e-14,
        atol=2.0e-14 * max(1.0, abs(expected)),
    )


def test_nonfinite_events_are_masked_per_coefficient():
    theta1 = np.array([0.4, np.nan, 1.2, 2.0])
    phi1 = np.array([0.1, np.nan, -0.3, 0.7])
    theta2 = np.array([0.8, 1.1, 1.5, 2.4])
    phi2 = np.array([-0.2, 0.4, 0.9, -1.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modes, coefficients, valid_fractions = inclusive_angular_coefficients(
            theta1,
            phi1,
            theta2,
            phi2,
            weights,
            l_max=1,
            return_valid_fractions=True,
        )

    assert any("removed fraction" in str(item.message) for item in caught)

    constant_index = modes.index((0, 0))
    np.testing.assert_allclose(
        coefficients[constant_index, constant_index],
        weights.sum(),
    )
    assert valid_fractions[constant_index, constant_index] == 1.0

    alpha_index = modes.index((1, 0))
    beta_index = modes.index((1, 0))
    assert valid_fractions[alpha_index, beta_index] == 0.75

    basis = symmetric_angular_harmonic(
        theta1, phi1, theta2, phi2, 1, 0, 1, 0
    )
    valid = np.isfinite(weights) & np.isfinite(basis)
    expected = 4.0 * np.pi * np.sum(
        weights[valid] * np.conjugate(basis[valid])
    )
    np.testing.assert_allclose(
        coefficients[alpha_index, beta_index],
        expected,
    )


def test_relative_uncertainty_uses_signed_weights_and_denominator_covariance():
    theta1 = np.array([0.3, 0.8, 1.4, 2.1, 2.7])
    phi1 = np.array([-0.5, 0.2, 1.0, -1.4, 2.2])
    theta2 = np.array([0.7, 1.1, 1.8, 2.4, 0.5])
    phi2 = np.array([0.4, -0.9, 1.7, 0.3, -2.0])
    weights = np.array([2.0, -0.4, 1.2, -0.3, 0.8])

    statistics = inclusive_angular_statistics(
        theta1, phi1, theta2, phi2, weights, l_max=1
    )
    alpha_index = statistics.modes.index((1, 0))
    beta_index = statistics.modes.index((1, 0))
    basis = symmetric_angular_harmonic(
        theta1, phi1, theta2, phi2, 1, 0, 1, 0
    )
    contributions = 4.0 * np.pi * weights * np.conjugate(basis)
    numerator = contributions.real.sum()
    denominator = weights.sum()
    relative = numerator / denominator
    variance = (
        np.sum(contributions.real**2)
        + relative**2 * np.sum(weights**2)
        - 2.0 * relative * np.sum(contributions.real * weights)
    ) / denominator**2
    expected_uncertainty = np.sqrt(max(variance, 0.0))

    np.testing.assert_allclose(
        statistics.relative_coefficients[alpha_index, beta_index].real,
        relative,
    )
    np.testing.assert_allclose(
        statistics.relative_uncertainties_real[alpha_index, beta_index],
        expected_uncertainty,
    )
    np.testing.assert_allclose(
        statistics.significances_real[alpha_index, beta_index],
        relative / expected_uncertainty,
    )
    np.testing.assert_allclose(statistics.relative_coefficients[0, 0], 1.0)
    assert statistics.relative_uncertainties_real[0, 0] == 0.0
    assert np.isnan(statistics.significances_real[0, 0])


def test_binned_projection_preserves_signed_inclusive_sum_and_weight_squared_error():
    observable = np.array([-2.0, -0.5, 0.2, 0.9, 3.0])
    harmonic_component = np.array([0.1, -0.2, 0.4, 0.3, -0.1])
    weights = np.array([1.0, -0.5, 2.0, -0.25, 0.75])
    edges = np.array([-1.0, 0.0, 1.0])
    result = binned_angular_coefficient(
        observable,
        harmonic_component,
        weights,
        edges,
        fold_flow=True,
    )
    contributions = 4.0 * np.pi * weights * harmonic_component

    np.testing.assert_allclose(result.values.sum(), contributions.sum())
    np.testing.assert_allclose(
        np.sum(result.uncertainties**2),
        np.sum(contributions**2),
    )
    assert result.valid_fraction == 1.0
