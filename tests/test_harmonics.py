import numpy as np

from offshell_angles import (
    angular_modes,
    angular_target,
    inclusive_angular_coefficients,
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
