import numpy as np

from offshell_angles import (
    angular_target,
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

