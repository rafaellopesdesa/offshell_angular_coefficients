"""Complex symmetric spherical-harmonic basis and BCE targets."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import sph_harm_y


def angular_modes(l_max: int) -> tuple[tuple[int, int], ...]:
    r"""Return canonical angular modes ordered by :math:`(\ell,m)`.

    The ordering is increasing in ``ell``, then increasing in ``m``.
    Consequently, an index pair ``i <= j`` implements

    .. math::

       \alpha \preceq \beta
       \iff
       \ell_1 < \ell_2
       \quad\text{or}\quad
       (\ell_1=\ell_2\ \text{and}\ m_1\leq m_2).
    """

    if not isinstance(l_max, (int, np.integer)) or l_max < 0:
        raise ValueError("l_max must be a non-negative integer")

    return tuple(
        (ell, m)
        for ell in range(int(l_max) + 1)
        for m in range(-ell, ell + 1)
    )


def inclusive_angular_coefficients(
    theta1,
    phi1,
    theta2,
    phi2,
    weights,
    *,
    l_max: int = 3,
    return_valid_fractions: bool = False,
):
    r"""Project all inclusive symmetric coefficients through ``l_max``.

    For the convention

    .. math::

       p(\Omega_1,\Omega_2)
       = \frac{1}{4\pi}
         \sum_{\alpha\preceq\beta}
         S_{\alpha\beta}\,
         \mathcal Y^{(+)}_{\alpha\beta},

    the Monte Carlo estimator is

    .. math::

       \widehat S_{\alpha\beta}
       = 4\pi\sum_i w_i
         \mathcal Y^{(+)*}_{\alpha\beta}(\Omega_{1,i},\Omega_{2,i}).

    Non-finite events are masked independently for each coefficient and a
    warning reports the affected fraction range. Returns the canonical mode
    tuple and a complex square matrix. Only entries with row index ``i <= j``
    are populated; the redundant lower triangle is filled with complex NaNs.
    If ``return_valid_fractions=True``, a matrix containing the retained event
    fraction for each coefficient is returned as a third value.
    """

    modes = angular_modes(l_max)
    theta1, phi1, theta2, phi2, weights = np.broadcast_arrays(
        np.asarray(theta1, dtype=np.float64),
        np.asarray(phi1, dtype=np.float64),
        np.asarray(theta2, dtype=np.float64),
        np.asarray(phi2, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    if weights.size == 0:
        raise ValueError("At least one event is required for the projection")

    theta1 = theta1.reshape(-1)
    phi1 = phi1.reshape(-1)
    theta2 = theta2.reshape(-1)
    phi2 = phi2.reshape(-1)
    weights = weights.reshape(-1)

    # Evaluate each one-particle harmonic only once per solid angle. Pairwise
    # masks are still constructed below because a non-finite value can affect
    # different (alpha, beta) coefficients differently.
    harmonics1 = np.stack(
        [_single_angular_harmonic(ell, m, theta1, phi1) for ell, m in modes]
    )
    harmonics2 = np.stack(
        [_single_angular_harmonic(ell, m, theta2, phi2) for ell, m in modes]
    )

    mode_count = len(modes)
    coefficients = np.full(
        (mode_count, mode_count),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    valid_fractions = np.full(
        (mode_count, mode_count),
        np.nan,
        dtype=np.float64,
    )
    finite_weights = np.isfinite(weights)

    for alpha_index in range(mode_count):
        for beta_index in range(alpha_index, mode_count):
            identical = int(alpha_index == beta_index)
            normalization = np.sqrt(2.0 * (1.0 + identical))
            basis = (
                harmonics1[alpha_index] * harmonics2[beta_index]
                + harmonics2[alpha_index] * harmonics1[beta_index]
            ) / normalization

            valid = finite_weights & np.isfinite(basis)
            valid_fractions[alpha_index, beta_index] = np.mean(valid)
            if np.any(valid):
                coefficients[alpha_index, beta_index] = (
                    4.0
                    * np.pi
                    * np.sum(
                        weights[valid] * np.conjugate(basis[valid]),
                        dtype=np.complex128,
                    )
                )

    upper_triangle = np.triu_indices(mode_count)
    upper_fractions = valid_fractions[upper_triangle]
    affected = upper_fractions < 1.0
    if np.any(affected):
        removed = 1.0 - upper_fractions[affected]
        affected_pairs = np.flatnonzero(affected)
        worst_flat_index = affected_pairs[np.argmax(removed)]
        worst_alpha = upper_triangle[0][worst_flat_index]
        worst_beta = upper_triangle[1][worst_flat_index]
        warnings.warn(
            "Inclusive projection excluded non-finite events independently for "
            f"{np.count_nonzero(affected)}/{len(upper_fractions)} coefficients; "
            f"the removed fraction ranges from {np.min(removed):.6%} to "
            f"{np.max(removed):.6%} (largest for alpha={modes[worst_alpha]}, "
            f"beta={modes[worst_beta]}).",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_valid_fractions:
        return modes, coefficients, valid_fractions
    return modes, coefficients


def _single_angular_harmonic(ell: int, m: int, theta, phi):
    """Evaluate one spherical harmonic, preserving the constant mode at NaNs."""

    theta, phi = np.broadcast_arrays(theta, phi)
    if ell == 0 and m == 0:
        return np.full(
            theta.shape,
            1.0 / np.sqrt(4.0 * np.pi),
            dtype=np.complex128,
        )
    return sph_harm_y(ell, m, theta, phi)


def symmetric_angular_harmonic(
    theta1,
    phi1,
    theta2,
    phi2,
    l1: int,
    m1: int,
    l2: int,
    m2: int,
):
    r"""Evaluate :math:`\mathcal{Y}^{(+)}_{l_1m_1;l_2m_2}`.

    SciPy's ``sph_harm_y`` convention is used: ``theta`` is the polar
    (colatitudinal) angle and ``phi`` is the azimuthal angle.
    """

    if abs(m1) > l1 or abs(m2) > l2:
        raise ValueError("Spherical-harmonic orders must satisfy |m_i| <= l_i")

    identical = int((l1 == l2) and (m1 == m2))
    normalization = np.sqrt(2.0 * (1.0 + identical))

    direct = _single_angular_harmonic(
        l1, m1, theta1, phi1
    ) * _single_angular_harmonic(l2, m2, theta2, phi2)
    exchanged = _single_angular_harmonic(
        l1, m1, theta2, phi2
    ) * _single_angular_harmonic(l2, m2, theta1, phi1)
    return (direct + exchanged) / normalization


def symmetric_angular_bound(l1: int, m1: int, l2: int, m2: int) -> float:
    r"""Return a rigorous bound on ``abs(Y_plus)``.

    The addition theorem implies

    .. math::

       |Y_l^m| \leq \sqrt{(2l+1)/(4\pi)}.

    The identical-index factor makes the bound tighter for ``(l1,m1) ==
    (l2,m2)``.
    """

    if abs(m1) > l1 or abs(m2) > l2:
        raise ValueError("Spherical-harmonic orders must satisfy |m_i| <= l_i")

    identical = int((l1 == l2) and (m1 == m2))
    bound1 = np.sqrt((2.0 * l1 + 1.0) / (4.0 * np.pi))
    bound2 = np.sqrt((2.0 * l2 + 1.0) / (4.0 * np.pi))
    return float(np.sqrt(2.0 / (1.0 + identical)) * bound1 * bound2)


def angular_target(
    theta1,
    phi1,
    theta2,
    phi2,
    l1: int,
    m1: int,
    l2: int,
    m2: int,
    *,
    component: str = "real",
    tolerance: float = 1.0e-12,
):
    r"""Return ``h``, ``t`` and ``M`` for density-ratio training.

    ``h`` is the requested real component of the conjugated basis function,
    as required by the projection integral.  The soft target is

    .. math:: t = \frac{1}{2}\left(1 + h/M\right).
    """

    y_star = np.conjugate(
        symmetric_angular_harmonic(
            theta1, phi1, theta2, phi2, l1, m1, l2, m2
        )
    )

    if component == "real":
        h = np.real(y_star)
    elif component == "imag":
        h = np.imag(y_star)
    else:
        raise ValueError("component must be either 'real' or 'imag'")

    bound = symmetric_angular_bound(l1, m1, l2, m2)
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -bound - tolerance) or np.any(h > bound + tolerance):
        raise RuntimeError(
            "The evaluated harmonic exceeds its analytic bound; check the "
            "angle and spherical-harmonic conventions."
        )

    target = np.clip(0.5 * (1.0 + h / bound), 0.0, 1.0)
    return h, target, bound

