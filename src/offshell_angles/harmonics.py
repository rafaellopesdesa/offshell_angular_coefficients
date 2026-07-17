"""Complex symmetric spherical-harmonic basis and BCE targets."""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


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

    direct = sph_harm_y(l1, m1, theta1, phi1) * sph_harm_y(
        l2, m2, theta2, phi2
    )
    exchanged = sph_harm_y(l1, m1, theta2, phi2) * sph_harm_y(
        l2, m2, theta1, phi1
    )
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

