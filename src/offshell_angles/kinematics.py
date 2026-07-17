"""Born projection and four-lepton angular conventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import vector


LEPTON_KEYS = ("electron_minus", "electron_plus", "muon_minus", "muon_plus")


@dataclass(frozen=True)
class BornProjectionDiagnostics:
    """Numerical checks for the ISR-removing Born projection."""

    raw_m4l: float
    born_m4l: float
    raw_y4l: float
    born_y4l: float
    raw_pt4l: float
    born_pt4l: float


def _sum_p4(momentum_map: Mapping[str, object]):
    iterator = iter(momentum_map.values())
    try:
        total = next(iterator)
    except StopIteration as exc:
        raise ValueError("At least one four-vector is required") from exc
    for momentum in iterator:
        total = total + momentum
    return total


def _spatial(momentum) -> np.ndarray:
    return np.array([momentum.px, momentum.py, momentum.pz], dtype=np.float64)


def _unit(value: np.ndarray, *, tolerance: float = 1.0e-14) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= tolerance:
        raise ValueError("Cannot define a direction from a zero-length vector")
    return np.asarray(value, dtype=np.float64) / norm


def _clip_cosine(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _signed_acos(cosine: float, orientation: float) -> float:
    angle = float(np.arccos(_clip_cosine(cosine)))
    if orientation < 0.0:
        return -angle
    return angle


def wrap_to_pi(angle: float) -> float:
    """Map an angle to the interval ``[-pi, pi)``."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def born_project_four_leptons(
    leptons: Mapping[str, object],
    *,
    check: bool = True,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-10,
):
    r"""Map a radiative color-singlet event to a Born-like configuration.

    The map follows Eqs. (2.14)--(2.18) of arXiv:2606.11083:

    1. boost longitudinally to set the four-lepton ``pz`` to zero;
    2. boost transversely to set its ``px`` and ``py`` to zero;
    3. apply the inverse of the first boost.

    The transformation is applied to every lepton.  It preserves all invariant
    masses as well as the four-lepton rapidity, while setting the four-lepton
    transverse momentum to zero.
    """

    missing = set(LEPTON_KEYS) - set(leptons)
    if missing:
        raise ValueError(f"Missing required leptons: {sorted(missing)}")

    selected = {key: leptons[key] for key in LEPTON_KEYS}
    total = _sum_p4(selected)
    if total.E <= 0.0 or total.mass <= 0.0:
        raise ValueError("The four-lepton system must be future-timelike")

    beta_l = vector.obj(x=0.0, y=0.0, z=float(total.pz / total.E))
    longitudinal_rest = {
        key: momentum.boostCM_of_beta3(beta_l)
        for key, momentum in selected.items()
    }
    total_l = _sum_p4(longitudinal_rest)

    beta_t = vector.obj(
        x=float(total_l.px / total_l.E),
        y=float(total_l.py / total_l.E),
        z=0.0,
    )
    zero_momentum = {
        key: momentum.boostCM_of_beta3(beta_t)
        for key, momentum in longitudinal_rest.items()
    }
    projected = {
        key: momentum.boost_beta3(beta_l)
        for key, momentum in zero_momentum.items()
    }
    projected_total = _sum_p4(projected)

    diagnostics = BornProjectionDiagnostics(
        raw_m4l=float(total.mass),
        born_m4l=float(projected_total.mass),
        raw_y4l=float(total.rapidity),
        born_y4l=float(projected_total.rapidity),
        raw_pt4l=float(total.pt),
        born_pt4l=float(projected_total.pt),
    )

    if check:
        scale = max(abs(total.E), 1.0)
        if not np.isclose(
            diagnostics.raw_m4l, diagnostics.born_m4l, rtol=rtol, atol=atol
        ):
            raise RuntimeError("Born projection did not preserve m4l")
        if not np.isclose(
            diagnostics.raw_y4l, diagnostics.born_y4l, rtol=rtol, atol=atol
        ):
            raise RuntimeError("Born projection did not preserve y4l")
        if diagnostics.born_pt4l > atol * scale:
            raise RuntimeError("Born projection did not remove four-lepton pT")

    return projected, diagnostics


def _helicity_frame(z_direction: np.ndarray, beam_direction: np.ndarray):
    r"""Build a right-handed helicity frame for one Z boson.

    ``z`` follows the Z direction in the four-lepton rest frame.  ``y`` is
    normal to the beam--Z production plane and ``x = y cross z``.  At the
    measure-zero collinear boundary the azimuthal origin is undefined; a
    deterministic transverse reference is used and the returned flag is true.
    """

    z_axis = _unit(z_direction)
    normal = np.cross(beam_direction, z_axis)
    degenerate = np.linalg.norm(normal) < 1.0e-12

    if not degenerate:
        y_axis = _unit(normal)
        x_axis = _unit(np.cross(y_axis, z_axis))
    else:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, z_axis)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        x_axis = _unit(reference - np.dot(reference, z_axis) * z_axis)
        y_axis = _unit(np.cross(z_axis, x_axis))

    return x_axis, y_axis, z_axis, degenerate


def _positive_lepton_angles(positive_lepton, z_boson, frame):
    x_axis, y_axis, z_axis, _ = frame
    positive_in_z_rest = positive_lepton.boostCM_of_p4(z_boson)
    direction = _unit(_spatial(positive_in_z_rest))
    cos_theta = _clip_cosine(float(np.dot(direction, z_axis)))
    theta = float(np.arccos(cos_theta))
    phi = float(
        np.arctan2(np.dot(direction, y_axis), np.dot(direction, x_axis))
    )
    return theta, phi, cos_theta


def standard_five_angles(
    leptons: Mapping[str, object],
    *,
    beam_direction=(0.0, 0.0, 1.0),
) -> dict[str, float | bool]:
    r"""Compute the five angles of arXiv:1208.4018.

    The paper labels the negative lepton as the fermion ``q_i1``.  Here V1 is
    always the dimuon system and V2 is always the dielectron system; no mass
    ordering is performed.
    """

    selected = {key: leptons[key] for key in LEPTON_KEYS}
    x_system = _sum_p4(selected)
    in_x_rest = {
        key: momentum.boostCM_of_p4(x_system)
        for key, momentum in selected.items()
    }
    z1 = in_x_rest["muon_minus"] + in_x_rest["muon_plus"]
    z2 = in_x_rest["electron_minus"] + in_x_rest["electron_plus"]

    beam = _unit(np.asarray(beam_direction, dtype=np.float64))
    q1 = _spatial(z1)
    q1_hat = _unit(q1)

    q11 = _spatial(in_x_rest["muon_minus"])
    q12 = _spatial(in_x_rest["muon_plus"])
    q21 = _spatial(in_x_rest["electron_minus"])
    q22 = _spatial(in_x_rest["electron_plus"])

    decay_normal1 = np.cross(q11, q12)
    decay_normal2 = np.cross(q21, q22)
    production_normal = np.cross(beam, q1_hat)
    decay_planes_defined = (
        np.linalg.norm(decay_normal1) > 1.0e-14
        and np.linalg.norm(decay_normal2) > 1.0e-14
    )
    production_plane_defined = np.linalg.norm(production_normal) > 1.0e-14

    if decay_planes_defined:
        n1 = _unit(decay_normal1)
        n2 = _unit(decay_normal2)
        phi = _signed_acos(
            -float(np.dot(n1, n2)),
            float(np.dot(q1_hat, np.cross(n1, n2))),
        )
    else:
        n1 = None
        phi = float("nan")

    if decay_planes_defined and production_plane_defined:
        n_sc = _unit(production_normal)
        phi1 = _signed_acos(
            float(np.dot(n1, n_sc)),
            float(np.dot(q1_hat, np.cross(n1, n_sc))),
        )
        psi = wrap_to_pi(phi1 + 0.5 * phi)
    else:
        phi1 = float("nan")
        psi = float("nan")

    mu_minus_z1 = in_x_rest["muon_minus"].boostCM_of_p4(z1)
    z2_in_z1 = z2.boostCM_of_p4(z1)
    electron_minus_z2 = in_x_rest["electron_minus"].boostCM_of_p4(z2)
    z1_in_z2 = z1.boostCM_of_p4(z2)

    cos_theta1 = -float(
        np.dot(_unit(_spatial(z2_in_z1)), _unit(_spatial(mu_minus_z1)))
    )
    cos_theta2 = -float(
        np.dot(_unit(_spatial(z1_in_z2)), _unit(_spatial(electron_minus_z2)))
    )

    return {
        "cos_theta_star": _clip_cosine(float(np.dot(beam, q1_hat))),
        "abs_cos_theta_star": abs(_clip_cosine(float(np.dot(beam, q1_hat)))),
        "theta1_standard": float(np.arccos(_clip_cosine(cos_theta1))),
        "theta2_standard": float(np.arccos(_clip_cosine(cos_theta2))),
        "Phi": phi,
        "Phi1": phi1,
        "Psi": psi,
        "standard_angles_degenerate": not (
            decay_planes_defined and production_plane_defined
        ),
    }


def angular_observables(
    leptons: Mapping[str, object],
    *,
    beam_direction=(0.0, 0.0, 1.0),
) -> dict[str, float | bool]:
    r"""Return masses and both angular-coordinate conventions.

    For the spherical-harmonic expansion, ``Omega1=(theta1,phi1)`` uses the
    positive muon in the Z1 rest frame and ``Omega2=(theta2,phi2)`` uses the
    positron in the Z2 rest frame.  Z1 is the dimuon system and Z2 the
    dielectron system.  The signed five-angle variables from arXiv:1208.4018
    are returned alongside these local helicity-frame coordinates.
    """

    selected = {key: leptons[key] for key in LEPTON_KEYS}
    x_system = _sum_p4(selected)
    in_x_rest = {
        key: momentum.boostCM_of_p4(x_system)
        for key, momentum in selected.items()
    }
    z1 = in_x_rest["muon_minus"] + in_x_rest["muon_plus"]
    z2 = in_x_rest["electron_minus"] + in_x_rest["electron_plus"]

    beam = _unit(np.asarray(beam_direction, dtype=np.float64))
    frame1 = _helicity_frame(_spatial(z1), beam)
    frame2 = _helicity_frame(_spatial(z2), beam)

    theta1, phi1, cos_theta1 = _positive_lepton_angles(
        in_x_rest["muon_plus"], z1, frame1
    )
    theta2, phi2, cos_theta2 = _positive_lepton_angles(
        in_x_rest["electron_plus"], z2, frame2
    )

    standard = standard_five_angles(selected, beam_direction=beam)
    output: dict[str, float | bool] = {
        "m_Z1": float((selected["muon_minus"] + selected["muon_plus"]).mass),
        "m_Z2": float(
            (selected["electron_minus"] + selected["electron_plus"]).mass
        ),
        "m_ZZ": float(x_system.mass),
        "y_ZZ": float(x_system.rapidity),
        "pt_ZZ": float(x_system.pt),
        "theta1": theta1,
        "phi1": phi1,
        "cos_theta1": cos_theta1,
        "theta2": theta2,
        "phi2": phi2,
        "cos_theta2": cos_theta2,
        "frame_degenerate": bool(frame1[3] or frame2[3]),
    }
    output.update(standard)
    return output


def projection_diagnostics_dict(diagnostics: BornProjectionDiagnostics):
    """Convert projection diagnostics to a flat dictionary."""

    return asdict(diagnostics)
