import numpy as np
import vector

from offshell_angles import angular_observables, born_project_four_leptons


def _two_body_system(parent_mass=350.0, mass1=90.0, mass2=60.0):
    momentum = np.sqrt(
        (parent_mass**2 - (mass1 + mass2) ** 2)
        * (parent_mass**2 - (mass1 - mass2) ** 2)
    ) / (2.0 * parent_mass)
    energy1 = np.sqrt(mass1**2 + momentum**2)
    energy2 = np.sqrt(mass2**2 + momentum**2)
    direction = np.array([0.8, 0.3, np.sqrt(1.0 - 0.8**2 - 0.3**2)])
    z1 = vector.obj(
        px=momentum * direction[0],
        py=momentum * direction[1],
        pz=momentum * direction[2],
        E=energy1,
    )
    z2 = vector.obj(px=-z1.px, py=-z1.py, pz=-z1.pz, E=energy2)
    return z1, z2


def _decay_to_massless_leptons(parent, cos_theta, phi):
    momentum = parent.mass / 2.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    positive = vector.obj(
        px=momentum * sin_theta * np.cos(phi),
        py=momentum * sin_theta * np.sin(phi),
        pz=momentum * cos_theta,
        E=momentum,
    )
    negative = vector.obj(
        px=-positive.px, py=-positive.py, pz=-positive.pz, E=momentum
    )
    beta = parent.to_beta3()
    return negative.boost_beta3(beta), positive.boost_beta3(beta)


def _recoiling_event():
    z1, z2 = _two_body_system()
    muon_minus, muon_plus = _decay_to_massless_leptons(z1, 0.25, 0.8)
    electron_minus, electron_plus = _decay_to_massless_leptons(z2, -0.4, -1.2)
    rest_event = {
        "muon_minus": muon_minus,
        "muon_plus": muon_plus,
        "electron_minus": electron_minus,
        "electron_plus": electron_plus,
    }
    recoil = vector.obj(x=0.12, y=-0.08, z=0.35)
    return {key: value.boost_beta3(recoil) for key, value in rest_event.items()}


def test_born_projection_preserves_invariants_and_removes_recoil():
    raw = _recoiling_event()
    born, diagnostics = born_project_four_leptons(raw)

    np.testing.assert_allclose(diagnostics.born_m4l, diagnostics.raw_m4l)
    np.testing.assert_allclose(diagnostics.born_y4l, diagnostics.raw_y4l)
    assert diagnostics.raw_pt4l > 1.0
    assert diagnostics.born_pt4l < 1.0e-10

    raw_z1 = raw["muon_minus"] + raw["muon_plus"]
    born_z1 = born["muon_minus"] + born["muon_plus"]
    raw_z2 = raw["electron_minus"] + raw["electron_plus"]
    born_z2 = born["electron_minus"] + born["electron_plus"]
    np.testing.assert_allclose([born_z1.mass, born_z2.mass], [raw_z1.mass, raw_z2.mass])


def test_angle_ranges_and_charge_conventions():
    born, _ = born_project_four_leptons(_recoiling_event())
    observables = angular_observables(born)

    assert 0.0 <= observables["theta1"] <= np.pi
    assert 0.0 <= observables["theta2"] <= np.pi
    assert -np.pi <= observables["phi1"] <= np.pi
    assert -np.pi <= observables["phi2"] <= np.pi
    assert -1.0 <= observables["cos_theta_star"] <= 1.0
    assert not observables["frame_degenerate"]
    assert not observables["standard_angles_degenerate"]

    # The harmonic angles use positive leptons, whereas the standard five-angle
    # polar definitions of arXiv:1208.4018 use negative leptons.
    np.testing.assert_allclose(
        observables["theta1_standard"], np.pi - observables["theta1"]
    )
    np.testing.assert_allclose(
        observables["theta2_standard"], np.pi - observables["theta2"]
    )

