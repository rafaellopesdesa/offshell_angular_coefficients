"""Utilities for Born-projected angular coefficients in H -> ZZ -> 2e2mu."""

from .harmonics import (
    angular_modes,
    angular_target,
    inclusive_angular_coefficients,
    symmetric_angular_bound,
    symmetric_angular_harmonic,
)
from .kinematics import (
    angular_observables,
    born_project_four_leptons,
    standard_five_angles,
)
from .lhe import (
    extract_event_particles,
    iter_lhe_records,
    load_lhe_dataframe,
    particle_four_vector,
)
from .training import (
    as_float32_features,
    class_balanced_validation_bce,
    recover_conditional_moment,
    prepare_weighted_classification,
    validation_loss_outlier_mask,
)

__all__ = [
    "angular_modes",
    "angular_observables",
    "as_float32_features",
    "angular_target",
    "born_project_four_leptons",
    "class_balanced_validation_bce",
    "extract_event_particles",
    "inclusive_angular_coefficients",
    "iter_lhe_records",
    "load_lhe_dataframe",
    "particle_four_vector",
    "prepare_weighted_classification",
    "recover_conditional_moment",
    "standard_five_angles",
    "symmetric_angular_bound",
    "symmetric_angular_harmonic",
    "validation_loss_outlier_mask",
]
