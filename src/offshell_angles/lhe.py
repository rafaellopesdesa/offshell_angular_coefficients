"""Read POWHEG LHE events and construct Born-projected observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pylhe
import vector

from .kinematics import (
    LEPTON_KEYS,
    angular_observables,
    born_project_four_leptons,
    projection_diagnostics_dict,
)


PDG_TO_LEPTON_KEY = {
    11: "electron_minus",
    -11: "electron_plus",
    13: "muon_minus",
    -13: "muon_plus",
}


@dataclass(frozen=True)
class ExtractedLHEEvent:
    """Physics objects retained from one LHE event."""

    leptons: dict[str, object]
    higgs_bosons: tuple[object, ...]
    z_bosons: tuple[object, ...]
    final_partons: tuple[object, ...]
    nominal_weight: float
    alternative_weights: dict[str, float]


def particle_four_vector(particle):
    """Convert a ``pylhe.LHEParticle`` to a ``vector`` four-vector."""

    return vector.obj(
        px=float(particle.px),
        py=float(particle.py),
        pz=float(particle.pz),
        E=float(particle.e),
    )


def extract_event_particles(event) -> ExtractedLHEEvent:
    """Select the final 2e2mu state plus optional Higgs, Z, and jet records."""

    lepton_candidates = {key: [] for key in LEPTON_KEYS}
    higgs_bosons = []
    z_bosons = []
    final_partons = []

    for particle in event.particles:
        momentum = particle_four_vector(particle)
        if particle.id == 25:
            higgs_bosons.append(momentum)
        if particle.id == 23:
            z_bosons.append(momentum)
        if particle.status == 1 and particle.id in PDG_TO_LEPTON_KEY:
            lepton_candidates[PDG_TO_LEPTON_KEY[particle.id]].append(momentum)
        if particle.status == 1 and (
            1 <= abs(particle.id) <= 6 or particle.id == 21
        ):
            final_partons.append(momentum)

    multiplicities = {
        key: len(candidates) for key, candidates in lepton_candidates.items()
    }
    invalid = {key: count for key, count in multiplicities.items() if count != 1}
    if invalid:
        raise ValueError(
            "Expected exactly one final-state lepton of each flavor and charge; "
            f"found {invalid}"
        )

    return ExtractedLHEEvent(
        leptons={key: candidates[0] for key, candidates in lepton_candidates.items()},
        higgs_bosons=tuple(higgs_bosons),
        z_bosons=tuple(z_bosons),
        final_partons=tuple(final_partons),
        nominal_weight=float(event.eventinfo.weight),
        alternative_weights={
            str(key): float(value) for key, value in event.weights.items()
        },
    )


def _object_summary(prefix: str, objects: tuple[object, ...]) -> dict[str, float | int]:
    output: dict[str, float | int] = {f"n_{prefix}_lhe": len(objects)}
    for index, momentum in enumerate(objects[:2]):
        output[f"{prefix}{index + 1}_lhe_m"] = float(momentum.mass)
        output[f"{prefix}{index + 1}_lhe_pt"] = float(momentum.pt)
        output[f"{prefix}{index + 1}_lhe_y"] = float(momentum.rapidity)
    return output


def _momentum_columns(prefix: str, momenta: dict[str, object]):
    output = {}
    for key, momentum in momenta.items():
        for component in ("px", "py", "pz", "E"):
            output[f"{prefix}_{key}_{component}"] = float(getattr(momentum, component))
    return output


def iter_lhe_records(
    path: str | Path,
    *,
    max_events: int | None = None,
    strict: bool = True,
    include_momenta: bool = False,
) -> Iterator[dict[str, object]]:
    """Yield one flat analysis record per valid LHE event.

    With ``strict=False``, events that are not an unambiguous final-state
    ``e+ e- mu+ mu-`` topology are skipped.  Other errors are never hidden.
    """

    lhe_file = pylhe.LHEFile.fromfile(Path(path), with_attributes=True, generator=True)
    accepted = 0
    for event_index, event in enumerate(lhe_file.events):
        if max_events is not None and accepted >= max_events:
            break
        try:
            extracted = extract_event_particles(event)
        except ValueError as exc:
            if strict:
                raise ValueError(f"LHE event {event_index}: {exc}") from exc
            continue

        born_leptons, diagnostics = born_project_four_leptons(extracted.leptons)
        record: dict[str, object] = {
            "event_index": event_index,
            "weight": extracted.nominal_weight,
            "n_alternative_weights": len(extracted.alternative_weights),
            "n_final_partons": len(extracted.final_partons),
        }
        record.update(_object_summary("higgs", extracted.higgs_bosons))
        record.update(_object_summary("z", extracted.z_bosons))
        record.update(projection_diagnostics_dict(diagnostics))
        record.update(angular_observables(born_leptons))
        if include_momenta:
            record.update(_momentum_columns("raw", extracted.leptons))
            record.update(_momentum_columns("born", born_leptons))

        accepted += 1
        yield record


def load_lhe_dataframe(
    path: str | Path,
    *,
    max_events: int | None = None,
    strict: bool = True,
    include_momenta: bool = False,
) -> pd.DataFrame:
    """Load selected events and their Born-projected observables into pandas."""

    records = list(
        iter_lhe_records(
            path,
            max_events=max_events,
            strict=strict,
            include_momenta=include_momenta,
        )
    )
    if not records:
        raise ValueError("No valid e+ e- mu+ mu- events were found in the LHE file")
    frame = pd.DataFrame.from_records(records)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return frame
