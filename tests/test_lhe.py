from pathlib import Path

import numpy as np

from offshell_angles import load_lhe_dataframe


LHE_TEXT = """<LesHouchesEvents version=\"3.0\">
<header></header>
<init>
 2212 2212 6500.0 6500.0 0 0 0 0 3 1
 1.0 0.0 1.0 1
</init>
<event>
 10 1 2.5 350.0 0.007297 0.118
 21 -1 0 0 501 502 0.0 0.0 6500.0 6500.0 0.0 0.0 9.0
 21 -1 0 0 502 501 0.0 0.0 -6500.0 6500.0 0.0 0.0 9.0
 25 2 1 2 0 0 15.0 10.0 15.0 115.869834 112.0 0.0 9.0
 23 2 3 3 0 0 10.0 0.0 30.0 72.360680 65.0 0.0 9.0
 23 2 3 3 0 0 5.0 10.0 -15.0 43.509154 39.0 0.0 9.0
 13 1 4 4 0 0 30.0 0.0 40.0 50.0 0.0 0.0 9.0
 -13 1 4 4 0 0 -20.0 0.0 -10.0 22.360680 0.0 0.0 9.0
 11 1 5 5 0 0 0.0 25.0 -10.0 26.925824 0.0 0.0 9.0
 -11 1 5 5 0 0 5.0 -15.0 -5.0 16.583124 0.0 0.0 9.0
 21 1 1 2 503 504 -15.0 -10.0 -15.0 23.452079 0.0 0.0 9.0
</event>
</LesHouchesEvents>
"""


def test_lhe_reader_uses_only_final_leptons_for_composite_objects(tmp_path: Path):
    path = tmp_path / "event.lhe"
    path.write_text(LHE_TEXT)
    events = load_lhe_dataframe(path, include_momenta=True)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["weight"] == 2.5
    assert "n_higgs_lhe" not in events
    assert "n_z_lhe" not in events
    assert "n_final_partons" not in events
    assert event["born_pt4l"] < 1.0e-10
    np.testing.assert_allclose(event["raw_m4l"], event["born_m4l"])
    np.testing.assert_allclose(event["raw_y4l"], event["born_y4l"])
    np.testing.assert_allclose(event["m_H"], event["m_ZZ"])
    # The explicit ID-25 record above declares m=112 GeV.  A different result
    # proves that the Higgs candidate is reconstructed from the four leptons.
    assert not np.isclose(event["m_H"], 112.0)
    for name in (
        "theta1",
        "phi1",
        "theta2",
        "phi2",
        "m_Z1",
        "m_Z2",
        "m_ZZ",
        "m_H",
    ):
        assert np.isfinite(event[name])
    assert "raw_muon_plus_E" in events
    assert "born_electron_minus_px" in events
