"""T5 loader: lookups, lap wrap, horizon calls, and every way a wrong file must be refused."""
import json
from pathlib import Path

import numpy as np
import pytest

from t5_reference import PlanningReference, build_meta, build_reference, centerline_from_t2, write_reference

RUN = Path("out/run_t1_t2_t3_t4")
COLS = ["e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm", "v_ref_mps"]


@pytest.fixture
def written(tmp_path):
    """A fresh planning_reference (synthetic track, T2 centre line) written into a temporary folder."""
    if not (RUN / "t4" / "velocity_profile.csv").exists():
        pytest.skip(f"{RUN} has no T4 output")
    cl = centerline_from_t2(RUN / "t2" / "centerline_t2.csv")
    rc, vc = RUN / "t3" / "racing_line.csv", RUN / "t4" / "velocity_profile.csv"
    table, report, problems, metas = build_reference(cl, rc, vc)
    csv, meta = write_reference(tmp_path, table, build_meta(cl, report, problems, metas, rc, vc))
    return csv, meta, table


def test_rows_come_back_exactly(written):
    csv, _, table = written
    ref = PlanningReference.from_csv(csv)
    got = ref.at(table[:, 0])
    for j, name in enumerate(COLS, start=1):
        assert np.allclose(got[name], table[:, j], atol=2e-6)      # 6 decimals in the file


def test_one_number_in_floats_out(written):
    ref = PlanningReference.from_csv(written[0])
    r = ref.at(3.21)
    assert all(isinstance(v, float) for v in r.values())
    assert ref(3.21) == r["v_ref_mps"]                             # speed-provider contract


def test_horizon_in_one_call(written):
    ref = PlanningReference.from_csv(written[0])
    s = 5.0 + 0.5 * np.arange(9)
    h = ref.at(s)
    assert h["v_ref_mps"].shape == (9,)
    assert np.allclose(h["v_ref_mps"], [ref(x) for x in s])


def test_wraps_round_the_lap(written):
    ref = PlanningReference.from_csv(written[0])
    L = ref.lap_length
    for s in (0.0, 7.3, L - 0.01):
        for k in (-1, 1, 3):                                       # lap -1, lap 2, lap 4
            assert np.allclose(list(ref.at(s + k * L).values())[:4], list(ref.at(s).values())[:4])


def test_finish_line_blends_last_and_first_row(written):
    csv, _, table = written
    ref = PlanningReference.from_csv(csv)
    s_last, L = table[-1, 0], ref.lap_length
    mid = ref.at(0.5 * (s_last + L))["e_y_ref_m"]                  # halfway from the last row to the finish
    assert mid == pytest.approx(0.5 * (table[-1, 1] + table[0, 1]), abs=2e-6)   # 6 decimals in the file


def test_e_psi_does_not_swing_through_zero():
    # two rows pointing almost backwards (+3.10 and -3.10 rad): halfway is +-pi, never 0
    table = np.array([[0.0, 0.0, 3.10, 0.0, 2.0], [1.0, 0.0, -3.10, 0.0, 2.0]])
    ref = PlanningReference(table, {"lap_length": 2.0})
    assert abs(abs(ref.at(0.5)["e_psi_ref_rad"]) - np.pi) < 1e-9


def test_refuses_an_edited_csv(written):
    csv, _, _ = written
    lines = csv.read_text().splitlines()
    lines[5] = lines[5].replace(lines[5].split(",")[-1], "9.999999")
    csv.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        PlanningReference.from_csv(csv)


def test_refuses_another_track(written):
    with pytest.raises(ValueError, match="track_hash"):
        PlanningReference.from_csv(written[0], track_hash="a44e46d238cf50c2")


def test_refuses_a_file_that_failed_its_checks(written):
    csv, meta_path, _ = written
    meta = json.loads(meta_path.read_text())
    meta.update(valid=False, problems=["test"])
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="self-checks"):
        PlanningReference.from_csv(csv)
    assert isinstance(PlanningReference.from_csv(csv, allow_invalid=True), PlanningReference)


def test_refuses_another_format(written):
    csv, meta_path, _ = written
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 99
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a planning_reference"):
        PlanningReference.from_csv(csv)
