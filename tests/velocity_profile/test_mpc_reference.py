"""The MPC handoff: racing line + profile indexed by centreline s, by geometric projection."""
import json

import numpy as np
import pytest

from t4_velocity_profile import MPCReference, VehicleLimits, compute_profile
from t4_velocity_profile.mpc_reference import build_reference, build_reference_meta, write_reference

R_C, OFFSET, W = 5.0, 0.4, 1.0          # centreline circle radius, racing line inside by OFFSET


def write_circle(path, R, n, start=0.0, header="s,x,y,w_left,w_right"):
    th = start + np.arange(n) * 2 * np.pi / n
    s = R * (th - start)
    # counter-clockwise: left = towards the centre
    cols = [s, R * np.cos(th), R * np.sin(th), np.full(n, W), np.full(n, W)]
    if "heading" in header:          # T3 layout: s,x,y,heading,kappa,w_left,w_right
        cols = [s, cols[1], cols[2], th + np.pi / 2, np.full(n, 1 / R), cols[3], cols[4]]
    np.savetxt(path, np.column_stack(cols), delimiter=",", fmt="%.9f", header=header, comments="")
    return 2 * np.pi * R


@pytest.fixture
def circle_run(tmp_path):
    track = tmp_path / "track.csv"
    write_circle(track, R_C, 400)
    race = tmp_path / "racing_line.csv"
    R_r = R_C - OFFSET
    # racing line starts a quarter lap later: its s = 0 is NOT the centreline's s = 0
    lap_r = write_circle(race, R_r, 300, start=np.pi / 2, header="s,x,y,heading,kappa,w_left,w_right")
    s = np.arange(300) * lap_r / 300
    p = compute_profile(s, np.full_like(s, 1 / R_r), lap_r, VehicleLimits(v_max=3.0))
    return track, race, p, R_r


def test_projection_gives_offset_and_centreline_s(circle_run):
    track, race, p, R_r = circle_run
    data, stats = build_reference(race, p, track, {})
    s_track, e_y, e_psi, kappa, v, a, s_race = data.T
    assert np.allclose(s_track, np.loadtxt(track, delimiter=",", skiprows=1)[:, 0])
    assert np.allclose(e_y, OFFSET, atol=1e-4)                  # inside the circle = left
    assert np.allclose(e_psi, 0.0, atol=1e-4)
    assert np.allclose(kappa, 1 / R_r)
    assert np.allclose(v, 3.0) and np.allclose(a, 0.0, atol=1e-9)
    # same angle on both circles: s_race = (s_track / R_C - pi/2) * R_r, wrapped
    lap_r = 2 * np.pi * R_r
    expect = np.mod((s_track / R_C - np.pi / 2) * R_r, lap_r)
    d = np.abs(s_race - expect)
    assert np.all(np.minimum(d, lap_r - d) < 1e-3)
    assert stats["round_trip_s_max_m"] < 1e-4
    assert stats["min_room_left_m"] == pytest.approx(W - OFFSET, abs=1e-4)


def test_written_reference_round_trips_through_the_loader(circle_run, tmp_path):
    track, race, p, _ = circle_run
    meta_in = {"map_version": "abc123", "boundary_hash": "def456"}
    data, stats = build_reference(race, p, track, meta_in)
    meta = build_reference_meta(stats, p, meta_in, track_csv=track, racing_csv=race)
    csv, meta_path = write_reference(tmp_path / "t4", data, meta)

    ref = MPCReference.from_csv(csv, track_hash="abc123", boundary_hash="def456")
    L = ref.lap_length
    assert L == pytest.approx(2 * np.pi * R_C, rel=1e-6)
    assert ref(0.0) == pytest.approx(ref(L)) == pytest.approx(ref(3 * L + 0.0))   # s_abs over laps
    r = ref.sample(np.linspace(0, 2 * L, 50))
    assert np.allclose(r["e_y_ref_m"], OFFSET, atol=1e-4)
    assert np.all((r["s_race_m"] >= 0) & (r["s_race_m"] < 2 * np.pi * (R_C - OFFSET)))
    assert json.loads(meta_path.read_text())["a_ref_semantics"].startswith("dv/dt")

    with pytest.raises(ValueError):
        MPCReference.from_csv(csv, track_hash="other")
    csv.write_text(csv.read_text().replace("3.000000", "3.100000", 1))
    with pytest.raises(ValueError):                          # edited file no longer matches its sha256
        MPCReference.from_csv(csv)


def test_rejects_line_driven_the_wrong_way(circle_run, tmp_path):
    track, race, p, _ = circle_run
    d = np.loadtxt(race, delimiter=",", skiprows=1)
    d[:, 1:3] = d[::-1, 1:3]                                     # same circle, clockwise
    wrong = tmp_path / "wrong_way.csv"
    np.savetxt(wrong, d, delimiter=",", fmt="%.9f", header="s,x,y,heading,kappa,w_left,w_right", comments="")
    with pytest.raises(ValueError, match="backwards"):
        build_reference(wrong, p, track, {})
