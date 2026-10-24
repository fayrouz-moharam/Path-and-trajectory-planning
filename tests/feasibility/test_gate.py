"""T6 gate: known answers on a circle, every check broken on purpose, and full gates on real outputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from t5_reference import build_meta, build_reference, centerline_from_t2, write_reference
from t6_feasibility import drive, gate_t2, run_checks
from t6_feasibility import checks as C

R, OFF, N = 3.0, 0.3, 377          # centre-line circle, racing line OFF to the left (inside), rows


def circle(v=3.0, e_y=OFF, e_psi=0.0, kappa=None, n=N):
    """Centre line: CCW circle of radius R. Reference rows: racing circle e_y to the left, constant v."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    table = np.column_stack([a * R, np.full(n, e_y), np.full(n, e_psi),
                             np.full(n, 1 / (R - e_y)) if kappa is None else kappa, np.broadcast_to(v, n)])
    return table, np.column_stack([R * np.cos(a), R * np.sin(a)]), a + np.pi / 2, np.full(n, 1 / R)


def results(table, c_xy, c_head, c_kappa, width=0.8, t4_lap_time=None):
    d = drive(table, c_xy, c_head, c_kappa)
    w = np.full(len(table), width)
    return {c.name.split(" (")[0]: c for c in run_checks(d, 2 * np.pi * R, w, w, t4_lap_time or float(d.dt.sum()))}


def test_drive_known_answers():
    d = drive(*circle(v=3.0))
    assert d.ds.sum() == pytest.approx(2 * np.pi * (R - OFF), rel=1e-4)          # the racing circle's length
    assert d.dt.sum() == pytest.approx(d.ds.sum() / 3.0)
    assert np.allclose(d.a_lon, 0.0) and np.allclose(d.steer_rate, 0.0)
    assert np.allclose(d.a_lat, 3.0 ** 2 / (R - OFF))
    assert np.allclose(d.steer, np.arctan(C.WHEELBASE / (R - OFF)))


def test_a_drivable_circle_passes_everything():
    r = results(*circle(v=3.0))
    assert all(c.passed for c in r.values()), [c.name for c in r.values() if not c.passed]


def _speed_jump(v=3.0):
    vv = np.full(N, v)
    vv[100] = v + 0.5                                       # +0.5 m/s within one 5 cm row: far over +2 m/s^2
    return vv


def _zigzag_kappa():
    k = np.full(N, 1 / (R - OFF))
    k[::2] += 0.3                                           # curvature flips every row: the steering can't follow
    return k


BROKEN = [
    ("speed max", dict(v=6.5)),
    ("speed min", dict(v=0.4)),
    ("steering angle", dict(kappa=np.full(N, 2.0), v=1.0)),            # 2 1/m > tan(0.4)/0.3 = 1.41
    ("steering rate", dict(kappa=_zigzag_kappa())),
    ("sideways acceleration", dict(v=5.6)),                             # 5.6^2 / 2.7 = 11.6 > 9.81
    ("acceleration", dict(v=_speed_jump())),
    ("grip ellipse", dict(v=_speed_jump(5.0))),                         # cornering near the limit + a speed jump
    ("room to the walls", dict(e_y=0.7)),                               # 0.8 m walls - 0.25 margin = 0.55 m
    ("Frenet valid", dict(e_y=2.8, kappa=np.full(N, 1 / 0.2))),        # 1 - e_y / R = 0.07 < 0.1
    ("columns agree", dict(e_psi=0.2)),                                 # angled to the centre line, but e_y never changes
    ("row spacing", dict(n=300)),                                       # 2 pi 3 / 300 = 6.3 cm > 5 cm
]


@pytest.mark.parametrize("name,change", BROKEN, ids=[b[0] for b in BROKEN])
def test_each_check_catches_its_problem(name, change):
    r = results(*circle(**change), width=10.0 if name == "Frenet valid" else 0.8)
    assert not r[name].passed, f"{name} should fail: worst {r[name].worst}, limit {r[name].limit}"


def test_braking_check_catches_hard_braking():
    v = np.full(N, 3.0)
    v[101:] = 2.5                                           # -0.5 m/s within one row
    assert not results(*circle(v=v))["braking"].passed


def test_finish_line_jump_is_caught():
    v = np.linspace(3.0, 3.3, N)                            # smooth inside the lap, then 3.3 -> 3.0 at the finish
    assert not results(*circle(v=v))["finish line smooth"].passed


def test_lap_time_mismatch_is_caught():
    assert not results(*circle(v=3.0), t4_lap_time=99.0)["lap time vs T4"].passed


# ---------------------------------------------------------------- full gates on real outputs

RUN = Path("out/run_t1_t2_t3_t4")


@pytest.fixture
def synthetic(tmp_path):
    """The synthetic track's T5 reference, written into a temporary folder (out/ is not touched)."""
    if not (RUN / "t4" / "velocity_profile.csv").exists():
        pytest.skip(f"{RUN} has no T4 output")
    cl = centerline_from_t2(RUN / "t2" / "centerline_t2.csv")
    rc, vc = RUN / "t3" / "racing_line.csv", RUN / "t4" / "velocity_profile.csv"
    table, report, problems, metas = build_reference(cl, rc, vc)
    csv, meta = write_reference(tmp_path, table, build_meta(cl, report, problems, metas, rc, vc))
    return csv, meta


def test_synthetic_track_passes_the_gate(synthetic):
    checks, info, d = gate_t2(synthetic[0], RUN / "t2" / "centerline_t2.csv", RUN / "t4" / "velocity_profile_meta.json")
    assert all(c.passed for c in checks), [(c.name, c.worst) for c in checks if not c.passed]
    assert d is not None and len(checks) == 17


def test_gate_refuses_a_tampered_file(synthetic):
    csv, _ = synthetic
    csv.write_text(csv.read_text().replace("\n0.050000,", "\n0.050000,9", 1))
    checks, info, d = gate_t2(csv, RUN / "t2" / "centerline_t2.csv", RUN / "t4" / "velocity_profile_meta.json")
    assert [c.passed for c in checks] == [False] and d is None and "sha256" in info["loader_error"]


def test_gate_notices_different_t4_limits(synthetic, tmp_path):
    t4 = json.loads((RUN / "t4" / "velocity_profile_meta.json").read_text())
    t4["limits"]["mu"] = 0.8
    other = tmp_path / "t4_meta.json"
    other.write_text(json.dumps(t4))
    checks, _, _ = gate_t2(synthetic[0], RUN / "t2" / "centerline_t2.csv", other)
    assert not next(c for c in checks if c.name == "T4 used the same limits").passed


def test_perception_closed1_passes_the_gate(tmp_path):
    """The real case: perception's closed1 track -> T4 (recomputed with today's code) -> T5 -> the gate."""
    pytest.importorskip("apex_track")
    import run_perception_t2_t3_t4 as run_perception
    from t4_velocity_profile import build_meta as t4_meta, compute_profile, read_t3, write_profile
    from t5_reference import centerline_from_apex
    from t6_feasibility import gate_apex

    rc = Path("out/run_perception_closed1/t3/racing_line.csv")
    if not rc.exists():
        pytest.skip("no closed1 racing line in out/")
    s, kappa, lap, meta3 = read_t3(rc)
    p = compute_profile(s, kappa, lap)
    vc, vm = write_profile(tmp_path / "t4", p, t4_meta(p, meta3, rc))
    yaml = run_perception.find_track_yaml("closed1")
    cl = centerline_from_apex(yaml)
    table, report, problems, metas = build_reference(cl, rc, vc)
    csv, _ = write_reference(tmp_path / "t5", table, build_meta(cl, report, problems, metas, rc, vc))
    checks, info, d = gate_apex(csv, yaml, vm)
    assert all(c.passed for c in checks), [(c.name, c.worst) for c in checks if not c.passed]
