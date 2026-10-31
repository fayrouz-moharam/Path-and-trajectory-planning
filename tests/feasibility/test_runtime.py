"""T6 runtime checks: OK on a good plan, the right light + reason for each problem, a real lap, timing."""
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from t6_feasibility import CyclePlan, check_cycle, plan_from_reference

H = "15f62f337c4c8787"                       # the track_hash both sides should carry


def good_plan():
    """9 stages 0.5 m apart from s = 10, 20 cm left of the centre line, in a 1.1 m wide corridor."""
    s = 10.0 + 0.5 * np.arange(9)
    return CyclePlan(s, np.full(9, 0.2), np.zeros(9), np.full(9, 0.3), np.full(9, 4.0),
                     np.full(9, -0.55), np.full(9, 0.55))


def verdict(plan=None, e_y=0.2, track_hash=H):
    return check_cycle(plan or good_plan(), 10.0, e_y, track_hash, H)


def test_good_plan_is_ok():
    v = verdict()
    assert v.ok and v.reasons == []


def changed(field, i, value):
    """good_plan() with one value of one field changed."""
    p = good_plan()
    arr = getattr(p, field).copy()
    arr[i] = value
    return replace(p, **{field: arr})


BROKEN = [
    ("NaN in the plan", dict(plan=changed("v_ref", 3, np.nan)), "NaN"),
    ("NaN in the state", dict(e_y=np.nan), "NaN"),
    ("wrong track", dict(track_hash="a44e46d238cf50c2"), "track mismatch"),
    ("preview not at the car", dict(plan=changed("s", 0, 9.0)), "start at the car"),
    ("preview going backwards", dict(plan=changed("s", 5, 11.0)), "start at the car"),
    ("corridor squeezed", dict(plan=changed("e_y_min", 4, 0.53)), "corridor only"),
    ("plan outside its corridor", dict(plan=changed("e_y_ref", 6, 0.7)), "e_y_ref"),
    ("car far outside", dict(e_y=0.70), "body reaches the wall"),
    ("too sharp to steer", dict(plan=changed("kappa_ref", 2, 1.6)), "sharper than the car"),
    ("too fast", dict(plan=changed("v_ref", 1, 6.5)), "above the cap"),
    ("too slow", dict(plan=changed("v_ref", 1, 0.3)), "below the minimum"),
    ("over this cycle's speed cap", dict(plan=replace(good_plan(), v_cap=3.5)), "above the cap"),
]


@pytest.mark.parametrize("name,args,reason", BROKEN, ids=[b[0] for b in BROKEN])
def test_each_problem_fails_with_its_reason(name, args, reason):
    v = verdict(**args)
    assert v.level == "FAIL" and any(reason in r for r in v.reasons), v.reasons


def test_car_slightly_outside_is_only_a_warning():
    v = verdict(e_y=0.60)                    # 5 cm outside its corridor: the 10 cm buffer still keeps the body clear
    assert v.level == "WARN" and "body still clear" in v.reasons[0]


def test_fail_beats_warn():
    v = verdict(e_y=0.60, track_hash="other")          # one WARN reason + one FAIL reason
    assert v.level == "FAIL" and "track mismatch" in v.reasons[0] and "still clear" in v.reasons[-1]


def test_fast_enough():
    p = good_plan()
    t0 = time.perf_counter()
    for _ in range(2000):
        check_cycle(p, 10.0, 0.2, H, H)
    assert (time.perf_counter() - t0) / 2000 < 1e-3      # well under 1 ms per cycle


# ---------------------------------------------------------------- built from a real reference

class FakeRef:
    """A stand-in for PlanningReference: a straight track, 0.2 m left of the centre line, 3 m/s."""
    lap_length = 50.0

    def at(self, s):
        s = np.asarray(s, float)
        return {"e_y_ref_m": np.full_like(s, 0.2), "e_psi_ref_rad": np.zeros_like(s),
                "kappa_ref_1pm": np.zeros_like(s), "v_ref_mps": np.full_like(s, 3.0), "s_track_m": s % 50.0}


def walls(s):
    return np.full(len(s), 0.8), np.full(len(s), 0.8)


def test_plan_from_reference_builds_the_preview():
    p = plan_from_reference(FakeRef(), 10.0, 4.0, walls)
    assert len(p.s) == 9 and p.s[0] == 10.0 and np.allclose(np.diff(p.s), 0.4)    # every 0.1 s at 4 m/s
    assert np.allclose(p.e_y_max, 0.55) and np.allclose(p.e_y_min, -0.55)          # walls 0.8 - margin 0.25
    assert check_cycle(p, 10.0, 0.2, H, H).ok


def test_standstill_preview_still_moves_forward():
    p = plan_from_reference(FakeRef(), 10.0, 0.0, walls)
    assert np.all(np.diff(p.s) > 0)


def test_whole_closed1_lap_is_ok(tmp_path):
    """The real case: closed1 racing line -> T4 (today's code) -> T5 reference -> a lap of build + check."""
    pytest.importorskip("apex_track")
    import run_perception_t2_t3_t4 as run_perception
    from t4_velocity_profile import build_meta as t4_meta, compute_profile, read_t3, write_profile
    from t5_reference import (PlanningReference, build_meta, build_reference, centerline_from_apex,
                              write_reference)
    from apex_track import load_track

    rc = Path("out/run_perception_closed1/t3/racing_line.csv")
    if not rc.exists():
        pytest.skip("no closed1 racing line in out/")
    s, kappa, lap, meta3 = read_t3(rc)
    p = compute_profile(s, kappa, lap)
    vc, _ = write_profile(tmp_path / "t4", p, t4_meta(p, meta3, rc))
    yaml = run_perception.find_track_yaml("closed1")
    cl = centerline_from_apex(yaml)
    table, report, problems, metas = build_reference(cl, rc, vc)
    csv, _ = write_reference(tmp_path / "t5", table, build_meta(cl, report, problems, metas, rc, vc))
    track = load_track(str(yaml))
    ref = PlanningReference.from_csv(csv, track_hash=track.track_hash)
    for row in table[::5]:                                    # every 25 cm of a lap, car on the racing line
        s_abs, e_y, v = row[0] + ref.lap_length, row[1], row[4]     # lap 2: s_abs past the lap length
        plan = plan_from_reference(ref, s_abs, v, track.width_at)
        verdict = check_cycle(plan, s_abs, e_y, track.track_hash, ref.meta["track_hash"])
        assert verdict.ok, (s_abs, verdict.reasons)
