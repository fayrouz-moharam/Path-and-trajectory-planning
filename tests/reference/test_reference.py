"""T5 checks: circles with a known answer, the tricky spots one by one, and the real T3/T4 outputs in out/."""
from pathlib import Path

import numpy as np
import pytest

from t5_reference import (build_reference, centerline_from_apex, centerline_from_t2, check_reference,
                          cross_racing_line, sample_at_crossings, wrap)

R, OFF = 3.0, 0.3          # centre-line circle radius; the racing circle sits OFF to the left (inside)


def circles(start_angle: float = 0.0, width: float = 0.8, n_c: int = 377, n_r: int = 170) -> dict:
    """Centre line: CCW circle of radius R starting at angle 0. Racing line: circle of radius R - OFF on its
    own grid, starting at start_angle. Speed on the racing line: 2 + sin(angle), so interpolation shows."""
    a = np.linspace(0.0, 2 * np.pi, n_c, endpoint=False)
    b = start_angle + np.arange(n_r) * 2 * np.pi / n_r
    r_xy = np.column_stack([(R - OFF) * np.cos(b), (R - OFF) * np.sin(b)])
    seg = np.hypot(*(np.roll(r_xy, -1, axis=0) - r_xy).T)
    return dict(a=a, c_xy=np.column_stack([R * np.cos(a), R * np.sin(a)]), c_head=wrap(a + np.pi / 2),
                r_xy=r_xy, r_s=np.concatenate([[0.0], np.cumsum(seg)[:-1]]), r_lap=float(seg.sum()),
                r_head=wrap(b + np.pi / 2), r_kappa=np.full(n_r, 1 / (R - OFF)), v=2.0 + np.sin(b),
                w=np.full(n_c, width))


def mapped(c: dict):
    e_y, seg, frac = cross_racing_line(c["c_xy"], c["c_head"], c["w"], c["w"], c["r_xy"])
    s_race, e_psi, kappa, v = sample_at_crossings(seg, frac, c["r_s"], c["r_head"], c["r_kappa"],
                                                  c["r_lap"], c["v"], c["c_head"])
    return e_y, s_race, e_psi, kappa, v


def checks(c: dict, e_y, s_race):
    return check_reference(c["c_xy"], c["c_head"], c["w"], c["w"], e_y, s_race, c["r_xy"], c["r_lap"], margin=0.25)


def test_wrap():
    got = wrap(np.array([0.3, 6.2, -np.pi, np.pi, 1.5 * np.pi]))
    assert np.allclose(got, [0.3, 6.2 - 2 * np.pi, np.pi, np.pi, -0.5 * np.pi])


def test_circle_known_answer():
    c = circles()
    e_y, s_race, e_psi, kappa, v = mapped(c)
    sag = (R - OFF) * (1 - np.cos(np.pi / 170))         # a straight segment bulges this much inside the arc
    assert np.all((e_y >= OFF - 1e-9) & (e_y <= OFF + sag + 1e-9))
    assert np.allclose(e_psi, 0.0, atol=1e-6)
    assert np.allclose(kappa, 1 / (R - OFF))
    assert np.allclose(v, 2.0 + np.sin(c["a"]), atol=1e-3)   # the racing line's speed at the same angle


def test_go_back_lands_on_the_line():
    c = circles()
    e_y, s_race, *_ = mapped(c)
    report, problems = checks(c, e_y, s_race)
    assert problems == [] and report["land_max_miss_m"] < 1e-6
    _, problems = checks(c, -e_y, s_race)                 # wrong sign -> lands on the other side
    assert any("misses the racing line" in p for p in problems)


def test_finish_line_wrap():
    # the racing line starts 0.4 rad after the centre line, so rows before that angle cross its CLOSING
    # segment: they must get s_race just below r_lap, never a value from the middle of the lap
    c = circles(start_angle=0.4)
    e_y, s_race, *_ = mapped(c)
    expect = ((c["a"] - 0.4) % (2 * np.pi)) / (2 * np.pi) * c["r_lap"]
    assert np.allclose(s_race, expect, atol=2e-3)
    report, problems = checks(c, e_y, s_race)
    assert report["s_race_wraps"] == 1
    assert problems == [p for p in problems if "not at the racing line's start" in p]   # only the start check


def test_s_race_going_backwards_is_reported():
    c = circles()
    e_y, s_race, *_ = mapped(c)
    s_bad = s_race.copy()
    s_bad[[100, 101]] = s_bad[[101, 100]]
    _, problems = checks(c, e_y, s_bad)
    assert any("goes backwards" in p for p in problems)


def test_nearest_crossing_is_kept():
    # walls 10 m away: the across-line also reaches the FAR side of the racing circle (t = 2R - OFF);
    # the near crossing (t = OFF) must win
    e_y, *_ = mapped(circles(width=10.0))
    assert np.all(e_y < 1.0)


def test_no_crossing_between_the_walls_raises():
    with pytest.raises(ValueError, match="does not cross"):
        mapped(circles(width=0.2))                        # racing line 0.3 m away, walls only 0.2 m


RUNS = [Path("out/run_t1_t2_t3_t4"), Path("out/run_t1_t2_t3_spielberg"), Path("out/run_t1_t2_t3_budapest")]


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_real_outputs_map_cleanly(run):
    if not (run / "t4" / "velocity_profile.csv").exists():
        pytest.skip(f"{run} has no T4 output")
    cl = centerline_from_t2(run / "t2" / "centerline_t2.csv")
    table, report, problems, _ = build_reference(cl, run / "t3" / "racing_line.csv", run / "t4" / "velocity_profile.csv")
    assert problems == [], problems
    assert np.array_equal(table[:, 0], cl.s)              # rows are the centre line's own s, untouched


def test_perception_closed1():
    pytest.importorskip("apex_track")
    import run_perception_t2_t3_t4 as run_perception
    run = Path("out/run_perception_closed1")
    cl = centerline_from_apex(run_perception.find_track_yaml("closed1"))
    table, report, problems, _ = build_reference(cl, run / "t3" / "racing_line.csv", run / "t4" / "velocity_profile.csv")
    assert problems == [], problems
    assert np.array_equal(table[:, 0], cl.s)              # perception's own s values
