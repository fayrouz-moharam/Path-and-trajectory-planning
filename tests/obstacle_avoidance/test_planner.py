"""T7 Half B: grid, DP, the two detours and one full cycle, on a made-up straight track (walls 0.8 m away,
so the car's centre may use +-0.55 m) with Half A's KeepOut boxes."""
import time

import numpy as np
import pytest

from t6_feasibility.runtime import check_cycle, plan_from_reference
from t7_avoidance.types import KeepOut, KeepOuts
from t7_obstacle_avoidance import grid as G
from t7_obstacle_avoidance.detour import METHODS
from t7_obstacle_avoidance.dp import dp_path
from t7_obstacle_avoidance.plan import Memory, plan_cycle

LAP = 50.0


class Ref:
    """A straight track whose racing line runs at e_ref (or e_ref(s)), 6 m/s."""
    def __init__(self, e_ref=0.0):
        self.e_ref = e_ref

    def at(self, s):
        s = np.mod(np.asarray(s, dtype=float), LAP)
        e = self.e_ref(s) if callable(self.e_ref) else np.full(s.shape, self.e_ref)
        return {"e_y_ref_m": e, "e_psi_ref_rad": np.zeros(s.shape), "kappa_ref_1pm": np.zeros(s.shape),
                "v_ref_mps": np.full(s.shape, 6.0), "s_track_m": s}


def walls(s):
    s = np.asarray(s, dtype=float)
    return np.full(s.shape, 0.8), np.full(s.shape, 0.8)


def kappa_at(s):
    return np.zeros(np.shape(s))


def zone(s_center, e_from, e_to, cid=1, half_s=0.4):
    gap_left, gap_right = 0.55 - e_to, e_from + 0.55
    cl, cr = gap_left < 0.05, gap_right < 0.05
    return KeepOut(cid, s_center - half_s, s_center + half_s, e_from, e_to, s_center, 0.5 * (e_from + e_to),
                   0.5 * (e_to - e_from), 1.0, cl or cr, True, gap_left, gap_right, cl, cr)


def ko(*zones, car_s=10.0, epoch=0, v_cap=6.0):
    return KeepOuts(car_s, "full", epoch, 8.0, v_cap, tuple(zones))


def run(*zones, car_s=10.0, e_y=0.0, vx=5.0, method="fastest", memory=None, ref=None, epoch=0):
    return plan_cycle(ko(*zones, car_s=car_s, epoch=epoch), car_s, e_y, vx, ref or Ref(), walls, kappa_at,
                      memory=memory, method=method)


def zone_never_in_corridor(plan, z):
    """At every stage beside the zone, the corridor lies wholly on one side of it."""
    for s, lo, hi in zip(plan.s, plan.e_y_min, plan.e_y_max):
        if z.s_from <= s <= z.s_to:
            assert hi < z.e_y_from or lo > z.e_y_to, f"zone {z.id} inside the corridor at s {s:.2f}"


# ---------------------------------------------------------------- B2 grid
def test_circle_blocks_a_subset_of_the_box():
    z = zone(13.0, -0.3, 0.3)
    e = np.arange(-0.55, 0.551, 0.05)
    for s in np.arange(12.4, 13.6, 0.25):
        box = G.blocked(z, s - 0.125, s + 0.125, e, shape="box")
        circle = G.blocked(z, s - 0.125, s + 0.125, e, shape="circle")
        assert not np.any(circle & ~box)
    assert G.blocked(z, 12.9, 13.1, e, shape="circle")[np.argmin(np.abs(e))]


def test_closed_side_is_blocked_to_the_wall_even_if_a_cell_is_free():
    z = zone(13.0, -0.2, 0.52)              # 3 cm on the left: Half A says closed, the grid would keep one cell
    assert z.closed_left
    e = np.arange(-0.55, 0.551, 0.05)
    b = G.blocked_with_sides(z, 12.9, 13.1, e)
    assert b[e > 0.5].all() and not b[e < -0.25].any()


def test_extra_widening_closes_more_cells():
    z = zone(13.0, -0.2, 0.2)
    e = np.arange(-0.55, 0.551, 0.05)
    assert G.blocked(z, 12.9, 13.1, e, widen=0.1).sum() > G.blocked(z, 12.9, 13.1, e).sum()


# ---------------------------------------------------------------- B3 DP
def test_dp_stays_on_the_racing_line_when_clear():
    w = G.window(10.0, Ref(0.1), walls, kappa_at)
    g = G.build_grid(w, [])
    r = dp_path(g.e, g.free, w.s, w.e_ref, 0.1)
    assert r.ok and np.allclose(r.e, 0.1, atol=0.026)


def test_dp_finds_no_path_when_two_zones_shut_the_track():
    a, b = zone(13.0, -0.2, 0.55, cid=1), zone(13.6, -0.55, 0.2, cid=2)   # left shut, then right shut, 0.6 m apart
    w = G.window(10.0, Ref(), walls, kappa_at)
    g = G.build_grid(w, [a, b])
    r = dp_path(g.e, g.free, w.s, w.e_ref, 0.0)
    assert not r.ok and 12.0 < w.s[r.blocked_at] < 14.0


# ---------------------------------------------------------------- full cycle
def test_clear_track_equals_the_reference():
    plan, info = run()
    ref = plan_from_reference(Ref(), 10.0, 5.0, walls)
    for a in ("s", "e_y_ref", "e_psi_ref", "kappa_ref", "v_ref", "e_y_min", "e_y_max"):
        assert np.allclose(getattr(plan, a), getattr(ref, a), atol=1e-9), a


@pytest.mark.parametrize("closed, side", [("left", "right"), ("right", "left")])
def test_closed_side_passes_the_other_way(closed, side):
    z = zone(12.5, -0.2, 0.55) if closed == "left" else zone(12.5, -0.55, 0.2)
    plan, info = run(z)
    assert info["sides"][z.id] == side
    zone_never_in_corridor(plan, z)


def test_both_open_picks_the_side_nearer_the_racing_line():
    z = zone(12.5, -0.05, 0.4)              # mostly left of the racing line: going right is the shorter way
    plan, info = run(z)
    assert info["sides"][z.id] == "right"


@pytest.mark.parametrize("method", METHODS)
def test_each_method_goes_round_a_centre_cone(method):
    z = zone(13.0, -0.35, 0.35)
    plan, info = run(z, method=method)
    assert info["path_found"] and info["method"].startswith(method)
    assert check_cycle(plan, 10.0, 0.0, None, None).level != "FAIL"
    zone_never_in_corridor(plan, z)


def test_fastest_tries_both_and_takes_the_quickest():
    z = zone(13.0, -0.35, 0.35)
    plan, info = run(z)
    assert set(info["times"]) | set(info["rejected"]) == set(METHODS)
    assert info["method"] == min(info["times"], key=info["times"].get)


def test_slalom_two_cones_opposite_sides():
    a, b = zone(12.5, -0.15, 0.55, cid=1), zone(15.5, -0.55, 0.15, cid=2)
    plan, info = run(a, b)
    assert info["path_found"] and info["sides"] == {1: "right", 2: "left"}
    zone_never_in_corridor(plan, a)
    zone_never_in_corridor(plan, b)


def test_track_shut_by_two_zones_stops_before_them():
    plan, info = run(zone(13.0, -0.2, 0.55, cid=1), zone(13.6, -0.55, 0.2, cid=2))
    assert not info["path_found"] and plan.v_cap < 6.0


def test_zone_past_the_finish_line():
    ref = Ref(lambda s: 0.1 * np.sin(2 * np.pi * s / LAP))
    z = zone(51.0, -0.3, 0.3)               # continuous s: 1 m into the next lap
    plan, info = run(z, car_s=48.0, ref=ref)
    assert info["path_found"] and z.id in info["sides"]
    zone_never_in_corridor(plan, z)


def test_side_is_kept_once_close_even_if_the_other_side_gets_nearer():
    mem = Memory()
    _, info = run(zone(14.0, -0.38, 0.42), memory=mem)             # right is nearer the racing line
    assert info["sides"][1] == "right"
    _, info = run(zone(14.0, -0.42, 0.38), car_s=12.8, memory=mem)  # now left is nearer, but 1.2 m to go
    assert info["sides"][1] == "right"


def test_new_epoch_forgets_the_sides():
    mem = Memory()
    run(zone(14.0, -0.38, 0.42), memory=mem)
    run(zone(14.0, -0.42, 0.38), car_s=12.8, memory=mem, epoch=1)
    assert mem.epoch == 1 and mem.side.get(1) == "left"


def test_cycle_time():
    zs = [zone(12.0 + 1.5 * k, -0.35 + 0.3 * (k % 2), 0.0 + 0.3 * (k % 2), cid=k) for k in range(4)]
    run(*zs)                                                         # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        run(*zs)
    per_cycle = (time.perf_counter() - t0) / 10
    print(f"fastest of both methods, 4 zones: {per_cycle * 1000:.1f} ms per cycle")
    assert per_cycle < 0.1
