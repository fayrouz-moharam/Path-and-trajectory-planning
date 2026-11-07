"""B4 + B5: one MPC cycle. KeepOuts (Half A) + the car's state -> CyclePlan (t6_feasibility.runtime) for the MPC.

    plan, info = plan_cycle(ko, s_abs, e_y, vx, ref, walls, kappa_at, memory=memory)

1. window + corridor grid (grid.py), sides locked for zones closer than LOCK_M (no flicker in the last metres)
2. cheapest path (dp.py); no path -> stay on the racing line and stop before the cut (Half A can't see a track
   blocked by two zones together)
3. corridor = the free strip around the path (Liniger): the zones are never inside it
4. bent racing line by METHOD ("graph" / "sampling"), or "fastest": both on the same grid and sides, each
   through check_cycle(), the quickest that passes wins
5. the 9 preview stages: e_y_ref, e_psi_ref, kappa_ref (T5 + the bend), v_ref (T5, the bend's grip, the cap)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from t6_feasibility.checks import V_MAX, V_MIN
from t6_feasibility.runtime import PREVIEW_DT, PREVIEW_STEPS, CyclePlan, check_cycle
from t7_avoidance.speed_policy import STOP_MARGIN_M, stop_speed

from . import detour as D
from .dp import dp_path, sides
from .grid import build_grid, window

METHOD = "fastest"       # "fastest": build graph AND sampling, check both, send the quicker that passes (~6 ms).
                         # "sampling" / "graph": only that one (sampling ~4 ms), graph as the backup if it fails.
                         # Each is faster on some cones (closed1: graph 34, sampling 33 of 67), so "fastest" takes
                         # the better of the two every cycle; 6 ms is still far under the 100 ms MPC cycle
LOCK_M = 1.5             # m: a zone whose start is this close keeps the side already chosen


@dataclass
class Memory:
    """What one cycle leaves for the next: the side of each zone and the path (Liniger's 'last plan')."""
    epoch: int | None = None
    side: dict = field(default_factory=dict)      # zone id -> "left" / "right"
    path_s: np.ndarray | None = None
    path_e: np.ndarray | None = None


def _corridor(free, e, path_e, wall_lo, wall_hi):
    """At each station, the run of free cells that holds the path: its two ends. An end that stops at a wall
    (not at a zone) is the wall band itself, not the last cell, so the racing line along the wall margin fits."""
    lo = np.empty(len(path_e))
    hi = np.empty(len(path_e))
    for i, pe in enumerate(path_e):
        j = int(np.argmin(np.abs(e - pe)))
        a = b = j
        while a > 0 and free[i, a - 1]:
            a -= 1
        while b < len(e) - 1 and free[i, b + 1]:
            b += 1
        lo[i] = wall_lo[i] if a == 0 or e[a - 1] < wall_lo[i] - 1e-9 else e[a]
        hi[i] = wall_hi[i] if b == len(e) - 1 or e[b + 1] > wall_hi[i] + 1e-9 else e[b]
    return lo, hi


def _lock_sides(grid, zones, memory, s_abs):
    """Block the other side of a zone the car is about to pass (side kept from earlier cycles)."""
    for z in zones:
        side = memory.side.get(z.id)
        if side is None or z.s_from - s_abs > LOCK_M or z.closed_left or z.closed_right:
            continue
        cells = grid.zone_cells.get(z.id)
        if cells is None:
            continue
        for i in np.flatnonzero(cells.any(axis=1)):
            body = grid.e[cells[i]]
            grid.free[i] &= ~(grid.e < body.min()) if side == "left" else ~(grid.e > body.max())


def _stages(w, s_abs, vx, n, dt):
    s = s_abs + max(vx, V_MIN) * dt * np.arange(n + 1)
    i = np.clip(np.searchsorted(w.s, s, side="right") - 1, 0, len(w.s) - 2)
    return s, i


def _cycle_plan(w, delta, lo, hi, s_abs, vx, v_cap, n, dt):
    """-> (the 9 preview stages for the MPC, the same plan on the whole window).
    The window (every station of w, 8 m every 0.25 m) is only handed out (info["window"], the LOCAL
    reference); the stages are computed exactly as before."""
    e_bent = w.e_ref + delta
    th_b, k_b, st_b = D.frenet_shape(w.s, e_bent, w.kappa_c)
    th_r, k_r, _ = D.frenet_shape(w.s, w.e_ref, w.kappa_c)
    kappa = w.kappa_ref + k_b - k_r                 # T5's curvature + what the bend adds (= T5 when not bent)
    v_bend, _ = D.speed_profile(w.s, kappa, st_b, V_MAX, v_cap)     # grip + braking; the MPC accelerates
    s, i = _stages(w, s_abs, vx, n, dt)
    lo_k = np.maximum(lo[i], lo[i + 1])           # both neighbouring stations: the stricter edge
    hi_k = np.minimum(hi[i], hi[i + 1])
    at = lambda a: np.interp(s, w.s, a)           # noqa: E731
    e_k = np.clip(at(e_bent), lo_k, hi_k)
    v_k = np.minimum(np.minimum(at(w.v_ref), at(v_bend)), v_cap)
    v_k = np.minimum(np.maximum(v_k, V_MIN), v_cap)
    plan = CyclePlan(s, e_k, at(w.e_psi_ref + th_b - th_r), at(kappa), v_k, lo_k, hi_k,
                     float(v_cap))
    # the same quantities at every window station (same rules as the stages, without the sampling)
    v_w = np.minimum(np.minimum(w.v_ref, v_bend), v_cap)
    win = {"s": w.s, "e_y_ref": np.clip(e_bent, lo, hi), "e_psi_ref": w.e_psi_ref + th_b - th_r,
           "kappa_ref": kappa, "v_ref": np.minimum(np.maximum(v_w, V_MIN), v_cap), "e_y_min": lo, "e_y_max": hi}
    return plan, win


def plan_cycle(ko, s_abs, e_y, vx, ref, walls, kappa_at, memory: Memory | None = None, method: str = METHOD,
               n: int = PREVIEW_STEPS, dt: float = PREVIEW_DT, track_hash=None, ref_track_hash=None):
    """One cycle. Returns (CyclePlan, info). memory is updated in place (pass the same one every cycle)."""
    memory = memory if memory is not None else Memory()
    if memory.epoch != ko.list_epoch:                                   # perception restarted its list
        memory.epoch, memory.side = ko.list_epoch, {}
    memory.side = {k: v for k, v in memory.side.items() if any(z.id == k and z.s_to > s_abs for z in ko.zones)}
    w = window(s_abs, ref, walls, kappa_at, h=min(8.0, ko.horizon_m))
    grid = build_grid(w, ko.zones)
    _lock_sides(grid, ko.zones, memory, s_abs)
    e_prev = None
    if memory.path_s is not None:
        e_prev = np.interp(w.s, memory.path_s, memory.path_e, left=np.nan, right=np.nan)
        e_prev = np.where(np.isnan(e_prev), w.e_ref, e_prev)
    dp = dp_path(grid.e, grid.free, w.s, w.e_ref, e_y, e_prev)
    info = {"path_found": dp.ok, "start_moved": dp.start_moved, "times": {}, "rejected": {}, "sides": {}}
    v_cap = min(ko.v_cap, V_MAX)

    if not dp.ok:                                                       # blocked: racing line, stop before it
        ds = w.s[1] - w.s[0]
        v_cap = min(v_cap, stop_speed(w.s[dp.blocked_at] - 0.5 * ds - s_abs - STOP_MARGIN_M))
        plan, win = _cycle_plan(w, np.zeros_like(w.s), w.lo, w.hi, s_abs, vx, v_cap, n, dt)
        info.update(method="none", reason=f"no way past: stop before s {w.s[dp.blocked_at]:.2f}", window=win)
        memory.path_s = memory.path_e = None
        return plan, info

    lo, hi = _corridor(grid.free, grid.e, dp.e, w.lo, w.hi)
    side = sides(grid.zone_cells, grid.e, dp.e)
    passes = []
    for zid, sd in side.items():
        idx = np.flatnonzero(grid.zone_cells[zid].any(axis=1))
        a, b = lo[idx].max(), hi[idx].min()                          # the narrowest part of the corridor there
        apex = min(a + D.APEX_MARGIN, b) if sd == "left" else max(b - D.APEX_MARGIN, a)
        passes.append(D.Pass(zid, sd, float(w.s[idx[0]]), float(w.s[idx[-1]]), float(apex)))
    passes.sort(key=lambda p: p.s_a)
    delta0 = float(np.clip(e_y, lo[0], hi[0]) - w.e_ref[0])
    make = {
        "graph": lambda: D.graph_detour(w.s, w.e_ref, lo, hi, dp.e),
        "sampling": lambda: D.sampling_detour(w.s, w.e_ref, lo, hi, passes, delta0, vx, w.kappa_c, vx, v_cap),
    }
    tried = D.METHODS if method == "fastest" else (method,)
    best = None
    for m in tried:
        delta = make[m]()
        if delta is None:
            info["rejected"][m] = "leaves the corridor"
            continue
        plan, win = _cycle_plan(w, delta, lo, hi, s_abs, vx, v_cap, n, dt)
        verdict = check_cycle(plan, s_abs, e_y, track_hash, ref_track_hash)
        if verdict.level == "FAIL":
            info["rejected"][m] = "; ".join(verdict.reasons)
            continue
        t = D.traverse_time(w.s, w.e_ref + delta, w.kappa_c, vx, v_cap)
        info["times"][m] = t
        if best is None or t < best[0]:
            best = (t, m, plan, win)
    if best is None and method != "fastest":                           # the asked method failed: use the DP path
        delta = D.graph_detour(w.s, w.e_ref, lo, hi, dp.e)
        if delta is not None:
            plan, win = _cycle_plan(w, delta, lo, hi, s_abs, vx, v_cap, n, dt)
            if check_cycle(plan, s_abs, e_y, track_hash, ref_track_hash).level != "FAIL":
                best = (D.traverse_time(w.s, w.e_ref + delta, w.kappa_c, vx, v_cap), "graph (fallback)", plan, win)
    if best is None:                                                    # nothing valid: racing line, slow down
        plan, win = _cycle_plan(w, np.zeros_like(w.s), lo, hi, s_abs, vx, min(v_cap, V_MIN), n, dt)
        info.update(method="none", reason="no detour passed the checks", window=win)
        return plan, info

    memory.side.update(side)
    memory.path_s, memory.path_e = w.s, dp.e
    info.update(method=best[1], sides={k: side[k] for k in side}, window=best[3])
    return best[2], info
