"""B4: the bent racing line around the zones -- two ways, both inside the same corridor and the same side choice.

  "graph"    Stahl et al. 2019 / Liniger: the DP path through the grid itself, smoothed and kept in the corridor.
  "sampling" Werling et al. 2010: for each zone, many quintic sideways moves (how far out, how long to get there
             and back); every one that stays in the corridor is timed and the quickest kept.

(A third, ForzaETH's spline, was dropped on 6 Nov 2026: slowest on closed1, 1.87 s vs 1.72 s past a cone.)
Each returns the offset from the racing line at the stations (delta = e_y - e_ref), or None. traverse_time()
scores any of them: the time to drive the window at the grip, acceleration and braking limits of T4 / T6.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from t6_feasibility.checks import A_ACCEL_MAX, A_BRAKE_MAX, G, MU, V_MAX, V_MIN

METHODS = ("graph", "sampling")
APEX_MARGIN = 0.01       # m: apex this far inside the corridor edge the zone makes (zone holds car + buffer)
RAMP_M = 1.0             # m: sampling's distance to bring the car's own offset back to the racing line (x alpha)
SMOOTH_M = 1.0           # m: graph's moving-average length
SAMPLE_OUT = (0.0, 0.05, 0.10)           # m further out than the apex, sampling
SAMPLE_RAMP = (0.75, 1.25, 2.0)          # m to get out / back (x alpha), sampling
TOL = 1e-6


# ---------------------------------------------------------------- geometry and time
def _slope(y, h):
    """dy/ds on stations h apart (same as np.gradient, faster for the short arrays here)."""
    g = np.empty_like(y)
    g[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    g[0], g[-1] = (y[1] - y[0]) / h, (y[-1] - y[-2]) / h
    return g


def frenet_shape(s, e, kappa_c):
    """Heading vs the centre line, curvature, and real path length per unit s of the line e(s)."""
    h = s[1] - s[0]
    de = _slope(e, h)
    d = 1.0 - kappa_c * e
    theta = np.arctan2(de, d)
    stretch = np.hypot(d, de)
    kappa = (kappa_c + _slope(theta, h)) / stretch
    return theta, kappa, stretch


def speed_profile(s, kappa, stretch, v0, v_cap=V_MAX):
    """Fastest speeds along the line: grip sqrt(mu g / |kappa|), the cap, +2 / -3 m/s^2."""
    v = np.minimum(np.sqrt(MU * G / np.maximum(np.abs(kappa), 1e-9)), min(v_cap, V_MAX)).tolist()
    step = (np.diff(s) * 0.5 * (stretch[1:] + stretch[:-1])).tolist()
    v[0] = min(v[0], max(v0, 0.0))
    for i in range(len(v) - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] * v[i] + 2 * A_ACCEL_MAX * step[i]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] * v[i + 1] + 2 * A_BRAKE_MAX * step[i]))
    return np.array(v), np.array(step)


def traverse_time(s, e, kappa_c, v0, v_cap=V_MAX):
    _, kappa, stretch = frenet_shape(s, e, kappa_c)
    v, step = speed_profile(s, kappa, stretch, v0, v_cap)
    v = np.maximum(v, V_MIN)
    return float(np.sum(2 * step / (v[1:] + v[:-1])))


def inside(delta, e_ref, lo, hi):
    e = e_ref + delta
    return bool(np.all(e >= lo - TOL) and np.all(e <= hi + TOL))


# ---------------------------------------------------------------- shared helpers
def _alpha(v):
    return 1.0 + min(max(v, 0.0) / V_MAX, 0.5)


def _smoothstep5(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)       # quintic: zero slope and curvature at both ends


def _combine(bumps):
    """Same-side bumps: the furthest out wins; opposite sides (a slalom) add up."""
    if not bumps:
        return 0.0
    b = np.array(bumps)
    return np.maximum(b, 0).max(axis=0) + np.minimum(b, 0).min(axis=0)


# ---------------------------------------------------------------- the two methods
@dataclass(frozen=True)
class Pass:
    """One zone as the grid sees it: the stations it narrows the corridor at, and the line to hold there."""
    zone_id: int
    side: str                # "left" / "right"
    s_a: float               # first and last station the zone blocks (its strip, not just its box)
    s_b: float
    apex: float              # e_y to hold beside the zone: just inside the corridor edge the zone makes


def graph_detour(s, e_ref, lo, hi, e_path):
    ds = s[1] - s[0]
    n = max(1, int(round(SMOOTH_M / ds)))
    delta = e_path - e_ref
    for _ in range(2):
        pad = np.concatenate([np.full(n, delta[0]), delta, np.full(n, delta[-1])])
        delta = np.convolve(pad, np.ones(2 * n + 1) / (2 * n + 1), mode="same")[n:-n]
    delta[0] = e_path[0] - e_ref[0]
    delta = np.clip(e_ref + delta, lo, hi) - e_ref
    return delta if inside(delta, e_ref, lo, hi) else None


def sampling_detour(s, e_ref, lo, hi, passes, delta0, v, kappa_c, v0, v_cap=V_MAX):
    a = _alpha(v)
    start = delta0 * (1 - _smoothstep5((s - s[0]) / (RAMP_M * a)))

    def bump(p, out, l_in, l_out):
        target = p.apex + (out if p.side == "left" else -out) - e_ref       # offset that holds e = apex + out
        up = _smoothstep5((s - (p.s_a - l_in * a)) / (l_in * a))
        down = 1 - _smoothstep5((s - p.s_b) / (l_out * a))
        return target * np.minimum(up, down)

    options = list(itertools.product(SAMPLE_OUT, SAMPLE_RAMP, SAMPLE_RAMP))
    bumps = [[bump(p, *o) for o in options] for p in passes]                 # every candidate, made once
    choice = [0] * len(passes)
    best, best_t = None, math.inf
    for k in range(len(passes)):                     # one pass of coordinate descent, zone by zone
        for o in range(len(options)):
            trial = choice[:k] + [o] + choice[k + 1:]
            delta = _combine([start] + [bumps[j][c] for j, c in enumerate(trial)])
            if not inside(delta, e_ref, lo, hi):
                continue
            t = traverse_time(s, e_ref + delta, kappa_c, v0, v_cap)
            if t < best_t:
                best, best_t, choice = delta, t, trial
    if not passes:
        best = start if inside(start, e_ref, lo, hi) else None
    return best
