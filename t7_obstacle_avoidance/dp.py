"""B3: the cheapest way through the free cells (Liniger, Domahidi, Morari 2015: a shortest path on a grid picks
the side; Stahl et al. 2019: cost = closeness to the racing line + smoothness, any free cell at the end counts).

One cell per station. A step from cell (i, j) to (i+1, j') costs
    DS * [ W_SLOPE * ((e_j' - e_j) / DS)^2  +  W_REF * (e_j' - e_ref)^2  +  W_PREV * (e_j' - e_prev)^2 ]
and is not allowed into a blocked cell or more than tan(PSI_MAX) * DS sideways. Solved backwards, so a cell
with no free way forward (a dead end) gets cost inf by itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

W_REF = 1.0              # stay near the racing line (also picks the side closer to it, as Half A's rule says)
W_SLOPE = 10.0           # don't jump sideways
W_PREV = 2.0             # stay near last cycle's path (Liniger: keeps the side from flickering)
PSI_MAX = 0.35           # rad: steepest the path may run against the track (tan = 0.37 -> 1 cell per 0.25 m)


@dataclass
class DPResult:
    ok: bool                 # False = no way past (blocked track): stop before it
    e: np.ndarray            # (M,) the path across the track at each station (nan when not ok)
    cost: float
    start_moved: bool        # the car's own cell was blocked: the path starts from the nearest free cell
    blocked_at: int | None   # not ok: first station with no reachable free cell


def dp_path(e: np.ndarray, free: np.ndarray, s: np.ndarray, e_ref: np.ndarray, e_car: float,
            e_prev: np.ndarray | None = None, w_ref: float = W_REF, w_slope: float = W_SLOPE,
            w_prev: float = W_PREV, psi_max: float = PSI_MAX) -> DPResult:
    m, n = free.shape
    ds = float(s[1] - s[0])
    de = float(e[1] - e[0])
    k_max = max(1, int(math.floor(ds * math.tan(psi_max) / de + 1e-9)))
    node = w_ref * (e[None, :] - e_ref[:, None]) ** 2
    if e_prev is not None:
        node = node + w_prev * (e[None, :] - e_prev[:, None]) ** 2
    node = ds * node
    J = np.full((m, n), np.inf)
    nxt = np.zeros((m, n), dtype=int)
    J[-1] = np.where(free[-1], node[-1], np.inf)
    for i in range(m - 2, -1, -1):
        best = np.full(n, np.inf)
        arg = np.zeros(n, dtype=int)
        for k in range(-k_max, k_max + 1):
            j_to = np.arange(n) + k
            ok = (j_to >= 0) & (j_to < n)
            cand = np.full(n, np.inf)
            cand[ok] = J[i + 1, j_to[ok]] + ds * w_slope * (k * de / ds) ** 2
            better = cand < best
            best[better], arg[better] = cand[better], j_to[better]
        J[i] = np.where(free[i], best + (node[i] if i > 0 else 0.0), np.inf)
        nxt[i] = arg
    # start: the car's own cell, or the nearest free one with a way forward
    reach = np.isfinite(J[0])
    if not reach.any():           # no way past: find where the car's reachable cells run out (stop before it)
        r = np.zeros(n, dtype=bool)
        r[int(np.argmin(np.where(free[0], np.abs(e - e_car), np.inf)))] = free[0].any()
        cut = 0
        for i in range(1, m):
            grown = r.copy()
            for k in range(1, k_max + 1):
                grown[k:] |= r[:-k]
                grown[:-k] |= r[k:]
            r = grown & free[i]
            if not r.any():
                cut = i
                break
        return DPResult(False, np.full(m, np.nan), math.inf, False, cut)
    j0_car = int(np.argmin(np.abs(e - e_car)))
    cand = np.flatnonzero(reach)
    j0 = int(cand[np.argmin(np.abs(e[cand] - e_car))])
    path = [j0]
    for i in range(m - 1):
        path.append(int(nxt[i, path[-1]]))
    return DPResult(True, e[np.array(path)], float(J[0, j0]), j0 != j0_car, None)


def sides(zone_cells: dict, e: np.ndarray, path_e: np.ndarray) -> dict:
    """Which side of each zone the path passes: 'left' (+e) or 'right', from the stations the zone blocks."""
    out = {}
    for zid, cells in zone_cells.items():
        votes = []
        for i in np.flatnonzero(cells.any(axis=1)):
            lo, hi = e[cells[i]].min(), e[cells[i]].max()
            if path_e[i] > hi:
                votes.append("left")
            elif path_e[i] < lo:
                votes.append("right")
        if votes:
            out[zid] = max(set(votes), key=votes.count)
    return out
