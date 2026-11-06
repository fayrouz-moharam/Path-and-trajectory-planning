"""B1 + B2: the stretch of track ahead of the car, and the corridor grid over it (Liniger's method).

B1 window(): stations every DS metres from the car's s_abs, H metres ahead (s never wraps; the lookups wrap).
B2 build_grid(): cells across the track every DE metres; a cell is FREE when the car's CENTRE may be there:
  - inside the walls minus WALL_MARGIN (0.25 m = half the car 0.15 + buffer 0.10, as T3 / T6), and
  - outside every keep-out zone, after the disparity-style widening below.

Disparity extender (Otterness 2019): grow each obstacle by half the car + a margin, so the gap shifts and the
car's centre can be planned as a point. Half A's zones ALREADY include exactly that (cone radius + 2.45 sigma +
half car 0.15 + buffer 0.10), so EXTRA_WIDEN_M is 0 by default: raise it only for extra margin on top, never
to add the car's width again (that would count it twice and close real gaps).

Each station blocks the whole s-strip [s - DS/2, s + DS/2] it stands for, not just its centre point, so the
path can never slip past the corner of a zone between two stations. A side Half A says is closed
(closed_left / closed_right) is blocked all the way to that wall, even when rounding left a free cell there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = 8.0                  # m: how far ahead the grid looks (Half A lists zones up to 8 m)
DS = 0.25                # m: station spacing along the track
DE = 0.05                # m: cell spacing across the track (= the reference's 0.05 m grid)
WALL_MARGIN = 0.25       # m: car centre to wall (half car 0.15 + buffer 0.10, as T6 checks.py)
EXTRA_WIDEN_M = 0.0      # m: extra disparity-style widening of every zone, on top of Half A's (see above)
SHAPE = "box"            # "box" or "circle": which of the zone's two shapes blocks cells (circle = more room)


@dataclass
class Window:
    """B1: the track ahead at the stations."""
    s: np.ndarray            # (M,) centre-line s of each station, continuous like s_abs
    e_ref: np.ndarray        # (M,) racing line (T5 e_y_ref), + left
    e_psi_ref: np.ndarray    # (M,) racing line heading vs the centre line
    kappa_ref: np.ndarray    # (M,) racing line curvature
    v_ref: np.ndarray        # (M,) T5 speed
    kappa_c: np.ndarray      # (M,) centre-line curvature
    lo: np.ndarray           # (M,) right edge for the car's centre: -w_right + WALL_MARGIN
    hi: np.ndarray           # (M,) left edge: w_left - WALL_MARGIN


def window(s_car: float, ref, walls, kappa_at, h: float = H, ds: float = DS) -> Window:
    """ref = PlanningReference (t5_reference.loader); walls(s) -> (w_left, w_right); kappa_at(s) -> centre-line
    curvature (apex_track's track.width_at / track.kappa_at). All three wrap the lap themselves."""
    s = s_car + ds * np.arange(int(round(h / ds)) + 1)
    r = ref.at(s)
    w_left, w_right = walls(s)
    return Window(s, np.asarray(r["e_y_ref_m"]), np.asarray(r["e_psi_ref_rad"]), np.asarray(r["kappa_ref_1pm"]),
                  np.asarray(r["v_ref_mps"]), np.asarray(kappa_at(s), dtype=float),
                  -(np.asarray(w_right, dtype=float) - WALL_MARGIN), np.asarray(w_left, dtype=float) - WALL_MARGIN)


def blocked(z, s_lo: float, s_hi: float, e: np.ndarray, widen: float = EXTRA_WIDEN_M,
            shape: str = SHAPE) -> np.ndarray:
    """Cells e (across the track) that zone z blocks anywhere in the station strip [s_lo, s_hi]."""
    if shape == "box":
        if z.s_to + widen * z.stretch < s_lo or z.s_from - widen * z.stretch > s_hi:
            return np.zeros(e.shape, dtype=bool)
        return (e >= z.e_y_from - widen) & (e <= z.e_y_to + widen)
    r = z.radius + widen                                 # circle: nearest point of the strip along s,
    s_near = min(max(z.s_center, s_lo), s_hi)            # back to real metres with the corner stretch
    d_s = (s_near - z.s_center) / z.stretch
    if d_s * d_s > r * r:
        return np.zeros(e.shape, dtype=bool)
    return np.abs(e - z.e_y_center) <= np.sqrt(r * r - d_s * d_s)


def blocked_with_sides(z, s_lo: float, s_hi: float, e: np.ndarray, **kw) -> np.ndarray:
    """blocked() + a closed side blocked up to its wall (Half A's flag beats the grid's rounding)."""
    b = blocked(z, s_lo, s_hi, e, **kw)
    if b.any():
        if z.closed_left:
            b |= e >= e[b].min()
        if z.closed_right:
            b |= e <= e[b].max()
    return b


@dataclass
class Grid:
    """B2: which cells the car's centre may use."""
    e: np.ndarray            # (N,) cell positions across the track, + left
    free: np.ndarray         # (M, N) bool
    zone_cells: dict         # zone id -> (M, N) bool: the zone's own cells, without a closed side (to read the side)


def build_grid(w: Window, zones, de: float = DE, **kw) -> Grid:
    ds = w.s[1] - w.s[0]
    e = np.arange(np.floor(w.lo.min() / de) * de, w.hi.max() + 0.5 * de, de)
    free = (e[None, :] >= w.lo[:, None] - 1e-9) & (e[None, :] <= w.hi[:, None] + 1e-9)
    zone_cells = {}
    for z in zones:
        if z.s_to < w.s[0] - ds or z.s_from > w.s[-1] + ds:      # beyond the window
            continue
        if kw.get("shape", SHAPE) == "box":          # same cells at every station it touches: one row, repeated
            widen = kw.get("widen", EXTRA_WIDEN_M)
            hit = (w.s + 0.5 * ds >= z.s_from - widen * z.stretch) & (w.s - 0.5 * ds <= z.s_to + widen * z.stretch)
            mid = 0.5 * (z.s_from + z.s_to)
            body = hit[:, None] & blocked(z, mid, mid, e, **kw)[None, :]
            sided = hit[:, None] & blocked_with_sides(z, mid, mid, e, **kw)[None, :]
        else:
            body = np.array([blocked(z, s - 0.5 * ds, s + 0.5 * ds, e, **kw) for s in w.s])
            sided = np.array([blocked_with_sides(z, s - 0.5 * ds, s + 0.5 * ds, e, **kw) for s in w.s])
        free &= ~sided
        zone_cells[z.id] = body
    return Grid(e, free, zone_cells)
