"""T5: describe the T3 racing line and T4 speeds from perception's centre line.

For every centre-line point C (perception's own s, never recomputed):
  B. the line across the track at C:  C + t * n,  n = left normal (-sin psi, cos psi)
  C. where it crosses the racing line (a closed chain of straight segments) -> e_y_ref = t
  D. where that crossing lies ON the racing line (s_race) -> heading, kappa, v_ref there
  E. e_psi_ref = racing-line heading - centre-line heading, wrapped to (-pi, pi]
The two s rulers are linked only through this physical crossing point.
"""
from __future__ import annotations

import numpy as np


def cross_racing_line(c_xy: np.ndarray, c_heading: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                      r_xy: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps B + C: where each centre-line point's across-line crosses the racing line.

    c_xy (N, 2), c_heading (N,): centre-line points and headings; w_left, w_right (N,): wall
    distances there (a crossing must lie between the walls). r_xy (M, 2): racing line, closed,
    no repeated end point. Returns per centre-line point:
      e_y  signed distance C -> crossing along the left normal (+ = left)
      seg  index j of the racing-line segment r[j] -> r[j+1] it crosses (j = M-1: back to r[0])
      frac how far along that segment the crossing lies, 0 <= frac < 1
    If the across-line crosses the racing line more than once, the crossing nearest C is kept.
    """
    c_xy, r_xy = np.asarray(c_xy, float), np.asarray(r_xy, float)
    n = np.column_stack([-np.sin(c_heading), np.cos(c_heading)])        # left normals (N, 2)
    d = np.roll(r_xy, -1, axis=0) - r_xy                                 # segment vectors, closed (M, 2)
    N = len(c_xy)
    e_y, seg, frac = np.empty(N), np.empty(N, dtype=int), np.empty(N)
    for lo in range(0, N, chunk):                                        # chunks keep memory small on long tracks
        hi = min(lo + chunk, N)
        nx, ny = n[lo:hi, 0:1], n[lo:hi, 1:2]                            # (k, 1)
        rx = r_xy[None, :, 0] - c_xy[lo:hi, 0:1]                         # segment start minus C (k, M)
        ry = r_xy[None, :, 1] - c_xy[lo:hi, 1:2]
        # solve  C + t n = r_j + u d_j   ->   [n, -d] [t, u] = r_j - C   (Cramer's rule)
        det = -nx * d[None, :, 1] + ny * d[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-rx * d[None, :, 1] + ry * d[None, :, 0]) / det
            u = (nx * ry - ny * rx) / det
        ok = ((np.abs(det) > 1e-12) & (u >= 0.0) & (u < 1.0)
              & (t <= w_left[lo:hi, None]) & (t >= -w_right[lo:hi, None]))
        dist = np.where(ok, np.abs(t), np.inf)
        j = np.argmin(dist, axis=1)                                      # nearest valid crossing
        rows = np.arange(hi - lo)
        missing = np.isinf(dist[rows, j])
        if missing.any():
            raise ValueError(f"centre-line point {lo + int(np.argmax(missing))}: its across-line "
                             "does not cross the racing line between the walls")
        e_y[lo:hi], seg[lo:hi], frac[lo:hi] = t[rows, j], j, u[rows, j]
    return e_y, seg, frac


def wrap(angle: np.ndarray) -> np.ndarray:
    """Angle(s) wrapped into (-pi, pi]."""
    return np.pi - np.mod(np.pi - angle, 2.0 * np.pi)


def sample_at_crossings(seg: np.ndarray, frac: np.ndarray, r_s: np.ndarray, r_heading: np.ndarray,
                        r_kappa: np.ndarray, r_lap: float, v_ref: np.ndarray,
                        c_heading: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steps D + E: racing-line values at the crossings found by cross_racing_line.

    seg, frac (N,): from cross_racing_line. r_s, r_heading, r_kappa (M,): racing_line.csv columns;
    r_lap: racing-line lap length; v_ref (M,): velocity_profile.csv on the SAME grid as r_s;
    c_heading (N,): centre-line headings. Returns per centre-line point:
      s_race  where the crossing is on the racing line [m]
      e_psi   racing-line heading - centre-line heading, in (-pi, pi]
      kappa   racing-line curvature there
      v       speed there
    """
    M = len(r_s)
    nxt = (seg + 1) % M                                      # the closing segment ends at point 0 ...
    s_end = np.where(nxt == 0, r_lap, r_s[nxt])              # ... which sits at s = r_lap, not at s = 0
    s_race = r_s[seg] + frac * (s_end - r_s[seg])
    heading = r_heading[seg] + frac * wrap(r_heading[nxt] - r_heading[seg])
    kappa = r_kappa[seg] + frac * (r_kappa[nxt] - r_kappa[seg])
    v = v_ref[seg] + frac * (v_ref[nxt] - v_ref[seg])
    e_psi = wrap(heading - c_heading)
    return s_race, e_psi, kappa, v


def check_reference(c_xy: np.ndarray, c_heading: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                    e_y: np.ndarray, s_race: np.ndarray, r_xy: np.ndarray, r_lap: float, margin: float,
                    tol_land: float = 1e-3, tol_wall: float = 5e-3, tol_start: float = 1e-2) -> tuple[dict, list[str]]:
    """Self-checks on the mapped reference. Returns (report, problems); problems == [] means all fine.

      1. s_race increases from row to row, wrapping round the lap exactly once
      2. C + e_y * n lands on the racing line (within tol_land)
      3. -(w_right - margin) <= e_y <= w_left - margin (within tol_wall)
      4. the first row is at the racing line's start (s_race ~ 0, within tol_start)
    """
    problems = []
    # 1. s_race keeps increasing; the step from the last row back to the first row counts too
    steps = np.diff(np.append(s_race, s_race[0]))
    wraps = steps < -0.5 * r_lap                             # the finish line: s_race drops by ~r_lap
    steps = np.where(wraps, steps + r_lap, steps)
    back = np.nonzero(steps <= 0)[0]
    if wraps.sum() != 1:
        problems.append(f"s_race wraps round the lap {int(wraps.sum())} times (want exactly 1)")
    if len(back):
        problems.append(f"s_race goes backwards at {len(back)} rows, first after row {back[0]}")
    # 2. go back and land on the racing line
    n = np.column_stack([-np.sin(c_heading), np.cos(c_heading)])
    miss = _dist_to_polyline(c_xy + e_y[:, None] * n, r_xy)
    if miss.max() > tol_land:
        problems.append(f"C + e_y n misses the racing line by {miss.max() * 1000:.1f} mm at row {int(np.argmax(miss))}")
    # 3. inside the walls, keeping the car margin
    room = np.minimum(w_left - margin - e_y, w_right - margin + e_y)
    if room.min() < -tol_wall:
        problems.append(f"e_y leaves the corridor (walls - {margin} m) by {-room.min() * 1000:.1f} mm "
                        f"at row {int(np.argmin(room))}")
    # 4. the first row is where the racing line starts
    start = min(s_race[0], r_lap - s_race[0])
    if start > tol_start:
        problems.append(f"first row is at s_race = {s_race[0]:.3f} m, not at the racing line's start")
    report = {"n_rows": len(e_y), "s_race_wraps": int(wraps.sum()), "s_race_min_step": float(steps.min()),
              "land_max_miss_m": float(miss.max()), "corridor_min_room_m": float(room.min()),
              "start_offset_m": float(start)}
    return report, problems


def _dist_to_polyline(p: np.ndarray, r_xy: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Distance from each point p (K, 2) to the closed polyline r_xy (M, 2)."""
    d = np.roll(r_xy, -1, axis=0) - r_xy
    L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    out = np.empty(len(p))
    for lo in range(0, len(p), chunk):
        q = p[lo:lo + chunk, None, :] - r_xy[None]                         # point minus segment start (k, M, 2)
        t = np.clip(np.einsum("kmj,mj->km", q, d) / L2, 0.0, 1.0)           # closest spot on each segment
        out[lo:lo + chunk] = np.min(np.linalg.norm(q - t[..., None] * d[None], axis=2), axis=1)
    return out
