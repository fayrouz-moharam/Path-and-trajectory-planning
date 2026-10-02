"""Step A: evenly spaced stations on the reference line, their left normals,
and the lateral offset limits alpha_min / alpha_max at each station.

A racing-line point is  p_i = r_i + alpha_i * n_i   (alpha > 0 = left).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Stations:
    s: np.ndarray          # (N,) arc length of each station, 0 ... L - ds (no station at L)
    xy: np.ndarray         # (N, 2) reference points r_i
    heading: np.ndarray    # (N,) psi_i
    kappa: np.ndarray      # (N,) curvature of the reference line (from T2)
    normals: np.ndarray    # (N, 2) left unit normals n_i = (-sin psi, cos psi)
    w_left: np.ndarray     # (N,) distance to the left wall
    w_right: np.ndarray    # (N,) distance to the right wall
    alpha_min: np.ndarray  # (N,) most-right allowed offset (<= 0 normally)
    alpha_max: np.ndarray  # (N,) most-left allowed offset (>= 0 normally)
    ds: float              # actual spacing = L / N
    margin: float          # closest the car's centre may get to a wall, see vehicle_margin()

    def __len__(self) -> int:
        return len(self.s)


def vehicle_margin(car_width: float, safety_buffer: float) -> float:
    """Closest the car's CENTRE may get to a wall: half the car's width (centre
    to its side) plus a safety buffer (tracking / localisation error)."""
    if car_width <= 0 or safety_buffer < 0:
        raise ValueError(f"need car_width > 0 and safety_buffer >= 0, "
                         f"got {car_width}, {safety_buffer}")
    return car_width / 2 + safety_buffer


def make_stations(track, spacing_target: float, margin: float, walls=None) -> Stations:
    """Sample `track` (anything with .length and .sample(s)) every ~spacing_target metres.

    walls: optional (left_wall, right_wall) from build_walls(). If given, the
    widths are re-measured by ray-casting from these stations to those fixed
    walls instead of taken from the track (used when the reference line is no
    longer the centreline, e.g. inside the iteration loop).
    """
    L = track.length
    n = int(round(L / spacing_target))
    if n < 3:
        raise ValueError(f"only {n} stations for L = {L:.2f} m; spacing_target too large")
    ds = L / n                      # exact spacing so that N * ds == L (closed loop)
    s = np.arange(n) * ds           # last station is L - ds; L itself would duplicate s = 0

    s, x, y, heading, kappa, w_left, w_right = track.sample(s)
    normals = np.column_stack([-np.sin(heading), np.cos(heading)])
    xy = np.column_stack([x, y])
    if walls is not None:
        w_left, w_right = widths_from_walls(xy, normals, *walls)

    alpha_max = w_left - margin
    alpha_min = -(w_right - margin)

    too_narrow = alpha_max <= alpha_min
    if too_narrow.any():
        raise ValueError(
            f"track narrower than 2 * margin = {2 * margin:.2f} m at {too_narrow.sum()} "
            f"stations, e.g. s = {np.round(s[too_narrow][:5], 2).tolist()} m"
        )

    # The centreline itself (alpha = 0) should normally be allowed. If it isn't,
    # a solution may still exist, but it means a wall is closer than the margin.
    # Only checked on the original track: inside the iteration loop (walls given)
    # the reference is the previous racing line, which sits ON the margin by design.
    # 1 mm tolerance so rounding at exactly the margin doesn't warn.
    centre_blocked = (alpha_min > 1e-3) | (alpha_max < -1e-3)
    if walls is None and centre_blocked.any():
        warnings.warn(
            f"alpha = 0 outside the bounds at {centre_blocked.sum()} stations "
            f"(a wall is closer than margin = {margin:.2f} m), e.g. "
            f"s = {np.round(s[centre_blocked][:5], 2).tolist()} m"
        )

    return Stations(s, xy, heading, kappa, normals,
                    w_left, w_right, alpha_min, alpha_max, ds, margin)


# ---------------------------------------------------------------- walls and widths

def build_walls(track, spacing: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """The real walls as fixed closed polylines, built ONCE from the original
    centreline: left = r + w_left*n, right = r - w_right*n. They never change."""
    n = max(int(round(track.length / spacing)), 3)
    s = np.arange(n) * (track.length / n)
    _, x, y, heading, _, w_left, w_right = track.sample(s)
    r = np.column_stack([x, y])
    normals = np.column_stack([-np.sin(heading), np.cos(heading)])
    return r + w_left[:, None] * normals, r - w_right[:, None] * normals


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D cross product a_x b_y - a_y b_x (works on any leading shape)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _ray_distance(origins: np.ndarray, dirs: np.ndarray, wall: np.ndarray,
                  chunk: int = 256) -> np.ndarray:
    """Distance t along each ray  o + t*d  (t > 0, d unit) to the nearest crossing
    with the closed polyline `wall`; inf where the ray misses.

    Ray o + t d meets edge a + u e (u in [0, 1]) where  t d - u e = a - o.
    Crossing both sides with e and with d gives
        t = ((a - o) x e) / (d x e),    u = ((a - o) x d) / (d x e).
    """
    a = wall
    e = np.roll(wall, -1, axis=0) - wall          # edge vectors, closing edge included
    out = np.full(len(origins), np.inf)
    for i0 in range(0, len(origins), chunk):      # chunks keep the (rays x edges) arrays small
        o = origins[i0:i0 + chunk, None, :]
        d = dirs[i0:i0 + chunk, None, :]
        ao = a[None] - o
        denom = _cross(d, e[None])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(ao, e[None]) / denom
            u = _cross(ao, d) / denom
        # small slack on u: a ray through a wall VERTEX is u = 1 on one edge and
        # u = 0 on the next, and rounding can push both just outside [0, 1]
        hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
        out[i0:i0 + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return out


def widths_from_walls(points: np.ndarray, normals: np.ndarray,
                      left_wall: np.ndarray, right_wall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w_left = distance along +n to the left wall, w_right = along -n to the right wall."""
    w_left = _ray_distance(points, normals, left_wall)
    w_right = _ray_distance(points, -normals, right_wall)
    miss = np.isinf(w_left) | np.isinf(w_right)
    if miss.any():
        raise ValueError(f"ray missed a wall at {miss.sum()} points, "
                         f"e.g. indices {np.flatnonzero(miss)[:5].tolist()}")
    return w_left, w_right
