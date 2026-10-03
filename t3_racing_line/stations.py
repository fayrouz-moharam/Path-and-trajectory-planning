"""Step A: evenly spaced stations on the reference line, their left normals,
and the lateral offset limits alpha_min / alpha_max at each station.

A racing-line point is  p_i = r_i + alpha_i * n_i   (alpha > 0 = left).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from t2_curvature.queryable_track import QueryableTrack


def track_from_points(x, y, w_left, w_right) -> QueryableTrack:
    """T2 track through a closed loop of points (first point NOT repeated).
    s = distance walked from point 0 along straight gaps -- the same way T1
    builds the s column of track.csv."""
    gaps = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(gaps)])
    return QueryableTrack.from_arrays(s, x, y, w_left, w_right)


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


def make_stations(track, spacing_target: float, margin: float, walls=None,
                  start_tol: float = 2e-3) -> Stations:
    """Sample `track` (a T2 QueryableTrack: .length, .sample_arrays(s)) every ~spacing_target metres.

    walls: optional (left_wall, right_wall) from build_walls(). If given, the
    limits come from walking sideways to these fixed walls (alpha_limits) instead
    of from the track's widths -- used in every round of the iteration loop.
    Then w_left = alpha_max + margin and w_right = margin - alpha_min.
    start_tol: how far inside the margin a station may start and still count as
    outside it (alpha_limits' tol).
    """
    L = track.length
    n = int(round(L / spacing_target))
    if n < 3:
        raise ValueError(f"only {n} stations for L = {L:.2f} m; spacing_target too large")
    ds = L / n                      # exact spacing so that N * ds == L (closed loop)
    s = np.arange(n) * ds           # last station is L - ds; L itself would duplicate s = 0

    s, x, y, heading, kappa, w_left, w_right = track.sample_arrays(s)
    normals = np.column_stack([-np.sin(heading), np.cos(heading)])
    xy = np.column_stack([x, y])
    if walls is None:
        # no walls given: trust the track's own widths (correct on a centreline
        # without folds or hairpin tips)
        alpha_max = w_left - margin
        alpha_min = -(w_right - margin)
    else:
        # walk sideways to the real walls (cannot miss a hairpin's tip); a station
        # slightly inside the margin gets a minimum move out instead of no room
        alpha_min, alpha_max, _, stuck = alpha_limits(xy, normals, walls, margin, tol=start_tol)
        if stuck.any():
            raise ValueError(
                f"{stuck.sum()} stations are inside the {margin:.2f} m margin with no way out "
                f"(the track is narrower than 2 * margin there?), "
                f"e.g. s = {np.round(s[stuck][:5], 2).tolist()} m"
            )
        w_left, w_right = alpha_max + margin, margin - alpha_min

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
    _, x, y, heading, _, w_left, w_right = track.sample_arrays(s)
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


# ---------------------------------------------------------------- room by walking sideways

def _densify_closed(poly: np.ndarray, max_gap: float) -> np.ndarray:
    """Closed polyline with extra points so no gap is longer than max_gap
    (the closing edge, last -> first, included). Original points are kept."""
    nxt = np.roll(poly, -1, axis=0)
    pieces = np.maximum(np.ceil(np.linalg.norm(nxt - poly, axis=1) / max_gap).astype(int), 1)
    seg = np.repeat(np.arange(len(poly)), pieces)              # which edge each new point is on
    first = np.cumsum(pieces) - pieces                         # index where each edge's points start
    frac = (np.arange(pieces.sum()) - first[seg]) / pieces[seg]  # 0, 1/k, 2/k, ... along that edge
    return poly[seg] + frac[:, None] * (nxt[seg] - poly[seg])


def alpha_limits(points: np.ndarray, normals: np.ndarray, walls, margin: float,
                 step: float = 0.01, max_dist: float = 3.0,
                 tol: float = 2e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed limits alpha_min <= alpha <= alpha_max for moving each point along its
    normal (+ = left) without coming closer than `margin` to ANY wall point.

    Walk sideways in `step` increments on both sides; at every step a KD-tree gives
    the distance to the nearest wall point. Unlike a ray that must CROSS a wall line,
    this cannot miss a hairpin's inside tip passing beside it, nor run on to a far wall.

    Start outside the margin (up to `tol` inside still counts, e.g. the previous
    racing line sitting on it): limits are [-room_right, +room_left], where room is
    how far the walk gets before reaching the margin.
    Start inside the margin: the point must be PUSHED OUT, so the walk goes AWAY from
    the wall (the side where the distance grows): it finds `exit` (back outside the
    margin) and `far` (the next wall), giving [+exit, +far] on the left or
    [-far, -exit] on the right -- a minimum move instead of no room at all.

    walls: (left_wall, right_wall) closed polylines (from build_walls).
    Returns (alpha_min, alpha_max, pushed, stuck): pushed = started inside the margin;
    stuck = started inside with no way out (its limits are meaningless). Limits are
    capped at max_dist (no wall found within it).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.vstack([_densify_closed(w, step) for w in walls]))
    alphas = np.arange(0.0, max_dist + step / 2, step)        # 0, 0.01, ..., max_dist
    n, last = len(points), len(alphas)
    side = {}
    for sign in (+1.0, -1.0):                                  # left first, then right
        walk = points[:, None, :] + sign * alphas[None, :, None] * normals[:, None, :]  # (N, S, 2)
        dist = tree.query(walk.reshape(-1, 2), workers=-1)[0].reshape(n, last)          # (N, S)
        outside = dist >= margin
        outside[:, 0] = dist[:, 0] >= margin - tol             # the start gets the tolerance
        start_ok = outside[:, 0]

        # exit: first step outside the margin (step 0 if the start is already OK)
        ex = np.where(outside.any(axis=1), outside.argmax(axis=1), last)
        exit_ = np.full(n, np.inf)
        found = ex < last
        exit_[found] = alphas[ex[found]]
        k = np.flatnonzero(found & (ex > 0))                   # refine: where dist reached margin
        e0, e1 = dist[k, ex[k] - 1], dist[k, ex[k]]
        exit_[k] = alphas[ex[k] - 1] + (margin - e0) / (e1 - e0) * step

        # far: first step back inside the margin AFTER the exit (the next wall)
        after = (~outside) & (np.arange(last)[None, :] > ex[:, None])
        fb = np.where(after.any(axis=1), after.argmax(axis=1), last)
        far = np.full(n, max_dist)
        i = np.flatnonzero(fb < last)
        j = fb[i]
        d0, d1 = dist[i, j - 1], dist[i, j]                    # last outside step, first inside step
        far[i] = np.maximum(alphas[j - 1] + (d0 - margin) / (d0 - d1) * step, 0.0)

        moving_away = dist[:, 1] > dist[:, 0]                  # does this side lead away from the wall?
        side[sign] = (start_ok, exit_, far, moving_away)

    ok_l, exit_l, far_l, away_l = side[+1.0]
    ok_r, exit_r, far_r, away_r = side[-1.0]
    alpha_min, alpha_max = -far_r, far_l.copy()                # normal case: start outside the margin
    pushed = ~(ok_l & ok_r)
    use_l = pushed & away_l & (~away_r | (exit_l <= exit_r))   # push out to the left (nearer exit wins)
    use_r = pushed & away_r & ~use_l                           # ... or to the right
    alpha_min[use_l], alpha_max[use_l] = exit_l[use_l], far_l[use_l]
    alpha_min[use_r], alpha_max[use_r] = -far_r[use_r], -exit_r[use_r]
    stuck = pushed & ~use_l & ~use_r
    return alpha_min, alpha_max, pushed, stuck
