"""Stage 2 — turn the accumulated point cloud into a closed, ordered
centerline with left/right widths, guided by the car's own pose trace.

1. Cut one lap out of the pose trace; smooth + resample it -> reference line.
   It is closed, ordered, and runs in the driving direction by construction.
2. Project every boundary point onto the reference: the sign of its lateral
   offset splits the unordered scatter into left / right wall points.
3. Per station, a robust statistic of those offsets gives the distance to
   each wall (median-filtered for spikes only -- smoothing is T2's job);
   the reference is shifted to the midpoint -> centerline.
4. Repeat 2-3 with the centerline as the reference (cfg.iterations times),
   then measure final widths against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from .config import T1Config
from .geometry import closed_arclength, dedupe, resample_closed, smooth_closed, tangents_normals


class ExtractionError(RuntimeError):
    pass


@dataclass
class WidthProfile:
    w_left: np.ndarray
    w_right: np.ndarray
    gap_fraction_left: float    # stations where no left wall point was seen (interpolated)
    gap_fraction_right: float
    n_points_used: int


@dataclass
class Track:
    s: np.ndarray        # (N,) arc length from start, metres
    xy: np.ndarray       # (N, 2) centerline, closed (last connects to first)
    w_left: np.ndarray   # (N,) distance to the wall on the left of the driving direction
    w_right: np.ndarray  # (N,)
    length: float        # perimeter
    ds: float            # actual uniform spacing (length / N)
    stats: dict = field(default_factory=dict)


def extract_lap(xy: np.ndarray, closure_radius: float, min_lap_length: float) -> tuple[np.ndarray, float]:
    """Return the first full lap of the pose trace (closing point excluded) and the closure gap."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    d0 = np.linalg.norm(xy - xy[0], axis=1)
    cand = (cum >= min_lap_length) & (d0 <= closure_radius)
    if not cand.any():
        far = cum >= min_lap_length
        closest = f"{d0[far].min():.2f} m" if far.any() else "n/a (trace too short)"
        raise ExtractionError(
            f"pose trace never returns within {closure_radius} m of its start after "
            f"{min_lap_length} m of travel (closest return: {closest}); is the test lap complete?"
        )
    first = int(np.argmax(cand))
    last = first
    while last + 1 < len(cand) and cand[last + 1]:
        last += 1
    end = first + int(np.argmin(d0[first:last + 1]))
    return xy[:end], float(d0[end])


def build_reference(lap: np.ndarray, cfg: T1Config) -> np.ndarray:
    ref = resample_closed(dedupe(lap, cfg.ds / 2), cfg.ds)
    ref = smooth_closed(ref, cfg.ref_smoothing / cfg.ds)
    return resample_closed(ref, cfg.ds)


def despike_closed(values: np.ndarray, window_samples: float) -> np.ndarray:
    """Circular median filter: removes width spikes without smoothing the shape (T2 does that)."""
    size = max(1, int(round(window_samples))) | 1  # odd
    return median_filter(values, size=size, mode="wrap")


def _fill_circular(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs, wrapping around the loop."""
    ok = ~np.isnan(values)
    if ok.all():
        return values
    n = len(values)
    idx = np.arange(n)
    xp = np.concatenate([idx[ok] - n, idx[ok], idx[ok] + n])
    fp = np.tile(values[ok], 3)
    return np.interp(idx, xp, fp)


def lateral_profile(ref: np.ndarray, points: np.ndarray, cfg: T1Config) -> WidthProfile:
    n = len(ref)
    _, normals = tangents_normals(ref)
    dist, idx = cKDTree(ref).query(points)
    lat = np.einsum("ij,ij->i", points - ref[idx], normals[idx])
    keep = (dist <= cfg.max_half_width) & (lat != 0)
    idx, lat = idx[keep], lat[keep]

    half_win = int(round(cfg.station_window / cfg.ds))
    out = []
    gaps = []
    for side_mask in (lat > 0, lat < 0):
        side_idx, side_off = idx[side_mask], np.abs(lat[side_mask])
        order = np.argsort(side_idx, kind="stable")
        side_idx, side_off = side_idx[order], side_off[order]
        bounds = np.searchsorted(side_idx, np.arange(n + 1))
        per_station = [side_off[bounds[i]:bounds[i + 1]] for i in range(n)]
        w = np.full(n, np.nan)
        for i in range(n):
            vals = np.concatenate([per_station[(i + k) % n] for k in range(-half_win, half_win + 1)])
            if len(vals):
                w[i] = np.percentile(vals, cfg.width_percentile)
        gap = float(np.isnan(w).mean())
        if gap == 1.0:
            side = "left" if len(out) == 0 else "right"
            raise ExtractionError(f"no boundary points found on the {side} side of the reference line")
        out.append(despike_closed(_fill_circular(w), cfg.width_median_window / cfg.ds))
        gaps.append(gap)

    return WidthProfile(out[0], out[1], gaps[0], gaps[1], int(keep.sum()))


def extract_track(poses: np.ndarray, points: np.ndarray, cfg: T1Config) -> Track:
    if len(points) == 0:
        raise ExtractionError("no boundary points after accumulation")
    lap, closure_error = extract_lap(poses[:, :2], cfg.closure_radius, cfg.min_lap_length)
    ref = build_reference(lap, cfg)

    for _ in range(cfg.iterations):
        prof = lateral_profile(ref, points, cfg)
        _, normals = tangents_normals(ref)
        centered = ref + normals * ((prof.w_left - prof.w_right) / 2)[:, None]
        ref = resample_closed(smooth_closed(centered, cfg.destaircase / cfg.ds), cfg.ds)

    prof = lateral_profile(ref, points, cfg)
    s = closed_arclength(ref)
    return Track(
        s=s[:-1],
        xy=ref,
        w_left=prof.w_left,
        w_right=prof.w_right,
        length=float(s[-1]),
        ds=float(s[-1] / len(ref)),
        stats={
            "lap_closure_error_m": closure_error,
            "lap_pose_samples": len(lap),
            "test_lap_length_m": float(closed_arclength(lap)[-1]),
            "gap_fraction_left": prof.gap_fraction_left,
            "gap_fraction_right": prof.gap_fraction_right,
            "n_points_used": prof.n_points_used,
        },
    )
