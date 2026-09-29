"""Closed-polyline helpers. A closed polyline is an (N, 2) array whose last
point connects back to the first (the first point is NOT repeated)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def closed_arclength(pts: np.ndarray) -> np.ndarray:
    """Cumulative length at each vertex plus the closing one: shape (N+1,), last = perimeter."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def dedupe(pts: np.ndarray, min_dist: float) -> np.ndarray:
    """Drop vertices closer than min_dist to the previously kept one (e.g. car standing still)."""
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= min_dist:
            keep.append(i)
    return pts[keep]


def resample_closed(pts: np.ndarray, ds: float) -> np.ndarray:
    """Resample to uniform spacing (as close to ds as divides the perimeter), keeping pts[0]."""
    s = closed_arclength(pts)
    n = max(int(round(s[-1] / ds)), 3)
    s_new = np.arange(n) * (s[-1] / n)
    closed = np.vstack([pts, pts[:1]])
    return np.column_stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])])


def smooth_closed(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return values.copy()
    return gaussian_filter1d(values, sigma_samples, axis=0, mode="wrap")


def tangents_normals(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents (direction of increasing index) and left normals."""
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def is_simple_closed(pts: np.ndarray) -> bool:
    """True if no two non-adjacent edges of the closed polyline intersect."""
    return len(self_intersections(pts)) == 0


def self_intersections(pts: np.ndarray) -> np.ndarray:
    """(K, 2) index pairs (i, j) of non-adjacent edges that cross or touch;
    edge i runs from pts[i] to pts[i+1] (wrapping)."""
    n = len(pts)
    a, b = pts, np.roll(pts, -1, axis=0)
    seg_len = np.linalg.norm(b - a, axis=1)
    # Two segments can only touch if their midpoints are within max segment length.
    pairs = cKDTree((a + b) / 2).query_pairs(r=seg_len.max(), output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]
    adjacent = (np.abs(i - j) == 1) | (np.abs(i - j) == n - 1)
    i, j = i[~adjacent], j[~adjacent]

    def orient(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    o1 = orient(a[i], b[i], a[j])
    o2 = orient(a[i], b[i], b[j])
    o3 = orient(a[j], b[j], a[i])
    o4 = orient(a[j], b[j], b[i])
    # Crossing or touching (a vertex lying on another edge also makes the loop non-simple).
    hit = (o1 * o2 <= 0) & (o3 * o4 <= 0)
    collinear = (o1 == 0) & (o2 == 0)
    if collinear.any():
        lo_i, hi_i = np.minimum(a[i], b[i]), np.maximum(a[i], b[i])
        lo_j, hi_j = np.minimum(a[j], b[j]), np.maximum(a[j], b[j])
        overlap = np.all((lo_i <= hi_j) & (lo_j <= hi_i), axis=1)
        hit = np.where(collinear, overlap, hit)
    return np.column_stack([i[hit], j[hit]])
