"""T1 sanity checks (planning_task_breakdown.md T1 step 11: closed,
non-self-intersecting, inside free space, widths above minimum, length
matches test-lap distance, uniform spacing). Each check reports value +
threshold so failures are diagnosable from track_meta.json alone.

Severity:
  "error"   -- the track is unusable; makes "valid" false.
  "warning" -- a real property of the track that downstream tasks must
               handle; flagged with its location, doesn't block (breakdown:
               "every defect handled or flagged, none silent").

A wall that folds over itself happens where a real corner is sharper than
centerline + width can represent (inside width > turn radius). T1 flags
where; T2's smoothing spline and T3/T6's inside-width < 1/|kappa| check are
where it gets handled. Curvature itself is deliberately NOT computed here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import T1Config
from .extract import Track
from .geometry import self_intersections, tangents_normals


def _check(passed: bool, value, threshold, note: str, severity: str = "error") -> dict:
    return {"passed": bool(passed), "severity": severity, "value": value, "threshold": threshold, "note": note}


def _fold_locations(pairs: np.ndarray, s: np.ndarray, gap: int = 20) -> list[float]:
    """Arc-length positions (m) of each cluster of self-intersecting edges."""
    if len(pairs) == 0:
        return []
    n = len(s)
    i, j = pairs.min(axis=1), pairs.max(axis=1)
    # A fold's two crossing edges lie either side of the corner: locate it at their
    # midpoint along the loop (the short way round).
    mid = np.where(j - i <= n // 2, (i + j) // 2, ((i + j + n) // 2) % n)
    mid = np.sort(mid)
    clusters = np.split(mid, np.nonzero(np.diff(mid) > gap)[0] + 1)
    return [round(float(s[int(np.median(c))]), 1) for c in clusters]


def validate(track: Track, points: np.ndarray, cfg: T1Config) -> dict:
    """points: the accumulated world-frame boundary points the track was built from."""
    _, normals = tangents_normals(track.xy)
    left = track.xy + normals * track.w_left[:, None]
    right = track.xy - normals * track.w_right[:, None]
    min_w = float(min(track.w_left.min(), track.w_right.min()))
    clearance = float(cKDTree(points).query(track.xy)[0].min())
    lap_len = track.stats["test_lap_length_m"]
    length_mismatch = abs(track.length - lap_len) / lap_len
    spacing = np.linalg.norm(np.diff(np.vstack([track.xy, track.xy[:1]]), axis=0), axis=1)
    spacing_dev = float(np.abs(spacing - track.ds).max() / track.ds)
    folds_left = _fold_locations(self_intersections(left), track.s)
    folds_right = _fold_locations(self_intersections(right), track.s)
    fold_note = ("s (m) of each spot where the {} wall, rebuilt as centerline +/- width, folds over "
                 "itself: corner sharper than the format can represent; T2/T3/T6 must handle")

    return {
        "lap_closed": _check(
            track.stats["lap_closure_error_m"] <= cfg.closure_radius,
            round(track.stats["lap_closure_error_m"], 4), cfg.closure_radius,
            "distance between first pose and lap-closing pose (m)"),
        "min_width": _check(
            min_w >= cfg.min_width, round(min_w, 4), cfg.min_width,
            "smallest of w_left / w_right (m)"),
        "inside_free_space": _check(
            clearance >= cfg.min_width, round(clearance, 4), cfg.min_width,
            "smallest distance from any centerline point to any observed wall point (m)"),
        "gap_fraction_left": _check(
            track.stats["gap_fraction_left"] <= cfg.max_gap_fraction,
            round(track.stats["gap_fraction_left"], 4), cfg.max_gap_fraction,
            "fraction of stations with no left wall point (width interpolated)"),
        "gap_fraction_right": _check(
            track.stats["gap_fraction_right"] <= cfg.max_gap_fraction,
            round(track.stats["gap_fraction_right"], 4), cfg.max_gap_fraction,
            "fraction of stations with no right wall point (width interpolated)"),
        "length_matches_test_lap": _check(
            length_mismatch <= cfg.max_length_mismatch, round(length_mismatch, 4), cfg.max_length_mismatch,
            "relative difference between centerline length and driven test-lap length"),
        "uniform_spacing": _check(
            spacing_dev <= cfg.max_spacing_deviation, round(spacing_dev, 4), cfg.max_spacing_deviation,
            "max |spacing - ds| / ds"),
        "centerline_simple": _check(
            len(self_intersections(track.xy)) == 0, None, None, "centerline does not self-intersect"),
        "left_wall_folds": _check(not folds_left, folds_left, None, fold_note.format("left"), "warning"),
        "right_wall_folds": _check(not folds_right, folds_right, None, fold_note.format("right"), "warning"),
    }


def is_valid(report: dict) -> bool:
    """Only errors make a track invalid; warnings are reported, not blocking."""
    return all(c["passed"] for c in report.values() if c["severity"] == "error")


def warnings(report: dict) -> list[str]:
    return [name for name, c in report.items() if c["severity"] == "warning" and not c["passed"]]
