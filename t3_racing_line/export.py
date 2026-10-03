"""Step F: turn the last iteration into the final racing line, write it to
racing_line.csv + racing_line_meta.json, and load it back as a T2 QueryableTrack.

The smooth spline through the last round's points already exists
(IterateResult.reference); here we only
  1. put s = 0 on the centreline's start/finish line,
  2. resample every ~final_spacing metres (x, y, heading, kappa from the spline),
  3. re-measure w_left / w_right from the racing line to the fixed walls, with the
     same sideways walk as the loop (alpha_limits): w = room to the margin + margin.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from t2_curvature.queryable_track import QueryableTrack

from .min_curvature import IterateResult
from .stations import alpha_limits

FORMAT_VERSION = 1
COLUMNS = ["s", "x", "y", "heading", "kappa", "w_left", "w_right"]


@dataclass
class RacingLine:
    s: np.ndarray        # (M,) arc length along the RACING LINE, 0 ... length - ds
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray  # wrapped to (-pi, pi]
    kappa: np.ndarray    # from the spline, left positive
    w_left: np.ndarray   # racing line -> left: room until `margin` from a wall, + margin
    w_right: np.ndarray  # racing line -> right: same
    length: float        # lap length of the racing line

    def as_array(self) -> np.ndarray:
        return np.column_stack([self.s, self.x, self.y, self.heading, self.kappa,
                                self.w_left, self.w_right])


# ---------------------------------------------------------------- build

def _start_on_line(line, centerline, step: float = 0.01) -> float:
    """s on `line` where it crosses the centreline's start/finish line
    (the straight line through centreline s = 0, along its normal)."""
    _, x0, y0, psi0, *_ = centerline.sample_arrays(0.0)
    t0 = np.array([np.cos(psi0), np.sin(psi0)])            # driving direction at the start

    s = np.linspace(0.0, line.length, int(np.ceil(line.length / step)) + 1)  # last = first point
    _, x, y, *_ = line.sample_arrays(s)
    ahead = (x - x0) * t0[0] + (y - y0) * t0[1]            # signed distance past the start line
    beside = np.abs(-(x - x0) * t0[1] + (y - y0) * t0[0])  # distance along the start line

    # crossings from behind (<= 0) to ahead (> 0); the start line also cuts the
    # track elsewhere (other side of the loop), so keep the crossing nearest r0
    i = np.flatnonzero((ahead[:-1] <= 0) & (ahead[1:] > 0))
    if len(i) == 0:
        raise ValueError("racing line never crosses the centreline's start/finish line")
    i = i[np.argmin(beside[i])]
    frac = -ahead[i] / (ahead[i + 1] - ahead[i])           # linear interpolation inside the step
    return float((s[i] + frac * (s[i + 1] - s[i])) % line.length)


def build_racing_line(result: IterateResult, centerline, final_spacing: float = 0.10) -> RacingLine:
    """Final racing line from the iteration result, uniform ~final_spacing, s = 0 on the start line."""
    line = result.reference
    L = line.length
    n = int(round(L / final_spacing))
    s = np.arange(n) * (L / n)                             # no duplicated endpoint

    s0 = _start_on_line(line, centerline)
    _, x, y, heading, kappa, _, _ = line.sample_arrays(s0 + s)    # wraps modulo L
    heading = np.where(heading <= -np.pi, heading + 2 * np.pi, heading)  # (-pi, pi]

    xy = np.column_stack([x, y])
    normals = np.column_stack([-np.sin(heading), np.cos(heading)])
    # same walk as the loop, with the REAL margin: w = room + margin. On a straight
    # wall this equals the distance to it; at a hairpin tip it is the distance at
    # which the tip comes within the margin (a ray could miss the tip).
    alpha_min, alpha_max, _, _ = alpha_limits(xy, normals, result.walls, result.margin)
    w_left, w_right = alpha_max + result.margin, result.margin - alpha_min
    return RacingLine(s, x, y, heading, kappa, w_left, w_right, L)


# ---------------------------------------------------------------- write

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_meta(line: RacingLine, result: IterateResult, *, kappa_max: float | None,
               final_spacing: float, car_width: float | None = None,
               safety_buffer: float | None = None, source_meta: dict | None = None,
               source_csv: str | Path | None = None) -> dict:
    """racing_line_meta.json content (spec section 6.2). source_meta = T1's track_meta.json."""
    src = source_meta or {}
    return {
        "format_version": FORMAT_VERSION,
        "method": "min_curvature_qp",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map_id": src.get("map_id"),
        "map_version": src.get("map_version"),
        "frame": src.get("frame", "map"),
        "source_track_version": _sha256(source_csv) if source_csv else None,
        "lap_length": round(line.length, 6),
        "n_points": len(line.s),
        "columns": COLUMNS,
        "margin": result.margin,
        "plan_buffer": result.plan_buffer,
        "car_width": car_width,
        "safety_buffer": safety_buffer,
        "kappa_max": kappa_max,
        "station_spacing": result.stations.ds,
        "final_spacing": line.length / len(line.s),
        "final_spacing_target": final_spacing,
        "iterations_run": len(result.history),
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "final_max_alpha": result.history[-1].max_alpha,
    }


def write_racing_line(out_dir: str | Path, line: RacingLine, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "racing_line.csv", out_dir / "racing_line_meta.json"
    np.savetxt(csv_path, line.as_array(), delimiter=",", fmt="%.6f",
               header=",".join(COLUMNS), comments="")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path


# ---------------------------------------------------------------- load

def read_racing_line(csv_path: str | Path) -> np.ndarray:
    """(M, 7) array with COLUMNS."""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)


def load_racing_line(csv_path: str | Path, meta_path: str | Path | None = None) -> QueryableTrack:
    """The racing line as a T2 QueryableTrack (centreline = racing line, widths = corridor),
    so T4 / the MPC use it like any track. The meta dict is attached as `.meta`."""
    d = read_racing_line(csv_path)
    tm = QueryableTrack.from_arrays(d[:, 0], d[:, 1], d[:, 2], d[:, 5], d[:, 6])
    tm.meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if meta_path else {}
    return tm
