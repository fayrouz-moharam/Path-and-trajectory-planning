"""T2 — track.csv -> a smooth, continuous, queryable track with curvature.

Input (from T1): track.csv with columns [s, x, y, w_left, w_right], metres,
map frame, following the driving direction, closed lap, s already close to
uniform arc-length spacing (T1's own "uniform_spacing" check enforces this).

Output: QueryableTrack, sampled at arbitrary s via a periodic cubic spline.
T3 (racing line), T4 (velocity profile), T5 (local trajectory) and T7
(obstacle avoidance) all query this for position, heading, curvature and
left/right width -- nothing downstream should touch track.csv directly.

Design mirrors the proven pattern documented in planning-racing-strategy-
technical-notes.md (apex.track / apex.coordinates.frenet): periodic cubic
spline in x(s), y(s), analytic curvature from the spline derivatives,
explicit widths queried separately from geometry. Kept dependency-free of
that other repo -- this is a standalone module for T2's own inputs/outputs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

REQUIRED_COLUMNS = ("s", "x", "y", "w_left", "w_right")


@dataclass(frozen=True)
class TrackSample:
    """One queried point on the track, SI units throughout."""

    s: float            # wrapped progress, [0, length)
    x: float             # m, map frame
    y: float             # m, map frame
    heading: float        # rad, atan2(dy, dx), counterclockwise from +x
    curvature: float      # 1/m, positive = left-hand bend
    w_left: float          # m, left half-width at s
    w_right: float         # m, right half-width at s


class TrackValidityError(ValueError):
    """track.csv failed a structural check before a spline was fit to it."""


class QueryableTrack:
    """Smooth, continuous, closed-lap track built from a T1 track.csv."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise TrackValidityError(f"track.csv missing required columns: {sorted(missing)}")
        if len(df) < 4:
            raise TrackValidityError("track.csv needs at least 4 points to fit a closed spline")

        s = df["s"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(s)) or not np.all(np.diff(s) > 0):
            raise TrackValidityError("s must be finite and strictly increasing")

        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)
        w_left = df["w_left"].to_numpy(dtype=np.float64)
        w_right = df["w_right"].to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise TrackValidityError("x, y must be finite")
        if np.any(w_left <= 0) or np.any(w_right <= 0):
            raise TrackValidityError("w_left, w_right must be strictly positive")

        # Close the lap: the wrap segment's length is the actual chord
        # distance from the last point back to the first, not an assumed
        # spacing. Using the input's median step here would silently distort
        # the spline right at the seam whenever the closing gap isn't
        # exactly one more nominal step (curvature/heading error was
        # measured >20x higher at the seam before this fix -- see
        # test_queryable_track.py's circle fixture).
        ds_close = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
        if ds_close <= 1e-9:
            raise TrackValidityError("first and last points coincide; cannot close the lap")
        self.length = float(s[-1] + ds_close)
        if self.length <= 0 or not np.isfinite(self.length):
            raise TrackValidityError("computed lap length is not positive/finite")

        # Periodic cubic spline needs the wrap point appended once (same
        # convention as the apex.track reference: ADR-018).
        s_ext = np.append(s, self.length)
        x_ext = np.append(x, x[0])
        y_ext = np.append(y, y[0])
        self._cx = CubicSpline(s_ext, x_ext, bc_type="periodic")
        self._cy = CubicSpline(s_ext, y_ext, bc_type="periodic")

        # Widths are interpolated, not splined: don't invent smoothness the
        # underlying measurement doesn't have.
        self._s_raw = s
        self._w_left = w_left
        self._w_right = w_right

    def _wrap(self, s: float) -> float:
        if not np.isfinite(s):
            raise TrackValidityError(f"s must be finite, got {s}")
        return float(s % self.length)

    def sample(self, s: float) -> TrackSample:
        sw = self._wrap(s)
        x = float(self._cx(sw))
        y = float(self._cy(sw))
        dx = float(self._cx(sw, 1))
        dy = float(self._cy(sw, 1))
        ddx = float(self._cx(sw, 2))
        ddy = float(self._cy(sw, 2))
        speed_sq = dx * dx + dy * dy
        if speed_sq <= 1e-12:
            raise TrackValidityError(f"degenerate tangent (near-zero speed) at s={sw}")
        heading = float(np.arctan2(dy, dx))
        curvature = float((dx * ddy - dy * ddx) / speed_sq**1.5)
        w_left = float(np.interp(sw, self._s_raw, self._w_left, period=self.length))
        w_right = float(np.interp(sw, self._s_raw, self._w_right, period=self.length))
        return TrackSample(sw, x, y, heading, curvature, w_left, w_right)

    def left_width(self, s: float) -> float:
        return self.sample(s).w_left

    def right_width(self, s: float) -> float:
        return self.sample(s).w_right

    def sample_many(self, s_values) -> list[TrackSample]:
        return [self.sample(s) for s in s_values]

    @classmethod
    def from_arrays(cls, s, x, y, w_left, w_right) -> "QueryableTrack":
        """Build the same track from arrays in memory instead of a track.csv file
        (e.g. T3's racing line between optimisation rounds). The arrays are
        written to an in-memory CSV buffer and read by the normal constructor,
        so the same checks and spline apply; nothing is written to disk.
        pandas' text round trip can change the last digit of a float
        (relative ~1e-15, i.e. ~1e-12 m on a 500 m track)."""
        buf = io.StringIO()
        pd.DataFrame({"s": s, "x": x, "y": y, "w_left": w_left, "w_right": w_right}).to_csv(
            buf, index=False, float_format="%.17g")
        buf.seek(0)
        return cls(buf)

    def sample_arrays(self, s_values):
        """Vectorised sample(): an array of s in, arrays out -- same maths as
        sample(), computed for all values at once (much faster than sample_many).
        Returns (s, x, y, heading, curvature, w_left, w_right), each shaped like s_values."""
        s_arr = np.asarray(s_values, dtype=np.float64)
        if not np.all(np.isfinite(s_arr)):
            raise TrackValidityError("s must be finite")
        sw = np.mod(s_arr, self.length)
        x, y = self._cx(sw), self._cy(sw)
        dx, dy = self._cx(sw, 1), self._cy(sw, 1)
        ddx, ddy = self._cx(sw, 2), self._cy(sw, 2)
        speed_sq = dx * dx + dy * dy
        if np.any(speed_sq <= 1e-12):
            bad = sw[speed_sq <= 1e-12]
            raise TrackValidityError(f"degenerate tangent (near-zero speed) at s={bad[:5].tolist()}")
        heading = np.arctan2(dy, dx)
        curvature = (dx * ddy - dy * ddx) / speed_sq**1.5
        w_left = np.interp(sw, self._s_raw, self._w_left, period=self.length)
        w_right = np.interp(sw, self._s_raw, self._w_right, period=self.length)
        return sw, x, y, heading, curvature, w_left, w_right
