"""T4 in / out: read T3's racing line, write velocity_profile.csv + velocity_profile_meta.json,
and hand the profile to the MPC as a speed_reference(track_s) -> float callable."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .profile import COLUMNS, VelocityProfile

FORMAT_VERSION = 1
APEX_SPEED_BOUNDS = (0.5, 6.0)   # shared synthetic speed domain agreed with Control (2026-10-19)


# ---------------------------------------------------------------- read T3

def read_t3(csv_path: str | Path, meta_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """s, kappa, lap_length, meta from T3's racing_line.csv (+ racing_line_meta.json)."""
    csv_path = Path(csv_path)
    meta_path = Path(meta_path) if meta_path else csv_path.with_name("racing_line_meta.json")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    d = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    col = {name: i for i, name in enumerate(header)}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    s, kappa = d[:, col["s"]], d[:, col["kappa"]]
    lap = float(meta.get("lap_length") or (s[-1] + (s[-1] - s[0]) / (len(s) - 1)))
    return s, kappa, lap, meta


# ---------------------------------------------------------------- write

def build_meta(p: VelocityProfile, t3_meta: dict, t3_csv: str | Path | None = None,
               kappa_sigma_m: float = 0.0) -> dict:
    lo, hi = APEX_SPEED_BOUNDS
    return {
        "format_version": FORMAT_VERSION,
        "method": "friction_ellipse_forward_backward",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "s_axis": "racing_line",          # NOT the centreline s the MPC uses (that is t5_reference's planning_reference.csv)
        "map_id": t3_meta.get("map_id"),
        "map_version": t3_meta.get("map_version"),
        "source_track_version": t3_meta.get("source_track_version"),
        "source_racing_line": hashlib.sha256(Path(t3_csv).read_bytes()).hexdigest() if t3_csv else None,
        "t3_converged": t3_meta.get("converged"),
        "columns": COLUMNS,
        "lap_length": round(p.lap_length, 6),
        "n_points": len(p.s),
        "ds": p.lap_length / len(p.s),
        "lap_time": round(p.lap_time, 4),
        "v_min": float(p.v_ref.min()),
        "v_max": float(p.v_ref.max()),
        "v_mean": p.lap_length / p.lap_time,
        "a_ref_min": float(p.a_ref.min()),
        "a_ref_max": float(p.a_ref.max()),
        "a_lat_peak": float(np.abs(p.a_lat).max()),
        "friction_use_peak": float(p.friction_use.max()),
        "kappa_sigma_m": kappa_sigma_m,
        "limits": asdict(p.limits),
        "apex_speed_bounds": list(APEX_SPEED_BOUNDS),
        "inside_apex_bounds": bool(p.v_ref.min() >= lo and p.v_ref.max() <= hi),
    }


def write_profile(out_dir: str | Path, p: VelocityProfile, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "velocity_profile.csv", out_dir / "velocity_profile_meta.json"
    np.savetxt(csv_path, p.as_array(), delimiter=",", fmt="%.6f", header=",".join(COLUMNS), comments="")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path


# ---------------------------------------------------------------- hand to the MPC

class SpeedReference:
    """speed_reference(track_s) -> v [m/s], same contract as apex's ConstantSpeed.

    Linear interpolation on the profile's grid, wrapping modulo lap_length.
    `accel(track_s)` gives a_ref the same way (feed-forward for the MPC's a_cmd).
    """

    def __init__(self, s: np.ndarray, v: np.ndarray, a: np.ndarray, lap_length: float):
        self.lap_length = float(lap_length)
        self._s = np.append(s, lap_length)       # close the lap for interpolation
        self._v = np.append(v, v[0])
        self._a = np.append(a, a[0])

    @classmethod
    def from_profile(cls, p: VelocityProfile) -> "SpeedReference":
        return cls(p.s, p.v_ref, p.a_ref, p.lap_length)

    @classmethod
    def from_csv(cls, csv_path: str | Path, meta_path: str | Path | None = None) -> "SpeedReference":
        csv_path = Path(csv_path)
        meta_path = Path(meta_path) if meta_path else csv_path.with_name("velocity_profile_meta.json")
        d = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        lap = json.loads(meta_path.read_text(encoding="utf-8"))["lap_length"]
        return cls(d[:, COLUMNS.index("s")], d[:, COLUMNS.index("v_ref")], d[:, COLUMNS.index("a_ref")], lap)

    def __call__(self, track_s: float) -> float:
        return float(np.interp(float(track_s) % self.lap_length, self._s, self._v))

    def accel(self, track_s: float) -> float:
        return float(np.interp(float(track_s) % self.lap_length, self._s, self._a))
