"""T4 -> MPC handoff: the racing line + velocity profile re-indexed by CENTRELINE arc length.

APEX's state estimate gives the car's Frenet state against perception's centreline
(StateEstimate.s_abs, e_y, e_psi). T3/T4 work on the racing line's own arc length, so the
MPC needs, at each centreline s, the racing-line point that projects there
(APEX Planning -> Control interface alignment, 2026-10-19, section 01):

    s_track_m        centreline arc length, same s as StateEstimate.s_abs (mod lap)
    e_y_ref_m        racing-line lateral offset from the centreline, + = left
    e_psi_ref_rad    racing-line heading minus centreline tangent heading, wrapped to (-pi, pi]
    kappa_ref_1pm    racing-line curvature at that physical point, + = left
    v_ref_mps        planned speed at that point
    a_ref_mps2       planned dv/dt along the racing line (see A_REF_SEMANTICS)
    s_race_m         racing-line arc length of the same point (traceability only)

The mapping is geometric: every racing-line point is projected onto the centreline
(closest point, tangent made exactly perpendicular), never "same number of metres".
The projection uses perception's own apex_track FrenetTrack when it is installed and its
track_hash matches track_meta.json, so s is bit-for-bit the s the MPC sees; otherwise a
periodic cubic spline through track.csv (mm-level agreement on closed1).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from t2_curvature.queryable_track import QueryableTrack

from .profile import VelocityProfile

FORMAT = "apex_planning_reference"
FORMAT_VERSION = 1
COLUMNS = ["s_track_m", "e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm", "v_ref_mps", "a_ref_mps2", "s_race_m"]
A_REF_SEMANTICS = ("dv/dt along the racing line from the velocity profile (kinematic). "
                   "Not Fx/m: no drag, rolling resistance or slope in it. Equals Fx/m only for a "
                   "drag-free point mass; conversion to be agreed with Control, or ignore the column.")
DENSE_DS = 0.02           # m, racing-line sampling used for the projection


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- projectors: (x, y) -> (s, e_y, theta)

class SplineProjector:
    """Centreline = periodic cubic spline through track.csv (T2's QueryableTrack, no smoothing)."""

    name = "track.csv periodic cubic spline"

    def __init__(self, track_csv: str | Path):
        self.track = QueryableTrack(str(track_csv), smoothing_sigma_m=0.0)
        self.L = self.track.length
        self._s = np.arange(0.0, self.L, 0.01)
        _, self._x, self._y, *_ = self.track.sample_arrays(self._s)

    def project(self, x, y):
        from scipy.spatial import cKDTree
        x, y = np.asarray(x, float), np.asarray(y, float)
        _, i = cKDTree(np.column_stack([self._x, self._y])).query(np.column_stack([x, y]))
        s = self._s[i]
        for _ in range(10):        # Newton on f(s) = (p - c(s)) . T(s) = 0, f'(s) = -(1 - kappa e_y)
            _, cx, cy, th, k, _, _ = self.track.sample_arrays(s)
            dx, dy = x - cx, y - cy
            f = dx * np.cos(th) + dy * np.sin(th)
            e_y = -dx * np.sin(th) + dy * np.cos(th)
            s = (s + np.clip(f / np.maximum(1.0 - k * e_y, 0.2), -0.05, 0.05)) % self.L
        _, cx, cy, th, _, _, _ = self.track.sample_arrays(s)
        e_y = -(x - cx) * np.sin(th) + (y - cy) * np.cos(th)
        return s, e_y, th


class ApexProjector:
    """Centreline = perception's apex_track FrenetTrack (the geometry StateEstimate uses)."""

    name = "apex_track FrenetTrack"

    def __init__(self, frenet_track):
        self.ft = frenet_track
        self.L = float(frenet_track.L)

    @classmethod
    def from_track_meta(cls, track_meta: dict) -> "ApexProjector | None":
        """None unless track.csv came from apex_track, the package imports and the hash matches."""
        yaml_path = track_meta.get("source_track_yaml")
        if not yaml_path or not str(track_meta.get("source", "")).startswith("perception"):
            return None
        try:
            from apex_track import load_track
            ft = load_track(yaml_path)
        except (ImportError, OSError, ValueError):
            return None
        if ft.track_hash != track_meta.get("map_version"):
            raise ValueError(f"apex_track track_hash {ft.track_hash} != track.csv map_version "
                             f"{track_meta.get('map_version')}: re-run T1 on the current track")
        return cls(ft)

    def project(self, x, y):
        out = np.empty((len(x), 3))
        for j, (xj, yj) in enumerate(zip(x, y)):
            self.ft.reset()
            _, s, e_y, e_psi, _ = self.ft.project(float(xj), float(yj), 0.0)
            out[j] = s, e_y, -e_psi                  # yaw 0 -> e_psi = -theta
        return out[:, 0], out[:, 1], out[:, 2]


# ---------------------------------------------------------------- build

def build_reference(racing_csv: str | Path, p: VelocityProfile, track_csv: str | Path,
                    track_meta: dict | None = None) -> tuple[np.ndarray, dict]:
    """(N, 7) array in COLUMNS order, one row per track.csv point, + validation stats."""
    track_meta = track_meta or {}
    projector = ApexProjector.from_track_meta(track_meta) or SplineProjector(track_csv)
    centre = np.loadtxt(track_csv, delimiter=",", skiprows=1, ndmin=2)
    s_track = centre[:, 0]
    L_c = projector.L

    # racing line as a smooth curve, sampled densely
    rl = np.loadtxt(racing_csv, delimiter=",", skiprows=1, ndmin=2)
    head = Path(racing_csv).read_text(encoding="utf-8").splitlines()[0].split(",")
    c = {n: i for i, n in enumerate(head)}
    race = QueryableTrack.from_arrays(rl[:, c["s"]], rl[:, c["x"]], rl[:, c["y"]],
                                      np.ones(len(rl)), np.ones(len(rl)))
    L_r = p.lap_length
    sig = np.linspace(0.0, L_r, int(np.ceil(L_r / DENSE_DS)), endpoint=False)
    _, rx, ry, rth, _, _, _ = race.sample_arrays(sig * race.length / L_r)

    s_c, e_y, th_c = projector.project(rx, ry)
    e_psi = _wrap(rth - th_c)

    # centreline s along the racing line must move forward only (one lap, no folds)
    u = s_c[0] + np.concatenate([[0.0], np.cumsum(_wrap((np.diff(s_c)) / L_c * 2 * np.pi) * L_c / (2 * np.pi))])
    step = np.diff(u)
    if np.any(step <= 0):
        k = int(np.argmin(step))
        raise ValueError(f"racing line goes backwards against the centreline near s_race = {sig[k]:.2f} m "
                         f"(1 - kappa e_y <= 0?); cannot index it by centreline s")
    closing = (s_c[0] + L_c) - u[-1]
    if not 0 < closing <= 3 * step.max():
        raise ValueError(f"racing line does not close one centreline lap (gap {closing:.3f} m)")
    u = np.append(u, s_c[0] + L_c)
    sig_ext = np.append(sig, L_r)

    # racing-line s of each centreline grid point, then every field at that s
    sig_k = np.interp((s_track - u[0]) % L_c + u[0], u, sig_ext)
    ey_k = np.interp(sig_k, sig_ext, np.append(e_y, e_y[0]))
    psi_k = np.interp(sig_k, sig_ext, np.unwrap(np.append(e_psi, e_psi[0])))
    per = dict(period=L_r)
    kap_k = np.interp(sig_k, p.s, p.kappa, **per)
    v_k = np.interp(sig_k, p.s, p.v_ref, **per)
    a_k = np.interp(sig_k, p.s, p.a_ref, **per)
    data = np.column_stack([s_track, ey_k, _wrap(psi_k), kap_k, v_k, a_k, np.mod(sig_k, L_r)])

    # checks: round trip, room to the walls, seam
    _, bx, by, *_ = race.sample_arrays(sig_k * race.length / L_r)
    s_back, ey_back, _ = projector.project(bx, by)
    rt_s = np.abs(_wrap((s_back - s_track) / L_c * 2 * np.pi) * L_c / (2 * np.pi))
    left_room = centre[:, 3] - ey_k
    right_room = centre[:, 4] + ey_k
    dv = np.abs(np.diff(np.append(v_k, v_k[0])))
    stats = {
        "projector": projector.name,
        "centreline_lap_length_m": L_c,
        "racing_line_lap_length_m": L_r,
        "round_trip_s_max_m": float(rt_s.max()),
        "round_trip_e_y_max_m": float(np.abs(ey_back - ey_k).max()),
        "e_y_range_m": [float(ey_k.min()), float(ey_k.max())],
        "e_psi_max_abs_rad": float(np.abs(data[:, 2]).max()),
        "min_room_left_m": float(left_room.min()),
        "min_room_right_m": float(right_room.min()),
        "seam_dv_mps": float(dv[-1]),
        "max_row_dv_mps": float(dv.max()),
    }
    return data, stats


def build_reference_meta(stats: dict, p: VelocityProfile, track_meta: dict, *, track_csv, racing_csv,
                         profile_csv=None) -> dict:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "s_axis": "centreline",
        "frame": track_meta.get("frame", "map"),
        "map_id": track_meta.get("map_id"),
        "track_hash": track_meta.get("map_version"),
        "boundary_hash": track_meta.get("boundary_hash"),
        "lap_length_m": stats["centreline_lap_length_m"],
        "n_points": int(len(np.loadtxt(track_csv, delimiter=",", skiprows=1, ndmin=2))),
        "columns": COLUMNS,
        "conventions": {
            "s_track_m": "centreline arc length, 0 at the track.csv start; StateEstimate.s_abs mod lap",
            "e_y_ref_m": "+ = left of the centreline",
            "e_psi_ref_rad": "racing-line heading - centreline tangent heading, (-pi, pi]",
            "kappa_ref_1pm": "racing-line curvature, + = left turn",
            "interpolation": "linear between rows, periodic in s_track_m",
        },
        "a_ref_semantics": A_REF_SEMANTICS,
        "vehicle": {**p.limits.__dict__, "a_lat_max": p.limits.a_lat_max},
        "lap_time_s": p.lap_time,
        "friction_use_peak": float(p.friction_use.max()),
        "validation": stats,
        "sources": {
            "track_csv": _sha(track_csv),
            "racing_line_csv": _sha(racing_csv),
            "velocity_profile_csv": _sha(profile_csv) if profile_csv else None,
        },
    }


def write_reference(out_dir: str | Path, data: np.ndarray, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "mpc_reference.csv", out_dir / "mpc_reference_meta.json"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f", header=",".join(COLUMNS), comments="")
    meta = {**meta, "sha256": _sha(csv_path)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path


# ---------------------------------------------------------------- loader for the MPC

class MPCReference:
    """Read mpc_reference.csv and look the reference up by the MPC's own s.

        ref = MPCReference.from_csv("out/.../t4/mpc_reference.csv", track_hash=track.track_hash,
                                    boundary_hash=track.boundary_hash)
        r = ref.sample(state.s_abs + v * dt * np.arange(9))   # dict of arrays, wraps mod lap
        speed_reference = ref                                  # ref(s) -> v_ref_mps
    """

    def __init__(self, data: np.ndarray, meta: dict):
        self.meta = meta
        self.lap_length = float(meta["lap_length_m"])
        self.columns = list(meta["columns"])
        self._s = np.append(data[:, 0], data[0, 0] + self.lap_length)
        cols = np.vstack([data, data[:1]])
        cols[:, self.columns.index("e_psi_ref_rad")] = np.unwrap(cols[:, self.columns.index("e_psi_ref_rad")])
        j = self.columns.index("s_race_m")
        cols[:, j] = np.unwrap(cols[:, j], period=meta["validation"]["racing_line_lap_length_m"])
        self._cols = cols

    @classmethod
    def from_csv(cls, csv_path: str | Path, meta_path: str | Path | None = None, *,
                 track_hash: str | None = None, boundary_hash: str | None = None) -> "MPCReference":
        """track_hash / boundary_hash: the MPC's own track; a mismatch raises ValueError."""
        csv_path = Path(csv_path)
        meta_path = Path(meta_path) if meta_path else csv_path.with_name("mpc_reference_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{meta_path}: not a {FORMAT} v{FORMAT_VERSION} file")
        if meta.get("sha256") and meta["sha256"] != _sha(csv_path):
            raise ValueError(f"{csv_path} does not match the sha256 in {meta_path.name}")
        for name, want in (("track_hash", track_hash), ("boundary_hash", boundary_hash)):
            if want is not None and meta.get(name) != want:
                raise ValueError(f"reference made for {name} {meta.get(name)}, MPC track has {want}")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        if data[0, 0] != 0.0 or np.any(np.diff(data[:, 0]) <= 0) or data[-1, 0] >= meta["lap_length_m"]:
            raise ValueError(f"{csv_path}: s_track_m must start at 0, increase and stay below the lap")
        return cls(data, meta)

    def sample(self, s_abs) -> dict[str, np.ndarray | float]:
        """Every column at centreline s (any s, e.g. s_abs over many laps). Scalar in, floats out."""
        s = np.mod(np.asarray(s_abs, dtype=float), self.lap_length)
        out = {}
        for j, name in enumerate(self.columns):
            v = np.interp(s, self._s, self._cols[:, j])
            if name == "e_psi_ref_rad":
                v = _wrap(v)
            elif name == "s_race_m":
                v = np.mod(v, self.meta["validation"]["racing_line_lap_length_m"])
            out[name] = float(v) if np.ndim(s_abs) == 0 else v
        out["s_track_m"] = float(s) if np.ndim(s_abs) == 0 else s
        return out

    def v_ref(self, s_abs):
        return self.sample(s_abs)["v_ref_mps"]

    def __call__(self, s_abs) -> float:
        return self.v_ref(s_abs)
