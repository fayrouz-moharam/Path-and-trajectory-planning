"""T5 in / out: perception's centre line + T3 racing line + T4 speeds -> planning_reference.csv
+ planning_reference_meta.json, the file the MPC reads (rows indexed by the centre line's s)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mapping import check_reference, cross_racing_line, sample_at_crossings

FORMAT_VERSION = 1
COLUMNS = ["s_track_m", "e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm", "v_ref_mps"]


# ---------------------------------------------------------------- step A: the centre line

@dataclass
class Centerline:
    """The centre-line points every row is written at (the MPC's Frenet frame)."""
    s: np.ndarray            # (N,) the centre line's own s -- never recomputed
    xy: np.ndarray           # (N, 2)
    heading: np.ndarray      # (N,) tangent angle; e_psi is measured against it
    w_left: np.ndarray       # (N,) corridor to the left / right wall
    w_right: np.ndarray
    lap_length: float
    track_id: str | None = None
    track_hash: str | None = None
    boundary_hash: str | None = None
    source: str = ""


def centerline_from_apex(track_yaml: str | Path, d_min: float | None = 0.1) -> Centerline:
    """Perception's centre line from apex_track: the same s as StateEstimate.s_abs."""
    from apex_track import load_track
    tr = load_track(str(track_yaml))
    heading = np.array([tr.point(s)[2] for s in tr.s0])        # the smooth tangent project() uses
    w_left, w_right = tr.width_at(tr.s0, d_min=d_min)
    return Centerline(tr.s0.copy(), tr.p.copy(), heading, w_left, w_right, tr.L, tr.track_id,
                      tr.track_hash, tr.boundary_hash, f"apex_track {Path(track_yaml).resolve()} (d_min {d_min})")


def centerline_from_t2(csv_path: str | Path) -> Centerline:
    """Our own T2 centre line (centerline_t2.csv), for runs without apex_track."""
    d = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)   # s,x,y,heading,curvature,w_left,w_right
    lap = d[-1, 0] + float(np.hypot(*(d[0, 1:3] - d[-1, 1:3])))
    return Centerline(d[:, 0], d[:, 1:3], d[:, 3], d[:, 5], d[:, 6], lap,
                      source=f"T2 {Path(csv_path).resolve()}")


# ---------------------------------------------------------------- steps B-E + checks

def build_reference(cl: Centerline, racing_csv: str | Path,
                    velocity_csv: str | Path) -> tuple[np.ndarray, dict, list[str], dict]:
    """(N, 5) table in COLUMNS order, the check report, the problems, and the T3 / T4 metas."""
    racing_csv, velocity_csv = Path(racing_csv), Path(velocity_csv)
    rl = np.loadtxt(racing_csv, delimiter=",", skiprows=1, ndmin=2)     # s,x,y,heading,kappa,w_left,w_right
    vp = np.loadtxt(velocity_csv, delimiter=",", skiprows=1, ndmin=2)   # s,kappa,v_lat,v_ref,a_ref,a_lat,t
    rl_meta = json.loads(racing_csv.with_name("racing_line_meta.json").read_text(encoding="utf-8"))
    vp_meta = json.loads(velocity_csv.with_name("velocity_profile_meta.json").read_text(encoding="utf-8"))
    if len(rl) != len(vp) or not np.allclose(rl[:, 0], vp[:, 0], atol=1e-5):
        raise ValueError("velocity_profile.csv is not on racing_line.csv's s grid -- rerun T4")
    if cl.track_hash and rl_meta.get("map_version") != cl.track_hash:
        raise ValueError(f"the racing line was made for track {rl_meta.get('map_version')}, "
                         f"the centre line is track {cl.track_hash}")

    r_lap = float(rl_meta["lap_length"])
    e_y, seg, frac = cross_racing_line(cl.xy, cl.heading, cl.w_left, cl.w_right, rl[:, 1:3])
    s_race, e_psi, kappa, v = sample_at_crossings(seg, frac, rl[:, 0], rl[:, 3], rl[:, 4], r_lap,
                                                  vp[:, 3], cl.heading)
    report, problems = check_reference(cl.xy, cl.heading, cl.w_left, cl.w_right, e_y, s_race,
                                       rl[:, 1:3], r_lap, float(rl_meta["margin"]))
    table = np.column_stack([cl.s, e_y, e_psi, kappa, v])
    return table, report, problems, {"racing_line": rl_meta, "velocity_profile": vp_meta}


# ---------------------------------------------------------------- write

def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_meta(cl: Centerline, report: dict, problems: list[str], metas: dict,
               racing_csv: str | Path, velocity_csv: str | Path) -> dict:
    rl_meta, vp_meta = metas["racing_line"], metas["velocity_profile"]
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": COLUMNS,
        "s_axis": "centre line: perception's own s (= StateEstimate.s_abs modulo lap_length)",
        "sign_convention": "e_y_ref, e_psi_ref, kappa_ref positive to the LEFT; "
                           "e_psi_ref = racing-line heading - centre-line heading, in (-pi, pi]",
        "track_id": cl.track_id,
        "track_hash": cl.track_hash,
        "boundary_hash": cl.boundary_hash,
        "centerline_source": cl.source,
        "lap_length": round(cl.lap_length, 6),
        "n_points": len(cl.s),
        "spacing_max": float(np.max(np.diff(np.append(cl.s, cl.lap_length)))),
        "source_racing_line": _sha256(racing_csv),
        "source_velocity_profile": _sha256(velocity_csv),
        "racing_line_length": rl_meta["lap_length"],
        "margin": rl_meta["margin"],
        "vehicle_limits": vp_meta.get("limits"),
        "vehicle_config_hash": None,           # no shared vehicle config file yet (Task 1)
        "a_ref": "not exported: the MPC reads a_ref as Fx/m; meaning still to be agreed",
        "checks": report,
        "problems": problems,
        "valid": not problems,
    }


def write_reference(out_dir: str | Path, table: np.ndarray, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "planning_reference.csv", out_dir / "planning_reference_meta.json"
    np.savetxt(csv_path, table, delimiter=",", fmt="%.6f", header=",".join(COLUMNS), comments="")
    meta = {**meta, "csv_sha256": _sha256(csv_path)}      # fingerprint of the file just written; the loader checks it
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path
