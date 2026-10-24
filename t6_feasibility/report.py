"""T6 report: open the reference the way the MPC does, run every full-lap check, write feasibility_report.json.

    checks, info, driven = gate_apex(".../t5/planning_reference.csv", track_yaml, ".../t4/velocity_profile_meta.json")
    write_report(".../t6/feasibility_report.json", checks, info)
    print_report(checks)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from t5_reference import PlanningReference

from . import checks as C

FORMAT_VERSION = 1


def gate(reference_csv: str | Path, c_s: np.ndarray, c_xy: np.ndarray, c_heading: np.ndarray,
         c_kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray, t4_meta_path: str | Path,
         track_hash: str | None = None, boundary_hash: str | None = None) -> tuple[list[C.Check], dict, C.Driven | None]:
    """Every check on one reference. c_* / w_* = the centre line and its corridor at every centre-line point.
    The file is opened with the MPC's own loader first: if that refuses it, nothing else is checked."""
    reference_csv = Path(reference_csv)
    info = {"reference": str(reference_csv.resolve()), "track_hash": track_hash, "boundary_hash": boundary_hash}
    try:
        ref = PlanningReference.from_csv(reference_csv, track_hash=track_hash, boundary_hash=boundary_hash)
    except ValueError as e:
        info["loader_error"] = str(e)
        return [C.Check("file opens with the MPC's loader", False, 0.0, 1.0, -1.0, 0.0, "-")], info, None
    table = np.loadtxt(reference_csv, delimiter=",", skiprows=1, ndmin=2)
    info["csv_sha256"] = ref.meta["csv_sha256"]
    checks = [C.Check("file opens with the MPC's loader", True, 1.0, 1.0, 0.0, 0.0, "-")]

    # the rows must sit exactly on the centre line's own points (T5 writes them there)
    off = np.abs(table[:, 0] - c_s).max() if len(table) == len(c_s) else np.inf
    checks.append(C.Check("rows on the centre line's s", bool(off <= 1e-5), float(off), 1e-5, 1e-5 - float(off),
                          0.0, "m"))
    if not checks[-1].passed:
        return checks, info, None

    # the limits T6 uses vs the ones T4 computed the speeds with (both should be control's shared numbers)
    t4 = json.loads(Path(t4_meta_path).read_text(encoding="utf-8"))
    mine = {"mu": C.MU, "a_accel_max": C.A_ACCEL_MAX, "a_brake_max": C.A_BRAKE_MAX, "v_max": C.V_MAX, "v_min": C.V_MIN}
    differ = [k for k, v in mine.items() if abs(t4["limits"].get(k, np.nan) - v) > 1e-9]
    info["limits_t4"], info["limits_differ"] = t4["limits"], differ
    checks.append(C.Check("T4 used the same limits", not differ, float(len(differ)), 0.0, -float(len(differ)),
                          0.0, "limits differing"))

    d = C.drive(table, c_xy, c_heading, c_kappa)
    checks += C.run_checks(d, ref.lap_length, np.asarray(w_left, float), np.asarray(w_right, float),
                           float(t4["lap_time"]))
    return checks, info, d


def gate_apex(reference_csv: str | Path, track_yaml: str | Path, t4_meta_path: str | Path,
              d_min: float | None = 0.1) -> tuple[list[C.Check], dict, C.Driven | None]:
    """The real case: perception's centre line, corner-cut walls and hashes from apex_track."""
    from apex_track import load_track
    tr = load_track(str(track_yaml))
    heading = np.array([tr.point(s)[2] for s in tr.s0])
    w_left, w_right = tr.width_at(tr.s0, d_min=d_min)
    checks, info, d = gate(reference_csv, tr.s0, tr.p, heading, tr.kappa_at(tr.s0), w_left, w_right, t4_meta_path,
                           tr.track_hash, tr.boundary_hash)
    info["centerline"] = f"apex_track {Path(track_yaml).resolve()} (d_min {d_min})"
    return checks, info, d


def gate_t2(reference_csv: str | Path, centerline_csv: str | Path,
            t4_meta_path: str | Path) -> tuple[list[C.Check], dict, C.Driven | None]:
    """Runs without apex_track: our own T2 centre line (centerline_t2.csv: s,x,y,heading,curvature,w_left,w_right)."""
    c = np.loadtxt(centerline_csv, delimiter=",", skiprows=1, ndmin=2)
    checks, info, d = gate(reference_csv, c[:, 0], c[:, 1:3], c[:, 3], c[:, 4], c[:, 5], c[:, 6], t4_meta_path)
    info["centerline"] = f"T2 {Path(centerline_csv).resolve()}"
    return checks, info, d


def write_report(path: str | Path, checks: list[C.Check], info: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": all(c.passed for c in checks),
        "n_checks": len(checks),
        "n_failed": sum(not c.passed for c in checks),
        "limits": {k: getattr(C, k) for k in ("MU", "G", "A_ACCEL_MAX", "A_BRAKE_MAX", "STEER_MAX", "STEER_RATE_MAX",
                                               "WHEELBASE", "V_MIN", "V_MAX", "WALL_MARGIN")},
        "limits_source": "control's shared synthetic vehicle (alignment v2, 2026-10-19); replace with vehicle_limits.yaml",
        "tolerances": {k: getattr(C, k) for k in ("SPACING_MAX", "TOL_SPACING", "TOL_FRICTION", "TOL_ACCEL", "TOL_WALL",
                                                   "D_MIN", "TOL_FRENET", "TOL_LAP_TIME")},
        "checks": [asdict(c) for c in checks],
        **info,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_report(checks: list[C.Check]) -> None:
    for c in checks:
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:34s} worst {c.worst:10.4f} {c.unit:16s} "
              f"limit {c.limit:9.4f}   margin {c.margin:+9.4f}   at s {c.at_s:7.2f} m")
    n_failed = sum(not c.passed for c in checks)
    print(f"  {'GATE PASSED' if not n_failed else f'GATE FAILED ({n_failed} of {len(checks)} checks)'}")
