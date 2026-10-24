"""Run T2 + T3 + T4 on perception's track: apex_track track folder -> track.csv (instead of T1)
-> spline + curvature (T2) -> minimum-curvature racing line (T3) -> velocity profile (T4).

    1. pip install -e <grad-perception>/apex_track   (once)
    2. edit the SETTINGS below
    3. python run_perception_t2_t3_t4.py

T1 part = t1_track.perception_out: perception's centre line + walls -> track/track.csv + track_meta.json.
T2 part = run_t1_t2.run_t2() unchanged, smoothing OFF: apex_track already smoothed the line, and
          smoothing it again would move our s away from the s in perception's StateEstimate.
T3 part = run_t1_t2_t3.run_t3() unchanged (car settings CAR_WIDTH, SAFETY_BUFFER, ... live there).
T4 part = run_t1_t2_t3_t4.run_t4() unchanged (vehicle limits live there).
T5 part = t5_reference: the racing line + speeds seen from perception's centre line ->
          t5/planning_reference.csv + planning_reference_meta.json (+ t5.png), the file the MPC reads.
T6 part = t6_feasibility: the full-lap gate -- can control's agreed car drive that file? ->
          t6/feasibility_report.json (+ t6.png). Limits are control's synthetic values (t6_feasibility/checks.py).

Exit code: 0 = all fine, 2 = no track (apex_track not installed / bad track folder),
3 = T2 rejected the track, 4 = T3 did not settle, 5 = speed profile leaves the shared 0.5-6.0 m/s
domain, 6 = planning_reference failed its self-checks (files still written, meta "valid": false),
7 = the full-lap gate failed (the report says which check, the worst value and where).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import run_t1_t2
import run_t1_t2_t3
import run_t1_t2_t3_t4
from t1_track.perception_out import load_perception_track, write_perception_track
from t5_reference import build_meta, build_reference, centerline_from_apex, write_reference
from t6_feasibility import gate_apex, margins, print_report, write_report

# ============================================================== SETTINGS
TRACK = "closed1"             # track folder inside apex_track/tracks/ ("closed1", "levine", later our own)
TRACK_YAML = None             # or a full path to a track.yaml elsewhere (overrides TRACK)
D_MIN = 0.1                   # apex_track's tight-corner limit on the inside walls (None = raw walls)
OUT_DIR = Path(f"out/run_perception_{TRACK}")
PLOT = True
# =======================================================================


def find_track_yaml(track: str) -> Path:
    """apex_track/tracks/<track>/track.yaml next to the installed (pip install -e) package."""
    import apex_track
    return Path(apex_track.__file__).resolve().parents[1] / "tracks" / track / "track.yaml"


def run_perception(out_dir: Path, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
                   d_min: float | None = D_MIN) -> int:
    """Perception's track -> out_dir/track.csv + track_meta.json (the T1 step)."""
    try:
        yaml_path = Path(track_yaml) if track_yaml else find_track_yaml(track)
        data, info = load_perception_track(yaml_path, d_min=d_min)
    except (ImportError, OSError, ValueError) as e:
        print(f"no track from perception: {e}", file=sys.stderr)
        return 2
    csv_path, meta_path = write_perception_track(out_dir, data, info, yaml_path)
    print(f"  track {info['track_id']} from {yaml_path}")
    print(f"  {info['n_points']} points, lap {info['lap_length']:.2f} m, "
          f"track_hash {info['track_hash']}, boundary_hash {info['boundary_hash']}")
    print(f"  inside walls cut by d_min {d_min}: {info['n_cut']} points; "
          f"virtual walls: {info['n_virtual_left']} left, {info['n_virtual_right']} right")
    print(f"wrote {csv_path} and {meta_path}")
    return 0


def run(out_dir: Path = OUT_DIR, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
        d_min: float | None = D_MIN, plot: bool = PLOT) -> int:
    out_dir = Path(out_dir)
    print("===== T1: track from perception (apex_track)")
    if run_perception(out_dir / "track", track=track, track_yaml=track_yaml, d_min=d_min):
        return 2
    track_csv = out_dir / "track" / "track.csv"

    print("\n===== T2: spline + curvature (no extra smoothing)")
    t2_code = run_t1_t2.run_t2(track_csv, out_dir / "t2", sigma=0.0, plot=plot)
    if t2_code:
        return t2_code

    print("\n===== T3: minimum-curvature racing line")
    t3_code = run_t1_t2_t3.run_t3(track_csv, out_dir / "t3", sigma=0.0, plot=plot)

    print("\n===== T4: velocity profile")
    t4_code = run_t1_t2_t3_t4.run_t4(out_dir / "t3", out_dir / "t4", plot=plot)

    print("\n===== T5: planning reference on perception's centre line")
    t5_code = run_t5(out_dir, track=track, track_yaml=track_yaml, d_min=d_min, plot=plot)

    print("\n===== T6: full-lap gate (control's synthetic car)")
    t6_code = run_t6(out_dir, track=track, track_yaml=track_yaml, d_min=d_min, plot=plot)
    return t5_code or t6_code or t3_code or t4_code


def run_t6(run_dir: Path, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
           d_min: float | None = D_MIN, plot: bool = PLOT) -> int:
    """run_dir/t5/planning_reference.csv -> every full-lap check -> run_dir/t6/feasibility_report.json."""
    yaml_path = Path(track_yaml) if track_yaml else find_track_yaml(track)
    checks, info, driven = gate_apex(run_dir / "t5" / "planning_reference.csv", yaml_path,
                                     run_dir / "t4" / "velocity_profile_meta.json", d_min=d_min)
    print_report(checks)
    path = write_report(run_dir / "t6" / "feasibility_report.json", checks, info)
    print(f"wrote {path}")
    if plot and driven is not None:
        plot_t6(run_dir / "t6" / "t6.png", driven, yaml_path, d_min)
        print(f"plot -> {run_dir / 't6' / 't6.png'}")
    return 0 if all(c.passed for c in checks) else 7


def plot_t6(path: Path, d, track_yaml: Path, d_min: float | None) -> None:
    """Each limit along the lap: grip used, acceleration, steering rate, room to the walls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from apex_track import load_track

    from t6_feasibility import checks as C

    tr = load_track(str(track_yaml))
    w_left, w_right = tr.width_at(tr.s0, d_min=d_min)
    m = margins(d, tr.L, w_left, w_right)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    a = ax[0, 0]
    a.plot(d.s, m["grip"], lw=1.2)
    a.axhline(1.0, color="tab:red", lw=1, label="limit")
    a.axhspan(1.0, 1.0 + C.TOL_FRICTION, color="tab:red", alpha=0.1, label=f"tolerance {C.TOL_FRICTION:.0%}")
    a.set(ylabel="grip used (friction ellipse)", title="grip: speeding up / braking and cornering together")
    a.legend(loc="lower left", fontsize=8)

    a = ax[0, 1]
    a.plot(d.s, d.a_lon, lw=1.2)
    a.axhline(C.A_ACCEL_MAX, color="tab:red", lw=1, label=f"+{C.A_ACCEL_MAX} / -{C.A_BRAKE_MAX} m/s^2")
    a.axhline(-C.A_BRAKE_MAX, color="tab:red", lw=1)
    a.set(ylabel="a_lon [m/s^2]  (+ speeding up)", title="acceleration and braking along the racing line")
    a.legend(loc="lower left", fontsize=8)

    a = ax[1, 0]
    a.plot(d.s, np.abs(d.steer_rate), lw=1.2, label="|steering rate|")
    a.axhline(C.STEER_RATE_MAX, color="tab:red", lw=1, label=f"slew limit {C.STEER_RATE_MAX} rad/s")
    a.plot(d.s, np.abs(d.steer), lw=1.0, color="tab:orange", label="|steering angle| [rad]")
    a.axhline(C.STEER_MAX, color="tab:orange", ls="--", lw=1, label=f"angle limit {C.STEER_MAX} rad")
    a.set(xlabel="s_track [m] (perception's centre line)", ylabel="rad/s  |  rad", title="steering")
    a.legend(loc="upper left", fontsize=8)

    a = ax[1, 1]
    a.plot(d.s, m["room"], lw=1.2)
    a.axhline(0.0, color="tab:red", lw=1, label=f"walls - {C.WALL_MARGIN} m margin")
    a.set(xlabel="s_track [m] (perception's centre line)", ylabel="room [m]",
          title="room left to the nearer wall (after the margin)")
    a.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_t5(run_dir: Path, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
           d_min: float | None = D_MIN, plot: bool = PLOT) -> int:
    """T3 racing line + T4 speeds -> run_dir/t5/planning_reference.csv on perception's centre-line s."""
    yaml_path = Path(track_yaml) if track_yaml else find_track_yaml(track)
    cl = centerline_from_apex(yaml_path, d_min=d_min)
    racing_csv, velocity_csv = run_dir / "t3" / "racing_line.csv", run_dir / "t4" / "velocity_profile.csv"
    table, report, problems, metas = build_reference(cl, racing_csv, velocity_csv)
    meta = build_meta(cl, report, problems, metas, racing_csv, velocity_csv)
    csv_path, meta_path = write_reference(run_dir / "t5", table, meta)

    s, e_y, e_psi, kappa, v = table.T
    print(f"  {len(s)} rows, every <= {meta['spacing_max']:.3f} m of the centre line ({cl.lap_length:.2f} m), "
          f"track_hash {cl.track_hash}")
    print(f"  e_y_ref {e_y.min():+.3f} .. {e_y.max():+.3f} m, e_psi_ref {np.degrees(e_psi).min():+.1f} .. "
          f"{np.degrees(e_psi).max():+.1f} deg, kappa_ref {kappa.min():+.3f} .. {kappa.max():+.3f} 1/m, "
          f"v_ref {v.min():.2f} .. {v.max():.2f} m/s")
    print(f"  checks: s_race wraps {report['s_race_wraps']}x (smallest step {report['s_race_min_step'] * 100:.1f} cm), "
          f"lands on the racing line within {report['land_max_miss_m'] * 1000:.2f} mm, "
          f"closest to the corridor edge {report['corridor_min_room_m'] * 1000:+.1f} mm, "
          f"start offset {report['start_offset_m'] * 1000:.1f} mm")
    print(f"wrote {csv_path} and {meta_path}")
    if plot:
        plot_t5(run_dir / "t5" / "t5.png", cl, racing_csv, table, meta["margin"])
        print(f"plot -> {run_dir / 't5' / 't5.png'}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return 6 if problems else 0


def plot_t5(path: Path, cl, racing_csv: Path, table: np.ndarray, margin: float) -> None:
    """Map with the C -> X segments every 20th row, e_y_ref in its corridor, e_psi_ref + kappa_ref, v_ref."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rl = np.loadtxt(racing_csv, delimiter=",", skiprows=1, ndmin=2)
    s, e_y, e_psi, kappa, v = table.T
    n = np.column_stack([-np.sin(cl.heading), np.cos(cl.heading)])
    left, right = cl.xy + cl.w_left[:, None] * n, cl.xy - cl.w_right[:, None] * n
    X = cl.xy + e_y[:, None] * n                               # where each row lands on the racing line
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    for wall in (left, right):
        a.plot(*np.vstack([wall, wall[:1]]).T, color="0.3", lw=1)
    a.plot(*np.vstack([cl.xy, cl.xy[:1]]).T, "--", color="0.5", lw=1, label="centre line (perception)")
    a.plot(*np.vstack([rl[:, 1:3], rl[:1, 1:3]]).T, color="tab:red", lw=1.4, label="racing line (T3)")
    for i in range(0, len(s), 20):
        a.plot([cl.xy[i, 0], X[i, 0]], [cl.xy[i, 1], X[i, 1]], color="tab:blue", lw=0.8)
    a.plot([], [], color="tab:blue", label="C -> X = e_y_ref (every 20th row)")
    a.set(aspect="equal", title="each centre-line row and where it meets the racing line")
    a.legend(loc="upper left", fontsize=8)

    a = ax[0, 1]
    a.fill_between(s, -(cl.w_right - margin), cl.w_left - margin, color="0.9", label=f"corridor (walls - {margin} m)")
    a.plot(s, e_y, color="tab:red", lw=1.4, label="e_y_ref")
    a.axhline(0.0, ls="--", color="0.5", lw=0.8)
    a.set(xlabel="s_track [m] (perception's centre line)", ylabel="e_y_ref [m]  (+ left)",
          title="offset of the racing line from the centre line")
    a.legend(loc="lower right", fontsize=8)

    a = ax[1, 0]
    a.plot(s, np.degrees(e_psi), lw=1.2, label="e_psi_ref [deg]")
    a.set(xlabel="s_track [m]", ylabel="e_psi_ref [deg]  (+ left)", title="heading and curvature of the racing line")
    b = a.twinx()
    b.plot(s, kappa, color="tab:orange", lw=1.0)
    b.set_ylabel("kappa_ref [1/m]", color="tab:orange")

    a = ax[1, 1]
    a.plot(s, v, color="tab:green", lw=1.4)
    a.set(xlabel="s_track [m]", ylabel="v_ref [m/s]", title="speed (T4), placed on the centre line's s")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
