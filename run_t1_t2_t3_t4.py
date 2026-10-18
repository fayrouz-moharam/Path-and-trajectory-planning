"""Run T1 + T2 + T3 + T4 in one go: test-lap log -> track.csv (T1) -> spline + curvature (T2)
-> minimum-curvature racing line (T3) -> velocity profile (T4).

    1. edit the SETTINGS below
    2. python run_t1_t2_t3_t4.py

T1 part = run_t1.run() unchanged (log, track/, T1 checks, T1 plots).
T2 part = run_t1_t2.run_t2() unchanged (t2/centerline_t2.csv, t2/t2.png).
T3 part = run_t1_t2_t3.run_t3() unchanged (t3/racing_line.csv + racing_line_meta.json + t3.png).
          The T3 car settings (CAR_WIDTH, MAX_STEER, ...) live in run_t1_t2_t3.py.
T4 part = the velocity profile on the T3 racing line (run_t4 below), with the vehicle limits
          in SETTINGS: t4/velocity_profile.csv + velocity_profile_meta.json + t4.png.

Exit code: 0 = all fine, 1 = T1 check failed, 2 = no track, 3 = T2 rejected the track,
4 = T3 did not settle within its rounds, 5 = the speed profile leaves apex's [1, 3] m/s bounds
(the racing line and the profile are still written for 4 and 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import run_t1
import run_t1_t2
import run_t1_t2_t3
from t4_velocity_profile import (SpeedReference, VehicleLimits, build_meta, compute_profile,
                                 read_t3, write_profile)
from t4_velocity_profile.export import APEX_SPEED_BOUNDS

# ============================================================== SETTINGS
SOURCE = "synth"              # "synth", "realtrack" or "log" (same meaning as in run_t1.py)
OUT_DIR = Path("out/run_t1_t2_t3_t4")
SEED = 0
PLOT = True

# used when SOURCE = "realtrack" / "log" (same as run_t1.py)
MAP_YAML = Path("out/real_tracks/Spielberg_map.yaml")
CENTERLINE_CSV = Path("out/real_tracks/Spielberg_centerline.csv")
LOG_PATH = Path("out/gym_spielberg/log.jsonl")

# T2
SMOOTHING_SIGMA_M = 0.2       # Gaussian smoothing of the T1 centerline; T3 starts from this line (0 = off)

# T4 (placeholders until the vehicle is measured; apex vehicle yaml is all null)
LIMITS = VehicleLimits(
    mu=0.8,                   # tyre-road friction
    a_accel_max=3.0,          # m/s^2
    a_brake_max=4.0,          # m/s^2 (positive)
    v_max=6.0,                # m/s
    v_min=0.5,                # m/s
)
KAPPA_SIGMA_M = 0.0           # extra smoothing of T3's kappa before the corner limit (0 = off)
# =======================================================================


def run_t4(t3_dir: Path, out_dir: Path, *, limits: VehicleLimits = LIMITS, sigma: float = KAPPA_SIGMA_M,
           plot: bool = PLOT) -> int:
    """T3's racing_line.csv -> T4 velocity profile -> t4/ files."""
    csv = t3_dir / "racing_line.csv"
    s, kappa, lap, t3_meta = read_t3(csv)
    print(f"T4 on {csv}: {len(s)} points, lap {lap:.2f} m, map {t3_meta.get('map_version')}")
    if t3_meta.get("converged") is False:
        print("  WARNING: T3 did not converge; the racing line may still move")

    p = compute_profile(s, kappa, lap, limits, kappa_sigma_m=sigma)
    meta = build_meta(p, t3_meta, csv, kappa_sigma_m=sigma)
    csv_out, meta_out = write_profile(out_dir, p, meta)

    print(f"  limits: mu {limits.mu}, a_lat_max {limits.a_lat_max:.2f}, accel {limits.a_accel_max}, "
          f"brake {limits.a_brake_max} m/s^2, v_max {limits.v_max} m/s")
    print(f"  lap time {p.lap_time:.2f} s, mean {meta['v_mean']:.2f} m/s, "
          f"v {meta['v_min']:.2f} .. {meta['v_max']:.2f} m/s")
    print(f"  a_ref {meta['a_ref_min']:.2f} .. {meta['a_ref_max']:.2f} m/s^2, "
          f"peak |a_lat| {meta['a_lat_peak']:.2f} m/s^2, peak friction use {meta['friction_use_peak']:.3f}")
    at_vmax = p.v_ref >= limits.v_max - 1e-6
    at_corner = (p.v_ref >= p.v_lat - 1e-6) & ~at_vmax
    print(f"  points at v_max {at_vmax.mean() * 100:.0f} %, at the corner limit {at_corner.mean() * 100:.0f} %, "
          f"accelerating / braking {100 - (at_vmax | at_corner).mean() * 100:.0f} %")
    print(f"wrote {csv_out} and {meta_out}")

    # the callable the MPC takes as speed_reference=...
    ref = SpeedReference.from_profile(p)
    print(f"  speed_reference(0) = {ref(0.0):.3f} m/s, speed_reference(lap) = {ref(lap):.3f} m/s (wraps)")

    if plot:
        plot_t4(out_dir / "t4.png", p)
        print(f"plot -> {out_dir / 't4.png'}")

    code = 0
    if t3_meta.get("converged") is False:
        code = 4
    if not meta["inside_apex_bounds"]:
        lo, hi = APEX_SPEED_BOUNDS
        print(f"  NOTE: v_ref leaves apex's [{lo}, {hi}] m/s bounds; agree the bound with the MPC owner")
        code = code or 5
    return code


def plot_t4(path: Path, p) -> None:
    """2x2: v(s) with the corner limit, a_ref(s), curvature, g-g diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = p.limits
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    a.plot(p.s, np.minimum(p.v_lat, L.v_max), color="0.6", lw=1, label="corner limit sqrt(mu g / |k|)")
    a.plot(p.s, p.v_ref, lw=1.6, label="v_ref (after both passes)")
    lo, hi = APEX_SPEED_BOUNDS
    a.axhspan(lo, hi, color="tab:green", alpha=0.08, label=f"apex bounds [{lo}, {hi}] m/s")
    a.set(xlabel="s along racing line [m]", ylabel="v [m/s]",
          title=f"velocity profile, lap time {p.lap_time:.2f} s")
    a.legend(loc="lower right", fontsize=8)

    a = ax[0, 1]
    a.plot(p.s, p.a_ref, lw=1)
    a.axhline(L.a_accel_max, ls="--", color="tab:green", lw=0.8)
    a.axhline(-L.a_brake_max, ls="--", color="tab:red", lw=0.8)
    a.set(xlabel="s [m]", ylabel="a_ref [m/s^2]", title="longitudinal acceleration (+ throttle, - brake)")

    a = ax[1, 0]
    a.plot(p.s, p.kappa, lw=1)
    a.set(xlabel="s [m]", ylabel="kappa [1/m]", title="racing-line curvature (from T3)")

    a = ax[1, 1]
    th = np.linspace(0, 2 * np.pi, 200)
    ax_acc, ax_brk = min(L.a_accel_max, L.a_lat_max), min(L.a_brake_max, L.a_lat_max)
    ex = np.where(np.sin(th) >= 0, ax_acc, ax_brk) * np.sin(th)
    a.plot(L.a_lat_max * np.cos(th), ex, color="0.6", lw=1, label="friction ellipse")
    sc = a.scatter(p.a_lat, p.a_ref, c=p.v_ref, s=3, cmap="viridis")
    fig.colorbar(sc, ax=a, label="v [m/s]")
    a.set(xlabel="a_lat [m/s^2]", ylabel="a_long [m/s^2]", title="g-g diagram", aspect="equal")
    a.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(source: str = SOURCE, out_dir: Path = OUT_DIR, *, seed: int = SEED, plot: bool = PLOT,
        sigma: float = SMOOTHING_SIGMA_M, limits: VehicleLimits = LIMITS) -> int:
    out_dir = Path(out_dir)
    print(f"===== T1: build the track  (source = {source})")
    code = run_t1.run(source, out_dir, plot=plot, seed=seed, map_yaml=MAP_YAML,
                      centerline_csv=CENTERLINE_CSV, log_path=LOG_PATH)
    if code == 2:
        return 2
    track_csv = out_dir / "track" / "track.csv"

    print("\n===== T2: spline + curvature (+ smoothing)")
    t2_code = run_t1_t2.run_t2(track_csv, out_dir / "t2", sigma=sigma, plot=plot)
    if t2_code:
        return t2_code

    print("\n===== T3: minimum-curvature racing line")
    t3_code = run_t1_t2_t3.run_t3(track_csv, out_dir / "t3", sigma=sigma, plot=plot)

    print("\n===== T4: velocity profile")
    t4_code = run_t4(out_dir / "t3", out_dir / "t4", limits=limits, plot=plot)
    return t3_code or t4_code or code


if __name__ == "__main__":
    sys.exit(run())
