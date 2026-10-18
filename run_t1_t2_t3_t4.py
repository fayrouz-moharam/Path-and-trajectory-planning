"""Run T1 + T2 + T3 + T4 in one go: test-lap log -> track.csv (T1) -> spline + curvature (T2)
-> minimum-curvature racing line (T3) -> velocity profile (T4).

    1. edit the SETTINGS below
    2. python run_t1_t2_t3_t4.py

T1 part = run_t1.run() unchanged (log, track/, T1 checks, T1 plots).
T2 part = run_t1_t2.run_t2() unchanged (t2/centerline_t2.csv, t2/t2.png).
T3 part = run_t1_t2_t3.run_t3() unchanged (t3/racing_line.csv + racing_line_meta.json + t3.png).
          The T3 car settings (CAR_WIDTH, MAX_STEER, ...) live in run_t1_t2_t3.py.
T4 part = run_t4.run() on the T3 output, with the vehicle limits below:
          t4/velocity_profile.csv + velocity_profile_meta.json + t4.png.

Exit code: 0 = all fine, 1 = T1 check failed, 2 = no track, 3 = T2 rejected the track,
4 = T3 did not settle within its rounds, 5 = the speed profile leaves apex's [1, 3] m/s bounds
(the racing line and the profile are still written for 4 and 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import run_t1
import run_t1_t2
import run_t1_t2_t3
import run_t4
from t4_velocity_profile import VehicleLimits

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
    t4_code = run_t4.run(out_dir / "t3", limits, sigma=KAPPA_SIGMA_M, plot=plot)
    return t3_code or t4_code or code


if __name__ == "__main__":
    sys.exit(run())
