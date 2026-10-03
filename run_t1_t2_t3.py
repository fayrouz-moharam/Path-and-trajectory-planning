"""Run T1 + T2 + T3 in one go: test-lap log -> track.csv (T1) -> spline + curvature (T2)
-> minimum-curvature racing line (T3).

    1. edit the SETTINGS below
    2. python run_t1_t2_t3.py

T1 part = run_t1.run() unchanged (log, track/, T1 checks, T1 plots).
T2 part = run_t1_t2.run_t2() unchanged (t2/centerline_t2.csv, t2/t2.png).
T3 part = the racing line, built on T2's smoothed centerline:
          t3/racing_line.csv + t3/racing_line_meta.json + t3/t3.png.

Exit code: 0 = all fine, 1 = T1 check failed, 2 = no track, 3 = T2 rejected the track,
4 = T3 did not settle within its rounds (the racing line is still written).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import run_t1
import run_t1_t2
from t2_curvature.queryable_track import QueryableTrack
from t3_racing_line.export import RacingLine, build_meta, build_racing_line, write_racing_line
from t3_racing_line.min_curvature import IterateResult, iterate_min_curvature, kappa_max_from_steering
from t3_racing_line.stations import vehicle_margin

# ============================================================== SETTINGS
SOURCE = "synth"              # "synth", "realtrack" or "log" (same meaning as in run_t1.py)
OUT_DIR = Path("out/run_t1_t2_t3")
SEED = 0
PLOT = True

# used when SOURCE = "realtrack" / "log" (same as run_t1.py)
MAP_YAML = Path("out/real_tracks/Spielberg_map.yaml")
CENTERLINE_CSV = Path("out/real_tracks/Spielberg_centerline.csv")
LOG_PATH = Path("out/gym_spielberg/log.jsonl")

# T2
SMOOTHING_SIGMA_M = 0.2       # Gaussian smoothing of the T1 centerline; T3 starts from this line (0 = off)

# T3 (spec placeholders until the vehicle team gives real numbers)
CAR_WIDTH = 0.30              # m
SAFETY_BUFFER = 0.05          # m, extra distance kept from the walls (tracking / localisation error)
MAX_STEER = 0.4               # rad
WHEELBASE = 0.33              # m
STATION_SPACING = 0.15        # m between optimisation stations
FINAL_SPACING = 0.10          # m between points in racing_line.csv
# =======================================================================


def run_t3(track_csv: Path, out_dir: Path, *, sigma: float = SMOOTHING_SIGMA_M,
           plot: bool = PLOT) -> int:
    """T1's track.csv -> T2 centerline (smoothed) -> T3 racing line -> t3/ files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    centerline = QueryableTrack(str(track_csv), smoothing_sigma_m=sigma)
    margin = vehicle_margin(CAR_WIDTH, SAFETY_BUFFER)
    kappa_max = kappa_max_from_steering(MAX_STEER, WHEELBASE)
    print(f"  start line: T2 centerline (sigma = {sigma} m), {centerline.length:.2f} m")
    print(f"  car: margin to walls {margin:.2f} m (half width {CAR_WIDTH / 2:.2f} + buffer {SAFETY_BUFFER:.2f}), "
          f"kappa_max {kappa_max:.2f} 1/m (tightest radius {1 / kappa_max:.2f} m)")

    with warnings.catch_warnings(record=True) as caught:   # "did not settle" -> printed below
        warnings.simplefilter("always")
        res = iterate_min_curvature(centerline, STATION_SPACING, margin, kappa_max=kappa_max)
    print_rounds(res)
    for w in caught:
        print(f"  WARNING: {w.message}")

    line = build_racing_line(res, centerline, FINAL_SPACING)
    track_meta = json.loads((track_csv.parent / "track_meta.json").read_text(encoding="utf-8"))
    meta = build_meta(line, res, kappa_max=kappa_max, final_spacing=FINAL_SPACING, car_width=CAR_WIDTH,
                      safety_buffer=SAFETY_BUFFER, source_meta=track_meta, source_csv=track_csv)
    meta["t2_smoothing_sigma_m"] = sigma
    csv_path, meta_path = write_racing_line(out_dir, line, meta)
    print_summary(centerline, line, margin, kappa_max)
    print(f"wrote {csv_path} and {meta_path}")

    if plot:
        plot_t3(out_dir / "t3.png", centerline, line, res, margin, kappa_max)
        print(f"plot -> {out_dir / 't3.png'}")
    return 0 if res.converged else 4


def print_rounds(res: IterateResult) -> None:
    """One line per optimisation round: how much the line still moved and how good it is."""
    print("  round    mu      max move   peak |k|   bending (integral k^2 ds)")
    for h in res.history:
        print(f"  {h.iteration:5d}  {h.mu:7.4f}  {h.max_alpha * 100:7.2f} cm  {h.line_peak_kappa:7.3f}    "
              f"{h.bending:8.3f}")
    print(f"  stopped after {len(res.history)} rounds: {res.stop_reason} "
          f"({'converged' if res.converged else 'NOT converged'})")


def compare_lines(centerline: QueryableTrack, line: RacingLine, step: float = 0.05) -> dict:
    """Put the racing line on the centerline's s axis so the two can be compared point by point.

    offset:  for every centerline point, sideways distance to the racing line (+ = left).
    line_k:  curvature of the racing line, at the centerline s of its nearest centerline point.
    """
    s_c, x_c, y_c, h_c, k_c, wl_c, wr_c = centerline.sample_arrays(np.arange(0.0, centerline.length, step))
    c_xy = np.column_stack([x_c, y_c])
    normals = np.column_stack([-np.sin(h_c), np.cos(h_c)])

    l_xy = np.column_stack([line.x, line.y])
    _, j = cKDTree(l_xy).query(c_xy)                       # nearest racing-line point
    offset = np.einsum("ij,ij->i", l_xy[j] - c_xy, normals)

    _, i = cKDTree(c_xy).query(l_xy)                       # nearest centerline point
    order = np.argsort(s_c[i])
    return dict(s=s_c, k=k_c, w_left=wl_c, w_right=wr_c, offset=offset,
                line_s=s_c[i][order], line_k=line.kappa[order])


def print_summary(centerline: QueryableTrack, line: RacingLine, margin: float, kappa_max: float) -> None:
    c = compare_lines(centerline, line)
    bend_c = np.sum(c["k"] ** 2) * 0.05
    bend_l = np.sum(line.kappa ** 2) * (line.length / len(line.s))
    closest = min(line.w_left.min(), line.w_right.min())
    print("  centerline -> racing line:")
    print(f"    lap length   {centerline.length:8.2f} m  -> {line.length:8.2f} m")
    print(f"    peak |k|     {np.abs(c['k']).max():8.3f}    -> {np.abs(line.kappa).max():8.3f} 1/m  "
          f"(car limit {kappa_max:.3f})")
    print(f"    bending      {bend_c:8.3f}    -> {bend_l:8.3f}  (integral k^2 ds, lower = smoother)")
    print(f"    largest shift from the centerline: {np.abs(c['offset']).max():.2f} m")
    print(f"    closest the line gets to a wall: {closest:.3f} m  (margin {margin:.2f} m)")


def plot_t3(path: Path, centerline: QueryableTrack, line: RacingLine, res: IterateResult,
            margin: float, kappa_max: float) -> None:
    """2x3 figure: track with centerline + racing line, sideways shift, curvature,
    per-round convergence, and a zoom on the tightest corner."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def loop(a):  # repeat the first point so the drawn line closes the lap
        return np.vstack([a, a[:1]])

    c = compare_lines(centerline, line)
    _, cx, cy, ch, *_ = centerline.sample_arrays(c["s"])
    # limits of the allowed area: walls moved inwards by the margin (same band as panel 2)
    normals = np.column_stack([-np.sin(ch), np.cos(ch)])
    allowed_left = np.column_stack([cx, cy]) + normals * (c["w_left"] - margin)[:, None]
    allowed_right = np.column_stack([cx, cy]) - normals * (c["w_right"] - margin)[:, None]
    c_xy, l_xy = np.column_stack([cx, cy]), np.column_stack([line.x, line.y])
    left_wall, right_wall = res.walls

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    (a1, a2, a3), (a4, a5, a6) = axes

    def draw_track(ax):
        ax.plot(*loop(left_wall).T, c="0.3", lw=1, label="walls")
        ax.plot(*loop(right_wall).T, c="0.3", lw=1)
        ax.plot(*loop(allowed_left).T, ":", c="tab:blue", lw=1.2,
                label=f"allowed area for the car centre (walls - {margin:.2f} m)")
        ax.plot(*loop(allowed_right).T, ":", c="tab:blue", lw=1.2)
        ax.plot(*loop(c_xy).T, "--", c="0.55", lw=1.2, label="centerline (T2)")
        ax.plot(*loop(l_xy).T, c="tab:red", lw=1.8, label="racing line (T3)")

    # 1. Whole track.
    draw_track(a1)
    a1.plot(line.x[0], line.y[0], "ko", ms=6, label="s = 0")
    a1.annotate("", xy=l_xy[min(20, len(l_xy) - 1)], xytext=l_xy[0], arrowprops={"arrowstyle": "->", "lw": 1.5})
    a1.set_aspect("equal")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)  # below the map
    a1.set_title("1. Centerline vs racing line\n(the line uses the whole width to make the corners gentler)")
    a1.set_xlabel("x [m]")
    a1.set_ylabel("y [m]")

    # 2. Sideways shift of the racing line, inside the allowed band.
    a2.fill_between(c["s"], -(c["w_right"] - margin), c["w_left"] - margin, color="0.9",
                    label=f"allowed (walls minus {margin:.2f} m margin)")
    a2.axhline(0, c="0.55", ls="--", lw=1, label="centerline")
    a2.plot(c["s"], c["offset"], c="tab:red", lw=1.2, label="racing line")
    a2.set_title("2. Where the racing line sits across the track\n(+ = left of the centerline)")
    a2.set_ylabel("sideways offset [m]")
    a2.legend(loc="upper right", fontsize=8)

    # 3. Curvature: centerline vs racing line.
    a3.plot(c["s"], c["k"], c="0.6", lw=0.8, label="centerline")
    a3.plot(c["line_s"], c["line_k"], c="tab:red", lw=1.2, label="racing line")
    for sign in (1, -1):
        a3.axhline(sign * kappa_max, c="k", ls=":", lw=1, label="car limit kappa_max" if sign == 1 else None)
    a3.set_title("3. Curvature along the lap\n(racing line = smaller peaks = higher corner speed)")
    a3.set_ylabel("curvature [1/m]")
    a3.legend(loc="upper right", fontsize=8)

    # 4-5. Optimisation rounds.
    rounds = [h.iteration for h in res.history]
    a4.semilogy(rounds, [max(h.max_alpha, 1e-6) * 100 for h in res.history], "o-", c="tab:blue")
    a4.axhline(1.0, c="k", ls=":", lw=1, label="stop when < 1 cm")
    a4.set_title(f"4. How far the line still moved each round\n({res.stop_reason})")
    a4.set_xlabel("round")
    a4.set_ylabel("max move [cm]  (log scale)")
    a4.legend(loc="upper right", fontsize=8)

    a5.plot(rounds, [h.line_peak_kappa for h in res.history], "o-", c="tab:red", label="racing line")
    a5.axhline(np.abs(c["k"]).max(), c="0.55", ls="--", lw=1, label="centerline")
    a5.axhline(kappa_max, c="k", ls=":", lw=1, label="car limit kappa_max")
    a5.set_title("5. Peak |curvature| after each round")
    a5.set_xlabel("round")
    a5.set_ylabel("peak |curvature| [1/m]")
    a5.legend(loc="upper right", fontsize=8)

    # 6. Zoom on the tightest corner of the centerline.
    i = int(np.argmax(np.abs(c["k"])))
    draw_track(a6)
    r = 3.0
    a6.set_xlim(cx[i] - r, cx[i] + r)
    a6.set_ylim(cy[i] - r, cy[i] + r)
    a6.set_aspect("equal")
    a6.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    a6.set_title(f"6. Zoom on the tightest corner (centerline s = {c['s'][i]:.1f} m)\n"
                 f"centerline radius {1 / abs(c['k'][i]):.2f} m")
    a6.set_xlabel("x [m]")
    a6.set_ylabel("y [m]")

    for ax in (a2, a3):
        ax.set_xlabel("s [m]  (distance along the centerline)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(source: str = SOURCE, out_dir: Path = OUT_DIR, *, seed: int = SEED, plot: bool = PLOT,
        sigma: float = SMOOTHING_SIGMA_M) -> int:
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
    t3_code = run_t3(track_csv, out_dir / "t3", sigma=sigma, plot=plot)
    return t3_code or code


if __name__ == "__main__":
    sys.exit(run())
