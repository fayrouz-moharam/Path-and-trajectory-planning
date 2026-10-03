"""Run T1 + T2 in one go: test-lap log -> track.csv (T1) -> smooth spline track with curvature (T2).

    1. edit the SETTINGS below
    2. python run_t1_t2.py

T1 part = run_t1.run() unchanged (log, track.csv, T1 checks, T1 plots).
T2 part = QueryableTrack on that track.csv, once raw and once smoothed, sampled
every T2_DS metres, written to t2/centerline_t2.csv and drawn in t2/t2.png.

Exit code: 0 = both fine, 1 = T1 track built but a T1 check failed, 2 = no track, 3 = T2 rejected the track.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import run_t1
from t2_curvature.queryable_track import QueryableTrack, TrackValidityError

# ============================================================== SETTINGS
SOURCE = "synth"              # "synth", "realtrack" or "log" (same meaning as in run_t1.py)
OUT_DIR = Path("out/run_t1_t2")
SEED = 0
PLOT = True

# used when SOURCE = "realtrack" / "log" (same as run_t1.py)
MAP_YAML = Path("out/real_tracks/Spielberg_map.yaml")
CENTERLINE_CSV = Path("out/real_tracks/Spielberg_centerline.csv")
LOG_PATH = Path("out/gym_spielberg/log.jsonl")

SMOOTHING_SIGMA_M = 0.2       # T2 Gaussian smoothing of the T1 centerline (0 = off)
T2_DS = 0.05                  # spacing (m) at which the T2 spline is sampled for the csv + plots
# =======================================================================

COLUMNS = ["s", "x", "y", "heading", "curvature", "w_left", "w_right"]


def sample_track(track: QueryableTrack, ds: float) -> dict:
    """Sample the T2 spline every ds metres around the whole lap -> dict of arrays (COLUMNS)."""
    s = np.arange(0.0, track.length, ds)
    return dict(zip(COLUMNS, track.sample_arrays(s)))


def write_t2_csv(path: Path, samples: dict) -> None:
    data = np.column_stack([samples[c] for c in COLUMNS])
    np.savetxt(path, data, delimiter=",", fmt="%.6f", header=",".join(COLUMNS), comments="")


def print_t2_report(name: str, track: QueryableTrack, samples: dict) -> None:
    k = samples["curvature"]
    i = int(np.argmax(np.abs(k)))
    print(f"  {name}: length {track.length:.2f} m, {len(k)} samples")
    print(f"    curvature: max |k| {abs(k[i]):.3f} 1/m (radius {1 / abs(k[i]):.2f} m) at s = {samples['s'][i]:.2f} m, "
          f"RMS {np.sqrt(np.mean(k ** 2)):.3f} 1/m")
    print(f"    widths:    min w_left {samples['w_left'].min():.3f} m, min w_right {samples['w_right'].min():.3f} m")


def centerline_shift(raw_track: QueryableTrack, smooth: dict, step: float = 0.005) -> np.ndarray:
    """Distance (m) from every smoothed sample to the raw spline. The raw spline is
    sampled every `step` (5 mm) so the gap between its samples doesn't count as shift."""
    _, x, y, *_ = raw_track.sample_arrays(np.arange(0.0, raw_track.length, step))
    dist, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([smooth["x"], smooth["y"]]))
    return dist


def plot_t2(path: Path, t1_csv: np.ndarray, raw: dict, smooth: dict | None, sigma: float) -> None:
    """2x3 figure: map coloured by curvature, curvature / heading / widths along s,
    how far smoothing moved the centerline, and a zoom on the tightest corner."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    final = smooth if smooth is not None else raw
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    (a1, a2, a3), (a4, a5, a6) = axes

    def loop(a):  # repeat the first point so the drawn line closes the lap
        return np.vstack([a, a[:1]])

    def walls(d):
        n = np.column_stack([-np.sin(d["heading"]), np.cos(d["heading"])])  # left normal
        xy = np.column_stack([d["x"], d["y"]])
        return loop(xy + n * d["w_left"][:, None]), loop(xy - n * d["w_right"][:, None])

    # 1. Map: T2 centerline coloured by signed curvature, walls from T2 widths.
    xy = np.column_stack([final["x"], final["y"]])
    segs = np.stack([xy, np.roll(xy, -1, axis=0)], axis=1)
    lim = np.abs(final["curvature"]).max()
    lc = LineCollection(segs, cmap="coolwarm", norm=plt.Normalize(-lim, lim), lw=2.5)
    lc.set_array(final["curvature"])
    a1.add_collection(lc)
    left, right = walls(final)
    a1.plot(*left.T, c="0.4", lw=0.8, label="walls (T2 widths)")
    a1.plot(*right.T, c="0.4", lw=0.8)
    a1.plot(final["x"][0], final["y"][0], "ko", ms=6, label="s = 0")
    fig.colorbar(lc, ax=a1, label="curvature [1/m]  (+ = left bend)", shrink=0.8)
    a1.set_aspect("equal")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title("1. T2 centerline spline, coloured by curvature\n(red = left bend, blue = right bend)")
    a1.set_xlabel("x [m]")
    a1.set_ylabel("y [m]")

    # 2. Curvature along s: raw spline vs smoothed spline.
    a2.plot(raw["s"], raw["curvature"], c="0.6", lw=0.8, label="raw spline (sigma = 0)")
    if smooth is not None:
        a2.plot(smooth["s"], smooth["curvature"], c="tab:blue", lw=1.2, label=f"smoothed (sigma = {sigma} m)")
    a2.set_title("2. Curvature along the lap\n(smoothing removes the jitter from T1's measured centerline)")
    a2.set_ylabel("curvature [1/m]")

    # 3. Heading along s (unwrapped: one lap = +-2 pi in total).
    a3.plot(final["s"], np.unwrap(final["heading"]), c="tab:purple", lw=1.2)
    a3.set_title("3. Heading along the lap (unwrapped)\nslope of this curve = curvature")
    a3.set_ylabel("heading [rad]")

    # 4. Widths along s: smoothing shifts the centerline, widths change so the walls stay put.
    a4.plot(raw["s"], raw["w_left"], c="tab:green", lw=0.8, alpha=0.5, label="w_left raw")
    a4.plot(raw["s"], raw["w_right"], c="tab:red", lw=0.8, alpha=0.5, label="w_right raw")
    if smooth is not None:
        a4.plot(smooth["s"], smooth["w_left"], c="tab:green", lw=1.2, label="w_left smoothed")
        a4.plot(smooth["s"], smooth["w_right"], c="tab:red", lw=1.2, label="w_right smoothed")
    a4.set_ylim(0, max(raw["w_left"].max(), raw["w_right"].max()) * 1.3)
    a4.set_title("4. Widths along the lap\n(centerline moves -> widths compensate, walls stay put)")
    a4.set_ylabel("width [m]")

    # 5. How far smoothing moved each point (distance to the raw spline, see centerline_shift).
    if smooth is not None:
        shift = smooth["shift"]
        a5.plot(smooth["s"], shift * 100, c="tab:orange", lw=1.2)
        a5.set_title(f"5. Centerline shift caused by smoothing\nmax {shift.max() * 100:.1f} cm "
                     f"(largest in tight corners)")
    else:
        a5.set_title("5. Centerline shift caused by smoothing\n(smoothing is off)")
    a5.set_ylabel("shift [cm]")

    # 6. Zoom on the tightest corner: T1 points, raw spline, smoothed spline.
    i = int(np.argmax(np.abs(final["curvature"])))
    cx, cy, r = final["x"][i], final["y"][i], 1.5
    a6.plot(t1_csv[:, 1], t1_csv[:, 2], ".", c="k", ms=4, label="T1 track.csv points")
    a6.plot(*loop(np.column_stack([raw["x"], raw["y"]])).T, c="0.6", lw=1.5, label="raw spline")
    if smooth is not None:
        a6.plot(*loop(xy).T, c="tab:blue", lw=1.5, label="smoothed spline")
    a6.plot(*left.T, c="tab:green", lw=1, label="left wall")
    a6.plot(*right.T, c="tab:red", lw=1, label="right wall")
    a6.set_xlim(cx - r, cx + r)
    a6.set_ylim(cy - r, cy + r)
    a6.set_aspect("equal")
    a6.legend(loc="upper right", fontsize=8)
    a6.set_title(f"6. Zoom on the tightest corner (s = {final['s'][i]:.1f} m)\n"
                 f"radius {1 / abs(final['curvature'][i]):.2f} m")
    a6.set_xlabel("x [m]")
    a6.set_ylabel("y [m]")

    for ax in (a2, a3, a4, a5):
        ax.set_xlabel("s [m]  (distance along centerline)")
        ax.grid(alpha=0.3)
    for ax in (a2, a4):
        ax.legend(loc="upper right", fontsize=8)
    for ax in (a1, a6):
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_t2(track_csv: Path, out_dir: Path, *, sigma: float = SMOOTHING_SIGMA_M, ds: float = T2_DS,
           plot: bool = PLOT) -> int:
    """T1's track.csv -> T2 tracks (raw + smoothed) -> centerline_t2.csv (+ t2.png)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_track = QueryableTrack(str(track_csv))
        smooth_track = QueryableTrack(str(track_csv), smoothing_sigma_m=sigma) if sigma > 0 else None
    except TrackValidityError as e:
        print(f"T2 rejected the track: {e}", file=sys.stderr)
        return 3

    raw = sample_track(raw_track, ds)
    print_t2_report("raw spline (no smoothing)", raw_track, raw)
    smooth = None
    if smooth_track is not None:
        smooth = sample_track(smooth_track, ds)
        print_t2_report(f"smoothed spline (sigma = {sigma} m)", smooth_track, smooth)
        print(f"    smoothing moved the centerline by at most {smooth_track.max_centerline_shift * 100:.1f} cm")
        smooth["shift"] = centerline_shift(raw_track, smooth)

    final = smooth if smooth is not None else raw
    csv_path = out_dir / "centerline_t2.csv"
    write_t2_csv(csv_path, final)
    print(f"wrote {csv_path}  ({'smoothed' if smooth is not None else 'raw'}, columns {','.join(COLUMNS)})")

    if plot:
        t1_points = np.loadtxt(track_csv, delimiter=",", skiprows=1, ndmin=2)
        plot_t2(out_dir / "t2.png", t1_points, raw, smooth, sigma)
        print(f"plot -> {out_dir / 't2.png'}")
    return 0


def run(source: str = SOURCE, out_dir: Path = OUT_DIR, *, seed: int = SEED, plot: bool = PLOT,
        sigma: float = SMOOTHING_SIGMA_M) -> int:
    out_dir = Path(out_dir)
    print(f"===== T1: build the track  (source = {source})")
    code = run_t1.run(source, out_dir, plot=plot, seed=seed, map_yaml=MAP_YAML,
                      centerline_csv=CENTERLINE_CSV, log_path=LOG_PATH)
    if code == 2:
        return 2

    print("\n===== T2: spline + curvature (+ smoothing)")
    t2_code = run_t2(out_dir / "track" / "track.csv", out_dir / "t2", sigma=sigma, plot=plot)
    return t2_code or code


if __name__ == "__main__":
    sys.exit(run())
