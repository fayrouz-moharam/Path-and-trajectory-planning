"""Checks for QueryableTrack (T2), in the same "[ok]/[FAIL]" spirit as T1's
own validation output. Run: python test_queryable_track.py
"""

import numpy as np

from t2_curvature.make_synthetic_track_csv import circle_csv, oval_csv
from t2_curvature.queryable_track import QueryableTrack, TrackValidityError

RESULTS = []


def check(name, value, threshold, ok):
    RESULTS.append((name, value, threshold, ok))
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name} (value={value:.6g}, threshold={threshold:.6g})")


def check_circle():
    print("\n--- circle fixture: exact analytic curvature = 1/R ---")
    radius = 5.0
    nominal_length = circle_csv("circle_track.csv", radius=radius)
    track = QueryableTrack("circle_track.csv")

    check("length_matches_nominal", abs(track.length - nominal_length), 0.05, abs(track.length - nominal_length) < 0.05)

    s_test = np.linspace(0, track.length, 200, endpoint=False)
    curv_err = [abs(track.sample(s).curvature - 1.0 / radius) for s in s_test]
    check("max_curvature_error", max(curv_err), 1e-3, max(curv_err) < 1e-3)

    r_err = [abs(np.hypot(track.sample(s).x, track.sample(s).y) - radius) for s in s_test]
    check("max_radius_error", max(r_err), 1e-3, max(r_err) < 1e-3)

    p0, pL = track.sample(0.0), track.sample(track.length - 1e-9)
    wrap_err = np.hypot(p0.x - pL.x, p0.y - pL.y)
    check("periodicity_position", wrap_err, 1e-2, wrap_err < 1e-2)

    heading_err = [
        abs(np.arctan2(np.sin(track.sample(s).heading - (s / radius + np.pi / 2)),
                        np.cos(track.sample(s).heading - (s / radius + np.pi / 2))))
        for s in s_test
    ]
    check("max_heading_error", max(heading_err), 1e-3, max(heading_err) < 1e-3)


def check_oval():
    print("\n--- oval fixture: closer to the real synthetic track shape ---")
    nominal_length = oval_csv("oval_track.csv")
    track = QueryableTrack("oval_track.csv")
    check("length_matches_nominal", abs(track.length - nominal_length), 0.1, abs(track.length - nominal_length) < 0.1)

    s_test = np.linspace(0, track.length, 500, endpoint=False)
    curvatures = [track.sample(s).curvature for s in s_test]
    check("finite_curvature_everywhere", max(abs(c) for c in curvatures) if all(np.isfinite(curvatures)) else float("inf"),
          10.0, all(np.isfinite(curvatures)) and max(abs(c) for c in curvatures) < 10.0)

    p0, pL = track.sample(0.0), track.sample(track.length - 1e-9)
    wrap_err = np.hypot(p0.x - pL.x, p0.y - pL.y)
    check("periodicity_position", wrap_err, 0.05, wrap_err < 0.05)

    widths = [(track.left_width(s), track.right_width(s)) for s in s_test]
    check("widths_match_input", max(abs(wl - 0.9) + abs(wr - 0.9) for wl, wr in widths), 1e-6,
          all(abs(wl - 0.9) < 1e-6 and abs(wr - 0.9) < 1e-6 for wl, wr in widths))


def check_rejects_bad_input():
    print("\n--- input validation ---")
    import pandas as pd

    pd.DataFrame({"s": [0, 1, 2], "x": [0, 1, 2], "y": [0, 0, 0]}).to_csv("bad_missing_cols.csv", index=False)
    try:
        QueryableTrack("bad_missing_cols.csv")
        ok = False
    except TrackValidityError:
        ok = True
    check("rejects_missing_columns", 0.0, 0.0, ok)

    pd.DataFrame({
        "s": [0, 0.05, 0.03, 0.1], "x": [0, 1, 2, 3], "y": [0, 0, 0, 0],
        "w_left": [0.9] * 4, "w_right": [0.9] * 4,
    }).to_csv("bad_nonmonotonic_s.csv", index=False)
    try:
        QueryableTrack("bad_nonmonotonic_s.csv")
        ok = False
    except TrackValidityError:
        ok = True
    check("rejects_nonmonotonic_s", 0.0, 0.0, ok)


def plot_oval():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track = QueryableTrack("oval_track.csv")
    s_dense = np.linspace(0, track.length, 2000)
    samples = track.sample_many(s_dense)
    xs = [p.x for p in samples]
    ys = [p.y for p in samples]
    curv = [p.curvature for p in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(xs, ys)
    axes[0].axis("equal")
    axes[0].set_title("T2 sampled centerline (oval fixture)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[1].plot(s_dense, curv)
    axes[1].set_title("curvature(s)")
    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("curvature [1/m]")
    fig.tight_layout()
    fig.savefig("t2_oval_check.png", dpi=120)
    print("\nwrote t2_oval_check.png")


if __name__ == "__main__":
    check_circle()
    check_oval()
    check_rejects_bad_input()
    plot_oval()

    n_fail = sum(1 for *_, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        raise SystemExit(1)
