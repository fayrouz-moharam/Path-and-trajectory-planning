"""Debug figures.

plot_track  -- final result: accumulated points, pose trace, centerline, walls.
plot_stages -- one panel per pipeline stage, to see what each step does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .accumulate import Accumulated, to_world
from .config import T1Config
from .extract import Track
from .geometry import tangents_normals
from .log_io import Frame


def _issues(track: Track, validation: dict | None) -> list[tuple[float, str]]:
    """(s, description) for every failed warning that has locations (wall folds)."""
    out = []
    for name, c in (validation or {}).items():
        if c.get("severity") == "warning" and not c["passed"]:
            out += [(float(s), name) for s in c["value"]]
    return sorted(out)


def plot_issues(path: str | Path, track: Track, acc: Accumulated, validation: dict,
                ground_truth=None, background=None, half_size: float = 3.0) -> bool:
    """One zoomed panel per warning location. Returns False if there was nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    issues = _issues(track, validation)
    if not issues:
        return False
    _, normals = tangents_normals(track.xy)
    left = track.xy + normals * track.w_left[:, None]
    right = track.xy - normals * track.w_right[:, None]

    fig, axes = plt.subplots(1, len(issues), figsize=(6 * len(issues), 6.5), squeeze=False)
    for k, (ax, (s_issue, name)) in enumerate(zip(axes[0], issues)):
        cx, cy = track.xy[np.searchsorted(track.s, s_issue) % len(track.s)]
        if background is not None:
            ax.imshow(~background.occupied, cmap="gray", extent=background.extent, vmin=0, vmax=1,
                      interpolation="nearest")
        ax.scatter(acc.points[:, 0], acc.points[:, 1], s=4, c="0.55", label="wall points")
        if ground_truth is not None:
            ax.plot(*ground_truth.ref.T, ":", c="c", lw=1.5, label="published centerline")
        ax.plot(*track.xy.T, c="tab:blue", lw=1.5, label="our centerline")
        ax.plot(*left.T, c="tab:green", lw=1.5, label="our left wall (centerline + w_left)")
        ax.plot(*right.T, c="tab:red", lw=1.5, label="our right wall (centerline - w_right)")
        ax.set_xlim(cx - half_size, cx + half_size)
        ax.set_ylim(cy - half_size, cy + half_size)
        ax.set_aspect("equal")
        side = "left" if name.startswith("left") else "right"
        ax.set_title(f"{k + 1}. {side} wall folds at s = {s_issue:.1f} m\n"
                     f"corner tighter than the {side} width: the rebuilt wall loops over itself")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def plot_track(path: str | Path, track: Track, acc: Accumulated, ground_truth=None, background=None,
               validation: dict | None = None) -> None:
    """ground_truth: optional synth.GroundTruth to overlay (dashed).
    background: optional realtrack.OccupancyMap drawn underneath (the map "photo").
    validation: optional report; warning locations are circled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, normals = tangents_normals(track.xy)
    left = track.xy + normals * track.w_left[:, None]
    right = track.xy - normals * track.w_right[:, None]

    def loop(a):
        return np.vstack([a, a[:1]])

    fig, ax = plt.subplots(figsize=(10, 8))
    if background is not None:
        ax.imshow(~background.occupied, cmap="gray", extent=background.extent, vmin=0, vmax=1,
                  interpolation="nearest", zorder=0)
    ax.scatter(acc.points[:, 0], acc.points[:, 1], s=1, c="0.6", label="accumulated boundary points")
    ax.plot(acc.poses[:, 0], acc.poses[:, 1], lw=0.8, c="tab:orange", label="pose trace")
    if ground_truth is not None:
        ax.plot(*loop(ground_truth.midline).T, "--", lw=1, c="k", label="ground truth (midline, walls)")
        ax.plot(*loop(ground_truth.left_wall).T, "--", lw=0.8, c="k")
        ax.plot(*loop(ground_truth.right_wall).T, "--", lw=0.8, c="k")
    ax.plot(*loop(track.xy).T, lw=1.5, c="tab:blue", label="centerline")
    ax.plot(*loop(left).T, lw=1, c="tab:green", label="left boundary")
    ax.plot(*loop(right).T, lw=1, c="tab:red", label="right boundary")
    ax.plot(*track.xy[0], "ko", ms=6, label="s = 0")
    ax.annotate("", xy=track.xy[5], xytext=track.xy[0], arrowprops={"arrowstyle": "->", "lw": 1.5})
    for k, (s_issue, what) in enumerate(_issues(track, validation)):
        p = track.xy[np.searchsorted(track.s, s_issue) % len(track.s)]
        ax.add_patch(plt.Circle(p, 3.0 if background is not None else 0.6, fill=False, ec="magenta", lw=2,
                                label="warning (see track_issues.png)" if k == 0 else None))
        ax.annotate(f"{k + 1}", p, xytext=(8, 8), textcoords="offset points", color="magenta", weight="bold")
    ax.set_aspect("equal")
    if background is not None:  # zoom to the track, not the whole map image
        lo, hi = acc.points.min(axis=0) - 2, acc.points.max(axis=0) + 2
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"T1 track  (length {track.length:.2f} m, {len(track.s)} pts)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stages(path: str | Path, frames: list[Frame], acc: Accumulated, track: Track, cfg: T1Config) -> None:
    """2x3 grid: one raw frame -> all frames stacked -> cleaned -> left/right split
    -> final track -> width profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def loop(a):
        return np.vstack([a, a[:1]])

    def world_pts(fr):
        return to_world(fr.boundary_points, fr.pose) if cfg.points_frame == "vehicle" else fr.boundary_points

    raw = np.vstack([world_pts(fr) for fr in frames if len(fr.boundary_points)])
    _, normals = tangents_normals(track.xy)
    left = track.xy + normals * track.w_left[:, None]
    right = track.xy - normals * track.w_right[:, None]
    _, idx = cKDTree(track.xy).query(acc.points)
    is_left = np.einsum("ij,ij->i", acc.points - track.xy[idx], normals[idx]) > 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    (a1, a2, a3), (a4, a5, a6) = axes

    # 1. One frame, as the car sees it (car at origin, facing +x).
    fr = frames[len(frames) // 4]
    x, y, yaw = fr.pose
    c, s = np.cos(yaw), np.sin(yaw)
    local = (world_pts(fr) - (x, y)) @ np.array([[c, -s], [s, c]])
    a1.scatter(local[:, 0], local[:, 1], s=6, c="tab:purple")
    a1.plot(0, 0, marker=(3, 0, -90), ms=14, c="tab:orange")
    a1.set_title(f"1. One frame (t = {fr.t:.2f} s) seen from the car\n"
                 f"{len(local)} wall points; car at origin facing +x")
    a1.set_xlabel("forward [m]")
    a1.set_ylabel("left [m]")

    # 2. All frames stacked in the map frame, before cleaning.
    a2.scatter(raw[:, 0], raw[:, 1], s=0.3, c="0.4", alpha=0.3)
    a2.plot(acc.poses[:, 0], acc.poses[:, 1], lw=0.8, c="tab:orange")
    a2.set_title(f"2. Stage 1: all {len(frames)} frames stacked in map frame\n"
                 f"{len(raw):,} raw points (duplicates + noise + junk)")

    # 3. After voxel downsample + isolated-point removal.
    a3.scatter(acc.points[:, 0], acc.points[:, 1], s=1, c="0.3")
    a3.set_title(f"3. Stage 1: cleaned\n{len(raw):,} -> {len(acc.points):,} points "
                 f"({cfg.voxel_size * 100:.0f} cm grid + junk removal)")

    # 4. Left / right split relative to the centerline.
    a4.scatter(*acc.points[is_left].T, s=1, c="tab:green", label="left wall points")
    a4.scatter(*acc.points[~is_left].T, s=1, c="tab:red", label="right wall points")
    a4.plot(acc.poses[:, 0], acc.poses[:, 1], lw=0.8, c="tab:orange", label="car path (guide)")
    a4.legend(loc="upper right", fontsize=8, markerscale=6)
    a4.set_title("4. Stage 2: points split left / right\nby the side of the guide line they fall on")

    # 5. Final track.
    a5.scatter(acc.points[:, 0], acc.points[:, 1], s=0.5, c="0.75")
    a5.plot(*loop(track.xy).T, lw=1.5, c="tab:blue", label="centerline")
    a5.plot(*loop(left).T, lw=1, c="tab:green", label="left wall (x,y + w_left)")
    a5.plot(*loop(right).T, lw=1, c="tab:red", label="right wall (x,y - w_right)")
    a5.plot(*track.xy[0], "ko", ms=6, label="s = 0")
    a5.annotate("", xy=track.xy[10], xytext=track.xy[0], arrowprops={"arrowstyle": "->", "lw": 1.5})
    a5.legend(loc="upper right", fontsize=8)
    a5.set_title(f"5. Output: centerline + widths\n{len(track.s)} points every {track.ds * 100:.1f} cm, "
                 f"length {track.length:.2f} m")

    # 6. Width profiles along s (what track.csv's last two columns contain).
    a6.plot(track.s, track.w_left, c="tab:green", label="w_left")
    a6.plot(track.s, track.w_right, c="tab:red", label="w_right")
    a6.set_xlabel("s [m]  (distance along centerline)")
    a6.set_ylabel("width [m]")
    a6.set_ylim(0, max(track.w_left.max(), track.w_right.max()) * 1.3)
    a6.grid(alpha=0.3)
    a6.legend(fontsize=8)
    a6.set_title("6. Output: widths along the track\n(track.csv columns w_left, w_right)")

    for ax in (a1, a2, a3, a4, a5):
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    for ax in (a2, a3, a4, a5):
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
