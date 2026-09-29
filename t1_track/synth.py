"""Synthetic test-lap generator: a known closed track plus a noisy per-frame
perception log in the T1_ASSUMPTIONS.md §3 format. Lets every stage be
checked against ground truth before real perception data exists."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import closed_arclength, resample_closed, tangents_normals
from .log_io import Frame


def _curvature(pts: np.ndarray) -> np.ndarray:
    """Signed curvature of the (exact, dense) synthetic curve -- only used to
    reject impossible synthetic tracks. T1 itself never computes curvature (T2's job)."""
    nxt, prv = np.roll(pts, -1, axis=0), np.roll(pts, 1, axis=0)
    d1 = (nxt - prv) / 2.0
    d2 = nxt - 2.0 * pts + prv
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3


@dataclass
class SynthConfig:
    seed: int = 0
    points_frame: str = "world"       # frame to express boundary points in
    # track shape: radius-modulated loop, x stretched
    base_radius: float = 4.0
    x_stretch: float = 1.4
    # half widths (m) at each station vary smoothly around these
    w_left: float = 0.9
    w_right: float = 0.8
    # driving
    speed: float = 2.0                # m/s
    dt: float = 0.05                  # s between frames
    laps: float = 1.15
    line_offset: float = 0.3          # amplitude of the car's weave around the centerline
    # sensor
    sensor_range: float = 4.0
    sensor_fov: float = np.deg2rad(270)
    keep_fraction: float = 0.3        # fraction of in-view wall points reported per frame
    point_noise: float = 0.02         # gaussian sigma on each boundary point
    outlier_rate: float = 0.01        # fraction of spurious points per frame
    pose_noise: float = 0.0           # gaussian sigma on logged x, y (0 = drift-free, per §2)


@dataclass
class GroundTruth:
    ref: np.ndarray         # (N, 2) generating curve (NOT the midline: offsets are asymmetric)
    left_wall: np.ndarray   # (N, 2)
    right_wall: np.ndarray  # (N, 2)

    @property
    def midline(self) -> np.ndarray:
        return (self.left_wall + self.right_wall) / 2

    @property
    def track_width(self) -> np.ndarray:
        return np.linalg.norm(self.left_wall - self.right_wall, axis=1)


def make_track(cfg: SynthConfig, ds: float = 0.05) -> GroundTruth:
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r = cfg.base_radius + 0.8 * np.sin(2 * th) + 0.5 * np.cos(3 * th)
    xy = resample_closed(np.column_stack([cfg.x_stretch * r * np.cos(th), r * np.sin(th)]), ds)
    s = closed_arclength(xy)
    phase = 2 * np.pi * s[:-1] / s[-1]
    wl = cfg.w_left + 0.2 * np.sin(phase)
    wr = cfg.w_right + 0.15 * np.cos(2 * phase)
    kappa = _curvature(xy)
    if np.any(np.where(kappa > 0, kappa * wl, -kappa * wr) >= 0.9):
        raise ValueError("synthetic track too tight for its widths")
    _, normals = tangents_normals(xy)
    return GroundTruth(xy, xy + normals * wl[:, None], xy - normals * wr[:, None])


def make_log(cfg: SynthConfig) -> tuple[list[Frame], GroundTruth]:
    rng = np.random.default_rng(cfg.seed)
    gt = make_track(cfg)
    _, normals = tangents_normals(gt.ref)
    ref_s = closed_arclength(gt.ref)
    walls = np.vstack([gt.left_wall, gt.right_wall])

    # Car path: generating curve plus a lateral weave that stays well inside the walls.
    weave = cfg.line_offset * np.sin(2 * np.pi * 4 * ref_s[:-1] / ref_s[-1])
    path = gt.ref + normals * weave[:, None]
    path_s = closed_arclength(path)
    closed_path = np.vstack([path, path[:1]])

    n_frames = int(cfg.laps * path_s[-1] / (cfg.speed * cfg.dt))
    lo, hi = walls.min(axis=0) - 1, walls.max(axis=0) + 1
    obstacles_world = [gt.ref[len(gt.ref) // 3], gt.ref[2 * len(gt.ref) // 3]]

    frames = []
    for k in range(n_frames):
        s_k = (k * cfg.speed * cfg.dt) % path_s[-1]
        x = np.interp(s_k, path_s, closed_path[:, 0])
        y = np.interp(s_k, path_s, closed_path[:, 1])
        xa = np.interp(s_k + 0.05, path_s, closed_path[:, 0], period=path_s[-1])
        ya = np.interp(s_k + 0.05, path_s, closed_path[:, 1], period=path_s[-1])
        yaw = np.arctan2(ya - y, xa - x)

        rel = walls - (x, y)
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        visible = (np.linalg.norm(rel, axis=1) <= cfg.sensor_range) & (np.abs(bearing) <= cfg.sensor_fov / 2)
        pts = walls[visible]
        pts = pts[rng.random(len(pts)) < cfg.keep_fraction]
        pts = pts + rng.normal(0, cfg.point_noise, pts.shape)
        n_out = rng.binomial(max(len(pts), 1), cfg.outlier_rate)
        pts = np.vstack([pts, rng.uniform(lo, hi, (n_out, 2))])

        pose = np.array([x, y, yaw])
        if cfg.points_frame == "vehicle":
            c, s = np.cos(yaw), np.sin(yaw)
            pts = (pts - (x, y)) @ np.array([[c, -s], [s, c]])
        if cfg.pose_noise > 0:
            pose[:2] += rng.normal(0, cfg.pose_noise, 2)

        obstacles = [{"id": i, "x": round(float(o[0]), 3), "y": round(float(o[1]), 3)}
                     for i, o in enumerate(obstacles_world)
                     if np.linalg.norm(o - (x, y)) <= cfg.sensor_range]
        frames.append(Frame(
            t=k * cfg.dt,
            pose=pose,
            velocity=np.array([cfg.speed, 0.0]),
            boundary_points=pts,
            obstacles=obstacles,
        ))
    return frames, gt


def densify_ground_truth(gt: GroundTruth, max_step: float) -> GroundTruth:
    """Linearly subdivide every edge so vertices are <= max_step apart; left/right
    walls are subdivided together so their correspondence (midline, width) holds."""
    def sub(a, k):
        nxt = np.roll(a, -1, axis=0)
        f = np.arange(k) / k
        return (a[:, None, :] + (nxt - a)[:, None, :] * f[None, :, None]).reshape(-1, 2)

    step = max(np.linalg.norm(np.diff(a, axis=0), axis=1).max() for a in (gt.ref, gt.left_wall, gt.right_wall))
    k = max(1, int(np.ceil(step / max_step)))
    return GroundTruth(sub(gt.ref, k), sub(gt.left_wall, k), sub(gt.right_wall, k))


def compare_to_ground_truth(xy: np.ndarray, w_left: np.ndarray, w_right: np.ndarray, gt: GroundTruth) -> dict:
    """Errors of an extracted track vs ground truth (metres). Width error compares
    total track width, so it doesn't depend on how the midline is defined."""
    from scipy.spatial import cKDTree

    # Densify the ground truth to ~1 cm first: measuring to the nearest vertex of a
    # sparse polyline (e.g. 40 cm spacing) would report up to half the spacing as error.
    gt = densify_ground_truth(gt, 0.01)
    d_mid, i = cKDTree(gt.midline).query(xy)
    _, n = tangents_normals(xy)
    d_left = cKDTree(gt.left_wall).query(xy + n * w_left[:, None])[0]
    d_right = cKDTree(gt.right_wall).query(xy - n * w_right[:, None])[0]
    w_err = (w_left + w_right) - gt.track_width[i]

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2)))

    return {
        "centerline_rms_m": rms(d_mid), "centerline_max_m": float(d_mid.max()),
        "width_rms_m": rms(w_err), "width_max_abs_m": float(np.abs(w_err).max()),
        "width_mean_bias_m": float(w_err.mean()),
        "left_wall_rms_m": rms(d_left), "right_wall_rms_m": rms(d_right),
    }


def write_ground_truth(path, gt: GroundTruth) -> None:
    data = np.column_stack([gt.midline, gt.left_wall, gt.right_wall])
    np.savetxt(path, data, delimiter=",", fmt="%.4f",
               header="mid_x,mid_y,left_x,left_y,right_x,right_y", comments="")


def read_ground_truth(path) -> GroundTruth:
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return GroundTruth(ref=d[:, 0:2], left_wall=d[:, 2:4], right_wall=d[:, 4:6])  # ref not stored; midline stands in
