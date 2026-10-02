"""Test-lap log from a REAL circuit map (e.g. f1tenth_racetracks: real F1
circuits scaled 1:10, map.png/map.yaml occupancy grid + published centerline).

A car drives the published centerline (with a weave, like a real test lap)
and at every frame a simulated 2D LiDAR casts rays against the walls in the
map image; the first wall pixel each ray hits becomes a boundary point.
Walls behind other walls are occluded, as with a real LiDAR.

The track geometry is real; only the sensor readings are simulated. The
published centerline + widths are the ground truth to compare T1 against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import closed_arclength, tangents_normals
from .log_io import Frame
from .synth import GroundTruth


@dataclass
class OccupancyMap:
    occupied: np.ndarray   # (H, W) bool, row 0 = top of image (map_server convention)
    resolution: float      # metres per pixel
    origin: tuple[float, float]  # world (x, y) of the bottom-left pixel

    @property
    def extent(self) -> list[float]:
        h, w = self.occupied.shape
        ox, oy = self.origin
        return [ox, ox + w * self.resolution, oy, oy + h * self.resolution]

    def is_occupied(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Outside the image counts as free (the ray simply finds nothing)."""
        h, w = self.occupied.shape
        col = np.floor((x - self.origin[0]) / self.resolution).astype(int)
        row = h - 1 - np.floor((y - self.origin[1]) / self.resolution).astype(int)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(x.shape, dtype=bool)
        out[inside] = self.occupied[row[inside], col[inside]]
        return out


def _read_simple_yaml(path: Path) -> dict:
    """map_server yaml files are flat 'key: value' lines; avoids a pyyaml dependency."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def load_map(yaml_path: str | Path) -> OccupancyMap:
    import matplotlib.image as mpimg

    yaml_path = Path(yaml_path)
    meta = _read_simple_yaml(yaml_path)
    img = mpimg.imread(yaml_path.parent / meta["image"])
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    if img.max() > 1.0:
        img = img / 255.0
    if int(meta.get("negate", "0")):
        img = 1.0 - img
    occupied_thresh = float(meta.get("occupied_thresh", "0.65"))
    ox, oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
    # map_server: occupancy probability = (1 - brightness) for negate=0
    return OccupancyMap((1.0 - img) > occupied_thresh, float(meta["resolution"]), (ox, oy))


def load_centerline(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f1tenth/TUMFTM format: x_m, y_m, w_tr_right_m, w_tr_left_m -> (xy, w_left, w_right)."""
    d = np.loadtxt(csv_path, delimiter=",", comments="#", ndmin=2)
    return d[:, :2], d[:, 3], d[:, 2]


def ground_truth_from_centerline(xy: np.ndarray, w_left: np.ndarray, w_right: np.ndarray) -> GroundTruth:
    _, n = tangents_normals(xy)
    return GroundTruth(ref=xy, left_wall=xy + n * w_left[:, None], right_wall=xy - n * w_right[:, None])


@dataclass
class RealLapConfig:
    seed: int = 0
    points_frame: str = "vehicle"    # real car: vehicle
    speed: float = 2.0             # m/s (slow test lap)
    dt: float = 0.1                  # s between logged frames (10 Hz)
    laps: float = 1.1
    line_offset: float = 0.3         # weave amplitude around the centerline (m)
    weave_period: float = 15.0       # m of track per weave cycle
    fov: float = np.deg2rad(270)     # typical F1TENTH 2D LiDAR (Hokuyo UST-10LX)
    n_beams: int = 271               # 1 degree spacing
    max_range: float = 10.0
    range_noise: float = 0.02        # gaussian sigma on each range (m)


def raycast(m: OccupancyMap, pose: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
    """Range to the first occupied pixel along each beam; NaN where nothing is hit."""
    step = m.resolution / 4
    r = np.arange(step, max_range, step)
    a = pose[2] + angles
    px = pose[0] + np.outer(r, np.cos(a))
    py = pose[1] + np.outer(r, np.sin(a))
    hit = m.is_occupied(px, py)
    first = hit.argmax(axis=0)
    return np.where(hit.any(axis=0), r[first], np.nan)


def make_log_from_map(m: OccupancyMap, center: np.ndarray, cfg: RealLapConfig) -> list[Frame]:
    rng = np.random.default_rng(cfg.seed)
    _, normals = tangents_normals(center)
    s = closed_arclength(center)
    weave = cfg.line_offset * np.sin(2 * np.pi * s[:-1] / cfg.weave_period)
    path = center + normals * weave[:, None]
    path_s = closed_arclength(path)
    closed = np.vstack([path, path[:1]])
    angles = np.linspace(-cfg.fov / 2, cfg.fov / 2, cfg.n_beams)

    frames = []
    for k in range(int(cfg.laps * path_s[-1] / (cfg.speed * cfg.dt))):
        sk = (k * cfg.speed * cfg.dt) % path_s[-1]
        x = np.interp(sk, path_s, closed[:, 0])
        y = np.interp(sk, path_s, closed[:, 1])
        xa = np.interp(sk + 0.1, path_s, closed[:, 0], period=path_s[-1])
        ya = np.interp(sk + 0.1, path_s, closed[:, 1], period=path_s[-1])
        pose = np.array([x, y, np.arctan2(ya - y, xa - x)])

        ranges = raycast(m, pose, angles, cfg.max_range)
        ok = ~np.isnan(ranges)
        rr = ranges[ok] + rng.normal(0, cfg.range_noise, ok.sum())
        local = np.column_stack([rr * np.cos(angles[ok]), rr * np.sin(angles[ok])])  # vehicle frame
        if cfg.points_frame == "world":
            c, sn = np.cos(pose[2]), np.sin(pose[2])
            pts = local @ np.array([[c, sn], [-sn, c]]) + pose[:2]
        else:
            pts = local
        frames.append(Frame(t=round(k * cfg.dt, 4), pose=pose, velocity=np.array([cfg.speed, 0.0]),
                            boundary_points=pts, obstacles=[]))
    return frames
