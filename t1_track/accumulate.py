"""Stage 1 — merge the per-frame log into one world-frame boundary point cloud.

Pose is treated as drift-free (T1_ASSUMPTIONS.md §2); no loop-closure
correction is applied. Obstacles are ignored by T1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import T1Config
from .log_io import Frame


@dataclass
class Accumulated:
    t: np.ndarray       # (F,) timestamps
    poses: np.ndarray   # (F, 3) x, y, yaw in world frame
    points: np.ndarray  # (M, 2) filtered boundary points in world frame
    n_raw_points: int


def to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + (x, y)


def voxel_downsample(points: np.ndarray, cell: float, min_hits: int = 1) -> np.ndarray:
    """Replace all points in each grid cell by their centroid; drop sparsely hit cells."""
    if len(points) == 0:
        return points
    keys = np.floor(points / cell).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    sums = np.zeros((len(counts), 2))
    np.add.at(sums, inv, points)
    centroids = sums / counts[:, None]
    return centroids[counts >= min_hits]


def remove_isolated(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(points) == 0 or min_neighbors <= 0:
        return points
    n = cKDTree(points).query_ball_point(points, r=radius, return_length=True) - 1
    return points[n >= min_neighbors]


def accumulate(frames: list[Frame], cfg: T1Config) -> Accumulated:
    poses = np.array([fr.pose for fr in frames])
    chunks = []
    for fr in frames:
        if len(fr.boundary_points) == 0:
            continue
        pts = to_world(fr.boundary_points, fr.pose) if cfg.points_frame == "vehicle" else fr.boundary_points
        near = np.linalg.norm(pts - fr.pose[:2], axis=1) <= cfg.max_point_range
        chunks.append(pts[near])
    raw = np.vstack(chunks) if chunks else np.empty((0, 2))

    pts = voxel_downsample(raw, cfg.voxel_size, cfg.min_hits_per_voxel)
    pts = remove_isolated(pts, cfg.outlier_radius, cfg.outlier_min_neighbors)
    return Accumulated(
        t=np.array([fr.t for fr in frames]),
        poses=poses,
        points=pts,
        n_raw_points=len(raw),
    )
