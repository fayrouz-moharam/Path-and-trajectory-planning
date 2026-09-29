"""Reading / writing the per-frame perception log (T1_ASSUMPTIONS.md §3).

One JSON object per line:
    {"t": 12.34,
     "pose": {"x": 1.02, "y": 0.55, "yaw": 0.31},
     "velocity": {"vx": 0.9, "vy": 0.02},
     "boundary_points": [[1.10, 0.80], [1.15, 0.30]],
     "obstacles": [{"id": 3, "x": 2.1, "y": 0.6}]}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    t: float
    pose: np.ndarray             # (3,) x, y, yaw
    velocity: np.ndarray         # (2,) vx, vy -- exact form unconfirmed, carried through as-is
    boundary_points: np.ndarray  # (N, 2), frame given by T1Config.points_frame
    obstacles: list[dict] = field(default_factory=list)  # carried through, unused by T1


def parse_frame(obj: dict) -> Frame:
    pose = obj["pose"]
    pose_arr = np.array([pose["x"], pose["y"], pose["yaw"]], dtype=float)
    if not np.all(np.isfinite(pose_arr)):
        raise ValueError("non-finite pose")
    vel = obj.get("velocity") or {}
    pts = np.asarray(obj.get("boundary_points") or [], dtype=float).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return Frame(
        t=float(obj["t"]),
        pose=pose_arr,
        velocity=np.array([vel.get("vx", np.nan), vel.get("vy", np.nan)], dtype=float),
        boundary_points=pts,
        obstacles=list(obj.get("obstacles") or []),
    )


def frame_to_dict(fr: Frame) -> dict:
    return {
        "t": round(fr.t, 4),
        "pose": {"x": round(fr.pose[0], 4), "y": round(fr.pose[1], 4), "yaw": round(fr.pose[2], 5)},
        "velocity": {"vx": round(fr.velocity[0], 4), "vy": round(fr.velocity[1], 4)},
        "boundary_points": np.round(fr.boundary_points, 4).tolist(),
        "obstacles": fr.obstacles,
    }


def read_log(path: str | Path) -> tuple[list[Frame], int]:
    """Return (frames sorted by t, number of malformed lines skipped)."""
    frames: list[Frame] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(parse_frame(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                skipped += 1
                log.warning("%s:%d: skipping malformed frame (%s)", path, lineno, e)
    if not frames:
        raise ValueError(f"{path}: no valid frames")
    frames.sort(key=lambda fr: fr.t)
    return frames, skipped


def write_log(path: str | Path, frames: list[Frame]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(frame_to_dict(fr)) + "\n")
