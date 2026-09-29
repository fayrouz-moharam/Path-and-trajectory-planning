"""Writing the T1 deliverable: track.csv (s, x, y, w_left, w_right) + track_meta.json."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import T1Config
from .extract import Track
from .validate import is_valid, warnings

FORMAT_VERSION = 1


def write_track_csv(path: Path, track: Track) -> None:
    data = np.column_stack([track.s, track.xy, track.w_left, track.w_right])
    np.savetxt(path, data, delimiter=",", fmt="%.4f", header="s,x,y,w_left,w_right", comments="")


def read_track_csv(path: Path) -> np.ndarray:
    """(N, 5) array: s, x, y, w_left, w_right."""
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_meta(track: Track, cfg: T1Config, validation: dict, source: dict,
               map_version: str, start_pose: dict) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assumptions": "docs/T1_ASSUMPTIONS.md",
        "map_version": map_version,
        "frame": cfg.frame_id,
        "resolution_m": cfg.voxel_size,
        "start_pose": start_pose,
        "source_log": source,
        "track": {
            "closed": True,
            "length_m": round(track.length, 4),
            "n_points": len(track.s),
            "ds_m": round(track.ds, 6),
            "direction": "driving direction of the test lap; w_left/w_right are "
                         "relative to that direction",
            "s0": "centerline point laterally abreast of the first logged pose",
            "columns": ["s", "x", "y", "w_left", "w_right"],
            "units": "m",
        },
        "stats": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in track.stats.items()},
        "config": asdict(cfg),
        "valid": is_valid(validation),
        "warnings": warnings(validation),
        "validation": validation,
    }


def write_outputs(out_dir: Path, track: Track, meta: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "track.csv", out_dir / "track_meta.json"
    write_track_csv(csv_path, track)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path
