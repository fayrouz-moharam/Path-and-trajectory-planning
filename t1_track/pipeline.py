"""End-to-end T1: perception log -> track.csv + track_meta.json."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .accumulate import Accumulated, accumulate
from .config import T1Config
from .export import build_meta, sha256_file, write_outputs
from .extract import Track, extract_track
from .log_io import Frame, read_log
from .validate import validate


@dataclass
class BuildResult:
    track: Track
    accumulated: Accumulated
    meta: dict
    frames: list[Frame]  # raw frames, kept for the stage-by-stage plot


def build_track(log_path: str | Path, cfg: T1Config, out_dir: str | Path | None = None,
                map_version: str | None = None) -> BuildResult:
    """map_version: ID/version of the SLAM map the log belongs to (any new map
    means T1-T5 must be regenerated, breakdown §3). Defaults to the log's hash."""
    log_path = Path(log_path)
    frames, n_skipped = read_log(log_path)

    acc = accumulate(frames, cfg)
    track = extract_track(acc.poses, acc.points, cfg)
    track.stats["n_raw_points"] = acc.n_raw_points
    track.stats["n_filtered_points"] = len(acc.points)

    log_hash = sha256_file(log_path)
    source = {
        "path": str(log_path),
        "sha256": log_hash,
        "n_frames": len(frames),
        "n_skipped_lines": n_skipped,
        "t_start": frames[0].t,
        "t_end": frames[-1].t,
    }
    x0, y0, yaw0 = acc.poses[0]
    start_pose = {"x": round(float(x0), 4), "y": round(float(y0), 4), "yaw": round(float(yaw0), 5)}
    meta = build_meta(track, cfg, validate(track, acc.points, cfg), source,
                      map_version=map_version or f"log-sha256:{log_hash[:12]}", start_pose=start_pose)
    if out_dir is not None:
        write_outputs(Path(out_dir), track, meta)
    return BuildResult(track, acc, meta, frames)
