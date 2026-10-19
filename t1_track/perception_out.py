"""T1 from perception's apex_track package: centre line + walls -> track.csv columns.

Used INSTEAD of building the track from a log (pipeline.py) when perception has
already mapped the track. apex_track's track folder (track.yaml, centerline.csv,
boundaries.csv) gives everything T1 would produce; T2 -> T3 -> T4 run unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def load_perception_track(track_yaml: str | Path, d_min: float | None = 0.1) -> tuple[np.ndarray, dict]:
    """(N, 5) array [s, x, y, w_left, w_right] + track info, from an apex_track track.yaml.

    d_min: apex_track's tight-corner limit -- the INSIDE wall of a tight corner is cut
    so that 1 - kappa * e_y >= d_min. None = the raw walls from the map.
    """
    from apex_track import load_track       # perception's package (pip install -e <grad-perception>/apex_track)

    tr = load_track(str(track_yaml))
    if not tr.has_boundaries:
        raise ValueError(f"{track_yaml} has no boundaries.csv -- ask perception to make it")

    w_left, w_right = tr.width_at(tr.s0, d_min=d_min)
    data = np.column_stack([tr.s0, tr.p[:, 0], tr.p[:, 1], w_left, w_right])
    info = {
        "track_id": tr.track_id,
        "track_hash": tr.track_hash,
        "boundary_hash": tr.boundary_hash,
        "lap_length": tr.L,
        "n_points": tr.n,
        "d_min": d_min,
        "n_cut": int(np.sum((w_left < tr.w_left - 1e-9) | (w_right < tr.w_right - 1e-9))),
        "n_virtual_left": int(tr.left_virtual.sum()),
        "n_virtual_right": int(tr.right_virtual.sum()),
    }
    return data, info


def write_perception_track(out_dir: str | Path, data: np.ndarray, info: dict,
                           track_yaml: str | Path) -> tuple[Path, Path]:
    """load_perception_track's output -> out_dir/track.csv + track_meta.json (T1's deliverable)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "track.csv", out_dir / "track_meta.json"
    # 6 decimals (1 um), not T1's 4: T2 runs unsmoothed on this line, and 0.1 mm rounding on
    # a 5 cm grid shows up as curvature noise (closed1: up to 0.19 1/m with 4 decimals, 0.07 with 6)
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f", header="s,x,y,w_left,w_right", comments="")

    meta = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "perception apex_track",
        "source_track_yaml": str(Path(track_yaml).resolve()),
        "map_id": info["track_id"],
        "map_version": info["track_hash"],      # flows into racing_line_meta.json and velocity_profile_meta.json
        "boundary_hash": info["boundary_hash"],
        "frame": "map",
        "track": {
            "closed": True,
            "length_m": round(info["lap_length"], 4),
            "n_points": info["n_points"],
            "ds_m": round(info["lap_length"] / info["n_points"], 6),
            "direction": "apex_track lap direction; first point = start/finish line",
            "columns": ["s", "x", "y", "w_left", "w_right"],
            "units": "m",
        },
        "walls": {k: info[k] for k in ("d_min", "n_cut", "n_virtual_left", "n_virtual_right")},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path
