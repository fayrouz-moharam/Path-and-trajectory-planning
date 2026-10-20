"""Run T2 + T3 + T4 on perception's track: apex_track track folder -> track.csv (instead of T1)
-> spline + curvature (T2) -> minimum-curvature racing line (T3) -> velocity profile (T4).

    1. pip install -e <grad-perception>/apex_track   (once)
    2. edit the SETTINGS below
    3. python run_perception_t2_t3_t4.py

T1 part = t1_track.perception_out: perception's centre line + walls -> track/track.csv + track_meta.json.
T2 part = run_t1_t2.run_t2() unchanged, smoothing OFF: apex_track already smoothed the line, and
          smoothing it again would move our s away from the s in perception's StateEstimate.
T3 part = run_t1_t2_t3.run_t3() unchanged (car settings CAR_WIDTH, SAFETY_BUFFER, ... live there).
T4 part = run_t1_t2_t3_t4.run_t4() unchanged (vehicle limits live there).

Exit code: 0 = all fine, 2 = no track (apex_track not installed / bad track folder),
3 = T2 rejected the track, 4 = T3 did not settle, 5 = speed profile leaves the shared 0.5-6.0 m/s
domain, 6 = the racing line cannot be indexed by centreline s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import run_t1_t2
import run_t1_t2_t3
import run_t1_t2_t3_t4
from t1_track.perception_out import load_perception_track, write_perception_track

# ============================================================== SETTINGS
TRACK = "closed1"             # track folder inside apex_track/tracks/ ("closed1", "levine", later our own)
TRACK_YAML = None             # or a full path to a track.yaml elsewhere (overrides TRACK)
D_MIN = 0.1                   # apex_track's tight-corner limit on the inside walls (None = raw walls)
OUT_DIR = Path(f"out/run_perception_{TRACK}")
PLOT = True
# =======================================================================


def find_track_yaml(track: str) -> Path:
    """apex_track/tracks/<track>/track.yaml next to the installed (pip install -e) package."""
    import apex_track
    return Path(apex_track.__file__).resolve().parents[1] / "tracks" / track / "track.yaml"


def run_perception(out_dir: Path, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
                   d_min: float | None = D_MIN) -> int:
    """Perception's track -> out_dir/track.csv + track_meta.json (the T1 step)."""
    try:
        yaml_path = Path(track_yaml) if track_yaml else find_track_yaml(track)
        data, info = load_perception_track(yaml_path, d_min=d_min)
    except (ImportError, OSError, ValueError) as e:
        print(f"no track from perception: {e}", file=sys.stderr)
        return 2
    csv_path, meta_path = write_perception_track(out_dir, data, info, yaml_path)
    print(f"  track {info['track_id']} from {yaml_path}")
    print(f"  {info['n_points']} points, lap {info['lap_length']:.2f} m, "
          f"track_hash {info['track_hash']}, boundary_hash {info['boundary_hash']}")
    print(f"  inside walls cut by d_min {d_min}: {info['n_cut']} points; "
          f"virtual walls: {info['n_virtual_left']} left, {info['n_virtual_right']} right")
    print(f"wrote {csv_path} and {meta_path}")
    return 0


def run(out_dir: Path = OUT_DIR, *, track: str = TRACK, track_yaml: Path | None = TRACK_YAML,
        d_min: float | None = D_MIN, plot: bool = PLOT) -> int:
    out_dir = Path(out_dir)
    print("===== T1: track from perception (apex_track)")
    if run_perception(out_dir / "track", track=track, track_yaml=track_yaml, d_min=d_min):
        return 2
    track_csv = out_dir / "track" / "track.csv"

    print("\n===== T2: spline + curvature (no extra smoothing)")
    t2_code = run_t1_t2.run_t2(track_csv, out_dir / "t2", sigma=0.0, plot=plot)
    if t2_code:
        return t2_code

    print("\n===== T3: minimum-curvature racing line")
    t3_code = run_t1_t2_t3.run_t3(track_csv, out_dir / "t3", sigma=0.0, plot=plot)

    print("\n===== T4: velocity profile")
    t4_code = run_t1_t2_t3_t4.run_t4(out_dir / "t3", out_dir / "t4", plot=plot)
    return t3_code or t4_code


if __name__ == "__main__":
    sys.exit(run())
