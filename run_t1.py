"""Run T1 in one go: get a test-lap log, build the track, print the checks, draw it.

    1. edit the SETTINGS below
    2. python run_t1.py

Exit code: 0 = valid track, 1 = built but a check failed, 2 = no track (e.g. incomplete lap).
"""
from __future__ import annotations

import sys
from pathlib import Path

from t1_track import T1Config, build_track
from t1_track.extract import ExtractionError
from t1_track.log_io import write_log
from t1_track.synth import SynthConfig, compare_to_ground_truth, make_log, read_ground_truth, write_ground_truth

# ============================================================== SETTINGS
SOURCE = "synth"          # where the test-lap log comes from:
                          #   "synth"     -> fake track made by synth.py (known ground truth)
                          #   "realtrack" -> drive a simulated LiDAR around a real map (realtrack.py)
                          #   "log"       -> an existing log file (the real car's, or out/gym_*/log.jsonl)
OUT_DIR = Path("out/run") # everything is written here: log, ground truth, track/, plots
PLOT = True               # also write track.png + track_stages.png (+ track_issues.png on warnings)
SEED = 0                  # random seed for "synth" / "realtrack" (same seed = same log)

# used when SOURCE = "realtrack"
MAP_YAML = Path("out/real_tracks/Spielberg_map.yaml")
CENTERLINE_CSV = Path("out/real_tracks/Spielberg_centerline.csv")

# used when SOURCE = "log"
LOG_PATH = Path("out/gym_spielberg/log.jsonl")
GROUND_TRUTH_CSV = None   # optional, e.g. Path("out/spielberg/log_ground_truth.csv")
MAP_IMAGE_YAML = None     # optional map drawn under track.png, e.g. Path("out/real_tracks/Spielberg_map.yaml")

CONFIG = T1Config()       # every T1 setting (config.py); change some like T1Config(ds=0.10)
MAP_VERSION = None        # SLAM map ID for track_meta.json (None = the log's hash)
# =======================================================================


def get_log(source: str, out_dir: Path, seed: int, map_yaml: Path, centerline_csv: Path,
            log_path: Path | None, ground_truth_csv: Path | None, map_image_yaml: Path | None):
    """Make (or pick) the input log. Returns (log, ground truth csv or None, map yaml for the plot or None)."""
    if source == "log":
        return Path(log_path), ground_truth_csv, map_image_yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "log.jsonl"
    gt_csv = out_dir / "log_ground_truth.csv"
    if source == "synth":
        frames, gt = make_log(SynthConfig(seed=seed))
        background = None
    elif source == "realtrack":
        from t1_track.realtrack import (RealLapConfig, ground_truth_from_centerline, load_centerline,
                                        load_map, make_log_from_map)
        center, w_left, w_right = load_centerline(centerline_csv)
        frames = make_log_from_map(load_map(map_yaml), center, RealLapConfig(seed=seed))
        gt = ground_truth_from_centerline(center, w_left, w_right)
        background = map_yaml
    else:
        raise ValueError(f'SOURCE must be "synth", "realtrack" or "log", got {source!r}')
    write_log(log, frames)
    write_ground_truth(gt_csv, gt)
    print(f"log: {len(frames)} frames -> {log}")
    return log, gt_csv, background


def print_report(meta: dict) -> None:
    t = meta["track"]
    print(f"track: {t['length_m']:.2f} m, {t['n_points']} points, ds={t['ds_m']:.4f} m")
    for name, c in meta["validation"].items():
        if c["severity"] == "warning":
            status = "ok" if c["passed"] else "WARN"
            extra = f" at s = {c['value']} m" if c["value"] else ""
        else:
            status = "ok" if c["passed"] else "FAIL"
            extra = f" (value={c['value']}, threshold={c['threshold']})" if c["value"] is not None else ""
        print(f"  [{status}] {name}{extra}")


def run(source: str = SOURCE, out_dir: Path = OUT_DIR, *, plot: bool = PLOT, seed: int = SEED,
        map_yaml: Path = MAP_YAML, centerline_csv: Path = CENTERLINE_CSV, log_path: Path | None = LOG_PATH,
        ground_truth_csv: Path | None = GROUND_TRUTH_CSV, map_image_yaml: Path | None = MAP_IMAGE_YAML,
        cfg: T1Config = CONFIG, map_version: str | None = MAP_VERSION) -> int:
    out_dir = Path(out_dir)
    log, gt_csv, background = get_log(source, out_dir, seed, map_yaml, centerline_csv,
                                      log_path, ground_truth_csv, map_image_yaml)

    track_dir = out_dir / "track"
    try:
        res = build_track(log, cfg, track_dir, map_version=map_version)
    except ExtractionError as e:
        print(f"extraction failed: {e}", file=sys.stderr)
        return 2
    print_report(res.meta)
    print(f"wrote {track_dir / 'track.csv'} and {track_dir / 'track_meta.json'}")

    gt = read_ground_truth(gt_csv) if gt_csv else None
    if gt is not None:
        err = compare_to_ground_truth(res.track.xy, res.track.w_left, res.track.w_right, gt)
        print("vs ground truth (T1 target: centerline RMS <= 1 map cell, width RMS <= 5 cm):")
        print(f"  centerline RMS {err['centerline_rms_m'] * 100:.1f} cm (max {err['centerline_max_m'] * 100:.1f} cm)")
        print(f"  width      RMS {err['width_rms_m'] * 100:.1f} cm (max {err['width_max_abs_m'] * 100:.1f} cm, "
              f"mean bias {err['width_mean_bias_m'] * 100:+.1f} cm)")

    if plot:
        from t1_track.plot import plot_issues, plot_stages, plot_track
        bg = None
        if background:
            from t1_track.realtrack import load_map
            bg = load_map(background)
        m = res.meta
        plot_track(track_dir / "track.png", res.track, res.accumulated, gt, background=bg, validation=m["validation"])
        plot_issues(track_dir / "track_issues.png", res.track, res.accumulated, m["validation"], gt, bg)
        plot_stages(track_dir / "track_stages.png", res.frames, res.accumulated, res.track, cfg)
        print(f"plots -> {track_dir}")
    return 0 if res.meta["valid"] else 1


if __name__ == "__main__":
    sys.exit(run())
