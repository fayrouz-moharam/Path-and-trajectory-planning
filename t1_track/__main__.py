"""CLI.

    python -m t1_track synth out/synth_log.jsonl [--points-frame vehicle] [--seed 1]
    python -m t1_track build out/synth_log.jsonl -o out/track [--points-frame vehicle] [--plot]
    python -m t1_track realtrack Spielberg_map.yaml Spielberg_centerline.csv -o out/spielberg/log.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

from .config import POINTS_FRAMES, T1Config
from .extract import ExtractionError
from .log_io import write_log
from .pipeline import build_track
from .synth import SynthConfig, compare_to_ground_truth, make_log, read_ground_truth, write_ground_truth


def _cmd_synth(args) -> int:
    out = Path(args.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, gt = make_log(SynthConfig(seed=args.seed, points_frame=args.points_frame, laps=args.laps))
    write_log(out, frames)
    gt_path = out.with_name(out.stem + "_ground_truth.csv")
    write_ground_truth(gt_path, gt)
    print(f"wrote {len(frames)} frames -> {out}\nground truth -> {gt_path}")
    return 0


def _cmd_realtrack(args) -> int:
    from .realtrack import RealLapConfig, ground_truth_from_centerline, load_centerline, load_map, make_log_from_map

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    center, w_left, w_right = load_centerline(args.centerline)
    frames = make_log_from_map(load_map(args.map_yaml), center,
                               RealLapConfig(seed=args.seed, points_frame=args.points_frame))
    write_log(out, frames)
    gt_path = out.with_name(out.stem + "_ground_truth.csv")
    write_ground_truth(gt_path, ground_truth_from_centerline(center, w_left, w_right))
    print(f"wrote {len(frames)} frames -> {out}\nground truth (published centerline) -> {gt_path}")
    return 0


def _cmd_build(args) -> int:
    overrides = {f.name: getattr(args, f.name) for f in fields(T1Config) if getattr(args, f.name, None) is not None}
    cfg = T1Config(**overrides)
    out_dir = Path(args.out)
    try:
        res = build_track(args.log, cfg, out_dir, map_version=args.map_version)
    except ExtractionError as e:
        print(f"extraction failed: {e}", file=sys.stderr)
        return 2

    m = res.meta
    print(f"track: {m['track']['length_m']:.2f} m, {m['track']['n_points']} points, ds={m['track']['ds_m']:.4f} m")
    for name, c in m["validation"].items():
        if c["severity"] == "warning":
            status = "ok" if c["passed"] else "WARN"
            extra = f" at s = {c['value']} m" if c["value"] else ""
        else:
            status = "ok" if c["passed"] else "FAIL"
            extra = f" (value={c['value']}, threshold={c['threshold']})" if c["value"] is not None else ""
        print(f"  [{status}] {name}{extra}")
    print(f"wrote {out_dir / 'track.csv'} and {out_dir / 'track_meta.json'}")

    gt = read_ground_truth(args.ground_truth) if args.ground_truth else None
    if gt is not None:
        err = compare_to_ground_truth(res.track.xy, res.track.w_left, res.track.w_right, gt)
        print("vs ground truth (T1 target: centerline RMS <= 1 map cell, width RMS <= 5 cm):")
        print(f"  centerline RMS {err['centerline_rms_m'] * 100:.1f} cm (max {err['centerline_max_m'] * 100:.1f} cm)")
        print(f"  width      RMS {err['width_rms_m'] * 100:.1f} cm (max {err['width_max_abs_m'] * 100:.1f} cm, "
              f"mean bias {err['width_mean_bias_m'] * 100:+.1f} cm)")

    if args.plot:
        from .plot import plot_issues, plot_stages, plot_track
        bg = None
        if args.map_image:
            from .realtrack import load_map
            bg = load_map(args.map_image)
        plot_track(out_dir / "track.png", res.track, res.accumulated, gt, background=bg, validation=m["validation"])
        if plot_issues(out_dir / "track_issues.png", res.track, res.accumulated, m["validation"], gt, bg):
            print(f"warnings zoomed in -> {out_dir / 'track_issues.png'}")
        plot_stages(out_dir / "track_stages.png", res.frames, res.accumulated, res.track, cfg)
        print(f"plots -> {out_dir / 'track.png'}, {out_dir / 'track_stages.png'}")
    return 0 if m["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="t1_track", description="T1 track representation")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synth", help="generate a synthetic test-lap log")
    ps.add_argument("log", help="output .jsonl path")
    ps.add_argument("--points-frame", choices=POINTS_FRAMES, default="world")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--laps", type=float, default=1.15)
    ps.set_defaults(func=_cmd_synth)

    pb = sub.add_parser("build", help="build track.csv + track_meta.json from a log")
    pb.add_argument("log", help="input .jsonl perception log")
    pb.add_argument("-o", "--out", default="out", help="output directory")
    pb.add_argument("--points-frame", dest="points_frame", choices=POINTS_FRAMES)
    for name in ("ds", "voxel_size", "max_point_range", "closure_radius", "min_lap_length", "max_half_width",
                 "width_percentile", "min_width"):
        pb.add_argument("--" + name.replace("_", "-"), dest=name, type=float)
    pb.add_argument("--map-version", help="SLAM map ID/version to record in track_meta.json")
    pb.add_argument("--plot", action="store_true", help="also write track.png")
    pb.add_argument("--ground-truth", help="ground-truth csv: overlay on the plot and print errors")
    pb.add_argument("--map-image", help="map.yaml whose image is drawn under track.png")
    pb.set_defaults(func=_cmd_build)

    pr = sub.add_parser("realtrack", help="generate a test-lap log by driving a simulated LiDAR on a real map")
    pr.add_argument("map_yaml", help="map_server yaml (e.g. Spielberg_map.yaml)")
    pr.add_argument("centerline", help="published centerline csv: x_m, y_m, w_tr_right_m, w_tr_left_m")
    pr.add_argument("-o", "--out", required=True, help="output .jsonl path")
    pr.add_argument("--points-frame", choices=POINTS_FRAMES, default="world")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=_cmd_realtrack)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
