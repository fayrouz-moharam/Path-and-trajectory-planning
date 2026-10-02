"""Synthetic lap -> full pipeline -> compare against ground truth."""
import json

import numpy as np
import pytest

from t1_track import T1Config, build_track
from t1_track.export import read_track_csv
from t1_track.extract import Track
from t1_track.geometry import closed_arclength, resample_closed, tangents_normals
from t1_track.log_io import write_log
from t1_track.synth import GroundTruth, SynthConfig, compare_to_ground_truth, make_log
from t1_track.validate import is_valid, validate, warnings
import run_t1


def run(tmp_path, synth_cfg, cfg=None):
    frames, gt = make_log(synth_cfg)
    log = tmp_path / "log.jsonl"
    write_log(log, frames)
    cfg = cfg or T1Config(points_frame=synth_cfg.points_frame)
    return build_track(log, cfg, tmp_path / "out"), gt


@pytest.mark.parametrize("frame", ["world", "vehicle"])
def test_matches_ground_truth(tmp_path, frame):
    res, gt = run(tmp_path, SynthConfig(points_frame=frame, seed=1))
    tr = res.track
    assert res.meta["valid"], res.meta["validation"]

    err = compare_to_ground_truth(tr.xy, tr.w_left, tr.w_right, gt)
    # Breakdown T1 "done when": centerline RMS <= 1 cell (5 cm), width RMS <= 5 cm.
    assert err["centerline_rms_m"] <= 0.05, err
    assert err["width_rms_m"] <= 0.05, err
    # Tighter than the target in practice, and walls land on the correct side.
    assert err["centerline_rms_m"] < 0.02, err
    assert err["left_wall_rms_m"] < 0.03 and err["right_wall_rms_m"] < 0.03, err
    assert np.max(np.abs(tr.w_left - tr.w_right)) < 0.1
    # Length matches the true midline's.
    mid = gt.midline
    true_len = np.linalg.norm(np.diff(np.vstack([mid, mid[:1]]), axis=0), axis=1).sum()
    assert abs(tr.length - true_len) / true_len < 0.01


def test_wrong_points_frame_is_caught(tmp_path):
    """Vehicle-frame points read as world-frame must not silently produce a valid track."""
    frames, _ = make_log(SynthConfig(points_frame="vehicle"))
    log = tmp_path / "log.jsonl"
    write_log(log, frames)
    try:
        res = build_track(log, T1Config(points_frame="world"))
    except RuntimeError:
        return
    assert not res.meta["valid"]


def test_outputs_format(tmp_path):
    res, _ = run(tmp_path, SynthConfig())
    data = read_track_csv(tmp_path / "out" / "track.csv")
    header = (tmp_path / "out" / "track.csv").read_text().splitlines()[0]
    assert header == "s,x,y,w_left,w_right"
    assert data.shape[1] == 5
    assert data[0, 0] == 0.0 and np.all(np.diff(data[:, 0]) > 0)
    meta = json.loads((tmp_path / "out" / "track_meta.json").read_text())
    assert meta["track"]["n_points"] == len(data)
    assert meta["source_log"]["n_frames"] > 0 and len(meta["source_log"]["sha256"]) == 64
    assert meta["config"]["points_frame"] == "vehicle"
    # fields required by the breakdown's T1 output spec
    for key in ("map_version", "frame", "resolution_m", "start_pose"):
        assert key in meta
    assert meta["track"]["direction"] and meta["frame"] == "map"
    assert "inside_offset_curvature" not in meta["validation"]  # curvature is T2/T3/T6, not T1


def test_s0_near_first_pose_and_driving_direction(tmp_path):
    res, _ = run(tmp_path, SynthConfig())
    tr, poses = res.track, res.accumulated.poses
    assert np.linalg.norm(tr.xy[0] - poses[0, :2]) < 0.5
    heading = tr.xy[1] - tr.xy[0]
    assert heading @ [np.cos(poses[0, 2]), np.sin(poses[0, 2])] > 0


def test_pose_noise_still_valid(tmp_path):
    res, gt = run(tmp_path, SynthConfig(pose_noise=0.02, seed=3))
    assert res.meta["valid"], res.meta["validation"]
    tr = res.track
    assert compare_to_ground_truth(tr.xy, tr.w_left, tr.w_right, gt)["centerline_rms_m"] < 0.05


def test_sparse_ground_truth_does_not_inflate_error():
    """Real published centerlines have ~40 cm spacing; a perfect track must still score ~0."""
    c = rounded_square(r=3.0)
    _, n = tangents_normals(c)
    gt_sparse = GroundTruth(c[::8], (c + n * 1.1)[::8], (c - n * 1.1)[::8])  # 40 cm spacing
    err = compare_to_ground_truth(c, np.full(len(c), 1.1), np.full(len(c), 1.1), gt_sparse)
    assert err["centerline_rms_m"] < 0.005 and err["width_rms_m"] < 0.005, err


def rounded_square(side=10.0, r=0.5, ds=0.05):
    """Counter-clockwise square with corner radius r (left = inside)."""
    h = side / 2 - r
    arcs = []
    for (cx, cy), a0 in zip([(h, h), (-h, h), (-h, -h), (h, -h)], [0, 90, 180, 270]):
        th = np.radians(np.linspace(a0, a0 + 90, 30))
        arcs.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))
    return resample_closed(np.vstack(arcs), ds)


def test_sharp_corner_wall_fold_is_warning_not_error():
    """Corner radius 0.5 m but inside width 1.0 m: the rebuilt inside wall must fold
    at all 4 corners -> flagged as a warning with its locations; track stays valid."""
    c = rounded_square(r=0.5)
    s = closed_arclength(c)
    _, n = tangents_normals(c)
    track = Track(s=s[:-1], xy=c, w_left=np.full(len(c), 1.0), w_right=np.full(len(c), 1.0),
                  length=float(s[-1]), ds=float(s[-1] / len(c)),
                  stats={"lap_closure_error_m": 0.0, "test_lap_length_m": float(s[-1]),
                         "gap_fraction_left": 0.0, "gap_fraction_right": 0.0})
    report = validate(track, c - n * 1.0, T1Config())
    assert len(report["left_wall_folds"]["value"]) == 4
    assert report["left_wall_folds"]["severity"] == "warning"
    assert report["right_wall_folds"]["passed"]
    assert is_valid(report) and warnings(report) == ["left_wall_folds"]


def test_incomplete_lap_fails_cleanly(tmp_path, capsys):
    frames, _ = make_log(SynthConfig(laps=0.6))
    log = tmp_path / "log.jsonl"
    write_log(log, frames)
    assert run_t1.run("log", tmp_path / "out", log_path=log, plot=False) == 2
    assert "never returns" in capsys.readouterr().err


def test_run_script_roundtrip(tmp_path):
    assert run_t1.run("synth", tmp_path, plot=False) == 0
    assert (tmp_path / "log.jsonl").exists() and (tmp_path / "log_ground_truth.csv").exists()
    assert (tmp_path / "track" / "track.csv").exists() and (tmp_path / "track" / "track_meta.json").exists()
