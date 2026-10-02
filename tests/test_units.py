import json

import numpy as np
import pytest

from t1_track.accumulate import remove_isolated, to_world, voxel_downsample
from t1_track.extract import ExtractionError, _fill_circular, extract_lap
from t1_track.geometry import closed_arclength, is_simple_closed, resample_closed, tangents_normals
from t1_track.log_io import parse_frame, read_log

DOC_EXAMPLE = {"t": 12.34,
               "pose": {"x": 1.02, "y": 0.55, "yaw": 0.31},
               "velocity": {"vx": 0.9, "vy": 0.02},
               "boundary_points": [[1.10, 0.80], [1.15, 0.30]],
               "obstacles": [{"id": 3, "x": 2.1, "y": 0.6}]}


def circle(r=2.0, n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


# --- log io ---

def test_parse_doc_example():
    fr = parse_frame(DOC_EXAMPLE)
    assert fr.t == 12.34
    np.testing.assert_allclose(fr.pose, [1.02, 0.55, 0.31])
    assert fr.boundary_points.shape == (2, 2)
    assert fr.obstacles[0]["id"] == 3


def test_read_log_skips_bad_lines_and_sorts(tmp_path):
    p = tmp_path / "log.jsonl"
    late = dict(DOC_EXAMPLE, t=20.0)
    no_pose = {k: v for k, v in DOC_EXAMPLE.items() if k != "pose"}
    p.write_text("\n".join([json.dumps(late), "{not json", json.dumps(no_pose), "",
                            json.dumps(DOC_EXAMPLE)]))
    frames, skipped = read_log(p)
    assert skipped == 2
    assert [f.t for f in frames] == [12.34, 20.0]


def test_empty_boundary_points_ok():
    fr = parse_frame(dict(DOC_EXAMPLE, boundary_points=[]))
    assert fr.boundary_points.shape == (0, 2)


# --- accumulation ---

def test_to_world_rotation_and_translation():
    pose = np.array([1.0, 2.0, np.pi / 2])
    out = to_world(np.array([[1.0, 0.0], [0.0, 1.0]]), pose)
    np.testing.assert_allclose(out, [[1.0, 3.0], [0.0, 2.0]], atol=1e-12)


def test_voxel_downsample_merges_and_filters():
    pts = np.array([[0.01, 0.01], [0.02, 0.03], [0.51, 0.51]])
    out = voxel_downsample(pts, 0.1, min_hits=1)
    assert len(out) == 2
    out = voxel_downsample(pts, 0.1, min_hits=2)
    np.testing.assert_allclose(out, [[0.015, 0.02]])


def test_far_points_are_dropped():
    """Points beyond max_point_range (e.g. no-hit LiDAR beams at range_max) never reach the cloud."""
    from t1_track.accumulate import accumulate
    from t1_track.config import T1Config
    from t1_track.log_io import Frame
    wall = np.column_stack([np.linspace(0, 1, 40), np.full(40, 1.0)])
    far = np.array([[29.99, 0.0], [0.0, -30.01]])
    fr = Frame(0.0, np.zeros(3), np.zeros(2), np.vstack([wall, far]))
    acc = accumulate([fr], T1Config(points_frame="vehicle", outlier_min_neighbors=0))
    assert np.all(np.linalg.norm(acc.points, axis=1) < 10.0)
    assert len(acc.points) > 0


def test_remove_isolated():
    cluster = np.random.default_rng(0).normal(0, 0.02, (20, 2))
    pts = np.vstack([cluster, [[5.0, 5.0]]])
    assert len(remove_isolated(pts, 0.25, 3)) == 20


# --- geometry ---

def test_resample_closed_uniform_and_length():
    pts = resample_closed(circle(2.0, 50), 0.1)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert seg.std() < 1e-2 * seg.mean() + 1e-9
    assert abs(closed_arclength(pts)[-1] - 2 * np.pi * 2.0) < 0.05


def test_normals_point_left():
    c = circle(2.0)  # counter-clockwise
    _, n = tangents_normals(c)
    np.testing.assert_allclose(n, -c / 2.0, atol=1e-3)   # left of CCW travel = inward
    _, n_cw = tangents_normals(c[::-1])
    np.testing.assert_allclose(n_cw, c[::-1] / 2.0, atol=1e-3)  # left of CW travel = outward


def test_is_simple_closed():
    assert is_simple_closed(circle())
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    figure_eight = np.column_stack([np.sin(th), np.sin(th) * np.cos(th)])
    assert not is_simple_closed(figure_eight)          # crosses exactly at a shared vertex
    th2 = th + 0.003                                    # crossing strictly inside two edges
    assert not is_simple_closed(np.column_stack([np.sin(th2), np.sin(th2) * np.cos(th2)]))


# --- extraction helpers ---

def test_extract_lap_cuts_first_lap():
    c = circle(2.0, 200)
    trace = np.vstack([c, c[:30]])  # 1.15 laps
    lap, err = extract_lap(trace, closure_radius=0.3, min_lap_length=5.0)
    assert len(lap) == 200
    assert err < 1e-9


def test_extract_lap_rejects_open_trace():
    with pytest.raises(ExtractionError, match="never returns"):
        extract_lap(circle(2.0, 200)[:150], closure_radius=0.3, min_lap_length=5.0)


def test_fill_circular_wraps():
    v = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    # index 4 and index 0 (== 5) lie between 3.0 at index 3 and 1.0 at index 6 (== 1)
    np.testing.assert_allclose(_fill_circular(v), [5 / 3, 1.0, 2.0, 3.0, 7 / 3])
