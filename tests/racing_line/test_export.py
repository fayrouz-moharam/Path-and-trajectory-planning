"""Step 6: final racing line, start point, widths, CSV/JSON export and loader."""
import json

import numpy as np
import pytest

pytest.importorskip("osqp")

from t3_racing_line.export import (build_meta, build_racing_line, load_racing_line,
                                   read_racing_line, write_racing_line)
from t3_racing_line.min_curvature import iterate_min_curvature, kappa_max_from_steering
from t2_curvature.queryable_track import QueryableTrack
from t3_racing_line.stations import track_from_points

R, MARGIN = 5.0, 0.2
KAPPA_MAX = kappa_max_from_steering(max_steer=0.4, wheelbase=0.33)


def circle(n=600) -> QueryableTrack:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3   # start not on the x axis
    return track_from_points(R * np.cos(th), R * np.sin(th), np.full(n, 1.0), np.full(n, 0.8))


def t1_shape(n=4000) -> QueryableTrack:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 4.0 + 0.8 * np.sin(2 * th) + 0.5 * np.cos(3 * th)
    return track_from_points(1.4 * r * np.cos(th), r * np.sin(th), np.full(n, 0.9), np.full(n, 0.8))


def run(track):
    res = iterate_min_curvature(track, 0.15, MARGIN, kappa_max=KAPPA_MAX)
    return res, build_racing_line(res, track, final_spacing=0.10)


def test_circle_line_is_outer_edge():
    # outer wall at R + 0.8; the loop plans with margin + 1 cm buffer
    res, line = run(circle())
    r_line = R + 0.8 - (MARGIN + res.plan_buffer)                       # 5.59
    assert np.allclose(np.hypot(line.x, line.y), r_line, atol=2e-3)
    # kappa from the spline. 0.3 %: the sideways walk measures room to wall points
    # (1 cm steps), so the line wobbles by ~0.06 mm and kappa (a 2nd derivative) by ~0.2 %
    assert np.allclose(line.kappa, 1 / r_line, rtol=3e-3)
    assert np.allclose(line.w_right, MARGIN + res.plan_buffer, atol=2e-3)
    assert np.allclose(line.w_left, 1.8 - MARGIN - res.plan_buffer, atol=2e-3)


def test_uniform_spacing_and_heading_range():
    _, line = run(t1_shape())
    assert line.s[0] == 0.0
    assert np.allclose(np.diff(line.s), line.length / len(line.s))
    assert abs(np.diff(line.s)[0] - 0.10) < 0.001                          # ~0.10 m
    gaps = np.hypot(np.diff(line.x), np.diff(line.y))
    assert np.allclose(gaps, np.diff(line.s), rtol=1e-2)                   # s = real distance
    assert np.all((line.heading > -np.pi) & (line.heading <= np.pi))


def test_start_is_on_centreline_start_line():
    track = t1_shape()
    _, line = run(track)
    _, x0, y0, psi0, *_ = track.sample_arrays(0.0)
    along = (line.x[0] - x0) * np.cos(psi0) + (line.y[0] - y0) * np.sin(psi0)
    assert abs(along) < 1e-3                                              # abreast of r0
    assert np.hypot(line.x[0] - x0, line.y[0] - y0) < 1.0                 # and on this side


def test_widths_respect_margin():
    _, line = run(t1_shape())
    assert line.w_left.min() > MARGIN - 0.01 and line.w_right.min() > MARGIN - 0.01


def test_roundtrip_csv_trackmodel_csv(tmp_path):
    track = t1_shape()
    res, line = run(track)
    meta = build_meta(line, res, kappa_max=KAPPA_MAX, final_spacing=0.10, car_width=0.30,
                      safety_buffer=0.05)
    csv_path, meta_path = write_racing_line(tmp_path, line, meta)

    tm = load_racing_line(csv_path, meta_path)
    assert tm.meta["method"] == "min_curvature_qp" and tm.meta["converged"] is True
    assert abs(tm.length - line.length) < 1e-3

    # CSV -> T2 track -> sampled again at the same s: same numbers
    s, x, y, heading, kappa, w_left, w_right = tm.sample_arrays(line.s)
    orig = read_racing_line(csv_path)
    assert np.max(np.hypot(x - orig[:, 1], y - orig[:, 2])) < 1e-4
    assert np.max(np.abs(np.angle(np.exp(1j * (heading - orig[:, 3]))))) < 1e-3
    assert np.max(np.abs(kappa - orig[:, 4])) < 1e-2
    assert np.allclose(w_left, orig[:, 5]) and np.allclose(w_right, orig[:, 6])


def test_deterministic():
    track = t1_shape()
    a = run(track)[1].as_array()
    b = run(track)[1].as_array()
    assert np.array_equal(a, b)
