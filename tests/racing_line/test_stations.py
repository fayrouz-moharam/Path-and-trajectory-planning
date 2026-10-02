"""Step 1: make_stations -- spacing, normals, bounds, narrow-track error."""
import numpy as np
import pytest

from t3_racing_line.stations import build_walls, make_stations, vehicle_margin, widths_from_walls
from t3_racing_line.track_model_stub import TrackModel

R = 5.0


def circle(w_left=1.0, w_right=0.8, n=600) -> TrackModel:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)  # CCW = left-turning
    return TrackModel(R * np.cos(th), R * np.sin(th), np.full(n, w_left), np.full(n, w_right))


def test_uniform_spacing_no_duplicate_endpoint():
    tm = circle()
    st = make_stations(tm, spacing_target=0.15, margin=0.2)
    assert len(st) == round(tm.length / 0.15)
    assert np.isclose(len(st) * st.ds, tm.length)          # N * ds closes the loop exactly
    assert np.allclose(np.diff(st.s), st.ds)
    assert st.s[-1] < tm.length - st.ds / 2                # no station at s = L
    gaps = np.linalg.norm(np.diff(st.xy, axis=0), axis=1)  # neighbours really ~ds apart
    assert np.allclose(gaps, st.ds, rtol=1e-3)


def test_normals_are_unit_and_point_left():
    st = make_stations(circle(), 0.15, 0.2)
    assert np.allclose(np.linalg.norm(st.normals, axis=1), 1.0)
    # CCW circle: left = inside, so r + w_left * n sits on radius R - w_left.
    left_wall = st.xy + st.w_left[:, None] * st.normals
    assert np.allclose(np.linalg.norm(left_wall, axis=1), R - 1.0, atol=1e-3)


def test_bounds_and_centre_feasible():
    st = make_stations(circle(), 0.15, margin=0.2)
    assert np.allclose(st.alpha_max, 1.0 - 0.2)
    assert np.allclose(st.alpha_min, -(0.8 - 0.2))
    assert np.all((st.alpha_min <= 0) & (0 <= st.alpha_max))  # alpha = 0 allowed


def test_narrow_track_raises_with_location():
    with pytest.raises(ValueError, match=r"narrower .* s = \["):
        make_stations(circle(w_left=0.15, w_right=0.15), 0.15, margin=0.2)


def t1_shape(w_left=0.9, w_right=0.8, n=4000) -> TrackModel:
    """Same curve as t1_track.synth.make_track: smooth, with left and right bends."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 4.0 + 0.8 * np.sin(2 * th) + 0.5 * np.cos(3 * th)
    return TrackModel(1.4 * r * np.cos(th), r * np.sin(th), np.full(n, w_left), np.full(n, w_right))


def test_widths_from_walls_reproduce_track_widths():
    tm = t1_shape()
    st = make_stations(tm, 0.15, 0.2)
    w_left, w_right = widths_from_walls(st.xy, st.normals, *build_walls(tm))
    assert np.max(np.abs(w_left - 0.9)) < 0.01
    assert np.max(np.abs(w_right - 0.8)) < 0.01


def test_widths_from_walls_miss_raises():
    st = make_stations(circle(), 0.15, 0.2)
    far = np.array([[100.0, 100.0], [101.0, 100.0], [101.0, 101.0], [100.0, 101.0]])
    with pytest.raises(ValueError, match="ray missed"):
        widths_from_walls(st.xy, st.normals, far, far)


def test_vehicle_margin():
    assert np.isclose(vehicle_margin(car_width=0.30, safety_buffer=0.05), 0.20)  # spec placeholders
    with pytest.raises(ValueError):
        vehicle_margin(car_width=0.0, safety_buffer=0.05)
