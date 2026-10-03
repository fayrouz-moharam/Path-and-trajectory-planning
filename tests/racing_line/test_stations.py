"""Step 1: make_stations -- spacing, normals, bounds, narrow-track error."""
import numpy as np
import pytest

from t3_racing_line.stations import (_ray_distance, alpha_limits, build_walls, make_stations,
                                     track_from_points, vehicle_margin, widths_from_walls)
from t2_curvature.queryable_track import QueryableTrack

R = 5.0


def circle(w_left=1.0, w_right=0.8, n=600) -> QueryableTrack:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)  # CCW = left-turning
    return track_from_points(R * np.cos(th), R * np.sin(th), np.full(n, w_left), np.full(n, w_right))


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


def t1_shape(w_left=0.9, w_right=0.8, n=4000) -> QueryableTrack:
    """Same curve as t1_track.synth.make_track: smooth, with left and right bends."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 4.0 + 0.8 * np.sin(2 * th) + 0.5 * np.cos(3 * th)
    return track_from_points(1.4 * r * np.cos(th), r * np.sin(th), np.full(n, w_left), np.full(n, w_right))


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


# ---------------------------------------------------------------- the sideways walk (alpha_limits)

def test_alpha_limits_match_the_widths_on_a_circle():
    tm = circle()                                    # w_left 1.0, w_right 0.8
    st = make_stations(tm, 0.15, 0.2)
    lo, hi, pushed, stuck = alpha_limits(st.xy, st.normals, build_walls(tm), 0.2)
    assert np.allclose(hi, 1.0 - 0.2, atol=1e-3) and np.allclose(lo, -(0.8 - 0.2), atol=1e-3)
    assert not pushed.any() and not stuck.any()


def test_alpha_limits_push_a_point_out_of_the_margin():
    # 5 cm inside the right margin: it must move at least 5 cm LEFT, instead of "no room"
    tm = circle()
    st = make_stations(tm, 0.15, 0.2)
    p = st.xy[:3] - 0.65 * st.normals[:3]           # right wall is 0.8 away -> 0.15 from it
    lo, hi, pushed, stuck = alpha_limits(p, st.normals[:3], build_walls(tm), 0.2)
    assert np.allclose(lo, 0.05, atol=1e-3) and np.allclose(hi, 0.65 + 0.8, atol=1e-3)
    assert pushed.all() and not stuck.any()


def test_alpha_limits_see_a_hairpin_tip_that_a_ray_misses():
    # left wall = a thin spike whose tip is 5 cm beside the sideways line (like a hairpin's
    # inside tip); the old ray passes it and hits nothing, the walk stops 0.2 m from it
    tip = np.array([0.05, 0.6])
    left = np.array([tip, [1.0, 0.63], [1.0, 0.57]])
    right = np.array([[-5.0, -1.0], [5.0, -1.0], [5.0, -1.1], [-5.0, -1.1]])
    p, n = np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.isinf(_ray_distance(p, n, left))
    lo, hi, _, _ = alpha_limits(p, n, (left, right), 0.2)
    assert np.isclose(hi[0], 0.6 - np.sqrt(0.2**2 - 0.05**2), atol=1e-3)   # 0.4064
    assert np.isclose(lo[0], -(1.0 - 0.2), atol=1e-3)
