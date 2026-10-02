"""Steps 2-3: linear curvature model, cost, and the QP on a circle."""
import numpy as np
import pytest

from t3_racing_line.min_curvature import (build_curvature_model, curvature_cost,
                                          iterate_min_curvature, kappa_max_from_steering,
                                          solve_min_curvature_qp)
from t3_racing_line.stations import make_stations
from t3_racing_line.track_model_stub import TrackModel

R = 5.0
KAPPA_MAX = kappa_max_from_steering(max_steer=0.4, wheelbase=0.33)  # spec placeholders


def circle_stations(w_left=1.0, w_right=0.8, margin=0.2):
    th = np.linspace(0, 2 * np.pi, 600, endpoint=False)  # CCW: left = inside
    tm = TrackModel(R * np.cos(th), R * np.sin(th), np.full(600, w_left), np.full(600, w_right))
    return make_stations(tm, 0.15, margin)


def true_kappa(p):
    """Curvature of the closed polyline p through each point's two neighbours
    (circle through 3 points), using p's OWN spacing -- no linearisation."""
    a, b = p - np.roll(p, 1, 0), np.roll(p, -1, 0) - p
    c = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return 2 * cross / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1))


def test_kappa_ref_matches_reference_curvature():
    st = circle_stations()
    kappa_ref, _ = build_curvature_model(st)
    assert np.allclose(kappa_ref, 1 / R, rtol=0.01)
    assert np.allclose(kappa_ref, st.kappa, rtol=0.01)


def test_moving_outward_lowers_curvature():
    # The sign that the spec's plain formula gets wrong (see build_curvature_model).
    st = circle_stations()
    kappa_ref, M = build_curvature_model(st)
    for c in (-0.3, 0.3):
        k = kappa_ref + M @ np.full(len(st), c)
        assert np.allclose(k, 1 / R + c / R**2, atol=1e-4)          # first order of 1/(R - c)
    assert (kappa_ref + M @ np.full(len(st), -0.3)).mean() < 1 / R  # outward (right) -> flatter


def test_linear_model_matches_true_curvature_for_small_alpha():
    st = circle_stations()
    kappa_ref, M = build_curvature_model(st)
    alpha = 0.05 * np.sin(3 * 2 * np.pi * st.s / (len(st) * st.ds))  # smooth, small
    p = st.xy + alpha[:, None] * st.normals
    assert np.max(np.abs(kappa_ref + M @ alpha - true_kappa(p))) < 1e-3


def test_cost_at_zero_is_reference_cost():
    st = circle_stations()
    kappa_ref, M = build_curvature_model(st)
    assert np.isclose(curvature_cost(kappa_ref, M, np.zeros(len(st))), kappa_ref @ kappa_ref)


def test_qp_circle_hugs_outer_edge():
    pytest.importorskip("osqp")
    st = circle_stations(w_left=1.0, w_right=0.8, margin=0.2)
    res = solve_min_curvature_qp(st, kappa_max=KAPPA_MAX)
    assert res.status == "solved"
    # CCW circle: outside = right = alpha_min = -(0.8 - 0.2)
    assert np.max(np.abs(res.alpha - st.alpha_min)) < 1e-3


def test_qp_infeasible_kappa_limit_raises():
    pytest.importorskip("osqp")
    st = circle_stations()
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_min_curvature_qp(st, kappa_max=0.1)  # R = 5 circle can't get below ~0.18


def t1_shape(n=4000) -> TrackModel:
    """Same curve as t1_track.synth.make_track: smooth, with left and right bends."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 4.0 + 0.8 * np.sin(2 * th) + 0.5 * np.cos(3 * th)
    return TrackModel(1.4 * r * np.cos(th), r * np.sin(th), np.full(n, 0.9), np.full(n, 0.8))


def test_iterate_converges_and_stays_inside():
    pytest.importorskip("osqp")
    tm = t1_shape()
    margin = 0.2
    res = iterate_min_curvature(tm, 0.15, margin, kappa_max=KAPPA_MAX)
    assert res.converged and len(res.history) <= 5
    # each round corrects less than the one before
    moves = [h.max_alpha for h in res.history]
    assert all(b < a for a, b in zip(moves, moves[1:]))
    # final line: at least `margin` from both real walls (1 cm slack for the linearisation)
    st = res.stations
    assert st.w_left.min() > margin - 0.01 and st.w_right.min() > margin - 0.01
    # straighter than the centreline
    assert np.abs(st.kappa).max() < np.abs(make_stations(tm, 0.15, margin).kappa).max()


def test_iterate_uses_full_width_with_an_apex():
    """This shape is one big left loop with small right-hand dents, so (like the
    circle) the line mostly hugs the OUTSIDE (right) wall to make the loop as big
    as possible, and it cuts the right-hand dents to their INSIDE (also the right
    wall) -- that's the apex: line curving toward the wall it touches."""
    pytest.importorskip("osqp")
    res = iterate_min_curvature(t1_shape(), 0.15, 0.2, kappa_max=KAPPA_MAX)
    st = res.stations
    touch_left, touch_right = st.w_left < 0.21, st.w_right < 0.21
    assert touch_right.mean() > 0.2                                    # hugs the outside
    apex = (touch_left & (st.kappa > 0.05)) | (touch_right & (st.kappa < -0.05))
    assert apex.any()                                                  # an apex exists


def test_kappa_max_from_steering():
    assert np.isclose(KAPPA_MAX, np.tan(0.4) / 0.33)       # ~1.29 1/m
    assert np.isclose(kappa_max_from_steering(np.pi / 4, 0.5), 2.0)  # tan 45 deg = 1 -> 1 / 0.5
    with pytest.raises(ValueError):
        kappa_max_from_steering(max_steer=0.4, wheelbase=0.0)
