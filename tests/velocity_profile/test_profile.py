"""T4 checks against cases with a known answer."""
import numpy as np
import pytest

from t4_velocity_profile import SpeedReference, VehicleLimits, compute_profile

LIM = VehicleLimits(mu=0.8, a_accel_max=3.0, a_brake_max=4.0, v_max=6.0, v_min=0.5)


def grid(lap, ds=0.05):
    n = int(round(lap / ds))
    return np.arange(n) * (lap / n)


def stadium(R=2.0, straight=20.0):
    """Two straights + two half circles, s = 0 at the start of the first straight."""
    lap = 2 * straight + 2 * np.pi * R
    s = grid(lap)
    k = np.zeros_like(s)
    k[(s % (lap / 2)) >= straight] = 1 / R
    return s, k, lap


def test_circle_constant_corner_speed():
    R = 3.0
    s = grid(2 * np.pi * R)
    p = compute_profile(s, np.full_like(s, 1 / R), 2 * np.pi * R, LIM)
    assert np.allclose(p.v_ref, np.sqrt(LIM.mu * LIM.g * R), rtol=1e-9)
    assert np.allclose(p.a_ref, 0, atol=1e-9)
    assert p.lap_time == pytest.approx(2 * np.pi * R / np.sqrt(LIM.mu * LIM.g * R))


def test_straight_line_hits_v_max():
    s = grid(100.0)
    p = compute_profile(s, np.zeros_like(s), 100.0, LIM)
    assert np.allclose(p.v_ref, LIM.v_max)


def test_stadium_accel_and_brake_distances():
    R, straight = 2.0, 20.0
    s, k, lap = stadium(R, straight)
    p = compute_profile(s, k, lap, LIM)
    v_c = np.sqrt(LIM.mu * LIM.g * R)
    on_straight = s < straight
    # never above the corner limit, corners driven at the corner speed
    assert np.all(p.v_ref <= p.v_lat + 1e-9)
    assert np.allclose(p.v_ref[k > 0], v_c, rtol=1e-6)
    # peak speed on the straight: accelerate a_acc then brake a_brk over 20 m
    a_acc, a_brk = LIM.a_accel_max, LIM.a_brake_max
    v_peak = min(LIM.v_max, np.sqrt(v_c ** 2 + 2 * straight * a_acc * a_brk / (a_acc + a_brk)))
    assert p.v_ref[on_straight].max() == pytest.approx(v_peak, rel=5e-3)
    # accel / brake stay inside the limits
    assert p.a_ref.max() <= a_acc + 1e-6 and p.a_ref.min() >= -a_brk - 1e-6


def test_lap_closes_and_friction_respected():
    s, k, lap = stadium()
    k = k + 0.05 * np.sin(2 * np.pi * 3 * s / lap)        # some left/right variation
    p = compute_profile(np.roll(s, 0), k, lap, LIM)
    ref = SpeedReference.from_profile(p)
    assert ref(0.0) == pytest.approx(ref(lap)) == pytest.approx(ref(-lap))
    # the exact brake pass keeps every step inside the friction ellipse
    assert p.friction_use.max() <= 1.0 + 1e-9


def test_start_point_does_not_matter():
    s, k, lap = stadium()
    shift = 137
    p1 = compute_profile(s, k, lap, LIM)
    p2 = compute_profile(s, np.roll(k, -shift), lap, LIM)
    assert np.allclose(np.roll(p1.v_ref, -shift), p2.v_ref)
    assert p1.lap_time == pytest.approx(p2.lap_time)


def test_rejects_bad_input():
    s = grid(10.0)
    with pytest.raises(ValueError):
        compute_profile(s[::-1], np.zeros_like(s), 10.0, LIM)
    with pytest.raises(ValueError):
        compute_profile(np.append(s, 10.0), np.zeros(len(s) + 1), 10.0, LIM)   # repeated end point
    with pytest.raises(ValueError):
        VehicleLimits(mu=0)


def test_shared_synthetic_config_stays_inside_grip():
    s, k, lap = stadium(R=1.0, straight=8.0)
    p = compute_profile(s, k, lap, VehicleLimits())      # defaults = the config shared with Control
    assert p.friction_use.max() <= 1.0 + 1e-9
    assert p.a_ref.max() <= 2.0 + 1e-9 and p.a_ref.min() >= -3.0 - 1e-9
    assert 0.5 <= p.v_ref.min() and p.v_ref.max() <= 6.0
