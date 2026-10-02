"""T3 spec section 3.2: what T3 assumes about the T2 track it is given,
checked on T2's QueryableTrack with circles whose answers are known exactly."""
import numpy as np

from t3_racing_line.stations import track_from_points

R = 5.0


def circle(ccw: bool = True, n: int = 600):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)   # first point not repeated
    if not ccw:
        th = -th
    return track_from_points(R * np.cos(th), R * np.sin(th), np.full(n, 1.0), np.full(n, 0.8))


def test_length_is_circumference():
    # T2 measures s along straight gaps between points (like T1): 600 points
    # on R = 5 m are 5.2 cm apart, so the polygon is ~0.003 % shorter than 2*pi*R.
    assert abs(circle().length - 2 * np.pi * R) < 2e-3


def test_curvature_sign_and_value():
    s = np.linspace(0, circle().length, 50, endpoint=False)
    assert np.allclose(circle(ccw=True).sample_arrays(s)[4], 1 / R, rtol=1e-3)    # left turn -> +
    assert np.allclose(circle(ccw=False).sample_arrays(s)[4], -1 / R, rtol=1e-3)  # right turn -> -


def test_s_is_metres_along_the_track():
    _, x, y, *_ = circle().sample_arrays([3.0, 3.1, 3.2])
    assert np.allclose(np.hypot(np.diff(x), np.diff(y)), 0.1, atol=1e-4)


def test_closed_and_continuous_at_the_seam():
    tm = circle()
    before, after = tm.sample_arrays(-1e-6), tm.sample_arrays(1e-6)
    assert np.hypot(before[1] - after[1], before[2] - after[2]) < 1e-5            # position
    assert abs(np.angle(np.exp(1j * (before[3] - after[3])))) < 1e-5              # heading
    assert abs(before[4] - after[4]) < 1e-4                                       # curvature


def test_left_normal_points_to_left_wall():
    # CCW circle: left = inside, so r + w_left * n lands on radius R - w_left.
    tm = circle()
    _, x, y, psi, _, wl, _ = tm.sample_arrays(np.linspace(0, tm.length, 20, endpoint=False))
    n = np.column_stack([-np.sin(psi), np.cos(psi)])
    wall = np.column_stack([x, y]) + wl[:, None] * n
    assert np.allclose(np.hypot(wall[:, 0], wall[:, 1]), R - 1.0, atol=1e-3)


def test_sample_arrays_matches_sample():
    # the vectorised call T3 uses must give exactly what T2's own sample() gives
    tm = circle()
    s = np.linspace(-3.0, tm.length + 3.0, 37)
    arrays = tm.sample_arrays(s)
    for k, p in enumerate(tm.sample_many(s)):
        assert [a[k] for a in arrays] == [p.s, p.x, p.y, p.heading, p.curvature, p.w_left, p.w_right]
