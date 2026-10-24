"""T6 full-lap gate: can the agreed car really drive the reference the MPC gets (planning_reference.csv)?

The file is indexed by the CENTRE line's s, but the car drives the RACING line: rows 5 cm apart on the
centre line can be 3 cm apart (inside of a corner) or 7 cm apart (outside) on the racing line. So every
check that involves time (acceleration, steering speed) first rebuilds the racing line from the file:
    X_i = C_i + e_y_i * n_i      (centre-line point + offset along its left normal)
and measures distances and times along X, never along s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ================================================================ LIMITS
# Control's shared SYNTHETIC vehicle ("Planning -> Control Interface Alignment v2", 2026-10-19).
# Not measured values. Replace with the shared vehicle_limits.yaml once it exists.
MU = 1.0                 # tyre-road friction
G = 9.81                 # m/s^2
A_ACCEL_MAX = 2.0        # m/s^2, longitudinal input upper bound (+2)
A_BRAKE_MAX = 3.0        # m/s^2, longitudinal input lower bound (-3), as a positive number
STEER_MAX = 0.40         # rad, steering angle
STEER_RATE_MAX = 1.0     # rad/s, steering slew
WHEELBASE = 0.30         # m
V_MIN, V_MAX = 0.5, 6.0  # m/s, shared synthetic speed domain
WALL_MARGIN = 0.25       # m, car centre to wall: half width 0.15 + buffer 0.10 (planning's provisional value)
# =======================================================================

# ================================================================ TOLERANCES
SPACING_MAX = 0.05       # m, rows at most this far apart on the centre line (control's requirement)
TOL_SPACING = 1e-4       # m: perception's grid fits a whole number of points into the lap (Levine 0.050019 m)
TOL_FRICTION = 0.01      # grip ellipse may read up to 1 % over 1 (interpolation onto the centre-line grid)
TOL_ACCEL = 0.01         # acceleration / braking may read up to 1 % over their limits (same interpolation)
TOL_WALL = 0.005         # m, same as T5's corridor check (T3 enforces its margin at stations 0.15 m apart)
D_MIN = 0.1              # 1 - kappa_c * e_y must stay above this (perception's d_min; the MPC divides by it)
TOL_FRENET = 0.02        # d(e_y)/ds vs (1 - kappa_c e_y) tan(e_psi): allowed difference (a slope)
TOL_LAP_TIME = 0.01      # lap time rebuilt from the file vs T4's own: within 1 %
# =======================================================================


@dataclass
class Driven:
    """The reference as the car drives it. Index i = row i, or the step from row i to row i+1
    (the last step goes from the last row back to the first: the lap is closed)."""
    s: np.ndarray            # (N,) centre-line s of each row (the file's s_track_m)
    xy: np.ndarray           # (N, 2) racing-line point of each row
    e_y: np.ndarray          # (N,) the file's columns, as read
    e_psi: np.ndarray
    kappa: np.ndarray        # racing-line curvature (kappa_ref)
    v: np.ndarray
    kappa_c: np.ndarray      # (N,) centre-line curvature at the row (from the centre line, not the file)
    ds: np.ndarray           # (N,) distance driven on the racing line, row i -> i+1 [m]
    dt: np.ndarray           # (N,) time for that step [s]
    a_lon: np.ndarray        # (N,) speeding up (+) / braking (-) on that step [m/s^2]
    a_lat: np.ndarray        # (N,) sideways acceleration at row i, v^2 * kappa, signed (+ left) [m/s^2]
    steer: np.ndarray        # (N,) steering angle at row i, atan(wheelbase * kappa) [rad]
    steer_rate: np.ndarray   # (N,) steering change on step i -> i+1, per second [rad/s]


def drive(table: np.ndarray, c_xy: np.ndarray, c_heading: np.ndarray, c_kappa: np.ndarray) -> Driven:
    """table (N, 5) = planning_reference.csv rows (s, e_y, e_psi, kappa, v); c_* = the centre line at those
    same N rows (points, headings, curvature). Rebuild the racing line and measure it in distance and time."""
    s, e_y, e_psi, kappa, v = (np.asarray(c, float) for c in table.T)
    n = np.column_stack([-np.sin(c_heading), np.cos(c_heading)])     # left normals
    xy = np.asarray(c_xy, float) + e_y[:, None] * n                  # racing-line points
    ds = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)                 # last step: back to the first row
    v_next = np.roll(v, -1)
    dt = 2.0 * ds / (v + v_next)                                     # constant acceleration on each step
    a_lon = (v_next ** 2 - v ** 2) / (2.0 * ds)
    steer = np.arctan(WHEELBASE * kappa)                             # kinematic bicycle: tan(steer) = L * kappa
    steer_rate = (np.roll(steer, -1) - steer) / dt
    return Driven(s, xy, e_y, e_psi, kappa, v, np.asarray(c_kappa, float), ds, dt, a_lon,
                  v ** 2 * kappa, steer, steer_rate)


@dataclass
class Check:
    """One line of the report."""
    name: str
    passed: bool
    worst: float             # the worst value found on the lap
    limit: float             # what it is compared with
    margin: float            # room left before the limit (+) or how far over it (-), in the check's unit
    at_s: float              # centre-line s of the worst value [m]
    unit: str


def _at_most(name, values, limit, d, unit, tol=0.0):
    """Every value must stay <= limit (+ tol); reports the largest."""
    i = int(np.argmax(values))
    w = float(values[i])
    return Check(name, bool(w <= limit + tol), w, limit, limit - w, float(d.s[i]), unit)


def _at_least(name, values, limit, d, unit, tol=0.0):
    """Every value must stay >= limit (- tol); reports the smallest."""
    i = int(np.argmin(values))
    w = float(values[i])
    return Check(name, bool(w >= limit - tol), w, limit, w - limit, float(d.s[i]), unit)


def margins(d: Driven, lap_length: float, w_left: np.ndarray, w_right: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row values the checks (and the T6 plot) compare with their limits:
      grip        share of the friction ellipse used: step i's acceleration with row i's cornering
      room        free space to the nearer wall after keeping WALL_MARGIN [m]
      frenet_D    1 - kappa_c * e_y (the MPC divides by it)
      frenet_gap  |d(e_y)/ds - (1 - kappa_c e_y) tan(e_psi)|: do the columns belong together?"""
    a_x_max = np.where(d.a_lon >= 0, min(A_ACCEL_MAX, MU * G), min(A_BRAKE_MAX, MU * G))
    s_next, s_prev = np.roll(d.s, -1), np.roll(d.s, 1)                 # periodic neighbours
    s_next[-1] += lap_length
    s_prev[0] -= lap_length
    dey_ds = (np.roll(d.e_y, -1) - np.roll(d.e_y, 1)) / (s_next - s_prev)
    return {
        "grip": np.hypot(d.a_lon / a_x_max, d.a_lat / (MU * G)),
        "room": np.minimum(w_left - WALL_MARGIN - d.e_y, w_right - WALL_MARGIN + d.e_y),
        "frenet_D": 1.0 - d.kappa_c * d.e_y,
        "frenet_gap": np.abs(dey_ds - (1.0 - d.kappa_c * d.e_y) * np.tan(d.e_psi)),
    }


def run_checks(d: Driven, lap_length: float, w_left: np.ndarray, w_right: np.ndarray,
               t4_lap_time: float) -> list[Check]:
    """All full-lap checks. w_left / w_right = the corridor at each row (perception's corner-cut widths);
    t4_lap_time = T4's own lap time (velocity_profile_meta.json)."""
    m = margins(d, lap_length, w_left, w_right)
    lap_time = float(d.dt.sum())

    checks = [
        _at_most("row spacing", np.diff(np.append(d.s, lap_length)), SPACING_MAX, d, "m", tol=TOL_SPACING),
        _at_most("speed max", d.v, V_MAX, d, "m/s"),
        _at_least("speed min", d.v, V_MIN, d, "m/s"),
        _at_most("steering angle", np.abs(d.steer), STEER_MAX, d, "rad"),
        _at_most("steering rate", np.abs(d.steer_rate), STEER_RATE_MAX, d, "rad/s"),
        _at_most("sideways acceleration", np.abs(d.a_lat), MU * G, d, "m/s^2", tol=1e-9),
        _at_most("acceleration", d.a_lon, A_ACCEL_MAX, d, "m/s^2", tol=TOL_ACCEL * A_ACCEL_MAX),
        _at_least("braking", d.a_lon, -A_BRAKE_MAX, d, "m/s^2", tol=TOL_ACCEL * A_BRAKE_MAX),
        _at_most("grip ellipse", m["grip"], 1.0, d, "-", tol=TOL_FRICTION),
        _at_least("room to the walls", m["room"], 0.0, d, "m", tol=TOL_WALL),
        _at_least("Frenet valid (1 - kappa_c e_y)", m["frenet_D"], D_MIN, d, "-"),
        _at_most("columns agree (Frenet relation)", m["frenet_gap"], TOL_FRENET, d, "-"),
    ]
    # finish line: the step from the last row back to the first must be no bigger than the largest step inside the lap
    worst = max((abs(c[0] - c[-1]) - np.abs(np.diff(c)).max(), name)
                for name, c in (("e_y", d.e_y), ("e_psi", d.e_psi), ("kappa", d.kappa), ("v", d.v)))
    checks.append(Check(f"finish line smooth ({worst[1]})", bool(worst[0] <= 1e-9), float(worst[0]), 0.0, -worst[0],
                        float(d.s[-1]), "excess step"))
    checks.append(Check("lap time vs T4", bool(abs(lap_time / t4_lap_time - 1.0) <= TOL_LAP_TIME), lap_time,
                        t4_lap_time, TOL_LAP_TIME - abs(lap_time / t4_lap_time - 1.0), 0.0, "s"))
    return checks
