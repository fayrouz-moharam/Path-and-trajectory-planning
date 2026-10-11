"""T4: velocity profile v_ref(s), a_ref(s) along the T3 racing line.

Three steps on the racing line's uniform s grid (ds = lap_length / M):

  1. corner limit   v_lat(s) = sqrt(a_lat_max / |kappa(s)|), a_lat_max = mu * g,
                    capped at v_max (straights: kappa ~ 0 -> v_max)
  2. forward pass   accelerate as hard as the tyres allow:
                    v[i+1] = min(v_lat[i+1], sqrt(v[i]^2 + 2 * a_x_free(v[i], kappa[i]) * ds))
  3. backward pass  brake as late as possible, same rule run from the end backwards
                    with the braking limit.

a_x_free is what is left of the friction ellipse after cornering:
    (a_x / a_x_max)^2 + (a_y / a_lat_max)^2 <= 1,   a_y = v^2 * |kappa|
    a_x_free = a_x_max * sqrt(1 - (a_y / a_lat_max)^2)
with a_x_max = min(a_accel_max, mu g) when speeding up, min(a_brake_max, mu g) when braking.

The lap is closed: both passes start at the slowest corner (where v = v_lat is
certain) and wrap round the whole lap, so v at the end of the lap = v at s = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class VehicleLimits:
    """Longitudinal / lateral limits. Placeholders until the vehicle is measured
    (configs/vehicles/generic_1_10_racecar.yaml in apex: all null today)."""
    mu: float = 0.8              # tyre-road friction coefficient
    a_accel_max: float = 3.0     # m/s^2, motor / traction limit when speeding up
    a_brake_max: float = 4.0     # m/s^2, braking deceleration (positive number)
    v_max: float = 6.0           # m/s, top speed (motor / ESC)
    v_min: float = 0.5           # m/s, floor so the time t(s) stays finite
    g: float = G

    def __post_init__(self):
        for name in ("mu", "a_accel_max", "a_brake_max", "v_max", "v_min", "g"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.v_min >= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be below v_max ({self.v_max})")

    @property
    def a_lat_max(self) -> float:
        return self.mu * self.g


@dataclass
class VelocityProfile:
    s: np.ndarray          # (M,) racing-line arc length, same grid as racing_line.csv
    kappa: np.ndarray      # (M,) curvature used (after optional smoothing)
    v_lat: np.ndarray      # (M,) corner limit (step 1)
    v_ref: np.ndarray      # (M,) final speed (after both passes)
    a_ref: np.ndarray      # (M,) longitudinal acceleration on the interval s[i] -> s[i+1]
    a_lat: np.ndarray      # (M,) lateral acceleration v^2 * kappa (signed, + = left)
    t: np.ndarray          # (M,) time since s = 0
    lap_length: float
    lap_time: float
    limits: VehicleLimits

    @property
    def friction_use(self) -> np.ndarray:
        """Fraction of the friction ellipse used at each point (<= 1 by construction)."""
        L = self.limits
        a_x_max = np.where(self.a_ref >= 0, min(L.a_accel_max, L.a_lat_max), min(L.a_brake_max, L.a_lat_max))
        return np.sqrt((self.a_ref / a_x_max) ** 2 + (self.a_lat / L.a_lat_max) ** 2)

    def as_array(self) -> np.ndarray:
        return np.column_stack([self.s, self.kappa, self.v_lat, self.v_ref, self.a_ref, self.a_lat, self.t])


COLUMNS = ["s", "kappa", "v_lat", "v_ref", "a_ref", "a_lat", "t"]


def smooth_periodic(values: np.ndarray, ds: float, sigma_m: float) -> np.ndarray:
    """Periodic Gaussian smoothing (sigma in metres along s). 0 = off."""
    if sigma_m <= 0:
        return values.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(values, sigma_m / ds, mode="wrap")


def corner_limit(kappa: np.ndarray, limits: VehicleLimits) -> np.ndarray:
    """Step 1: v_lat = sqrt(mu g / |kappa|), capped at v_max."""
    with np.errstate(divide="ignore"):
        v = np.sqrt(limits.a_lat_max / np.abs(kappa))
    return np.clip(v, limits.v_min, limits.v_max)


def _a_x_free(v: float, kappa: float, a_x_max: float, a_lat_max: float) -> float:
    """Longitudinal acceleration left over after cornering at speed v (friction ellipse)."""
    use = (v * v * abs(kappa)) / a_lat_max
    return a_x_max * np.sqrt(max(0.0, 1.0 - use * use))


def _pass(v_lim: np.ndarray, kappa: np.ndarray, ds: float, a_x_max: float, a_lat_max: float) -> np.ndarray:
    """One forward pass on an open array starting at v_lim[0]."""
    v = v_lim.copy()
    for i in range(len(v) - 1):
        reach = np.sqrt(v[i] ** 2 + 2.0 * _a_x_free(v[i], kappa[i], a_x_max, a_lat_max) * ds)
        v[i + 1] = min(v[i + 1], reach)
    return v


def compute_profile(s: np.ndarray, kappa: np.ndarray, lap_length: float,
                    limits: VehicleLimits = VehicleLimits(), kappa_sigma_m: float = 0.0) -> VelocityProfile:
    """v_ref(s), a_ref(s), t(s) on the racing line's own s grid (closed lap)."""
    s, kappa = np.asarray(s, float), np.asarray(kappa, float)
    if s.ndim != 1 or s.shape != kappa.shape or len(s) < 3:
        raise ValueError("s and kappa must be 1-D arrays of the same length (>= 3)")
    if not np.all(np.diff(s) > 0) or s[-1] >= lap_length:
        raise ValueError("s must increase strictly and stay below lap_length (no repeated end point)")
    if not np.all(np.isfinite(kappa)):
        raise ValueError("kappa has NaN / inf values")
    ds = lap_length / len(s)
    if np.max(np.abs(np.diff(s) - ds)) > 1e-3 * ds + 1e-6:
        raise ValueError("s must be uniformly spaced (T3 writes it that way)")

    k = smooth_periodic(kappa, ds, kappa_sigma_m)
    v_lat = corner_limit(k, limits)
    a_acc = min(limits.a_accel_max, limits.a_lat_max)
    a_brk = min(limits.a_brake_max, limits.a_lat_max)

    # roll so index 0 is the slowest corner: there v = v_lat for sure, which closes the lap
    i0 = int(np.argmin(v_lat))
    vl, kr = np.roll(v_lat, -i0), np.roll(k, -i0)
    vl_c, k_c = np.append(vl, vl[0]), np.append(kr, kr[0])          # one wrap: last = first point

    v_fwd = _pass(vl_c, k_c, ds, a_acc, limits.a_lat_max)
    # backward pass = forward pass on the reversed lap (braking is acceleration backwards)
    v_bwd = _pass(v_fwd[::-1], k_c[::-1], ds, a_brk, limits.a_lat_max)[::-1]
    v_r = np.roll(v_bwd[:-1], i0)

    v_next = np.roll(v_r, -1)
    a_ref = (v_next ** 2 - v_r ** 2) / (2.0 * ds)
    dt = 2.0 * ds / (v_r + v_next)
    t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    return VelocityProfile(s=s, kappa=k, v_lat=v_lat, v_ref=v_r, a_ref=a_ref, a_lat=v_r ** 2 * k,
                           t=t, lap_length=float(lap_length), lap_time=float(dt.sum()), limits=limits)
