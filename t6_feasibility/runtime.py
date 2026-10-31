"""T6 runtime checks: a fast "is everything still safe?" check on EVERY MPC cycle (~10 Hz), while driving.

The full-lap gate (checks.py) proves the stored reference is drivable before the race. While the car
drives, things can still go wrong between that file and the MPC: a NaN from a bad message, the wrong
track, a corridor narrowed to nothing (by obstacle avoidance, later), the car already outside its
corridor. check_cycle() looks at ONE cycle -- the MPC's preview + the car's state -- and answers
OK / WARN / FAIL with the reasons, in well under a millisecond. It only judges: what the car does about
WARN / FAIL is control's (see "Planning runtime check: OK / WARN / FAIL for the MPC").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .checks import STEER_MAX, V_MAX, V_MIN, WALL_MARGIN, WHEELBASE

# ================================================================ THRESHOLDS
MIN_CORRIDOR = 0.05                      # m: the corridor must stay at least this wide at every preview stage
BODY_SLACK = WALL_MARGIN - 0.15          # m: car centre may be this far outside its corridor before its body
                                         #    reaches a wall (the 0.10 m buffer): up to here WARN, beyond FAIL
KAPPA_MAX = math.tan(STEER_MAX) / WHEELBASE   # 1/m, sharpest curvature the wheels can steer (1.41)
PREVIEW_STEPS = 8        # control's preview: 8 steps ...
PREVIEW_DT = 0.1         # ... of 0.1 s (their 10 Hz release, 0.8 s ahead)
# =======================================================================

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class CyclePlan:
    """One cycle's preview: what the MPC is about to follow, one entry per preview stage (9 today)."""
    s: np.ndarray            # (K,) centre-line s of each stage; s[0] = the car's s_abs (may run past the lap)
    e_y_ref: np.ndarray      # (K,) where to be, + left of the centre line [m]
    e_psi_ref: np.ndarray    # (K,) which way to point [rad]
    kappa_ref: np.ndarray    # (K,) how much to turn [1/m]
    v_ref: np.ndarray        # (K,) how fast [m/s]
    e_y_min: np.ndarray      # (K,) corridor for the car's CENTRE: right edge (walls - margin) [m]
    e_y_max: np.ndarray      # (K,) left edge [m]
    v_cap: float = V_MAX     # speed cap for this cycle (later: lowered by obstacle avoidance)


@dataclass
class Verdict:
    """OK / WARN / FAIL + the reasons (worst first: every FAIL reason, then every WARN reason)."""
    level: str
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.level == OK


def check_cycle(plan: CyclePlan, s_abs: float, e_y: float, track_hash: str | None,
                ref_track_hash: str | None) -> Verdict:
    """The 8 runtime checks on one cycle. s_abs, e_y, track_hash = the car's StateEstimate;
    ref_track_hash = the reference's (PlanningReference.meta["track_hash"])."""
    arrays = (plan.s, plan.e_y_ref, plan.e_psi_ref, plan.kappa_ref, plan.v_ref, plan.e_y_min, plan.e_y_max)
    # 1. no broken numbers -- checked first: nothing else can be trusted with a NaN in it
    if not (all(np.all(np.isfinite(a)) for a in arrays) and all(map(math.isfinite, (s_abs, e_y, plan.v_cap)))):
        return Verdict(FAIL, ["a number is NaN / inf in the car's state or the preview"])
    fail, warn = [], []
    # 2. same track
    if track_hash != ref_track_hash:
        fail.append(f"track mismatch: StateEstimate {track_hash}, reference {ref_track_hash}")
    # 3. the preview starts at the car and moves forward
    if abs(plan.s[0] - s_abs) > 1e-6 or np.any(np.diff(plan.s) <= 0):
        fail.append(f"preview must start at the car (s {s_abs:.2f}) and move forward")
    # 4. the corridor is open at every stage
    width = plan.e_y_max - plan.e_y_min
    i = int(np.argmin(width))
    if width[i] < MIN_CORRIDOR:
        fail.append(f"corridor only {width[i] * 100:.1f} cm wide at s {plan.s[i]:.2f}")
    # 5. the planned path stays inside its corridor
    out = np.maximum(plan.e_y_min - plan.e_y_ref, plan.e_y_ref - plan.e_y_max)
    i = int(np.argmax(out))
    if out[i] > 1e-6:
        fail.append(f"e_y_ref {out[i] * 100:.1f} cm outside its corridor at s {plan.s[i]:.2f}")
    # 6. the car itself is inside its corridor (stage 0 = where the car is)
    car_out = max(plan.e_y_min[0] - e_y, e_y - plan.e_y_max[0])
    if car_out > BODY_SLACK:
        fail.append(f"car {car_out * 100:.1f} cm outside its corridor at s {s_abs:.2f}: its body reaches the wall")
    elif car_out > 0:
        warn.append(f"car {car_out * 100:.1f} cm outside its corridor at s {s_abs:.2f} (body still clear of the wall)")
    # 7. the wheels can steer it
    i = int(np.argmax(np.abs(plan.kappa_ref)))
    if abs(plan.kappa_ref[i]) > KAPPA_MAX + 1e-9:
        fail.append(f"curvature {plan.kappa_ref[i]:+.2f} 1/m at s {plan.s[i]:.2f} is sharper than the car can steer")
    # 8. the speed is allowed
    v_hi = min(plan.v_cap, V_MAX)
    if plan.v_ref.max() > v_hi + 1e-9:
        fail.append(f"speed {plan.v_ref.max():.2f} m/s above the cap {v_hi:.2f}")
    if plan.v_ref.min() < V_MIN - 1e-9:
        fail.append(f"speed {plan.v_ref.min():.2f} m/s below the minimum {V_MIN}")
    return Verdict(FAIL if fail else WARN if warn else OK, fail + warn)


def plan_from_reference(ref, s_abs: float, vx: float, walls, n: int = PREVIEW_STEPS,
                        dt: float = PREVIEW_DT) -> CyclePlan:
    """This cycle's preview straight from planning_reference.csv (what runs until obstacle avoidance exists).

    ref   = PlanningReference (t5_reference.loader)
    s_abs, vx = the car's StateEstimate: where it is (centre-line s) and its forward speed
    walls = walls(s) -> (w_left, w_right) at centre-line s, any s (must wrap the lap), e.g. apex_track's
            track.width_at
    The stages are where the car will be every dt seconds at its current speed (never slower than V_MIN,
    so the stages always move forward), like control's own preview."""
    s = s_abs + max(vx, V_MIN) * dt * np.arange(n + 1)
    r = ref.at(s)
    w_left, w_right = walls(s)
    return CyclePlan(s, r["e_y_ref_m"], r["e_psi_ref_rad"], r["kappa_ref_1pm"], r["v_ref_mps"],
                     -(np.asarray(w_right) - WALL_MARGIN), np.asarray(w_left) - WALL_MARGIN)
