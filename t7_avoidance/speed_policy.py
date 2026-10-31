"""T7 Half A, A4: the speed cap -- the highest speed planning allows right now, and why.

Only a LIMIT (a ceiling over T4's v_ref) plus the reason: the MPC decides how to brake.
Three rules; the lowest cap wins:
  1. the cone list can't be trusted (incomplete / stale): be able to stop before a cone that is first
     reported only ASSUMED_DETECTION_M ahead
  2. a zone ahead leaves no gap on either side (MIN_GAP_M, from A3): be able to stop STOP_MARGIN_M before it
  3. a gap is open but tighter than TIGHT_GAP_M: at most TIGHT_GAP_V (switchable with TIGHT_GAP_RULE)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from t6_feasibility.checks import A_BRAKE_MAX, V_MAX

from .perception_obstacles import INCOMPLETE, MIN_GAP_M, STALE
from .types import KeepOuts

# ================================================================ A4 SETTINGS (agreed 2026-10-29)
REACTION_S = 0.3           # s from "a cone is there" to "the car starts braking" (scan + perception + one cycle)
ASSUMED_DETECTION_M = 2.8  # m: how close an unknown cone may be when first reported (worst case seen, run B)
STOP_MARGIN_M = 0.3        # m: extra room kept when stopping before a cone / zone
TIGHT_GAP_M = 0.10         # m: a gap narrower than this (our buffer) is "tight" ...
TIGHT_GAP_V = 3.0          # m/s: ... and is passed at most this fast
TIGHT_GAP_RULE = True      # switch rule 3 off once closed-loop tests show the MPC tracks precisely enough
# =======================================================================


@dataclass(frozen=True)
class SpeedCap:
    v_cap: float             # m/s: the car may not go faster than this right now
    reason: str              # why (for the logs and for control)


def stop_speed(distance_m: float) -> float:
    """Highest speed from which the car can still stop within distance_m: it drives REACTION_S at that speed,
    then brakes at A_BRAKE_MAX:  v * t + v^2 / (2 a) = d  ->  v = a * (-t + sqrt(t^2 + 2 d / a))."""
    if distance_m <= 0:
        return 0.0
    t, a = REACTION_S, A_BRAKE_MAX
    return a * (-t + math.sqrt(t * t + 2.0 * distance_m / a))


def speed_cap(ko: KeepOuts) -> SpeedCap:
    """A4: the cap for this cycle from A3's KeepOuts (trust word + each zone's gaps, which A3 measured)."""
    caps = [SpeedCap(V_MAX, "clear: normal racing")]
    if ko.trust in (INCOMPLETE, STALE):                                                    # rule 1
        caps.append(SpeedCap(stop_speed(ASSUMED_DETECTION_M - STOP_MARGIN_M),
                             f"cone list {ko.trust}: a cone may appear only {ASSUMED_DETECTION_M} m ahead"))
    for z in ko.zones:
        gap = max(z.gap_left, z.gap_right)                                                 # the better side
        if gap < MIN_GAP_M:                                                                # rule 2
            caps.append(SpeedCap(stop_speed(z.s_from - ko.car_s_abs - STOP_MARGIN_M),
                                 f"cone {z.id} blocks the track at s {z.s_center:.2f}: stop before it"))
        elif TIGHT_GAP_RULE and gap < TIGHT_GAP_M:                                         # rule 3
            caps.append(SpeedCap(TIGHT_GAP_V, f"tight gap ({gap * 100:.0f} cm) beside cone {z.id} "
                                              f"at s {z.s_center:.2f}"))
    return min(caps, key=lambda c: c.v_cap)


def with_speed_cap(ko: KeepOuts, cap: SpeedCap) -> KeepOuts:
    """The KeepOuts package with A4's cap filled in (what Half B receives)."""
    return replace(ko, v_cap=cap.v_cap)
