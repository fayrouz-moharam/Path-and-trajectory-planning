"""T7 handover boxes between Half A (obstacles, trust, speed cap) and Half B (the path around them).
Agreed in "T7 obstacle avoidance: split and handover" (2026-10-24); circle fields added 2026-10-25."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeepOut:
    """One cone's forbidden zone: the car's CENTRE must not enter it. Given in two shapes -- use either."""
    id: int                  # perception's obstacle id
    # the BOX (simple): along the track (centre-line s, continuous like s_abs) and across it (+ left) [m]
    s_from: float
    s_to: float
    e_y_from: float
    e_y_to: float
    # the CIRCLE (exact, more room): centre and radius in real metres; in a corner its along-track half-length
    # in s is radius * stretch (the box above already includes that)
    s_center: float
    e_y_center: float
    radius: float
    stretch: float           # 1 m of road there = stretch metres of centre-line s (1 on a straight)
    near_wall: bool          # the zone reaches the corridor edge on one side: that side is closed
    seen_now: bool           # seen in perception's latest scan (False = remembered)


@dataclass(frozen=True)
class KeepOuts:
    """Everything Half B needs from Half A, every cycle."""
    car_s_abs: float         # car position the box was built for [m]
    trust: str               # "full" / "frozen" / "incomplete" / "stale" (Half A's notebook word)
    list_epoch: int | None   # perception's epoch the zones come from
    horizon_m: float         # zones are listed up to this far ahead [m]
    v_cap: float             # speed cap from Half A's safety rules (A4); Half B may only lower it [m/s]
    zones: tuple[KeepOut, ...]   # sorted by s_from
