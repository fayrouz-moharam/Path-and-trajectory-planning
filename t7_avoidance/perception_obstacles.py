"""T7 Half A: perception's obstacles -> what planning can trust -> keep-out zones (the speed cap is in policy.py).

A1: READ perception's data into plain Python objects --
  - live:     her /obstacles/fused message (apex_msgs/ObstacleArray, schema apex-obstacles/0.2, contract v0.2; the
              v0.1 fields are unchanged, the new camera fields are not used yet; /obstacles works too) with
              obstacles_from_ros, and her /state_estimate message (apex_msgs/StateEstimate, contract v0.3) with
              state_from_ros -- read by attribute names only, so no ROS import is needed (works off the car too)
  - recorded: her CSV runs (planning_package: obstacle_messages.csv + obstacles.csv + state_estimate.csv)
and REPLAY a recording: obstacle and state messages merged in time order, as the car received them.
The recordings miss a few fields (track_hash, max_range_m, blind_strip_m): those are optional (None).
A2: REMEMBER the cones across messages (ObstacleMemory) and say how far that list can be trusted.
A3: turn every cone that matters now into a forbidden zone for the car's centre (keep_outs -> KeepOuts).
"""
from __future__ import annotations

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .types import KeepOut, KeepOuts

# /obstacles status (contract v0.1): only OK means the list was updated with this scan
STATUS_OK, STATUS_FROZEN_LOCALIZATION, STATUS_FROZEN_SCAN, STATUS_RESET = 0, 1, 2, 3
# StateEstimate mode
MODE_INIT, MODE_OK, MODE_DEGRADED, MODE_LOST = 0, 1, 2, 3


@dataclass(frozen=True)
class Obstacle:
    """One confirmed obstacle (a cone), as perception reports it."""
    id: int                  # stays the same while listed, never reused
    x: float                 # centre, map frame [m]
    y: float
    radius: float            # half its size as the LiDAR saw it [m] (>= 0.075)
    std_xy: float            # 1-sigma position uncertainty per axis [m], includes the car's pose uncertainty
    s_track: float           # perception's centre-line s, in [0, L) [m]
    e_y: float               # from the centre line, + left [m]
    detections: int          # scans it was found in
    seen_now: bool           # found in THIS scan (False = remembered while out of sight)


@dataclass(frozen=True)
class ObstacleMessage:
    """One /obstacles message (one LiDAR scan)."""
    stamp_us: int                       # when the scan was taken [us]
    sequence_id: int                    # +1 per message; a gap = a dropped message
    status: int                         # STATUS_OK / FROZEN_LOCALIZATION / FROZEN_SCAN / RESET
    list_epoch: int                     # +1 every time perception empties the list
    obstacles: tuple[Obstacle, ...]
    track_hash: str | None = None       # live only (not in the recordings)
    max_range_m: float | None = None    # live only: nothing farther than this is detected (8 m)
    blind_strip_m: float | None = None  # live only: nothing closer than this to a wall is detected (0.10 m)


@dataclass(frozen=True)
class CarState:
    """The parts of one StateEstimate message Half A needs."""
    stamp_us: int
    s_abs: float             # continuous centre-line s (keeps growing over laps) [m]
    s_track: float           # s_abs modulo the lap, in [0, L) [m]
    e_y: float               # + left [m]
    vx: float                # forward speed [m/s]
    mode: int                # MODE_INIT / OK / DEGRADED / LOST
    flags: int
    track_hash: str | None = None    # live messages only (the CSV recordings don't have it)


def state_from_ros(msg) -> CarState:
    """A live apex_msgs/StateEstimate -> CarState (attribute names from the contract; no ROS import)."""
    return CarState(int(msg.stamp_us), float(msg.s_abs_m), float(msg.s_track_m), float(msg.e_y_m),
                    float(msg.vx_mps), int(msg.mode), int(msg.flags), str(msg.track_hash))


def obstacles_from_ros(msg) -> ObstacleMessage:
    """A live apex_msgs/ObstacleArray -> ObstacleMessage (attribute names from the contract; no ROS import)."""
    obstacles = tuple(Obstacle(int(o.id), float(o.x_m), float(o.y_m), float(o.radius_m), float(o.std_xy_m),
                               float(o.s_track_m), float(o.e_y_m), int(o.detections), bool(o.seen_now))
                      for o in msg.obstacles)
    return ObstacleMessage(int(msg.stamp_us), int(msg.sequence_id), int(msg.status), int(msg.list_epoch), obstacles,
                           str(msg.track_hash), float(msg.max_range_m), float(msg.blind_strip_m))


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_recording(run_dir: str | Path) -> tuple[list[ObstacleMessage], list[CarState]]:
    """One recorded run (a folder with obstacle_messages.csv, obstacles.csv, state_estimate.csv)."""
    run_dir = Path(run_dir)
    by_seq = defaultdict(list)                                  # obstacles.csv: one row per obstacle per message
    for r in _rows(run_dir / "obstacles.csv"):
        by_seq[int(r["sequence_id"])].append(Obstacle(
            int(r["id"]), float(r["x_m"]), float(r["y_m"]), float(r["radius_m"]), float(r["std_xy_m"]),
            float(r["s_track_m"]), float(r["e_y_m"]), int(r["detections"]), r["seen_now"] == "1"))
    messages = []
    for r in _rows(run_dir / "obstacle_messages.csv"):
        seq = int(r["sequence_id"])
        obstacles = tuple(by_seq.get(seq, ()))
        if len(obstacles) != int(r["n_obstacles"]):
            raise ValueError(f"{run_dir}: message {seq} says {r['n_obstacles']} obstacles, obstacles.csv has "
                             f"{len(obstacles)}")
        messages.append(ObstacleMessage(int(r["stamp_us"]), seq, int(r["status"]), int(r["list_epoch"]), obstacles))
    states = [CarState(int(r["stamp_us"]), float(r["s_abs_m"]), float(r["s_track_m"]), float(r["e_y_m"]),
                       float(r["vx_mps"]), int(r["mode"]), int(r["flags"]))
              for r in _rows(run_dir / "state_estimate.csv")]
    return messages, states


def replay(messages: list[ObstacleMessage], states: list[CarState]) -> Iterator[ObstacleMessage | CarState]:
    """Both streams merged in time order, as the car received them (a state and an obstacle message with the
    same stamp: the state first, so the car's position is known when the obstacles arrive)."""
    keyed = ([(m.stamp_us, 1, i, m) for i, m in enumerate(messages)],
             [(s.stamp_us, 0, i, s) for i, s in enumerate(states)])
    for *_, item in heapq.merge(*(sorted(k) for k in keyed)):
        yield item


# ================================================================ A2 SETTINGS (agreed 2026-10-24)
STALE_AFTER_S = 0.43       # s without an obstacle message -> "stale" (3 missed scans at 7 Hz; re-tune on the car)
RECOVER_DISTANCE_M = 5.0   # m driven with OK messages after a reset (or the start) before the list is trusted again
STANDSTILL_V = 0.1         # m/s: below this the car counts as standing still (added 2026-10-29) ...
STANDSTILL_LOOK_S = 1.0    # s ... and this long standing still with OK messages also recovers the list: the LiDAR
                           #   has looked at the track ahead (so a standing start after boot has no speed cap)
# =======================================================================

# the word at the top of the notebook page (same names as the KeepOuts box agreed with Half B)
FULL, FROZEN, INCOMPLETE, STALE = "full", "frozen", "incomplete", "stale"


class ObstacleMemory:
    """A2: the notebook of cones, and how far it can be trusted right now.

    Feed it every message in replay order: on_state() for each StateEstimate, on_obstacles() for each
    /obstacles message. Ask it any time: .obstacles (the list) and .trust() (FULL / FROZEN / INCOMPLETE / STALE).
      OK          -> write perception's new (complete) list
      FROZEN      -> keep the old list (cones don't move)
      RESET / new epoch -> tear the page out: empty list, INCOMPLETE until the car has driven
                     RECOVER_DISTANCE_M with OK messages, or has stood still STANDSTILL_LOOK_S with OK
                     messages (the start of the run counts as a reset too)
      silence     -> keep the list, but STALE after STALE_AFTER_S without a message
    Most cautious wins: STALE, then INCOMPLETE, then FROZEN, then FULL."""

    def __init__(self):
        self.obstacles: tuple[Obstacle, ...] = ()   # the notebook: perception's latest trusted list
        self.epoch: int | None = None               # perception's list_epoch the notebook belongs to
        self.last_status: int | None = None         # status of the latest obstacle message
        self.last_message_us: int | None = None     # when the latest obstacle message was taken
        self.now_us: int | None = None              # latest time seen on either stream
        self.recovering = True                      # page torn out (or the start): cones may be missing
        self.driven_m = 0.0                         # metres driven with OK messages while recovering
        self.still_s = 0.0                          # seconds stood still with OK messages while recovering
        self._last_s: float | None = None
        self._last_t_us: int | None = None

    def on_state(self, state: CarState) -> None:
        """A car state: while recovering and the obstacle messages are OK, count the metres driven -- or, while the
        car stands still, the seconds it has been looking. Either one long enough ends the recovery."""
        self.now_us = state.stamp_us
        if self.recovering and self._last_s is not None and self.last_status == STATUS_OK:
            self.driven_m += max(0.0, state.s_abs - self._last_s)
            if abs(state.vx) < STANDSTILL_V:
                self.still_s += (state.stamp_us - self._last_t_us) / 1e6
            else:
                self.still_s = 0.0                                  # moving: only the distance rule counts
            if self.driven_m >= RECOVER_DISTANCE_M or self.still_s >= STANDSTILL_LOOK_S:
                self.recovering = False
        self._last_s, self._last_t_us = state.s_abs, state.stamp_us

    def on_obstacles(self, msg: ObstacleMessage) -> None:
        """An obstacle message: tear out, keep or rewrite the notebook."""
        self.now_us = msg.stamp_us if self.now_us is None else max(self.now_us, msg.stamp_us)
        self.last_message_us = msg.stamp_us
        if msg.status == STATUS_RESET or (self.epoch is not None and msg.list_epoch != self.epoch):
            self.obstacles = ()                                     # tear the page out
            self.recovering, self.driven_m = True, 0.0
        if msg.status != STATUS_OK:
            self.still_s = 0.0                                      # looking only counts with OK messages
        self.epoch, self.last_status = msg.list_epoch, msg.status
        if msg.status == STATUS_OK:
            self.obstacles = msg.obstacles                          # perception's complete list, as it is now

    def trust(self, now_us: int | None = None) -> str:
        """The word at the top of the page, at time now_us (default: the latest time seen)."""
        now = self.now_us if now_us is None else now_us
        if self.last_message_us is None or now - self.last_message_us > STALE_AFTER_S * 1e6:
            return STALE
        if self.recovering:
            return INCOMPLETE
        if self.last_status in (STATUS_FROZEN_LOCALIZATION, STATUS_FROZEN_SCAN):
            return FROZEN
        return FULL


# ================================================================ A3 SETTINGS (agreed 2026-10-25)
HORIZON_M = 8.0          # zones listed up to this far ahead of the car (perception's detection range)
CONFIDENCE_K = 2.45      # x std_xy: perception's 95 % circle (obstacle contract v0.1)
HALF_CAR = 0.15          # m, half the car's 0.30 m width (planning's provisional value)
BUFFER = 0.10            # m, safety buffer (same as T3 / T6; control's 0.08 m soft margin still to agree)
MIN_D = 0.1              # 1 - kappa * e_y never below this when stretching (perception's d_min)
MIN_GAP_M = 0.05         # m of free room for the car's centre below which a side counts as closed (= runtime check)
# =======================================================================


def keep_outs(memory: ObstacleMemory, car: CarState, lap_length: float, kappa_at, width_at,
              v_cap: float = 6.0) -> KeepOuts:
    """A3: every cone in the notebook that matters now -> its forbidden zone, packed for Half B.

    kappa_at(s) / width_at(s) = perception's centre-line curvature and corner-cut walls (apex_track's track).
    A zone is kept while any part of it is ahead of the car, up to HORIZON_M (a cone beside the car stays
    until its zone no longer reaches the car)."""
    zones = []
    for o in memory.obstacles:
        r = o.radius + CONFIDENCE_K * o.std_xy + HALF_CAR + BUFFER          # the bubble, in real metres
        ahead = (o.s_track - car.s_track) % lap_length                       # 0 .. L ahead of the car ...
        if ahead > lap_length / 2:
            ahead -= lap_length                                              # ... or just behind it (negative)
        k = kappa_at(o.s_track)
        d = max(MIN_D, min(1 - k * (o.e_y - r), 1 - k * (o.e_y + r)))      # most stretched point of the bubble
        half_s = r / d                                                       # the bubble's half-length in s
        if ahead + half_s < 0 or ahead - half_s > HORIZON_M:
            continue                                                         # already passed, or too far ahead
        s_c = car.s_abs + ahead                                              # on the car's continuous s
        w_left, w_right = width_at(o.s_track)
        corridor = HALF_CAR + BUFFER                                         # car centre keeps this from walls
        gap_left = (w_left - corridor) - (o.e_y + r)                         # room between zone and left edge
        gap_right = (o.e_y - r) - (-(w_right - corridor))                    # ... and the right edge
        closed_left, closed_right = gap_left < MIN_GAP_M, gap_right < MIN_GAP_M
        zones.append(KeepOut(o.id, s_c - half_s, s_c + half_s, o.e_y - r, o.e_y + r,
                             s_c, o.e_y, r, 1 / d, closed_left or closed_right, o.seen_now,
                             gap_left, gap_right, closed_left, closed_right))
    zones.sort(key=lambda z: z.s_from)
    return KeepOuts(car.s_abs, memory.trust(), memory.epoch, HORIZON_M, v_cap, tuple(zones))
