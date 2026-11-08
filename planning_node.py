"""The planning node: listens to perception on the car and (later) plans every MPC cycle.

    PART 1 -- the mailbox (every message, built): /state_estimate and /obstacles/fused arrive ->
              translate (state_from_ros / obstacles_from_ros) -> same track as our reference? (track_hash) ->
              feed Half A's notebook (ObstacleMemory) -> keep the newest car state
    PART 2 -- the clock (every 0.1 s): newest state + notebook -> Half A zones + speed cap ->
              Half B plan_cycle (uses T5's reference and T6's check_cycle) -> T6 verdict on the chosen plan
              -> print (later: publish to control)

    1. edit the SETTINGS below (which planning_reference.csv, which track folder)
    2. in a ROS 2 shell where Fatma's apex_msgs is built and sourced:  python3 planning_node.py
    For now it only prints one line per second: what arrived, the car, the trust word, the cones,
    and what the last cycle decided. For debugging it also draws (PUBLISH_MARKERS) in the map frame:
    /planning/racing_line (T5's lap, once, latched) and /planning/markers (every cycle: planned path coloured
    by T6's verdict, Half A's zone boxes, a status label above the car). Add both to RViz / Foxglove.

The logic is two classes without ROS (tested on any laptop: tests/test_planning_node.py):
Listener (PART 1) and Planner (PART 2). ROS is imported only in main(), when the node really starts.
"""
from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from t5_reference import PlanningReference
from t6_feasibility.checks import WALL_MARGIN
from t6_feasibility.runtime import FAIL, WARN, CyclePlan, Verdict, check_cycle
from t7_avoidance.perception_obstacles import (CarState, ObstacleMemory, keep_outs, obstacles_from_ros,
                                               state_from_ros)
from t7_avoidance.speed_policy import SpeedCap, speed_cap, with_speed_cap
from t7_avoidance.types import KeepOuts
from t7_obstacle_avoidance.plan import Memory, plan_cycle

# ============================================================== SETTINGS
REFERENCE = Path("out/run_perception_closed1/t5/planning_reference.csv")   # the reference we race on
TRACK = "closed1"                   # its apex_track track folder (apex_track/tracks/<TRACK>/) ...
TRACK_YAML = None                   # ... or a full path to a track.yaml anywhere (overrides TRACK)
D_MIN = 0.1                         # apex_track's tight-corner limit on the inside walls (as T5 / Half B)
STATE_TOPIC = "/state_estimate"     # Fatma's topic names (fixed in her code)
OBSTACLE_TOPIC = "/obstacles/fused"  # contract v0.2: the LiDAR list + the camera check (her fusion_node must run;
                                     # "/obstacles" = the same list without the camera fields)
CYCLE_S = 0.1                       # PART 2 runs this often (the MPC's cycle)
STATE_MAX_AGE_S = 0.05              # newest car state older than this -> no cycle (Fatma's valid_for_ms)
PRINT_EVERY_S = 1.0                 # one status line per second
PUBLISH_MARKERS = True              # draw the plan for RViz / Foxglove (debug only; control never reads it)
MARKER_TOPIC = "/planning/markers"          # every cycle: planned path, zones, status label
RACING_LINE_TOPIC = "/planning/racing_line"  # once at start (latched): T5's racing line around the lap
PUBLISH_GLOBAL_REFERENCE = True     # send our T5 reference once at start (latched), for Omar's reference_adapter
GLOBAL_REFERENCE_TOPIC = "/planning/global_reference"
PUBLISH_LOCAL_REFERENCE = True      # every cycle: Half B's plan 8 m ahead (valid = T6 not FAIL)
LOCAL_REFERENCE_TOPIC = "/planning/local_reference"
MSG_PACKAGE = "apex_planning_msgs"  # where ReferenceTrajectory lives: ours (ros/apex_planning_msgs) until the
                                    # team's apex_interfaces exists -> then just change this name
# =======================================================================

MODE_NAMES = ["INIT", "OK", "DEGRADED", "LOST"]


class Listener:
    """PART 1 without ROS: what happens to every message that arrives.

    ref_track_hash = the track our planning_reference was made for. A message from another track is
    dropped (never reaches Half A) and counted as wrong_track."""

    def __init__(self, ref_track_hash: str):
        self.ref_track_hash = ref_track_hash
        self.memory = ObstacleMemory()          # Half A's notebook: needs EVERY message
        self.car: CarState | None = None        # the newest car state (for PART 2)
        self.last_wrong_hash: str | None = None
        self.counts = self._zero()

    @staticmethod
    def _zero() -> dict:
        return {"state": 0, "obstacles": 0, "wrong_track": 0}

    def on_state_msg(self, msg) -> bool:
        """One /state_estimate message. True = used, False = dropped (another track)."""
        car = state_from_ros(msg)
        if car.track_hash != self.ref_track_hash:
            return self._wrong(car.track_hash)
        self.memory.on_state(car)
        self.car = car
        self.counts["state"] += 1
        return True

    def on_obstacles_msg(self, msg) -> bool:
        """One obstacle-list message (OBSTACLE_TOPIC). True = used, False = dropped (another track)."""
        obs = obstacles_from_ros(msg)
        if obs.track_hash != self.ref_track_hash:
            return self._wrong(obs.track_hash)
        self.memory.on_obstacles(obs)
        self.counts["obstacles"] += 1
        return True

    def _wrong(self, track_hash: str | None) -> bool:
        self.counts["wrong_track"] += 1
        self.last_wrong_hash = track_hash
        return False

    def status_line(self, elapsed_s: float) -> str:
        """What arrived since the last line (in Hz), the newest car state, the trust word, the cones.
        Starts the counting again."""
        c, self.counts = self.counts, self._zero()
        hz = lambda n: n / elapsed_s if elapsed_s > 0 else 0.0  # noqa: E731
        line = f"state {hz(c['state']):5.1f} Hz | obstacles {hz(c['obstacles']):4.1f} Hz | "
        if self.car is None:
            line += "no car state yet"
        else:
            car = self.car
            mode = MODE_NAMES[car.mode] if 0 <= car.mode < len(MODE_NAMES) else str(car.mode)
            line += f"s_abs {car.s_abs:7.2f} m  e_y {car.e_y:+.2f} m  vx {car.vx:4.2f} m/s  {mode}"
        line += f" | trust {self.memory.trust()} | cones {len(self.memory.obstacles)}"
        if c["wrong_track"]:
            line += (f" | WRONG TRACK: {c['wrong_track']} messages dropped (they say {self.last_wrong_hash}, "
                     f"our reference is {self.ref_track_hash})")
        return line


@dataclass
class CycleResult:
    """What one cycle decided (PART 2)."""
    plan: CyclePlan             # Half B's preview for the MPC
    verdict: Verdict            # T6 on that plan: OK / WARN / FAIL + reasons
    cap: SpeedCap               # Half A's speed cap + reason
    trust: str                  # Half A's trust word
    zones: int                  # forbidden zones ahead
    method: str                 # Half B: which detour ("none" = racing line, no detour)
    cycle_ms: float
    ko: KeepOuts                # Half A's full handover (the zones themselves, for the drawing)
    window: dict                # Half B's chosen plan on its whole window (8 m every 0.25 m): the LOCAL reference


class Planner:
    """PART 2 without ROS: every CYCLE_S, the newest car state + Half A's notebook -> one CyclePlan.

    track = perception's apex_track track (walls + curvature), ref = our T5 PlanningReference."""

    def __init__(self, track, ref: PlanningReference):
        self.track, self.ref = track, ref
        self.ref_track_hash = ref.meta["track_hash"]
        self.walls = lambda s: track.width_at(s, d_min=D_MIN)     # corner-cut walls, as T5 / Half B
        self.half_b = Memory()                                    # Half B's sides + last path, across cycles
        self.last: CycleResult | str = "no cycle yet"             # the latest result (or why there was none)
        self.counts = self._zero()

    @staticmethod
    def _zero() -> dict:
        return {"cycles": 0, "skipped": 0, WARN: 0, FAIL: 0}

    def cycle(self, listener: Listener, now_us: int) -> CycleResult | str:
        """One cycle at time now_us (same clock as the messages' stamps). A CycleResult, or a short
        reason why there was no cycle."""
        result = self._cycle(listener, now_us)
        self.last = result
        if isinstance(result, str):
            self.counts["skipped"] += 1
        else:
            self.counts["cycles"] += 1
            if result.verdict.level in (WARN, FAIL):
                self.counts[result.verdict.level] += 1
        return result

    def _cycle(self, listener: Listener, now_us: int) -> CycleResult | str:
        car = listener.car
        if car is None:
            return "waiting for the car"
        age_s = (now_us - car.stamp_us) / 1e6
        if age_s > STATE_MAX_AGE_S:
            return f"car state {age_s * 1000:.0f} ms old: no cycle"
        t0 = time.perf_counter()
        ko = keep_outs(listener.memory, car, self.track.L, self.track.kappa_at, self.walls)   # Half A zones
        cap = speed_cap(ko)                                                                     # Half A cap
        plan, info = plan_cycle(with_speed_cap(ko, cap), car.s_abs, car.e_y, car.vx, self.ref,   # Half B
                                self.walls, self.track.kappa_at, memory=self.half_b,
                                track_hash=car.track_hash, ref_track_hash=self.ref_track_hash)
        verdict = check_cycle(plan, car.s_abs, car.e_y, car.track_hash, self.ref_track_hash)   # T6
        return CycleResult(plan, verdict, cap, ko.trust, len(ko.zones), info["method"],
                           (time.perf_counter() - t0) * 1000, ko, info["window"])

    def status_part(self) -> str:
        """The PART 2 half of the status line: the cycles since the last line + the latest one.
        Starts the counting again."""
        c, self.counts = self.counts, self._zero()
        line = f"cycles {c['cycles']} (skipped {c['skipped']}, WARN {c[WARN]}, FAIL {c[FAIL]})"
        last = self.last
        if isinstance(last, str):
            return line + f" | last: {last}"
        line += (f" | last: {last.verdict.level}, plan {last.method}, zones {last.zones}, "
                 f"cap {last.cap.v_cap:.2f} m/s ({last.cap.reason}), {last.cycle_ms:.1f} ms")
        if last.verdict.reasons:
            line += " | " + "; ".join(last.verdict.reasons)
        return line


# ---------------------------------------------------------------- the debug drawing (no ROS here)

def to_map(track, s, e_y) -> np.ndarray:
    """Track positions (s along the centre line, e_y + left of it) -> map x, y [m], shape (N, 2).
    Same formula as apex_track's pose_from_frenet: the centre-line point, then e_y sideways (to the left)."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    e_y = np.broadcast_to(np.asarray(e_y, dtype=float), s.shape)
    out = np.empty((len(s), 2))
    for i, (si, ei) in enumerate(zip(s, e_y)):
        cx, cy, th = track.point(si)                   # wraps, so s_abs past the lap works
        out[i] = cx - ei * np.sin(th), cy + ei * np.cos(th)
    return out


def racing_line_xy(track, ref: PlanningReference, step_m: float = 0.1) -> np.ndarray:
    """T5's racing line around the whole lap in map x, y, closed (the last point = the first)."""
    s = np.append(np.arange(0.0, ref.lap_length, step_m), 0.0)
    return to_map(track, s, ref.at(s)["e_y_ref_m"])


def zone_outline(track, z, n: int = 10) -> np.ndarray:
    """One zone's BOX (what Half B avoids) in map x, y, closed. Its sides follow the track's curve,
    so each long side is n points along s, not just the 4 corners."""
    s = np.linspace(z.s_from, z.s_to, n)
    right = to_map(track, s, z.e_y_from)               # right edge, going forward
    left = to_map(track, s[::-1], z.e_y_to)            # left edge, coming back
    return np.vstack([right, left, right[:1]])


@dataclass
class Drawing:
    """One cycle's picture in map x, y [m]: what the ROS layer turns into markers."""
    path: np.ndarray            # (K, 2) the planned points ahead (path[0] = the car)
    level: str                  # T6 verdict on that path: OK / WARN / FAIL (= its colour)
    zones: list                 # one closed outline (N, 2) per zone
    label: str                  # the status text shown above the car


def drawing(track, result: CycleResult) -> Drawing:
    """One CycleResult -> its picture."""
    p = result.plan
    label = (f"{result.verdict.level} | cap {result.cap.v_cap:.1f} m/s ({result.cap.reason}) | "
             f"plan {result.method} | trust {result.trust}")
    return Drawing(to_map(track, p.s, p.e_y_ref), result.verdict.level,
                   [zone_outline(track, z) for z in result.ko.zones], label)


# ---------------------------------------------------------------- the GLOBAL reference for Omar (no ROS here)

SOURCE_GLOBAL, SOURCE_LOCAL = 0, 1      # ReferenceTrajectory v0.1 (proposed to Omar 2026-11-08)


def global_reference_fields(track, ref: PlanningReference, version: int = 1) -> dict:
    """Our T5 reference -> every field of a GLOBAL ReferenceTrajectory, by its name in the message.

    The points are exactly the T5 file's rows (every 5 cm), plus their map x, y and the corridor
    (walls minus WALL_MARGIN, the same walls as T5 / T6 / Half B)."""
    s = ref._s[:-1]                                     # the file's own s values (the loader adds the lap end)
    r = ref.at(s)
    w_left, w_right = (np.asarray(w, dtype=float) for w in track.width_at(s, d_min=D_MIN))
    xy = to_map(track, s, r["e_y_ref_m"])
    problems = ref.meta.get("problems") or []
    return {
        "source": SOURCE_GLOBAL,
        "track_id": ref.meta["track_id"], "track_hash": ref.meta["track_hash"],
        "boundary_hash": ref.meta["boundary_hash"], "lap_length_m": ref.lap_length,
        "valid": bool(ref.meta.get("valid")), "reason": "; ".join(map(str, problems)), "version": int(version),
        "s_m": s, "x_m": xy[:, 0], "y_m": xy[:, 1],
        "e_y_ref_m": r["e_y_ref_m"], "e_psi_ref_rad": r["e_psi_ref_rad"],
        "kappa_ref_1pm": r["kappa_ref_1pm"], "v_ref_mps": r["v_ref_mps"],
        "e_y_min_m": -(w_right - WALL_MARGIN), "e_y_max_m": w_left - WALL_MARGIN,
    }


def local_reference_fields(track, ref: PlanningReference, result: CycleResult | str, version: int) -> dict:
    """One cycle -> every field of a LOCAL ReferenceTrajectory (same names as GLOBAL).

    The points are Half B's window: 8 m ahead of the car every 0.25 m (33 points). s_m is the car's
    running s (it keeps growing past the lap end, no jump back to 0). valid = T6 did not say FAIL;
    reason = T6's reasons. A skipped cycle (result is a str) -> valid False, no points, reason = why."""
    head = {"source": SOURCE_LOCAL, "track_id": ref.meta["track_id"], "track_hash": ref.meta["track_hash"],
            "boundary_hash": ref.meta["boundary_hash"], "lap_length_m": ref.lap_length, "version": int(version)}
    if isinstance(result, str):
        empty = np.zeros(0)
        return {**head, "valid": False, "reason": result,
                **{k: empty for k in ("s_m", "x_m", "y_m", "e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm",
                                      "v_ref_mps", "e_y_min_m", "e_y_max_m")}}
    w = result.window
    xy = to_map(track, w["s"], w["e_y_ref"])
    return {**head, "valid": result.verdict.level != FAIL, "reason": "; ".join(result.verdict.reasons),
            "s_m": w["s"], "x_m": xy[:, 0], "y_m": xy[:, 1],
            "e_y_ref_m": w["e_y_ref"], "e_psi_ref_rad": w["e_psi_ref"], "kappa_ref_1pm": w["kappa_ref"],
            "v_ref_mps": w["v_ref"], "e_y_min_m": w["e_y_min"], "e_y_max_m": w["e_y_max"]}


def plain(fields: dict) -> dict:
    """numpy numbers and arrays -> plain Python floats / ints / bools / lists, which a ROS message accepts."""
    out = {}
    for name, value in fields.items():
        if isinstance(value, np.ndarray):
            out[name] = [float(v) for v in value]
        elif isinstance(value, (np.floating, np.integer, np.bool_)):
            out[name] = value.item()
        else:
            out[name] = value
    return out


def load_track_and_reference():
    """Perception's track + our T5 reference; refuses to start when the reference was made for another track."""
    import apex_track
    from apex_track import load_track
    path = Path(TRACK_YAML) if TRACK_YAML else \
        Path(apex_track.__file__).resolve().parents[1] / "tracks" / TRACK / "track.yaml"
    track = load_track(str(path))
    ref = PlanningReference.from_csv(REFERENCE, track_hash=track.track_hash,      # ValueError on a mismatch
                                     boundary_hash=track.boundary_hash)
    return track, ref


def main() -> int:
    import rclpy                                  # ROS only from here on
    from apex_msgs.msg import ObstacleArray, StateEstimate
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import Marker, MarkerArray

    colours = {"OK": (0.1, 0.8, 0.1), WARN: (1.0, 0.8, 0.0), FAIL: (1.0, 0.1, 0.1)}

    def line(stamp, ns, xy, rgb, width, z=0.05):
        """One LINE_STRIP marker in the map frame through the points xy (N, 2)."""
        mk = Marker()
        mk.header.stamp, mk.header.frame_id = stamp, "map"
        mk.ns, mk.id, mk.type, mk.action = ns, 0, Marker.LINE_STRIP, Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = width
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = *rgb, 1.0
        mk.points = [Point(x=float(x), y=float(y), z=z) for x, y in xy]
        return mk

    def cycle_markers(stamp, d: Drawing | None) -> MarkerArray:
        """Clear the last picture, then draw this cycle's (d None = skipped cycle: just clear)."""
        clear = Marker()
        clear.header.stamp, clear.header.frame_id, clear.action = stamp, "map", Marker.DELETEALL
        arr = MarkerArray(markers=[clear])
        if d is None:
            return arr
        arr.markers.append(line(stamp, "path", d.path, colours.get(d.level, (1, 1, 1)), 0.06, z=0.10))
        for k, box in enumerate(d.zones):
            mk = line(stamp, "zones", box, (1.0, 0.5, 0.0), 0.03)
            mk.id = k
            arr.markers.append(mk)
        txt = Marker()
        txt.header.stamp, txt.header.frame_id = stamp, "map"
        txt.ns, txt.id, txt.type, txt.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(d.path[0, 0]), float(d.path[0, 1]), 0.6
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.2
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = d.label
        arr.markers.append(txt)
        return arr

    track, ref = load_track_and_reference()
    listener = Listener(ref.meta["track_hash"])
    planner = Planner(track, ref)

    class PlanningNode(Node):
        def __init__(self):
            super().__init__("planning")
            self.create_subscription(StateEstimate, STATE_TOPIC, self.on_state, 10)
            self.create_subscription(ObstacleArray, OBSTACLE_TOPIC, self.on_obstacles, 10)
            self.create_timer(CYCLE_S, self.run_cycle)
            self.create_timer(PRINT_EVERY_S, self.print_status)
            self.last_print = self.get_clock().now()
            self.marker_pub = None
            if PUBLISH_MARKERS:
                self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
                latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)   # late viewers get it too
                self.line_pub = self.create_publisher(Marker, RACING_LINE_TOPIC, latched)
                self.line_pub.publish(line(self.get_clock().now().to_msg(), "racing_line",
                                           racing_line_xy(track, ref), (0.2, 0.5, 1.0), 0.03, z=0.02))
            if PUBLISH_GLOBAL_REFERENCE:
                ReferenceTrajectory = importlib.import_module(f"{MSG_PACKAGE}.msg").ReferenceTrajectory
                latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)   # late readers get it
                self.global_pub = self.create_publisher(ReferenceTrajectory, GLOBAL_REFERENCE_TOPIC, latched)
                m = ReferenceTrajectory()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "map"
                for name, value in plain(global_reference_fields(track, ref)).items():
                    setattr(m, name, value)
                self.global_pub.publish(m)
                self.get_logger().info(f"GLOBAL reference sent on {GLOBAL_REFERENCE_TOPIC} ({MSG_PACKAGE}): "
                                       f"{len(m.s_m)} points, valid {m.valid}, track {m.track_id} {m.track_hash}")
            self.local_pub = None
            if PUBLISH_LOCAL_REFERENCE:
                self.ReferenceTrajectory = importlib.import_module(f"{MSG_PACKAGE}.msg").ReferenceTrajectory
                self.local_pub = self.create_publisher(self.ReferenceTrajectory, LOCAL_REFERENCE_TOPIC, 10)
                self.local_version = 0
            self.get_logger().info(f"reference {REFERENCE} on track {track.track_id} (track_hash "
                                   f"{listener.ref_track_hash}); listening to {STATE_TOPIC} and {OBSTACLE_TOPIC}; "
                                   f"a cycle every {CYCLE_S} s")

        def run_cycle(self):
            now = self.get_clock().now()
            result = planner.cycle(listener, now.nanoseconds // 1000)
            if self.local_pub is not None:              # skipped cycles too: valid False + why
                self.local_version += 1
                m = self.ReferenceTrajectory()
                m.header.stamp = now.to_msg()
                m.header.frame_id = "map"
                for name, value in plain(local_reference_fields(track, ref, result, self.local_version)).items():
                    setattr(m, name, value)
                self.local_pub.publish(m)
            if self.marker_pub is not None:
                d = None if isinstance(result, str) else drawing(track, result)
                self.marker_pub.publish(cycle_markers(now.to_msg(), d))

        def on_state(self, msg):
            if not listener.on_state_msg(msg):
                self.wrong_track(msg.track_hash)

        def on_obstacles(self, msg):
            if not listener.on_obstacles_msg(msg):
                self.wrong_track(msg.track_hash)

        def wrong_track(self, track_hash):
            self.get_logger().error(f"message from track {track_hash}, our reference is "
                                    f"{listener.ref_track_hash}: dropped", throttle_duration_sec=1.0)

        def print_status(self):
            now = self.get_clock().now()
            elapsed = (now - self.last_print).nanoseconds * 1e-9
            self.last_print = now
            self.get_logger().info(listener.status_line(elapsed) + " || " + planner.status_part())

    rclpy.init()
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
