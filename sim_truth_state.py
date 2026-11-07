"""SIMULATOR ONLY: a stand-in for Fatma's localization that publishes the simulator's TRUE car state.

Why: on a 4-core laptop the simulator + her full localization chain (state pipeline, AMCL, wheel odometry,
fake sensors) do not run in real time -> her /state_estimate goes LOST and arrives hundreds of ms late.
This node replaces that whole chain with one cheap step, so our planning node can be tested live:

    /ego_racecar/odom (simulator's true pose + speed)
        -> apex_track projection (the same one her state node uses) -> s_abs, e_y, e_psi, lap
        -> /state_estimate (apex_msgs/StateEstimate, contract v0.3, mode OK, our track's hash)

Her obstacle node and test driver work unchanged (the driver steers from the true odometry anyway; the
obstacle node needs a /state_estimate with mode OK). It does NOT test her localization -- that is hers.
Never run it on the car (there is no /ego_racecar/odom there, so it would just stay silent).

    in a ROS 2 shell (NO venv), with the simulator running and her localization NOT running:
    python3 sim_truth_state.py

The logic is one class without ROS (TruthState, tested in tests/test_sim_truth_state.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# ============================================================== SETTINGS
TRACK = "closed1"                   # apex_track track folder (must be the simulator's map)
ODOM_TOPIC = "/ego_racecar/odom"    # the simulator's true odometry
STATE_TOPIC = "/state_estimate"     # what her state node would publish
PUBLISH_HZ = 100.0                  # her rate (the simulator's odometry comes faster: thinned to this)
SCHEMA = "apex-state-estimate/1.1"  # contract v0.3
VALID_FOR_MS = 50                   # contract v0.3
STD_XY_M = 0.02                     # small, honest-looking uncertainties (the truth has none, but 0 could
STD_PSI_RAD = 0.01                  # look like "not filled in" to a reader)
STD_V_MPS = 0.02
# =======================================================================

FLAG_PROJECTION_RESEEDED = 256      # bit 8, as in the contract


class TruthState:
    """The simulator's true pose -> every number in a StateEstimate (no ROS here).

    track = apex_track track. Its projection remembers the last s, so call fields() in time order."""

    def __init__(self, track):
        self.track = track
        self.seq = 0

    def fields(self, x: float, y: float, yaw: float, vx: float, vy: float, r: float, stamp_us: int) -> dict:
        """One pose (map frame) -> the StateEstimate fields, by their names in the message."""
        s_abs, s_track, e_y, e_psi, _ = self.track.project(x, y, yaw)
        lap = round((s_abs - s_track) / self.track.L)          # s_abs = lap * L + s_track
        self.seq += 1
        return {
            "schema_version": SCHEMA, "sequence_id": self.seq, "stamp_us": int(stamp_us),
            "track_id": self.track.track_id, "track_hash": self.track.track_hash,
            "x_m": x, "y_m": y, "psi_rad": _wrap(yaw),
            "vx_mps": vx, "vy_mps": vy, "r_radps": r,
            "s_abs_m": s_abs, "e_y_m": e_y, "e_psi_rad": e_psi, "s_track_m": s_track, "lap": lap,
            "ax_mps2": 0.0, "ay_mps2": 0.0,                     # her node sends 0 too
            "std_xy_m": STD_XY_M, "std_psi_rad": STD_PSI_RAD, "std_vx_mps": STD_V_MPS, "std_vy_mps": STD_V_MPS,
            "std_r_radps": STD_PSI_RAD, "std_e_y_m": STD_XY_M, "std_e_psi_rad": STD_PSI_RAD,
            "mode": 1,                                          # MODE_OK: the truth is never lost
            "flags": FLAG_PROJECTION_RESEEDED if self.track.reseeded else 0,
            "valid_for_ms": VALID_FOR_MS,
        }


def _wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def load_track():
    import apex_track
    from apex_track import load_track as load
    return load(str(Path(apex_track.__file__).resolve().parents[1] / "tracks" / TRACK / "track.yaml"))


def main() -> int:
    import rclpy                                  # ROS only from here on
    from apex_msgs.msg import StateEstimate
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    truth = TruthState(load_track())

    class SimTruthState(Node):
        def __init__(self):
            super().__init__("sim_truth_state")
            self.pub = self.create_publisher(StateEstimate, STATE_TOPIC, 10)
            self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, 10)
            self.last_us = -10**18
            self.get_logger().info(f"SIMULATOR ONLY: {ODOM_TOPIC} -> {STATE_TOPIC} on track {truth.track.track_id} "
                                   f"(hash {truth.track.track_hash}), {PUBLISH_HZ:.0f} Hz, mode OK")

        def on_odom(self, odom):
            t_us = odom.header.stamp.sec * 1_000_000 + odom.header.stamp.nanosec // 1000
            if t_us - self.last_us < 1e6 / PUBLISH_HZ - 500:     # thin to PUBLISH_HZ (0.5 ms slack)
                return
            self.last_us = t_us
            p, q, tw = odom.pose.pose.position, odom.pose.pose.orientation, odom.twist.twist
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            m = StateEstimate()
            m.header.stamp = odom.header.stamp
            m.header.frame_id = "map"
            for name, value in truth.fields(p.x, p.y, yaw, tw.linear.x, tw.linear.y, tw.angular.z, t_us).items():
                setattr(m, name, value)
            now = self.get_clock().now().nanoseconds // 1000
            m.publish_us = int(now)                              # last thing before sending, as hers
            self.pub.publish(m)

    rclpy.init()
    node = SimTruthState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
