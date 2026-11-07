"""sim_truth_state (simulator-only stand-in for Fatma's localization), without ROS, on the real closed1 track."""
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from planning_node import Listener
from sim_truth_state import TruthState, load_track

apex_track = pytest.importorskip("apex_track")


@pytest.fixture()
def track():
    return load_track()                       # a fresh one each test: the projection remembers the last s


def pose_at(track, s, e_y=0.0):
    """Map pose e_y left of the centre line at s, pointing along it."""
    cx, cy, th = track.point(s)
    return cx - e_y * math.sin(th), cy + e_y * math.cos(th), th


def test_fills_every_field_of_her_message(track):
    msg = Path(apex_track.__file__).resolve().parents[2] / "apex_msgs" / "msg" / "StateEstimate.msg"
    if not msg.exists():
        pytest.skip(f"no {msg}")
    names = set()
    for line in msg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_/]+)\s+([a-z][a-z0-9_]*)\s*(#|$)", line)   # "type name", not CONSTANT = 1
        if m:
            names.add(m.group(2))
    f = TruthState(track).fields(*pose_at(track, 5.0), 1.5, 0.0, 0.0, 1_000_000)
    assert set(f) <= names                                  # no made-up names
    assert names - set(f) == {"header", "publish_us"}       # only these are set by the ROS part


def test_a_pose_on_the_track_gives_its_s_and_e_y(track):
    t = TruthState(track)
    for s, e in [(2.0, 0.0), (13.0, 0.3), (30.0, -0.4)]:
        f = t.fields(*pose_at(track, s, e), 1.5, 0.0, 0.0, 0)
        assert abs(f["s_track_m"] - s) < 1e-3 and abs(f["e_y_m"] - e) < 1e-3 and abs(f["e_psi_rad"]) < 1e-3
        assert f["mode"] == 1 and f["track_hash"] == track.track_hash


def test_crossing_the_start_line_counts_a_lap(track):
    t = TruthState(track)
    s_path = (track.L - 1.0, track.L - 0.3, track.L + 0.3, track.L + 1.0)      # across the line
    out = [t.fields(*pose_at(track, s % track.L), 1.5, 0.0, 0.0, 0) for s in s_path]
    assert [f["lap"] for f in out] == [0, 0, 1, 1]
    assert all(abs(f["s_abs_m"] - (f["lap"] * track.L + f["s_track_m"])) < 1e-9 for f in out)
    assert abs(out[-1]["s_abs_m"] - (track.L + 1.0)) < 1e-3     # continuous: no jump back to 0


def test_our_planning_node_accepts_it(track):
    L = Listener(track.track_hash)
    f = TruthState(track).fields(*pose_at(track, 10.0, 0.2), 1.5, 0.0, 0.0, 2_000_000)
    assert L.on_state_msg(SimpleNamespace(**f))
    assert abs(L.car.s_abs - 10.0) < 1e-3 and abs(L.car.e_y - 0.2) < 1e-3 and L.car.mode == 1
