"""planning_node without ROS, with fake messages in Fatma's field names (StateEstimate v0.3, ObstacleArray v0.1):
PART 1 (Listener): every message -> translate -> same track? -> Half A's notebook + newest state.
PART 2 (Planner): every cycle -> Half A zones + speed cap -> Half B plan_cycle -> T6 verdict (real closed1 track)."""
from types import SimpleNamespace

import numpy as np
import pytest

from planning_node import Listener, Planner, drawing, racing_line_xy, to_map
from t7_avoidance.perception_obstacles import FULL, INCOMPLETE, STALE, STATUS_OK

OURS, OTHER = "15f62f337c4c8787", "a44e46d238cf50c2"       # closed1, Levine


def state(t_ms, s_abs=10.0, vx=2.0, mode=1, track_hash=OURS):
    return SimpleNamespace(stamp_us=int(t_ms * 1000), s_abs_m=s_abs, s_track_m=s_abs % 47.29, e_y_m=0.05,
                           vx_mps=vx, mode=mode, flags=0, track_hash=track_hash)


def cone(i=1):
    return SimpleNamespace(id=i, x_m=1.0, y_m=2.0, radius_m=0.075, std_xy_m=0.02, s_track_m=15.0, e_y_m=0.3,
                           detections=4, seen_now=True)


def obstacles(t_ms, cones=(), status=STATUS_OK, epoch=0, track_hash=OURS):
    return SimpleNamespace(stamp_us=int(t_ms * 1000), sequence_id=0, status=status, list_epoch=epoch,
                           obstacles=list(cones), track_hash=track_hash, max_range_m=8.0, blind_strip_m=0.1)


def test_our_track_reaches_half_a_and_the_newest_state_is_kept():
    L = Listener(OURS)
    assert L.on_state_msg(state(0, s_abs=10.0)) and L.on_state_msg(state(10, s_abs=10.02))
    assert L.on_obstacles_msg(obstacles(10, [cone()]))
    assert L.car.s_abs == 10.02 and L.car.track_hash == OURS          # the newest one, for PART 2
    assert len(L.memory.obstacles) == 1                               # Half A's notebook got the cone


def test_another_track_is_dropped_and_never_reaches_half_a():
    L = Listener(OURS)
    assert not L.on_state_msg(state(0, track_hash=OTHER))
    assert not L.on_obstacles_msg(obstacles(0, [cone()], track_hash=OTHER))
    assert L.car is None and L.memory.obstacles == ()
    assert L.counts["wrong_track"] == 2 and L.last_wrong_hash == OTHER


def test_half_a_trust_moves_as_the_messages_arrive():
    L = Listener(OURS)
    assert L.memory.trust() == STALE                                  # nothing heard yet
    L.on_state_msg(state(0, s_abs=0.0))
    L.on_obstacles_msg(obstacles(0))
    assert L.memory.trust() == INCOMPLETE                             # the start counts as a reset
    for k in range(1, 40):                                            # drive 6 m with OK messages
        L.on_state_msg(state(k * 100, s_abs=k * 0.15))
        if k % 2 == 0:
            L.on_obstacles_msg(obstacles(k * 100))
    assert L.memory.trust() == FULL


def test_status_line_counts_per_second_and_starts_again():
    L = Listener(OURS)
    for k in range(100):
        L.on_state_msg(state(k * 10))
    for k in range(7):
        L.on_obstacles_msg(obstacles(k * 140, [cone()]))
    L.on_state_msg(state(1000, track_hash=OTHER))
    line = L.status_line(1.0)
    assert "state 100.0 Hz" in line and "obstacles  7.0 Hz" in line and "cones 1" in line and " OK" in line
    assert "WRONG TRACK: 1 messages dropped" in line and OTHER in line
    assert "state   0.0 Hz" in L.status_line(1.0)                     # counting started again


def test_status_line_before_any_state():
    assert "no car state yet" in Listener(OURS).status_line(1.0)


# ---------------------------------------------------------------- PART 2: Planner, on the real closed1 track

@pytest.fixture(scope="module")
def closed1():
    pytest.importorskip("apex_track")
    import planning_node
    return planning_node.load_track_and_reference()          # the node's own SETTINGS: closed1 + its T5 file


def drive(track, ref, listener, s_from, s_to, dt_us=100_000):
    """The car on the racing line at T5's speed, with OK (empty) obstacle messages; returns the last stamp."""
    t, s = 0, s_from
    while s <= s_to + 1e-9:
        r = ref.at(s)
        listener.on_state_msg(SimpleNamespace(stamp_us=t, s_abs_m=s, s_track_m=s % track.L,
                                              e_y_m=r["e_y_ref_m"], vx_mps=r["v_ref_mps"], mode=1, flags=0,
                                              track_hash=ref.meta["track_hash"]))
        listener.on_obstacles_msg(obstacles(t, track_hash=ref.meta["track_hash"]))
        t, s = t + dt_us, s + 0.1
    return t - dt_us


def test_no_cycle_without_a_car_or_with_an_old_one(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    assert P.cycle(L, 0) == "waiting for the car"
    t = drive(track, ref, L, 4.0, 10.0)
    assert "ms old: no cycle" in P.cycle(L, t + 60_000)              # 60 ms > Fatma's 50 ms
    assert P.counts["skipped"] == 2 and P.counts["cycles"] == 0


def test_clear_track_follows_the_racing_line(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    t = drive(track, ref, L, 4.0, 10.0)                              # 6 m with OK messages -> trust full
    r = P.cycle(L, t + 10_000)
    assert r.trust == FULL and r.zones == 0 and r.cap.v_cap == 6.0 and r.verdict.level == "OK"
    assert abs(r.plan.s[0] - 10.0) < 1e-9 and len(r.plan.s) == 9     # 9 stages, starting at the car
    assert np.allclose(r.plan.e_y_ref, ref.at(r.plan.s)["e_y_ref_m"], atol=0.02)   # on T5's racing line


def cycle_with_cone_at_13(track, ref):
    """Drive to s 10, then a cone 3 m ahead right on the racing line; one cycle. -> (result, the cone's e_y)."""
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    t = drive(track, ref, L, 4.0, 10.0)
    e_cone = float(ref.at(13.0)["e_y_ref_m"])
    cone_13 = SimpleNamespace(id=1, x_m=0.0, y_m=0.0, radius_m=0.075, std_xy_m=0.03, s_track_m=13.0,
                              e_y_m=e_cone, detections=5, seen_now=True)
    L.on_obstacles_msg(obstacles(t + 20_000, [cone_13], track_hash=ref.meta["track_hash"]))
    return P.cycle(L, t + 30_000), e_cone


def test_a_cone_on_the_racing_line_is_passed_around(closed1):
    track, ref = closed1
    r, e_cone = cycle_with_cone_at_13(track, ref)
    assert r.zones == 1 and r.method != "none" and r.verdict.level != "FAIL"
    near = np.abs(r.plan.s - 13.0) < 0.4                              # the stages beside the cone
    assert np.all(np.abs(r.plan.e_y_ref[near] - e_cone) > 0.3)        # the path keeps away from it


def test_status_part_reports_the_last_cycle(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    t = drive(track, ref, L, 4.0, 10.0)
    P.cycle(L, t + 10_000)
    line = P.status_part()
    assert line.startswith("cycles 1 (skipped 0, WARN 0, FAIL 0)") and "last: OK" in line and "cap 6.00" in line
    assert P.counts["cycles"] == 0                                    # counting started again


# ---------------------------------------------------------------- the debug drawing (map x, y for RViz)

def test_to_map_is_apex_tracks_own_formula(closed1):
    track, _ = closed1
    from apex_track.vectors import pose_from_frenet
    s, e = np.array([0.5, 13.0, 30.0, 47.0, 60.0]), np.array([0.0, 0.3, -0.4, 0.1, -0.2])   # 60 = past the lap
    xy = to_map(track, s, e)
    for (x, y), si, ei in zip(xy, s, e):
        px, py, _ = pose_from_frenet(track, si, ei, 0.0)
        assert abs(x - px) < 1e-9 and abs(y - py) < 1e-9
    centre = to_map(track, 13.0, 0.0)[0]
    left = to_map(track, 13.0, 0.3)[0]
    assert abs(np.hypot(*(left - centre)) - 0.3) < 1e-9                 # e_y 0.3 = 0.3 m sideways


def test_racing_line_is_one_closed_loop(closed1):
    track, ref = closed1
    xy = racing_line_xy(track, ref)
    assert np.allclose(xy[0], xy[-1])                                  # closed
    assert np.max(np.hypot(*np.diff(xy, axis=0).T)) < 0.2              # no jumps (0.1 m steps)


def test_clear_track_drawing(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    t = drive(track, ref, L, 4.0, 10.0)
    d = drawing(track, P.cycle(L, t + 10_000))
    assert d.level == "OK" and d.zones == [] and d.path.shape == (9, 2)
    on_line = to_map(track, P.last.plan.s, ref.at(P.last.plan.s)["e_y_ref_m"])
    assert np.max(np.hypot(*(d.path - on_line).T)) < 0.02              # drawn on the racing line
    assert d.label.startswith("OK | cap 6.0 m/s") and "trust full" in d.label


def test_cone_drawing_has_its_zone_around_the_cone(closed1):
    track, ref = closed1
    r, e_cone = cycle_with_cone_at_13(track, ref)
    d = drawing(track, r)
    assert len(d.zones) == 1
    box = d.zones[0]
    assert len(box) == 21 and np.allclose(box[0], box[-1])             # 10 + 10 points, closed
    cone_xy = to_map(track, r.ko.zones[0].s_center, e_cone)[0]
    assert np.all(box.min(axis=0) < cone_xy) and np.all(cone_xy < box.max(axis=0))   # the cone is inside
    z = r.ko.zones[0]
    assert abs(np.hypot(*(box[0] - box[-2])) - (z.e_y_to - z.e_y_from)) < 1e-6    # box as wide as the zone


def test_node_refuses_a_reference_made_for_another_track(monkeypatch):
    pytest.importorskip("apex_track")
    import planning_node
    monkeypatch.setattr(planning_node, "REFERENCE",                   # Levine's reference ...
                        planning_node.Path("out/run_perception_levine/t5/planning_reference.csv"))
    with pytest.raises(ValueError, match="track_hash"):              # ... on the closed1 track
        planning_node.load_track_and_reference()
