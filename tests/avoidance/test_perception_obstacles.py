"""T7 A1: reading perception's obstacle messages (live and recorded) and replaying them in time order."""
from types import SimpleNamespace

import pytest

from t7_avoidance.perception_obstacles import STATUS_OK, STATUS_RESET, CarState, ObstacleMessage, from_ros, read_recording, replay

MSG_HEADER = "sequence_id,stamp_us,status,list_epoch,n_obstacles,latency_ms\n"
OBS_HEADER = "sequence_id,stamp_us,id,x_m,y_m,radius_m,std_xy_m,s_track_m,e_y_m,detections,seen_now\n"
STATE_HEADER = ("stamp_us,x_m,y_m,psi_rad,vx_mps,vy_mps,r_radps,s_abs_m,s_track_m,lap,e_y_m,e_psi_rad,"
                "std_xy_m,std_psi_rad,mode,flags\n")


def recording(tmp_path, messages, obstacles, states):
    """Write a tiny recorded run in Fatma's CSV format."""
    (tmp_path / "obstacle_messages.csv").write_text(MSG_HEADER + "".join(messages))
    (tmp_path / "obstacles.csv").write_text(OBS_HEADER + "".join(obstacles))
    (tmp_path / "state_estimate.csv").write_text(STATE_HEADER + "".join(states))
    return tmp_path


def test_reads_messages_and_groups_their_obstacles(tmp_path):
    run = recording(tmp_path,
                    ["0,1000,0,0,2,11.0\n", "1,2000,3,1,0,9.0\n"],
                    ["0,1000,7,1.0,2.0,0.075,0.02,5.9,0.42,3,1\n", "0,1000,8,3.0,4.0,0.090,0.03,9.1,-0.30,2,0\n"],
                    ["900,0,0,0,2.0,0,0,5.0,5.0,0,0.1,0,0.05,0.01,1,0\n"])
    msgs, states = read_recording(run)
    assert [m.status for m in msgs] == [STATUS_OK, STATUS_RESET] and [m.list_epoch for m in msgs] == [0, 1]
    first = msgs[0].obstacles
    assert [o.id for o in first] == [7, 8] and first[0].e_y == 0.42 and first[1].seen_now is False
    assert msgs[1].obstacles == () and msgs[0].track_hash is None             # not in the recordings
    assert states == [CarState(900, 5.0, 5.0, 0.1, 2.0, 1, 0)]


def test_refuses_a_message_whose_obstacle_count_is_wrong(tmp_path):
    run = recording(tmp_path, ["0,1000,0,0,2,11.0\n"],                        # says 2 obstacles ...
                    ["0,1000,7,1.0,2.0,0.075,0.02,5.9,0.42,3,1\n"], [])        # ... but only 1 is there
    with pytest.raises(ValueError, match="says 2 obstacles"):
        read_recording(run)


def test_replay_is_in_time_order_with_the_state_first_on_a_tie():
    msgs = [ObstacleMessage(2000, 0, STATUS_OK, 0, ()), ObstacleMessage(1000, 1, STATUS_OK, 0, ())]
    states = [CarState(t, 0.0, 0.0, 0.0, 1.0, 1, 0) for t in (500, 1000, 2500)]
    out = list(replay(msgs, states))
    assert [x.stamp_us for x in out] == [500, 1000, 1000, 2000, 2500]
    assert isinstance(out[1], CarState) and isinstance(out[2], ObstacleMessage)  # same stamp: state first


def test_from_ros_reads_a_live_message():
    cone = SimpleNamespace(id=4, x_m=1.0, y_m=2.0, radius_m=0.08, std_xy_m=0.02, s_track_m=19.3, e_y_m=0.44,
                           detections=5, seen_now=True)
    msg = SimpleNamespace(stamp_us=123, sequence_id=9, status=0, list_epoch=2, obstacles=[cone],
                          track_hash="15f62f337c4c8787", max_range_m=8.0, blind_strip_m=0.1)
    m = from_ros(msg)
    assert m.obstacles[0].s_track == 19.3 and m.track_hash == "15f62f337c4c8787" and m.blind_strip_m == 0.1


# ---------------------------------------------------------------- A2: the notebook (ObstacleMemory)

from t7_avoidance.perception_obstacles import (FROZEN, FULL, INCOMPLETE, RECOVER_DISTANCE_M, STALE,  # noqa: E402
                                               STATUS_FROZEN_LOCALIZATION, Obstacle, ObstacleMemory)

CONE = Obstacle(1, 0.0, 0.0, 0.075, 0.02, 10.0, 0.4, 3, True)


def msg(t_ms, status=STATUS_OK, epoch=0, cones=(CONE,)):
    return ObstacleMessage(int(t_ms * 1000), 0, status, epoch, tuple(cones))


def state(t_ms, s_abs):
    return CarState(int(t_ms * 1000), s_abs, s_abs, 0.0, 2.0, 1, 0)


def driven_in(mem, t_ms, metres):
    """Car states every 10 ms, 0.05 m apart, with an OK message every 140 ms (perception keeps talking)."""
    s = mem._last_s or 0.0
    for k in range(int(metres / 0.05)):
        t = t_ms + 10 * k
        if k % 14 == 0:
            mem.on_obstacles(msg(t, cones=mem.obstacles or (CONE,)))
        mem.on_state(state(t, s + 0.05 * (k + 1)))
    return t_ms + 10 * int(metres / 0.05)


def test_start_is_incomplete_until_the_car_has_driven():
    mem = ObstacleMemory()
    assert mem.trust() == STALE                              # nothing heard yet
    mem.on_state(state(0, 0.0))
    mem.on_obstacles(msg(10))
    assert mem.trust() == INCOMPLETE and mem.obstacles == (CONE,)
    driven_in(mem, 20, RECOVER_DISTANCE_M + 0.1)
    assert mem.trust() == FULL


def test_standing_still_with_ok_messages_recovers_without_driving():
    """A standing start: 1 s of OK messages while the car stands still = the LiDAR has looked ahead."""
    mem = ObstacleMemory()
    for k in range(120):                                             # 1.2 s at the start line, car not moving
        t = 10 * k
        if k % 14 == 0:
            mem.on_obstacles(msg(t))
        mem.on_state(CarState(t * 1000, 0.0, 0.0, 0.0, 0.0, 1, 0))
    assert mem.trust() == FULL and mem.driven_m == 0.0


def test_standing_still_does_not_count_while_perception_is_not_ok():
    mem = ObstacleMemory()
    for k in range(200):                                             # 2 s standing, but perception FROZEN
        t = 10 * k
        if k % 14 == 0:
            mem.on_obstacles(msg(t, status=STATUS_FROZEN_LOCALIZATION, cones=()))
        mem.on_state(CarState(t * 1000, 0.0, 0.0, 0.0, 0.0, 1, 0))
    assert mem.trust() == INCOMPLETE


def test_frozen_keeps_the_list():
    mem = ObstacleMemory()
    t = driven_in(mem, 0, 6.0)
    mem.on_obstacles(msg(t, status=STATUS_FROZEN_LOCALIZATION, cones=()))
    assert mem.trust() == FROZEN and mem.obstacles == (CONE,)     # cones don't move: the old list stays


def test_reset_tears_the_page_out_and_ok_alone_is_not_enough():
    mem = ObstacleMemory()
    t = driven_in(mem, 0, 6.0)
    mem.on_obstacles(msg(t, status=STATUS_RESET, epoch=1, cones=()))
    assert mem.trust() == INCOMPLETE and mem.obstacles == ()
    mem.on_obstacles(msg(t + 140, epoch=1, cones=()))              # "OK with 0 cones" right after the reset
    assert mem.trust() == INCOMPLETE                                # still not trusted
    driven_in(mem, t + 150, RECOVER_DISTANCE_M + 0.1)
    assert mem.trust() == FULL


def test_a_new_epoch_also_tears_the_page_out():
    mem = ObstacleMemory()
    t = driven_in(mem, 0, 6.0)
    mem.on_obstacles(msg(t, epoch=1))                               # the RESET message itself was lost
    assert mem.trust() == INCOMPLETE


def test_nothing_is_counted_while_frozen():
    mem = ObstacleMemory()
    t = driven_in(mem, 0, 6.0)
    mem.on_obstacles(msg(t, status=STATUS_RESET, epoch=1, cones=()))
    s = mem._last_s
    for k in range(200):                                            # 10 m driven, but perception is FROZEN
        if k % 14 == 0:
            mem.on_obstacles(msg(t + 10 * k, status=STATUS_FROZEN_LOCALIZATION, epoch=1, cones=()))
        mem.on_state(state(t + 10 * k, s + 0.05 * (k + 1)))
    assert mem.driven_m == 0.0 and mem.trust() == INCOMPLETE


def test_silence_makes_it_stale_but_keeps_the_list():
    mem = ObstacleMemory()
    t = driven_in(mem, 0, 6.0)
    mem.on_state(state(t + 500, mem._last_s))                       # 0.5 s with no obstacle message
    assert mem.trust() == STALE and mem.obstacles == (CONE,)


def test_run_b_story():
    """Fatma's run B replayed: full before the fault, incomplete from the RESET, through the 'OK with 0 cones',
    full again only after RECOVER_DISTANCE_M of good messages. Skipped if the recording isn't available."""
    import bisect
    import os
    from pathlib import Path
    run = Path(os.environ.get("APEX_PLANNING_PACKAGE", "")) / "run_B_localization_fault"
    if not (run / "obstacle_messages.csv").exists():
        pytest.skip("set APEX_PLANNING_PACKAGE to the unpacked planning_package folder")
    msgs, states = read_recording(run)
    mem, times, words = ObstacleMemory(), [], []
    reset_us = next(m.stamp_us for m in msgs if m.status == STATUS_RESET)
    zero_ok_us = next(m.stamp_us for m in msgs if m.stamp_us > reset_us and m.status == STATUS_OK)
    for item in replay(msgs, states):
        (mem.on_state if isinstance(item, CarState) else mem.on_obstacles)(item)
        times.append(item.stamp_us)
        words.append(mem.trust())

    def word_at(t_us):                                              # the word in force at t (last item at or before it)
        return words[bisect.bisect_right(times, t_us) - 1]

    assert word_at(reset_us - 200_000) == FULL                     # 0.2 s before the fault
    assert word_at(reset_us) == INCOMPLETE
    assert word_at(zero_ok_us) == INCOMPLETE                        # the "OK with 0 cones" trap
    assert words[-1] == FULL                                        # trusted again by the end of the run


# ---------------------------------------------------------------- A3: forbidden zones (keep_outs)

from t7_avoidance.perception_obstacles import BUFFER, CONFIDENCE_K, HALF_CAR, HORIZON_M, keep_outs  # noqa: E402

LAP = 50.0


def straight(_s):
    return 0.0


def wide_walls(_s):
    return 1.5, 1.5


def cone(s, e_y, cid=1, radius=0.075, std=0.02):
    return Obstacle(cid, 0.0, 0.0, radius, std, s, e_y, 3, True)


def zones_for(cones, car_s, kappa_at=straight, width_at=wide_walls, laps=0):
    mem = ObstacleMemory()
    mem.on_obstacles(msg(0, cones=cones))
    car = CarState(0, car_s + laps * LAP, car_s, 0.0, 3.0, 1, 0)
    return keep_outs(mem, car, LAP, kappa_at, width_at)


R0 = 0.075 + CONFIDENCE_K * 0.02 + HALF_CAR + BUFFER                     # bubble of the default cone


def test_straight_box_is_the_bubble():
    z = zones_for([cone(12.0, 0.3)], car_s=10.0, laps=2).zones[0]           # lap 3: continuous s
    assert z.radius == pytest.approx(R0) and z.stretch == pytest.approx(1.0)
    assert z.s_center == pytest.approx(2 * LAP + 12.0)
    assert (z.s_from, z.s_to) == pytest.approx((z.s_center - R0, z.s_center + R0))
    assert (z.e_y_from, z.e_y_to) == pytest.approx((0.3 - R0, 0.3 + R0))


def test_inside_of_a_corner_stretches_the_zone():
    z = zones_for([cone(12.0, 0.5)], car_s=10.0, kappa_at=lambda s: 0.9).zones[0]   # left turn, cone on the inside
    d = 1 - 0.9 * (0.5 + R0)                                                  # its most stretched (inside) edge
    assert z.stretch == pytest.approx(1 / d) and z.s_to - z.s_from == pytest.approx(2 * R0 / d)
    assert z.s_to - z.s_from > 2 * R0                                         # longer than on a straight


def test_too_far_and_already_passed_are_skipped_beside_is_kept():
    ko = zones_for([cone(10.0 + HORIZON_M + R0 + 0.1, 0.0, cid=1),          # starts beyond the look-ahead
                    cone(10.0 - R0 - 0.1, 0.0, cid=2),                       # whole zone behind the car
                    cone(10.0 - 0.1, 0.0, cid=3)], car_s=10.0)               # just passed, zone still reaches the car
    assert [z.id for z in ko.zones] == [3]


def test_finish_line_cone_just_ahead():
    ko = zones_for([cone(1.0, 0.0)], car_s=LAP - 1.0, laps=0)               # car 1 m before the line, cone 1 m after
    assert ko.zones[0].s_center == pytest.approx(LAP + 1.0)                 # 2 m ahead, not 48 m behind


def test_near_wall_closes_a_side():
    walls = lambda s: (0.8, 0.8)                                             # noqa: E731
    ko = zones_for([cone(12.0, 0.6, cid=1), cone(15.0, 0.0, cid=2)], car_s=10.0, width_at=walls)
    near = {z.id: z.near_wall for z in ko.zones}
    assert near == {1: True, 2: False}                 # 0.6 + bubble reaches the left corridor edge; the middle doesn't


def test_which_side_is_closed():
    walls = lambda s: (0.8, 0.8)                                             # noqa: E731
    left_cone, right_cone, middle = (zones_for([cone(12.0, e)], car_s=10.0, width_at=walls).zones[0]
                                     for e in (0.6, -0.6, 0.0))
    assert left_cone.closed_left and not left_cone.closed_right          # cone near the left wall: use the right
    assert left_cone.gap_right == pytest.approx(0.6 - R0 + 0.55)         # room between the zone and the right edge
    assert right_cone.closed_right and not right_cone.closed_left        # mirror image: use the left
    assert not (middle.closed_left or middle.closed_right)               # 1.1 m corridor, 0.82 m zone: both open
    assert middle.gap_left == pytest.approx(middle.gap_right)


def test_box_carries_trust_epoch_and_sorted_zones():
    ko = zones_for([cone(15.0, 0.0, cid=2), cone(12.0, 0.0, cid=1)], car_s=10.0)
    assert [z.id for z in ko.zones] == [1, 2] and ko.trust == INCOMPLETE and ko.list_epoch == 0
    assert ko.horizon_m == HORIZON_M and ko.car_s_abs == 10.0


def test_run_b_zones_contain_the_true_cones():
    """Fatma's run B: every zone holds its TRUE cone's own bubble (cones_truth.csv), as a box and as a circle."""
    import csv
    import math
    import os
    from pathlib import Path
    pkg = Path(os.environ.get("APEX_PLANNING_PACKAGE", ""))
    if not (pkg / "cones_truth.csv").exists():
        pytest.skip("set APEX_PLANNING_PACKAGE to the unpacked planning_package folder")
    apex = pytest.importorskip("apex_track")
    import run_perception_t2_t3_t4 as run_perception
    tr = apex.load_track(str(run_perception.find_track_yaml("closed1")))
    truth = [(float(r["s"]), float(r["e_y"]), float(r["size_m"]) / 2) for r in csv.DictReader(open(pkg / "cones_truth.csv"))]
    msgs, states = read_recording(pkg / "run_B_localization_fault")
    mem, car, n = ObstacleMemory(), None, 0
    for item in replay(msgs, states):
        if isinstance(item, CarState):
            mem.on_state(item)
            car = item
            continue
        mem.on_obstacles(item)
        if car is None:
            continue
        for z in keep_outs(mem, car, tr.L, tr.kappa_at, tr.width_at).zones:
            gaps = [(((ts - z.s_center % tr.L + tr.L / 2) % tr.L - tr.L / 2) / z.stretch, te - z.e_y_center, rr)
                    for ts, te, rr in truth]
            ds, de, rr = min(gaps, key=lambda g: math.hypot(g[0], g[1]))     # its true cone
            need = rr + HALF_CAR + BUFFER
            assert abs(ds) + need <= z.radius + 1e-9 and abs(de) + need <= z.radius + 1e-9    # inside the box
            assert math.hypot(ds, de) + need <= z.radius + 1e-9                                # inside the circle
            n += 1
    assert n > 100
