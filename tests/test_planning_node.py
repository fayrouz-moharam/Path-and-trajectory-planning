"""planning_node without ROS, with fake messages in Fatma's field names (StateEstimate v0.3, ObstacleArray v0.1):
PART 1 (Listener): every message -> translate -> same track? -> Half A's notebook + newest state.
PART 2 (Planner): every cycle -> Half A zones + speed cap -> Half B plan_cycle -> T6 verdict (real closed1 track)."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import planning_node
from planning_node import (SOURCE_GLOBAL, SOURCE_LOCAL, Listener, Planner, drawing, global_reference_fields,
                           local_reference_fields, plain, racing_line_xy, to_map)
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


# ---------------------------------------------------------------- Half B's whole window (the LOCAL reference's data)

WINDOW_KEYS = {"s", "e_y_ref", "e_psi_ref", "kappa_ref", "v_ref", "e_y_min", "e_y_max"}


def check_window(r):
    """The window: 33 stations 0.25 m apart from the car, inside its corridor, and the 9 stages lie on it."""
    w, p = r.window, r.plan
    assert set(w) == WINDOW_KEYS and all(len(w[k]) == 33 for k in WINDOW_KEYS)
    assert w["s"][0] == pytest.approx(p.s[0]) and np.allclose(np.diff(w["s"]), 0.25)          # from the car, 8 m
    assert np.all(w["e_y_min"] <= w["e_y_ref"] + 1e-12) and np.all(w["e_y_ref"] <= w["e_y_max"] + 1e-12)
    assert np.all(w["v_ref"] <= p.v_cap + 1e-12) and np.all(w["v_ref"] > 0)
    at = lambda k: np.interp(p.s, w["s"], w[k])                                                # noqa: E731
    assert np.allclose(at("e_psi_ref"), p.e_psi_ref, atol=1e-9)        # same heading ...
    assert np.allclose(at("kappa_ref"), p.kappa_ref, atol=1e-9)        # ... and curvature as the stages
    assert np.allclose(np.clip(at("e_y_ref"), p.e_y_min, p.e_y_max), p.e_y_ref, atol=1e-9)   # the stage's own
    #                                                   corridor (the stricter of its two neighbouring stations)
    assert np.all(np.abs(at("v_ref") - p.v_ref) <= 0.05)               # stages: min after interpolating


def test_clear_track_window_is_the_racing_line(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    t = drive(track, ref, L, 4.0, 10.0)
    r = P.cycle(L, t + 10_000)
    check_window(r)
    assert np.allclose(r.window["e_y_ref"], ref.at(r.window["s"])["e_y_ref_m"], atol=1e-9)   # no bend at all


def test_cone_window_goes_around_the_cone(closed1):
    track, ref = closed1
    r, e_cone = cycle_with_cone_at_13(track, ref)
    check_window(r)
    near = np.abs(r.window["s"] - 13.0) < 0.4
    assert np.all(np.abs(r.window["e_y_ref"][near] - e_cone) > 0.3)    # the 8 m path keeps away from it too


# ---------------------------------------------------------------- the GLOBAL reference (ReferenceTrajectory v0.1)

AGREED_FIELDS = {"source", "track_id", "track_hash", "boundary_hash", "lap_length_m", "valid", "reason", "version",
                 "s_m", "x_m", "y_m", "e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm", "v_ref_mps",
                 "e_y_min_m", "e_y_max_m"}           # the format agreed 2026-11-08 (header is set by the ROS part)


def test_global_reference_has_exactly_the_agreed_fields(closed1):
    track, ref = closed1
    f = global_reference_fields(track, ref)
    assert set(f) == AGREED_FIELDS
    n = ref.meta["n_points"]
    assert all(len(f[k]) == n for k in AGREED_FIELDS if k.endswith(("_m", "_rad", "_1pm", "_mps")) and k != "lap_length_m")
    assert f["source"] == SOURCE_GLOBAL and f["valid"] and f["reason"] == "" and f["version"] == 1
    assert f["track_id"] == "closed1" and f["track_hash"] == track.track_hash and f["boundary_hash"] == track.boundary_hash
    assert f["lap_length_m"] == pytest.approx(track.L, abs=1e-5)


def test_global_reference_is_the_t5_file_row_by_row(closed1):
    track, ref = closed1
    f = global_reference_fields(track, ref)
    csv = np.loadtxt(planning_node.REFERENCE, delimiter=",", skiprows=1)
    for j, k in enumerate(["s_m", "e_y_ref_m", "e_psi_ref_rad", "kappa_ref_1pm", "v_ref_mps"]):
        assert np.allclose(f[k], csv[:, j], atol=1e-9), k                  # same numbers as the file


def test_global_reference_map_points_and_corridor(closed1):
    track, ref = closed1
    f = global_reference_fields(track, ref)
    xy = np.column_stack([f["x_m"], f["y_m"]])
    assert np.allclose(xy, to_map(track, f["s_m"], f["e_y_ref_m"]))      # x, y = the same points on the map
    assert np.max(np.hypot(*np.diff(xy, axis=0).T)) < 0.1                 # one smooth line, no jumps
    assert np.all(f["e_y_min_m"] < f["e_y_max_m"])                        # the corridor is open everywhere ...
    assert np.all(f["e_y_ref_m"] >= f["e_y_min_m"] - 0.005)               # ... and the racing line is inside it
    assert np.all(f["e_y_ref_m"] <= f["e_y_max_m"] + 0.005)               #     (T5's own 5 mm tolerance)


def test_global_reference_values_are_plain_python_for_ros(closed1):
    track, ref = closed1
    p = planning_node.plain(global_reference_fields(track, ref))
    allowed = (int, float, bool, str)
    for name, value in p.items():
        if isinstance(value, list):
            assert all(type(v) is float for v in value), name                # float64[]: plain floats only
        else:
            assert type(value) in allowed, (name, type(value))
    assert type(p["valid"]) is bool and type(p["version"]) is int and type(p["source"]) is int


def test_the_msg_file_has_exactly_the_agreed_fields():
    msg = Path(__file__).resolve().parents[1] / "ros" / "apex_planning_msgs" / "msg" / "ReferenceTrajectory.msg"
    names = set()
    for line in msg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_/\[\]]+)\s+([a-z][a-z0-9_]*)\s*(#|$)", line)   # "type name", not CONSTANT = 0
        if m:
            names.add(m.group(2))
    assert names == AGREED_FIELDS | {"header"}                             # the .msg and the node agree


def test_node_refuses_a_reference_made_for_another_track(monkeypatch):
    pytest.importorskip("apex_track")
    import planning_node
    monkeypatch.setattr(planning_node, "REFERENCE",                   # Levine's reference ...
                        planning_node.Path("out/run_perception_levine/t5/planning_reference.csv"))
    with pytest.raises(ValueError, match="track_hash"):              # ... on the closed1 track
        planning_node.load_track_and_reference()


# ---------------------------------------------------------------- the LOCAL reference (same message, every cycle)

def test_local_reference_is_the_window_with_the_agreed_fields(closed1):
    track, ref = closed1
    r, _ = cycle_with_cone_at_13(track, ref)
    f = local_reference_fields(track, ref, r, version=7)
    assert set(f) == AGREED_FIELDS
    assert f["source"] == SOURCE_LOCAL and f["version"] == 7 and f["valid"] == (r.verdict.level != "FAIL")
    assert f["track_hash"] == ref.meta["track_hash"] and f["lap_length_m"] == ref.lap_length
    assert len(f["s_m"]) == 33 and np.allclose(np.diff(f["s_m"]), 0.25)
    for k, w in {"s_m": "s", "e_y_ref_m": "e_y_ref", "e_psi_ref_rad": "e_psi_ref", "kappa_ref_1pm": "kappa_ref",
                 "v_ref_mps": "v_ref", "e_y_min_m": "e_y_min", "e_y_max_m": "e_y_max"}.items():
        assert np.array_equal(f[k], r.window[w])                      # exactly Half B's window, renamed
    xy = planning_node.to_map(track, f["s_m"], f["e_y_ref_m"])
    assert np.allclose(f["x_m"], xy[:, 0]) and np.allclose(f["y_m"], xy[:, 1])


def test_local_reference_s_keeps_growing_past_the_lap_end(closed1):
    track, ref = closed1
    L, P = Listener(ref.meta["track_hash"]), Planner(track, ref)
    s0 = ref.lap_length - 3.0
    t = drive(track, ref, L, s0 - 6.0, s0)
    f = local_reference_fields(track, ref, P.cycle(L, t + 10_000), version=1)
    assert np.all(np.diff(f["s_m"]) > 0) and f["s_m"][-1] > ref.lap_length   # no jump back to 0


def test_local_reference_fail_is_sent_invalid_with_the_reasons(closed1):
    from dataclasses import replace
    from t6_feasibility.runtime import Verdict
    track, ref = closed1
    r, _ = cycle_with_cone_at_13(track, ref)
    f = local_reference_fields(track, ref, replace(r, verdict=Verdict("FAIL", ["a", "b"])), version=1)
    assert not f["valid"] and f["reason"] == "a; b" and len(f["s_m"]) == 33


def test_local_reference_skipped_cycle_is_invalid_and_empty(closed1):
    track, ref = closed1
    f = plain(local_reference_fields(track, ref, "car state 120 ms old: no cycle", version=3))
    assert set(f) == AGREED_FIELDS and not f["valid"] and f["reason"] == "car state 120 ms old: no cycle"
    assert f["s_m"] == [] and f["v_ref_mps"] == [] and f["version"] == 3
