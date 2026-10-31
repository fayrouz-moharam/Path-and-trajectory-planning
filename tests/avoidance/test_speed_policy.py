"""T7 A4: the speed cap -- clear, list not trusted, a blocked cone, a tight gap, lowest wins, and run B."""
import math

import pytest

from t7_avoidance import speed_policy as P
from t7_avoidance.perception_obstacles import FROZEN, FULL, INCOMPLETE, STALE
from t7_avoidance.speed_policy import speed_cap, stop_speed, with_speed_cap
from t7_avoidance.types import KeepOut, KeepOuts

LAP = 50.0


def zone(s_center, e_y_from, e_y_to, cid=1, half_s=0.4):
    """A zone between walls 0.8 m away: the car's centre may use +-0.55 m, so the gaps are measured to that."""
    e_c = 0.5 * (e_y_from + e_y_to)
    gap_left, gap_right = 0.55 - e_y_to, e_y_from + 0.55
    closed_left, closed_right = gap_left < 0.05, gap_right < 0.05
    return KeepOut(cid, s_center - half_s, s_center + half_s, e_y_from, e_y_to, s_center, e_c,
                   0.5 * (e_y_to - e_y_from), 1.0, closed_left or closed_right, True,
                   gap_left, gap_right, closed_left, closed_right)


def package(trust=FULL, zones=(), car_s=10.0):
    return KeepOuts(car_s, trust, 0, 8.0, 6.0, tuple(zones))


def test_stop_speed_formula():
    v = stop_speed(2.5)
    assert v * P.REACTION_S + v * v / (2 * P.A_BRAKE_MAX) == pytest.approx(2.5)   # reaction + braking = distance
    assert stop_speed(0.0) == 0.0 and stop_speed(-1.0) == 0.0


def test_all_clear_is_normal_racing():
    cap = speed_cap(package())
    assert cap.v_cap == P.V_MAX and cap.reason.startswith("clear")


@pytest.mark.parametrize("trust", [INCOMPLETE, STALE])
def test_untrusted_list_caps_the_speed(trust):
    cap = speed_cap(package(trust=trust))
    assert cap.v_cap == pytest.approx(stop_speed(P.ASSUMED_DETECTION_M - P.STOP_MARGIN_M))
    assert math.isclose(cap.v_cap, 3.076, abs_tol=0.01) and trust in cap.reason


def test_frozen_list_does_not_slow_down():
    assert speed_cap(package(trust=FROZEN)).v_cap == P.V_MAX         # cones don't move


def test_blocked_cone_cap_falls_to_zero_as_the_car_gets_closer():
    blocked = zone(20.0, -0.60, 0.60)                                            # covers the whole corridor
    caps = [speed_cap(package(zones=[blocked], car_s=s)).v_cap for s in (12.0, 16.0, 18.9, 19.3)]
    assert caps == sorted(caps, reverse=True) and caps[-1] == pytest.approx(0.0, abs=1e-9)   # stop: 20 - 0.4 - 0.3
    # stopping from 6 m/s needs 6 * 0.3 + 6^2 / (2 * 3) = 7.8 m: farther than that the normal cap still wins
    assert speed_cap(package(zones=[blocked], car_s=10.0)).reason.startswith("clear")      # 9.3 m to stop
    assert "blocks the track" in speed_cap(package(zones=[blocked], car_s=12.0)).reason   # 7.3 m to stop


def test_tight_gap_caps_and_can_be_switched_off(monkeypatch):
    tight = zone(15.0, -0.47, 0.60)                                              # 8 cm left on the right side
    cap = speed_cap(package(zones=[tight]))
    assert cap.v_cap == P.TIGHT_GAP_V and "tight gap" in cap.reason
    monkeypatch.setattr(P, "TIGHT_GAP_RULE", False)
    assert speed_cap(package(zones=[tight])).v_cap == P.V_MAX


def test_wide_gap_does_not_slow_down():
    assert speed_cap(package(zones=[zone(15.0, 0.0, 0.60)])).v_cap == P.V_MAX   # 55 cm on the right


def test_lowest_cap_wins():
    cap = speed_cap(package(trust=INCOMPLETE, zones=[zone(15.0, -0.47, 0.60)]))
    assert cap.v_cap == P.TIGHT_GAP_V and "tight gap" in cap.reason             # 3.0 < 3.08


def test_with_speed_cap_fills_the_package():
    cap = speed_cap(package(trust=STALE))
    assert with_speed_cap(package(trust=STALE), cap).v_cap == cap.v_cap


def test_run_b_caps():
    """Fatma's run B through A1-A4: 3.08 m/s exactly while the list is untrusted, 6 m/s when trusted,
    and closed1's cones never block the track. Skipped if the recording isn't available."""
    import os
    from pathlib import Path
    pkg = Path(os.environ.get("APEX_PLANNING_PACKAGE", ""))
    if not (pkg / "run_B_localization_fault" / "obstacle_messages.csv").exists():
        pytest.skip("set APEX_PLANNING_PACKAGE to the unpacked planning_package folder")
    apex = pytest.importorskip("apex_track")
    import run_perception_t2_t3_t4 as run_perception
    from t7_avoidance.perception_obstacles import CarState, ObstacleMemory, keep_outs, read_recording, replay
    tr = apex.load_track(str(run_perception.find_track_yaml("closed1")))
    msgs, states = read_recording(pkg / "run_B_localization_fault")
    mem, car, seen = ObstacleMemory(), None, set()
    for item in replay(msgs, states):
        if isinstance(item, CarState):
            mem.on_state(item)
            car = item
        else:
            mem.on_obstacles(item)
        if car is None:
            continue
        ko = keep_outs(mem, car, tr.L, tr.kappa_at, tr.width_at)
        cap = speed_cap(ko)
        expected = P.V_MAX if ko.trust in (FULL, FROZEN) else stop_speed(P.ASSUMED_DETECTION_M - P.STOP_MARGIN_M)
        assert cap.v_cap == pytest.approx(expected), (ko.trust, cap)
        seen.add(ko.trust)
    assert {FULL, INCOMPLETE} <= seen
