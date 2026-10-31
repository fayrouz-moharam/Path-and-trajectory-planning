"""T7 A5: the replay report -- cycles every 0.1 s, the summary, the truth check, and a missing recording."""
from pathlib import Path

import run_t7_replay as R

MSG_HEADER = "sequence_id,stamp_us,status,list_epoch,n_obstacles,latency_ms\n"
OBS_HEADER = "sequence_id,stamp_us,id,x_m,y_m,radius_m,std_xy_m,s_track_m,e_y_m,detections,seen_now\n"
STATE_HEADER = ("stamp_us,x_m,y_m,psi_rad,vx_mps,vy_mps,r_radps,s_abs_m,s_track_m,lap,e_y_m,e_psi_rad,"
                "std_xy_m,std_psi_rad,mode,flags\n")


class FakeTrack:
    """A straight 50 m loop, walls 0.8 m each side: stands in for apex_track's track."""
    L = 50.0

    def kappa_at(self, s):
        return 0.0

    def width_at(self, s):
        return 0.8, 0.8


def tiny_recording(folder: Path, cone_e_y=0.4) -> Path:
    """2 s: the car stands 1.2 s (perception OK), then drives 2 m/s; one cone 6 m ahead of the start."""
    folder.mkdir(parents=True, exist_ok=True)
    msgs, obs, states = [], [], []
    for k in range(15):                                              # obstacle messages every ~144 ms
        t = 144_000 * k
        msgs.append(f"{k},{t},0,0,1,11.0\n")
        obs.append(f"{k},{t},1,0,0,0.075,0.02,6.0,{cone_e_y},3,1\n")
    for k in range(201):                                             # car states every 10 ms
        t = 10_000 * k
        moving = t > 1_200_000
        s = 2.0 * (t - 1_200_000) / 1e6 if moving else 0.0
        states.append(f"{t},0,0,0,{2.0 if moving else 0.0},0,0,{s},{s},0,0,0,0.05,0.01,1,0\n")   # 16 columns
    (folder / "obstacle_messages.csv").write_text(MSG_HEADER + "".join(msgs))
    (folder / "obstacles.csv").write_text(OBS_HEADER + "".join(obs))
    (folder / "state_estimate.csv").write_text(STATE_HEADER + "".join(states))
    return folder


def test_one_row_every_cycle_and_trust_recovers_standing_still(tmp_path):
    rows, packages = R.replay_half_a(tiny_recording(tmp_path / "run"), FakeTrack())
    assert len(rows) == len(packages) == 21                          # 2.0 s / 0.1 s, plus the first
    assert [r["t_s"] for r in rows[:3]] == [0.0, 0.1, 0.2]
    assert rows[-1]["trust"] == "full" and rows[-1]["v_cap_mps"] == 6.0        # 1 s standing with OK
    assert rows[0]["trust"] == "stale" and rows[0]["v_cap_mps"] < 6.0          # t = 0: no message heard yet
    assert rows[1]["trust"] == "incomplete" and rows[1]["v_cap_mps"] < 6.0     # heard, not yet looked 1 s


def test_summary_counts(tmp_path):
    rows, _ = R.replay_half_a(tiny_recording(tmp_path / "run"), FakeTrack())
    s = R.summarise(rows, R.CYCLE_S)
    assert s["cycles"] == 21 and s["time_capped_while_moving_s"] == 0.0        # trusted before it moved
    assert s["most_zones_at_once"] == 1 and "cone list incomplete" in s["cap_reasons"]
    assert sum(s["time_by_trust_s"].values()) == round(21 * R.CYCLE_S, 2)


def test_truth_check_passes_on_the_true_cone_and_fails_on_a_wrong_one(tmp_path):
    rows, packages = R.replay_half_a(tiny_recording(tmp_path / "run"), FakeTrack())
    truth = tmp_path / "cones_truth.csv"
    truth.write_text("label,x,y,s,e_y,size_m,gap_to_wall_m,clearance_m\nopen,0,0,6.0,0.4,0.15,0.3,0.1\n")
    ok = R.truth_check(packages, truth, FakeTrack.L)
    assert ok["zones_checked"] > 0 and ok["zones_missing_their_cone"] == 0
    truth.write_text("label,x,y,s,e_y,size_m,gap_to_wall_m,clearance_m\nopen,0,0,6.0,-0.3,0.15,0.3,0.1\n")
    assert R.truth_check(packages, truth, FakeTrack.L)["zones_missing_their_cone"] > 0   # 70 cm off: caught


def test_no_recording_exits_2(tmp_path):
    assert R.run(None) == 2
    assert R.run(tmp_path / "nothing_here") == 2
