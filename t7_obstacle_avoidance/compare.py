"""Which detour is fastest? Graph vs sampling on a real track, one cone at a time.

    python -m t7_obstacle_avoidance.compare            (from the repo folder)

A cone (radius 0.075 m, sigma 0.03 m) is put every STEP_M metres around the lap, at each of the OFFSETS
(on the racing line, and 0.3 m to either side of it). Half A's keep_outs() turns it into a zone, the car sits
CAR_BEFORE_M before it on the racing line at T5's speed, and each method plans that cycle. A method "fails" a
case when its line leaves the corridor or check_cycle() says FAIL. Its time = driving the 8 m window at the
car's grip / acceleration / braking limits. Writes OUT_DIR/<track>.csv (every case) and OUT_DIR/<track>_summary.txt
(the table printed), for every track in TRACKS. Each method is run alone, then "fastest" (the default: both, the
quicker that passes; "used" says which one it sent).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from t5_reference.loader import PlanningReference
from t7_avoidance.perception_obstacles import CarState, Obstacle, ObstacleMemory, keep_outs

from .detour import METHODS
from .plan import plan_cycle

# ============================================================== SETTINGS
TRACKS = {               # track name -> its T5 reference (perception's real tracks)
    "closed1": Path("out/run_perception_closed1/t5/planning_reference.csv"),
    "levine": Path("out/run_perception_levine/t5/planning_reference.csv"),
}
TRACK = "closed1"
REFERENCE = TRACKS[TRACK]
APEX_TRACK = Path("grad-perception/apex_track")   # perception's package (added to the path if not installed)
STEP_M = 2.0             # m between cone positions around the lap
OFFSETS = (0.0, -0.3, 0.3)   # m: cone relative to the racing line
CAR_BEFORE_M = 3.0       # m: the car this far before the cone
OUT_DIR = Path("out/t7_planner_compare")
# =======================================================================


def load(track_name=TRACK, reference=REFERENCE):
    if str(APEX_TRACK) not in sys.path and APEX_TRACK.exists():
        sys.path.insert(0, str(APEX_TRACK))
    from apex_track import load_track
    track = load_track(str(APEX_TRACK / "tracks" / track_name / "track.yaml"))
    ref = PlanningReference.from_csv(reference, track_hash=track.track_hash)
    return track, ref


def zones_for(track, ref, s_cone, e_cone, car_s):
    mem = ObstacleMemory()
    mem.obstacles = (Obstacle(1, 0.0, 0.0, 0.075, 0.03, s_cone % track.L, e_cone, 5, True),)
    mem.last_message_us = mem.now_us = 0
    mem.last_status, mem.recovering, mem.epoch = 0, False, 0
    r = ref.at(car_s)
    car = CarState(0, car_s, car_s % track.L, float(r["e_y_ref_m"]), float(r["v_ref_mps"]), 1, 0)
    walls = lambda s: track.width_at(s, d_min=0.1)      # noqa: E731
    return keep_outs(mem, car, track.L, track.kappa_at, walls)


def run(track_name=TRACK, reference=REFERENCE, out_dir=OUT_DIR):
    track, ref = load(track_name, reference)
    walls = lambda s: track.width_at(s, d_min=0.1)      # noqa: E731
    rows = []
    for s_cone in np.arange(CAR_BEFORE_M, track.L + CAR_BEFORE_M, STEP_M):
        for off in OFFSETS:
            e_cone = float(ref.at(s_cone)["e_y_ref_m"]) + off
            car_s = s_cone - CAR_BEFORE_M
            r = ref.at(car_s)
            ko = zones_for(track, ref, s_cone, e_cone, car_s)
            for m in METHODS + ("fastest",):          # each method alone, then the default "fastest"
                t0 = time.perf_counter()
                plan, info = plan_cycle(ko, car_s, float(r["e_y_ref_m"]), float(r["v_ref_mps"]), ref, walls,
                                        track.kappa_at, method=m, track_hash=track.track_hash,
                                        ref_track_hash=ref.meta["track_hash"])
                ms = (time.perf_counter() - t0) * 1000
                used = info["method"]
                ok = used != "none" if m == "fastest" else used == m
                rows.append({"s_cone": round(float(s_cone % track.L), 2), "offset": off, "method": m,
                             "ok": ok, "side": info["sides"].get(1, ""),
                             "used": used,
                             "time_s": round(info["times"][m if m != "fastest" else used], 4) if ok else "",
                             "why_not": "" if ok else info.get("reason") or info["rejected"].get(m, ""),
                             "cycle_ms": round(ms, 2)})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{track_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # summary: success rate, and time compared on the cases where both worked
    lines = []
    say = lambda s: (print(s), lines.append(s))      # noqa: E731
    alone = [r for r in rows if r["method"] != "fastest"]
    cases = {(r["s_cone"], r["offset"]) for r in rows}
    all_ok = [c for c in cases if all(r["ok"] for r in alone if (r["s_cone"], r["offset"]) == c)]
    say(f"{track_name}: {len(cases)} cone positions, both methods worked on {len(all_ok)}")
    say(f"{'method':10} {'worked':>8} {'mean time (s), common cases':>28} {'fastest on':>11} {'mean ms':>8}")
    for m in METHODS:
        mine = [r for r in rows if r["method"] == m]
        common = [r["time_s"] for r in mine if (r["s_cone"], r["offset"]) in all_ok]
        wins = sum(1 for c in all_ok if min((r for r in alone if (r["s_cone"], r["offset"]) == c),
                                             key=lambda r: r["time_s"])["method"] == m)
        say(f"{m:10} {sum(r['ok'] for r in mine):>4}/{len(mine):<3} "
              f"{(np.mean(common) if common else float('nan')):>28.4f} {wins:>11} "
              f"{np.mean([r['cycle_ms'] for r in mine]):>8.1f}")
    best = [r for r in rows if r["method"] == "fastest"]
    common = [r["time_s"] for r in best if (r["s_cone"], r["offset"]) in all_ok]
    used = {u: sum(r["used"] == u for r in best) for u in sorted({r["used"] for r in best})}
    say(f"{'fastest':10} {sum(r['ok'] for r in best):>4}/{len(best):<3} {np.mean(common):>28.4f} {'-':>11} "
        f"{np.mean([r['cycle_ms'] for r in best]):>8.1f}   (the default; used: {used})")
    fails = [r for r in rows if not r["ok"]]
    for r in fails:
        say(f"  {r['method']:9} failed at s {r['s_cone']:6.2f}, cone {r['offset']:+.1f} m: {r['why_not']}")
    (out_dir / f"{track_name}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / (track_name + '.csv')} and {track_name}_summary.txt")
    return rows


if __name__ == "__main__":
    for name, reference in TRACKS.items():
        run(name, reference)
