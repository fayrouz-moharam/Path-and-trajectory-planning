"""Replay a recorded drive through T7 Half A, as the car would have done it, every CYCLE_S:
perception's obstacle + state messages -> notebook + trust (A2) -> forbidden zones (A3) -> speed cap (A4).

    1. edit the SETTINGS below (which recording, which track)
    2. python run_t7_replay.py

A recording = a folder with obstacle_messages.csv, obstacles.csv and state_estimate.csv (perception's
recording format). If cones_truth.csv sits in it or one folder up, every zone is also checked against
the true cone positions.
Writes OUT_DIR/<recording name>/: timeline.csv (one row per cycle), summary.json, t7_replay.png.
Exit code: 0 = fine, 1 = a zone missed its true cone, 2 = no recording / no track.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import run_perception_t2_t3_t4 as perception_run
from t7_avoidance.perception_obstacles import (BUFFER, HALF_CAR, CarState, ObstacleMemory, keep_outs,
                                               read_recording, replay)
from t7_avoidance.speed_policy import speed_cap, with_speed_cap

# ============================================================== SETTINGS
RECORDING = None              # folder of a recorded drive (perception's CSV format) -- set before running
TRACK = "closed1"             # the apex_track track the recording was driven on
CYCLE_S = 0.1                 # how often Half A runs (the MPC's cycle)
OUT_DIR = Path("out/t7_replay")
PLOT = True
# =======================================================================

TRUST_LEVELS = ["full", "frozen", "incomplete", "stale"]          # for the plot, least to most cautious


def replay_half_a(recording: Path, track, cycle_s: float = CYCLE_S) -> tuple[list[dict], list]:
    """Every cycle_s of the recording: one row (what Half A decided) + the KeepOuts package it built."""
    msgs, states = read_recording(recording)
    mem, car, next_us, t0_us = ObstacleMemory(), None, None, None
    rows, packages = [], []
    for item in replay(msgs, states):
        if isinstance(item, CarState):
            mem.on_state(item)
            car = item
        else:
            mem.on_obstacles(item)
        if car is None:
            continue
        if t0_us is None:                                            # not "or": a recording may start at t = 0
            t0_us = next_us = item.stamp_us
        if item.stamp_us < next_us:
            continue                                                 # not time for the next cycle yet
        next_us += int(cycle_s * 1e6)
        start = time.perf_counter()
        ko = keep_outs(mem, car, track.L, track.kappa_at, track.width_at)          # A3
        cap = speed_cap(ko, track.width_at, track.L)                                # A4
        ko = with_speed_cap(ko, cap)
        cycle_ms = (time.perf_counter() - start) * 1e3
        rows.append({"t_s": round((item.stamp_us - t0_us) / 1e6, 3), "s_abs_m": round(car.s_abs, 3),
                     "speed_mps": round(car.vx, 3), "trust": ko.trust, "zones": len(ko.zones),
                     "v_cap_mps": round(cap.v_cap, 3), "reason": cap.reason, "cycle_ms": round(cycle_ms, 3)})
        packages.append(ko)
    return rows, packages


def truth_check(packages: list, truth_csv: Path, lap_length: float) -> dict:
    """Does every zone hold its TRUE cone's own bubble (cone + half car + buffer), as a box and as a circle?"""
    truth = [(float(r["s"]), float(r["e_y"]), float(r["size_m"]) / 2) for r in csv.DictReader(open(truth_csv))]
    n, missed, tightest = 0, 0, math.inf
    for ko in packages:
        for z in ko.zones:
            gaps = [(((ts - z.s_center % lap_length + lap_length / 2) % lap_length - lap_length / 2) / z.stretch,
                     te - z.e_y_center, r) for ts, te, r in truth]
            ds, de, r = min(gaps, key=lambda g: math.hypot(g[0], g[1]))          # its true cone
            spare = z.radius - (max(abs(ds), abs(de)) + r + HALF_CAR + BUFFER)     # box (the circle is inside it)
            spare = min(spare, z.radius - (math.hypot(ds, de) + r + HALF_CAR + BUFFER))
            n, missed, tightest = n + 1, missed + (spare < -1e-9), min(tightest, spare)
    return {"zones_checked": n, "zones_missing_their_cone": missed,
            "tightest_spare_m": None if n == 0 else round(tightest, 4)}


def summarise(rows: list[dict], cycle_s: float) -> dict:
    capped = [r for r in rows if r["v_cap_mps"] < 6.0]
    return {
        "duration_s": rows[-1]["t_s"] if rows else 0.0,
        "cycles": len(rows),
        "time_by_trust_s": {w: round(sum(r["trust"] == w for r in rows) * cycle_s, 2) for w in TRUST_LEVELS},
        "time_capped_s": round(len(capped) * cycle_s, 2),
        "time_capped_while_moving_s": round(sum(r["speed_mps"] > 0.1 for r in capped) * cycle_s, 2),
        "lowest_cap_mps": min((r["v_cap_mps"] for r in rows), default=None),
        "cap_reasons": sorted({r["reason"].split(":")[0] for r in capped}),
        "most_zones_at_once": max((r["zones"] for r in rows), default=0),
        "cycle_ms_mean": round(sum(r["cycle_ms"] for r in rows) / max(len(rows), 1), 3),
        "cycle_ms_max": max((r["cycle_ms"] for r in rows), default=0.0),
    }


def plot_replay(path: Path, rows: list[dict]) -> None:
    """Over time: trust word, speed cap with the car's speed, number of zones ahead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t_s"] for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax[0].step(t, [TRUST_LEVELS.index(r["trust"]) for r in rows], where="post")
    ax[0].set(yticks=range(4), yticklabels=TRUST_LEVELS, title="trust in the cone list (A2)")
    ax[1].plot(t, [r["v_cap_mps"] for r in rows], label="speed cap (A4)")
    ax[1].plot(t, [r["speed_mps"] for r in rows], label="car speed (recorded)", alpha=0.7)
    ax[1].set(ylabel="m/s", title="speed cap vs the car's speed")
    ax[1].legend(loc="lower right", fontsize=8)
    ax[2].step(t, [r["zones"] for r in rows], where="post")
    ax[2].set(xlabel="time [s]", ylabel="zones", title="forbidden zones within the look-ahead (A3)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(recording: Path | None = RECORDING, track_name: str = TRACK, out_dir: Path = OUT_DIR,
        plot: bool = PLOT) -> int:
    if recording is None or not (Path(recording) / "obstacle_messages.csv").exists():
        print(f"no recording at {recording}: set RECORDING to a recorded drive's folder", file=sys.stderr)
        return 2
    from apex_track import load_track
    recording = Path(recording)
    track = load_track(str(perception_run.find_track_yaml(track_name)))
    rows, packages = replay_half_a(recording, track)
    summary = summarise(rows, CYCLE_S)
    truth_csv = next((p for p in (recording / "cones_truth.csv", recording.parent / "cones_truth.csv")
                      if p.exists()), None)
    if truth_csv:
        summary["truth_check"] = truth_check(packages, truth_csv, track.L)

    out = Path(out_dir) / recording.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot:
        plot_replay(out / "t7_replay.png", rows)
    print(f"replayed {recording.name} on {track_name}: {summary['cycles']} cycles over {summary['duration_s']} s")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 1 if truth_csv and summary["truth_check"]["zones_missing_their_cone"] else 0


if __name__ == "__main__":
    sys.exit(run())
