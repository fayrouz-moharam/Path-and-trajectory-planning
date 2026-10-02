# Path-and-trajectory-planning

## Task 1 — Track representation (`t1_track/`)

Turns the per-frame perception log of a test lap (format in
[docs/T1_ASSUMPTIONS.md](docs/T1_ASSUMPTIONS.md) §3) into a closed, ordered
centerline with left/right widths:

- `track.csv` — `s,x,y,w_left,w_right` (metres; uniform spacing; closed loop,
  last row connects to first; `s` increases in the driving direction)
- `track_meta.json` — map version, frame, resolution, start pose, direction,
  length, spacing, source-log hash, config used, pipeline stats and
  validation results (`"valid": true/false`)

### Usage

```bash
pip install numpy scipy matplotlib pytest osqp
```

Everything runs from one script. Edit the `SETTINGS` block at the top of
[run_t1.py](run_t1.py), then:

```bash
python run_t1.py
```

| `SOURCE` | Test-lap log comes from |
|---|---|
| `"synth"` | a fake track made by `synth.py` (known ground truth) |
| `"realtrack"` | a simulated LiDAR driven around a real map (`realtrack.py`), e.g. Spielberg: set `MAP_YAML` / `CENTERLINE_CSV` |
| `"log"` | an existing log file (`LOG_PATH`): the real car's, or `out/gym_*/log.jsonl` |

Output goes to `OUT_DIR` (default `out/run/`): `log.jsonl` + ground truth,
`track/track.csv`, `track/track_meta.json` and the plots. Exit code
0 = valid, 1 = built but a check failed, 2 = no track (e.g. incomplete lap).

Boundary points are **relative to the car** (`points_frame = "vehicle"`,
confirmed by perception); a log in map coordinates needs
`CONFIG = T1Config(points_frame="world")`. Picking the wrong one fails validation.

Tests: `python -m pytest`

### Testing on real circuits (no real perception log exists yet)

Real F1 circuits at 1:10 scale with published centerlines:
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) (GPL-3.0,
download into `out/real_tracks/`, not committed). Set `SOURCE = "realtrack"`
in `run_t1.py`; the published centerline becomes the ground truth.

T1 was also cross-checked once on logs recorded in the independent official
simulator [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) v1.0.0 (the
recorder tool has since been removed; the logs `out/gym_spielberg/log.jsonl`,
`out/gym_monza/log.jsonl` can still be built with `SOURCE = "log"`).

Results (official simulator): centerline RMS 0.5 cm (Spielberg) / 0.9 cm
(Monza). Width bias +8 / −7 cm comes from the map drawing and the
simulator's sensor model, measured independently of T1.

### Pipeline

| Module | Stage |
|---|---|
| `log_io.py` | read JSONL log, skip malformed lines, sort by `t` |
| `accumulate.py` | **Stage 1**: vehicle→world transform (if needed), voxel downsample, isolated-point removal |
| `extract.py` | **Stage 2**: cut one lap from the pose trace → smoothed reference line; split points left/right by signed lateral offset; per-station median wall distances (de-spiked); re-centre (2 passes, light de-staircasing) |
| `validate.py` | closure, min width, inside free space, wall-coverage gaps, length vs test lap, uniform spacing, self-intersection (centerline + both walls) |
| `export.py` | `track.csv` + `track_meta.json` |
| `pipeline.py` | `build_track()`: runs all the stages above in order |
| `synth.py` | synthetic track + noisy perception log with ground truth |
| `realtrack.py` | test-lap log from a real map (simulated LiDAR) |
| `plot.py` | debug figures (`PLOT = True` in `run_t1.py`) |

Out of T1 scope (see `planning_task_breakdown.md`): spline fitting, heading,
curvature (T2); inside-width < 1/|κ| (T3/T6).

Tunables live in `t1_track/config.py` (`T1Config`).
