# Task 1 — Track Representation: Working Assumptions

Status: draft, decided in planning discussion before implementation started.
Owner: Person A (per planning_task_breakdown.md). Revisit once perception's
actual interface is confirmed.

## 1. Assumed perception interface (per-frame, during test lap)

Perception provides, per timestamp, matching the project description's listed
Perception outputs:

- **Vehicle pose:** `(x, y, yaw)`
- **Velocity estimate:** `(vx, vy)` — exact form (vx/vy vs. v/yaw_rate) not yet
  confirmed, doesn't block T1
- **Track boundaries:** a set of points detected as "wall" this frame
- **Obstacle locations:** positions of anything detected as an obstacle this
  frame

## 2. Simplifications assumed for now (deliberately deferred, not forgotten)

- **Boundary points are an unordered scatter** — not yet labeled left vs.
  right relative to the car. Splitting them by side is punted to a later
  step in the extraction stage.
- **Pose is treated as idealized / drift-free** — no loop-closure or map-frame
  correction applied yet. Real SLAM pose may drift over a full test lap; if
  so, accumulated boundary points will be slightly warped by lap-closure time.
  This is a known risk to revisit once perception's actual pose source
  (raw odometry vs. SLAM-corrected) is confirmed.

## 3. Per-frame log format (raw accumulation file)

Line-delimited JSON, one object per timestamp:

```json
{"t": 12.34,
 "pose": {"x": 1.02, "y": 0.55, "yaw": 0.31},
 "velocity": {"vx": 0.9, "vy": 0.02},
 "boundary_points": [[1.10, 0.80], [1.15, 0.30]],
 "obstacles": [{"id": 3, "x": 2.1, "y": 0.6}]}
```

This is the raw file produced by logging the test lap — not yet a track
representation. It feeds Stage 1 below.

## 4. T1 pipeline — revised two-stage split

The original breakdown (`planning_task_breakdown.md`, Task 1 / T1) assumed a
finished occupancy-grid map as input. Given the per-frame interface above,
T1 actually splits into:

**Stage 1 — Accumulation (new — not in the original T1 plan)**
Merge the per-frame log across the whole test lap into one point cloud of
boundary points (and obstacle points, if any appear during the test lap),
in a single consistent frame.

**Stage 2 — Extraction (the original T1 pipeline, adapted)**
Turn the accumulated point cloud into a closed, ordered centerline with
left/right widths. The original pipeline (binarize → morphology →
skeletonize → order → resample → widths → validate) assumed a clean
occupancy grid; with a raw point cloud as input, the binarize/skeletonize
steps need point-based equivalents (e.g. clustering or rasterizing the
scatter into a grid first). Downstream steps — ordering, resampling,
width computation, validation — carry over as originally planned.

**Output (unchanged from original spec):** `track.csv` (s, x, y, w_left,
w_right) + `track_meta.json`, per the original T1 deliverable.

## 5. Open items before Stage 2 can be finalized

1. Confirm with perception: are boundary points ever left/right-labeled, or
   always an unordered scatter?
2. Confirm with perception: is pose raw odometry or SLAM/loop-closed?
3. Confirm velocity estimate's exact form (vx/vy vs. v/yaw_rate).

See `planning_task_breakdown.md` §3 (Interface with perception) for the
original proposed data contract this refines.
