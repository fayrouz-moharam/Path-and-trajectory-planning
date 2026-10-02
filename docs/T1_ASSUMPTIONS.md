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
4. ~~Confirm with perception: are boundary/obstacle points in the world/map
   frame or relative to the vehicle pose?~~ **Answered 2026-10-02: relative
   to the vehicle** (`points_frame = "vehicle"`, now the default). Still open:
   relative to which point — base_link or the LiDAR? (see item 5, offset)
5. If perception publishes ROS 2 topics: plan is `ros2 bag record` during the
   test lap, then a small bag → JSONL converter (T1 stays ROS-free). Agree on
   topic names/message types, timestamps (scan ↔ pose matching) and the
   base_link → laser offset. Lesson from simulator testing: the converter
   must drop beams at/near `range_max` (and inf/NaN) — noisy no-hit returns
   otherwise become phantom wall points.

See `planning_task_breakdown.md` §3 (Interface with perception) for the
original proposed data contract this refines.

## 6. Implementation decisions (first implementation, `t1_track/`)

- **Point frame (open item 4):** perception confirmed (2026-10-02) that
  boundary points are relative to the vehicle, so the config flag
  `points_frame` now defaults to `vehicle` (also the default for the `synth`
  and `realtrack` test-log generators); `world` remains available. A wrong
  setting fails validation rather than producing a plausible-looking track.
- **Stage 2 uses the pose trace, not rasterize + skeletonize.** The car's
  own test-lap trajectory is already a closed, ordered loop in the driving
  direction. It is used as the reference line; the sign of each boundary
  point's lateral offset from it performs the left/right split (resolving
  §2's unordered-scatter simplification without perception labels), and
  per-station wall distances re-centre it into the centerline.
  Consequence: the test lap must be one complete, closed lap (checked).
  Known limitation: where two track sections come closer than
  `max_half_width` (hairpins, parallel straights), points may be assigned
  to the wrong section — revisit if the real track has such features.
- **Obstacles are ignored by T1.** They're parsed and carried in the log,
  but don't affect widths and aren't exported.
- **Velocity is unused by T1**, so open item 3 doesn't block it.
- **Relation to the breakdown's T1 pipeline:** the pose-guided approach is
  essentially the breakdown's step 10 fallback ("trajectory-centering …
  iterate to midpoint") promoted to the primary method, because our input is
  the per-frame point log rather than an occupancy grid.
- **Validation follows the breakdown's T1 step 11:** closed, non-self-
  intersecting (centerline and both walls), inside free space (centerline
  clearance to observed wall points), widths above minimum, length matches
  test-lap distance, uniform spacing — plus wall-coverage gap fractions.
- **Kept out of T1 on purpose (other tasks' scope):** no curvature, heading
  or spline output (T2); only light de-staircasing (10 cm gaussian on the
  centerline) and a median de-spike on widths — real smoothing is T2's; the
  inside-width < 1/|κ| check belongs to T3/T6.
