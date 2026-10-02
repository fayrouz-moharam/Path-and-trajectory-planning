# T2 — queryable track (Person B)

Turns T1's `track.csv` (`s, x, y, w_left, w_right`) into a smooth, continuous,
closed-lap track you can query at any `s`, with position, heading, curvature
and widths. T3 (racing line), T4 (velocity profile), T5 (local trajectory)
and T7 (obstacle avoidance) should all query this — nothing downstream
should read `track.csv` directly.

## Usage

```python
from queryable_track import QueryableTrack

track = QueryableTrack("track.csv")   # Ehab's real output
track.length                           # m, full lap
sample = track.sample(12.3)            # arbitrary s, wraps automatically
sample.x, sample.y, sample.heading, sample.curvature
sample.w_left, sample.w_right
track.left_width(12.3)                 # shortcut, same as sample(s).w_left
```

`sample(s)` accepts any finite `s`, including negative or > length — it wraps.

## Files

- `queryable_track.py` — the module. No dependency on T1's code or on
  `track.csv`'s exact origin, only its schema.
- `make_synthetic_track_csv.py` — generates test fixtures only (a circle with
  exact analytic curvature, and a rounded-rectangle oval). **Not real data** —
  swap in Ehab's actual `track.csv` before trusting this for T3+.
- `test_queryable_track.py` — 11 checks, all passing against the synthetic
  fixtures: `python test_queryable_track.py`. Re-run against the real
  `track.csv` once you have it (drop the circle/oval-specific analytic
  checks, keep the periodicity/finite-curvature/width ones).

## Design notes

- Periodic cubic spline in `x(s), y(s)` (same approach as `apex.track` in the
  MPC repo — see `planning-racing-strategy-technical-notes.md` — kept
  dependency-free here since T2's input is this repo's own `track.csv`, not
  apex's waypoint format).
- Widths are **linearly interpolated**, not splined — the measurement isn't
  known to be smooth, so the code doesn't invent smoothness it doesn't have.
- Lap closing uses the **actual chord distance** from the last point back to
  the first as the closing segment's length, not an assumed spacing. Using
  the median interior spacing instead measured 10–20x worse curvature/heading
  error right at the seam in testing — worth knowing if this gets
  reimplemented elsewhere.

## Known limitation

A periodic cubic spline is smooth (C2) everywhere, so at a point where the
*true* curvature jumps discontinuously — e.g. an idealized straight
tangent-joining-directly-into-a-circular-arc corner, like the synthetic oval
fixture here — the fit rings slightly near the seam (visible in
`t2_oval_check.png`). Real cone-derived centerlines from T1 are very unlikely
to have a perfectly sharp analytic transition like that, so this is mostly a
property of the idealized test fixture, not an expected problem with real
track data — but if a real track ever shows odd curvature spikes right at a
corner entry/exit, this is the first thing to check.
