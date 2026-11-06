# T7 Half B results (6 Nov 2026: spline removed, METHOD = "fastest")

Run on the laptop, from the repo folder:

- `python -m t7_obstacle_avoidance.compare` -> `closed1.csv`, `closed1_summary.txt`, `levine.csv`, `levine_summary.txt`
- `python -m pytest tests -v -rs` -> `pytest_all_output.txt` (175 passed, 3 skipped; run in the cloud copy of this
  exact code because the laptop VM has no pytest)

## Where the test inputs come from

| Test | Real (from perception / T5) | Made up |
|---|---|---|
| compare.py | closed1 and Levine tracks (grad-perception/apex_track), T5 racing line and speeds (out/run_perception_*/t5), Half A's keep_outs() | the cones: one at a time, every 2 m, on the racing line and 0.3 m either side |
| tests/obstacle_avoidance (18) | Half A's zone type and T6's check_cycle | a straight 50 m track, walls 0.8 m away, hand-placed zones |
| tests/avoidance (Half A) | perception's message format | small hand-written messages; the 3 tests on perception's recorded drive (run_B) are skipped: recording not here |

## Results

| Track | Method | Worked | Mean time past the cone* | Fastest in | ms per cycle |
|---|---|---|---|---|---|
| closed1 (72) | graph alone | 67 | 1.732 s | 34 | 2.1 |
| closed1 | sampling alone | 71 | 1.722 s | 33 | 4.8 |
| closed1 | **fastest (default)** | **71** | **1.666 s** | - | **5.1** |
| Levine (96) | graph alone | 87 | 1.711 s | 23 | 1.9 |
| Levine | sampling alone | 93 | 1.634 s | 61 | 4.2 |
| Levine | **fastest (default)** | **96** | **1.606 s** | - | **4.4** |

*over the cases where graph and sampling both worked (67 on closed1, 84 on Levine).
"fastest" builds both every cycle and sends the quicker one that passes check_cycle (column `used` in the csv).
