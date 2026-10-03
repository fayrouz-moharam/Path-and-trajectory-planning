# Real circuit maps

Downloaded from https://github.com/f1tenth/f1tenth_racetracks (GPL-3.0; see that
repository's LICENSE). Real F1 circuits at 1:10 scale: `<Track>_map.yaml` +
`<Track>_map.png` (occupancy map) and `<Track>_centerline.csv`
(x_m, y_m, w_tr_right_m, w_tr_left_m). Unmodified.

Used by `run_t1.py` / `run_t1_t2.py` / `run_t1_t2_t3.py` with `SOURCE = "realtrack"`.
Note: the Budapest and Catalunya csv widths (1.1 m) do not match their maps (~1.3 m).
