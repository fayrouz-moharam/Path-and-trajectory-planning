"""Generate a synthetic track.csv matching T1's schema, for testing T2 only.

Not T1's real output -- Fayrouz/Mohamed should re-run these checks against
the actual track.csv once it's shared. Two fixtures:
  - circle: exact analytic curvature (1/R everywhere), the strongest possible
    check on the spline/curvature math.
  - oval: rounded-rectangle shape closer to the real synthetic track.
"""

import numpy as np
import pandas as pd

DS = 0.05  # m, matches T1's "every 5.0 cm" spacing


def circle_csv(path, radius=5.0, w_left=0.9, w_right=0.9):
    length = 2 * np.pi * radius
    n = int(round(length / DS))
    s = np.arange(n) * DS
    theta = s / radius
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    df = pd.DataFrame(
        {"s": s, "x": x, "y": y, "w_left": w_left, "w_right": w_right}
    )
    df.to_csv(path, index=False)
    return length


def rounded_rect_boundary(t, x0, y0, x1, y1, r, n_samples=20000):
    """Densely sample a rounded-rectangle perimeter, evenly by arc length."""
    # Build the shape as arcs + straights, sample densely and finely,
    # then resample evenly by cumulative chord distance (good enough
    # approximation to arc length at this density).
    cx0, cy0 = x0 + r, y0 + r
    cx1, cy1 = x1 - r, y0 + r
    cx2, cy2 = x1 - r, y1 - r
    cx3, cy3 = x0 + r, y1 - r
    pts = []

    def arc(cx, cy, a0, a1):
        a = np.linspace(a0, a1, 200)
        pts.extend(zip(cx + r * np.cos(a), cy + r * np.sin(a)))

    def line(p0, p1):
        t = np.linspace(0, 1, 200)
        pts.extend(zip(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))

    # Start at bottom straight, go counterclockwise (positive curvature turns)
    line((cx0, y0), (cx1, y0))
    arc(cx1, cy1, -np.pi / 2, 0)
    line((x1, cy1), (x1, cy2))
    arc(cx2, cy2, 0, np.pi / 2)
    line((cx2, y1), (cx3, y1))
    arc(cx3, cy3, np.pi / 2, np.pi)
    line((x0, cy3), (x0, cy0))
    arc(cx0, cy0, np.pi, 1.5 * np.pi)

    pts = np.array(pts)
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
    total = d[-1]
    n = int(round(total / DS))
    s_even = np.arange(n) * DS
    x_even = np.interp(s_even, d, pts[:, 0])
    y_even = np.interp(s_even, d, pts[:, 1])
    return s_even, x_even, y_even, total


def oval_csv(path, w_left=0.9, w_right=0.9):
    s, x, y, length = rounded_rect_boundary(None, 0.0, 0.0, 14.0, 9.0, 3.0)
    df = pd.DataFrame({"s": s, "x": x, "y": y, "w_left": w_left, "w_right": w_right})
    df.to_csv(path, index=False)
    return length


if __name__ == "__main__":
    for path, fn in [("circle_track.csv", circle_csv), ("oval_track.csv", oval_csv)]:
        length = fn(path)
        print(f"wrote {path}, nominal length {length:.3f} m")
