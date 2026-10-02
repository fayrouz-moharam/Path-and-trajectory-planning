"""Tunable parameters for the T1 pipeline. Defaults assume a small-scale
(F1TENTH-like) car and track; everything is in metres / radians."""
from __future__ import annotations

from dataclasses import dataclass

POINTS_FRAMES = ("world", "vehicle")


@dataclass
class T1Config:
    # --- Stage 1: accumulation ---
    # Frame the logged boundary points are expressed in: "vehicle" = relative
    # to that frame's pose, transformed here (confirmed by perception
    # 2026-10-02, T1_ASSUMPTIONS.md open item 4); "world" = already global.
    points_frame: str = "vehicle"
    max_point_range: float = 10.0     # ignore points farther than this from the car: a heading
                                      # error of 1 deg moves a 30 m point by 52 cm, a 3 m one by 5 cm
    voxel_size: float = 0.05          # downsampling cell edge
    min_hits_per_voxel: int = 1       # drop cells observed fewer times than this
    outlier_radius: float = 0.25      # isolated-point filter search radius
    outlier_min_neighbors: int = 3    # neighbours needed within radius to survive

    # --- Lap detection on the pose trace ---
    closure_radius: float = 0.5       # how close the car must come back to its start
    min_lap_length: float = 5.0       # distance to travel before a return counts as a lap

    # --- Stage 2: extraction ---
    ds: float = 0.05                  # output centerline spacing (breakdown T1 step 8: ~5 cm)
    ref_smoothing: float = 0.5        # gaussian sigma (m) for the pose trace; guide line only, not output
    max_half_width: float = 3.0       # ignore points farther than this from the reference
    station_window: float = 0.3       # points within +/- this (m) of a station feed its width
    width_percentile: float = 50.0    # statistic of lateral offsets per station (50 = median)
    width_median_window: float = 0.25 # median filter (m) on width profiles: spike removal only;
                                      # real smoothing is T2's job
    destaircase: float = 0.10         # gaussian sigma (m) on the centerline: light de-staircasing
                                      # only (breakdown T1 step 8); removes mm-level jitter
    iterations: int = 2               # re-centre passes (reference -> centerline)

    # --- Output metadata ---
    frame_id: str = "map"             # frame the track is expressed in (breakdown §3: SLAM map frame)

    # --- Validation ---
    min_width: float = 0.2            # smallest acceptable w_left / w_right / wall clearance
    max_gap_fraction: float = 0.2     # max fraction of stations with no wall seen, per side
    max_length_mismatch: float = 0.10 # |centerline length - test-lap length| / test-lap length
    max_spacing_deviation: float = 0.05  # max |spacing - ds| / ds

    def __post_init__(self) -> None:
        if self.points_frame not in POINTS_FRAMES:
            raise ValueError(f"points_frame must be one of {POINTS_FRAMES}, got {self.points_frame!r}")
        if self.ds <= 0 or self.voxel_size <= 0:
            raise ValueError("ds and voxel_size must be positive")
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
