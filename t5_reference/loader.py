"""T5 loader: what the MPC imports to read planning_reference.csv.

    ref = PlanningReference.from_csv("out/run_perception_closed1/t5/planning_reference.csv",
                                     track_hash=track.track_hash, boundary_hash=track.boundary_hash)
    r = ref.at(state.s_abs + horizon_s)     # dict: e_y_ref_m, e_psi_ref_rad, kappa_ref_1pm, v_ref_mps (+ s_track_m)
    speed_reference = ref                   # ref(s_abs) -> v_ref_mps, the same contract as apex's ConstantSpeed
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .export import COLUMNS, FORMAT_VERSION
from .mapping import wrap


class PlanningReference:
    """planning_reference.csv in memory; look it up by the MPC's own s (any s_abs, any number of laps)."""

    def __init__(self, table: np.ndarray, meta: dict):
        self.meta = meta
        self.lap_length = float(meta["lap_length"])
        closed = np.vstack([table, table[:1]])                  # the first row again, at s = lap_length
        closed[-1, 0] = self.lap_length
        closed[:, 2] = np.unwrap(closed[:, 2])                   # e_psi: no 2 pi jump inside an interpolation
        self._s, self._cols = closed[:, 0], closed[:, 1:]

    @classmethod
    def from_csv(cls, csv_path: str | Path, meta_path: str | Path | None = None, *,
                 track_hash: str | None = None, boundary_hash: str | None = None,
                 allow_invalid: bool = False) -> "PlanningReference":
        """track_hash / boundary_hash: the MPC's own track (from apex_track); a mismatch raises ValueError.
        A file whose self-checks failed (meta "valid": false) is refused unless allow_invalid."""
        csv_path = Path(csv_path)
        meta_path = Path(meta_path) if meta_path else csv_path.with_name("planning_reference_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format_version") != FORMAT_VERSION or meta.get("columns") != COLUMNS:
            raise ValueError(f"{meta_path}: not a planning_reference v{FORMAT_VERSION} file")
        if meta.get("csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
            raise ValueError(f"{csv_path} is not the file {meta_path.name} was written for (sha256 differs)")
        for name, want in (("track_hash", track_hash), ("boundary_hash", boundary_hash)):
            if want is not None and meta.get(name) != want:
                raise ValueError(f"reference made for {name} {meta.get(name)}, the MPC's track has {want}")
        if not meta.get("valid") and not allow_invalid:
            raise ValueError(f"{csv_path} failed its self-checks: {meta.get('problems')}")
        if csv_path.read_text(encoding="utf-8").splitlines()[0].split(",") != COLUMNS:
            raise ValueError(f"{csv_path}: columns are not {COLUMNS}")
        table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        s = table[:, 0]
        if len(table) != meta["n_points"] or s[0] != 0.0 or np.any(np.diff(s) <= 0) or s[-1] >= meta["lap_length"]:
            raise ValueError(f"{csv_path}: s_track_m must start at 0, increase, and stay below lap_length")
        return cls(table, meta)

    def at(self, s_abs) -> dict:
        """The reference at centre-line s. One number in -> floats out; an array (a horizon) -> arrays out."""
        s = np.mod(np.asarray(s_abs, dtype=float), self.lap_length)
        vals = [np.interp(s, self._s, self._cols[:, j]) for j in range(4)]
        vals[1] = wrap(vals[1])                                  # e_psi back into (-pi, pi]
        out = dict(zip(COLUMNS[1:], vals))
        out["s_track_m"] = s
        if np.ndim(s_abs) == 0:
            out = {k: float(v) for k, v in out.items()}
        return out

    def __call__(self, s_abs):
        """Speed provider: ref(s_abs) -> v_ref_mps."""
        return self.at(s_abs)["v_ref_mps"]
