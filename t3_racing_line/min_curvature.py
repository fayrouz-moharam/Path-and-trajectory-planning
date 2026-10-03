"""Steps B-D: curvature as a linear function of alpha, the cost, and the QP.

Racing point:  p_i = r_i + alpha_i * n_i   (alpha > 0 = left of the reference)
Goal:          minimise  sum_i kappa_i^2  (+ lam * sum_i (alpha_{i+1} - alpha_i)^2)
subject to:    alpha_min <= alpha <= alpha_max      (stay inside the walls - margin)
               |kappa| <= kappa_max                  (optional, steering limit)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from t2_curvature.queryable_track import QueryableTrack

from .stations import Stations, build_walls, make_stations, track_from_points


# ---------------------------------------------------------------- vehicle limit

def kappa_max_from_steering(max_steer: float, wheelbase: float) -> float:
    """Tightest curvature the car can drive (kinematic bicycle model):
    turning radius R = wheelbase / tan(steer), so kappa_max = tan(max_steer) / wheelbase.
    Spec placeholders: tan(0.4) / 0.33 = 1.29 1/m (R = 0.78 m)."""
    if not (0 < max_steer < np.pi / 2) or wheelbase <= 0:
        raise ValueError(f"need 0 < max_steer < pi/2 and wheelbase > 0, "
                         f"got {max_steer}, {wheelbase}")
    return float(np.tan(max_steer) / wheelbase)


# ---------------------------------------------------------------- step B: kappa(alpha)

def build_curvature_model(st: Stations) -> tuple[np.ndarray, sparse.csc_matrix]:
    """Linear curvature model  kappa ~= kappa_ref + M @ alpha  (first order in alpha).

    kappa_ref_i = n_i . (r_{i-1} - 2 r_i + r_{i+1}) / ds^2     (curvature of the reference)
    M[i, i-1]   = n_i . n_{i-1} / ds^2
    M[i, i]     = -2 / ds^2  +  2 kappa_ref_i^2                 <-- correction, see below
    M[i, i+1]   = n_i . n_{i+1} / ds^2

    Correction (deviation from the T3 spec, section 5.2): the plain three-point
    formula n_i . (p_{i-1} - 2 p_i + p_{i+1}) / ds^2 divides by the REFERENCE
    spacing ds, but a line shifted outward is longer (its spacing is
    (1 - kappa*alpha) ds). Expanding the exact curvature of p = r + alpha n:
        exact      kappa_p = kappa + kappa^2 alpha + alpha''  + O(alpha^2)
        spec form  n . p'' = kappa - kappa^2 alpha + alpha''
    so the spec form is missing +2 kappa^2 alpha. Without it, moving outward on
    a circle appears to INCREASE curvature, and the QP hugs the inside wall.
    """
    r, n, ds = st.xy, st.normals, st.ds
    N = len(st)
    idx = np.arange(N)
    prev, nxt = (idx - 1) % N, (idx + 1) % N   # closed loop: neighbours wrap around

    kappa_ref = np.sum(n * (r[prev] - 2 * r + r[nxt]), axis=1) / ds**2

    m_prev = np.sum(n * n[prev], axis=1) / ds**2
    m_diag = -2.0 / ds**2 + 2.0 * kappa_ref**2
    m_next = np.sum(n * n[nxt], axis=1) / ds**2

    M = sparse.csc_matrix(
        (np.concatenate([m_prev, m_diag, m_next]),
         (np.tile(idx, 3), np.concatenate([prev, idx, nxt]))),
        shape=(N, N),
    )
    return kappa_ref, M


def difference_matrix(N: int) -> sparse.csc_matrix:
    """D1 @ alpha = alpha_{i+1} - alpha_i  (closed loop)."""
    idx = np.arange(N)
    return sparse.csc_matrix(
        (np.concatenate([-np.ones(N), np.ones(N)]),
         (np.tile(idx, 2), np.concatenate([idx, (idx + 1) % N]))),
        shape=(N, N),
    )


# ---------------------------------------------------------------- step C: cost

def curvature_cost(kappa_ref: np.ndarray, M: sparse.spmatrix, alpha: np.ndarray,
                   lam: float = 0.0) -> float:
    """sum kappa_i^2  (+ lam * sum (alpha_{i+1} - alpha_i)^2), with kappa from the linear model."""
    k = kappa_ref + M @ alpha
    c = float(k @ k)
    if lam > 0:
        d = difference_matrix(len(alpha)) @ alpha
        c += lam * float(d @ d)
    return c


# ---------------------------------------------------------------- step D: QP

@dataclass
class QPResult:
    alpha: np.ndarray      # (N,) optimal lateral offsets
    points: np.ndarray     # (N, 2) racing-line points r + alpha * n
    kappa: np.ndarray      # (N,) curvature predicted by the linear model
    cost: float            # curvature_cost at the solution
    status: str            # OSQP status string ("solved")
    solve_time: float      # s
    iterations: int        # OSQP iterations


# max_iter: the spec says 20000, but on a full lap (~3000 stations, T1 Monza) OSQP
# needs ~60000 iterations to reach 1e-6 -- the cost is a 4th-difference operator,
# which is badly conditioned. It stops as soon as it has converged.
DEFAULT_OSQP = dict(eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=100000, verbose=False)


def solve_min_curvature_qp(st: Stations, kappa_max: float | None = None, lam: float = 0.0,
                           osqp_settings: dict | None = None, mu: float = 0.0) -> QPResult:
    """One minimum-curvature QP around the reference line in `st`.

    OSQP standard form:  minimise 1/2 a^T P a + q^T a   s.t.  lo <= A a <= hi
    Expanding ||kappa_ref + M a||^2 = a^T (M^T M) a + 2 (M^T kappa_ref)^T a + const gives
        P = 2 M^T M  (+ 2 lam D1^T D1)  (+ 2 mu I),   q = 2 M^T kappa_ref.
    Everything is multiplied by ds^2 (cost by ds^4): M entries go from ~1/ds^2 (~44)
    to ~1, which is better for the solver. Scaling a cost does not move its minimum.

    mu: step penalty mu * sum alpha^2 -- "don't move far from the reference in one
    round". On long straights the curvature cost doesn't care where the line sits
    (flat bowl): the solver struggles (status "solved inaccurate") and rounds slide
    the line across the straight. mu makes the bowl round. At convergence alpha -> 0,
    so mu does not change the final line, only the way to it.
    """
    import osqp  # imported here so the rest of the module works without it

    kappa_ref, M = build_curvature_model(st)
    N = len(st)
    sc = st.ds**2
    Ms, ks = (M * sc).tocsc(), kappa_ref * sc

    P = 2.0 * (Ms.T @ Ms)
    q = 2.0 * (Ms.T @ ks)
    if lam > 0:
        D = difference_matrix(N)
        P = P + 2.0 * lam * sc**2 * (D.T @ D)
    if mu > 0:
        P = P + 2.0 * mu * sc**2 * sparse.identity(N, format="csc")

    # Rows 0..N-1: alpha itself.  Rows N..2N-1 (optional): kappa = kappa_ref + M alpha.
    A_blocks, lo, hi = [sparse.identity(N, format="csc")], [st.alpha_min], [st.alpha_max]
    if kappa_max is not None:
        A_blocks.append(Ms)
        lo.append((-kappa_max - kappa_ref) * sc)
        hi.append((kappa_max - kappa_ref) * sc)
    A = sparse.vstack(A_blocks, format="csc")

    prob = osqp.OSQP()
    prob.setup(sparse.triu(P, format="csc"), q, A, np.concatenate(lo), np.concatenate(hi),
               **(DEFAULT_OSQP | (osqp_settings or {})))
    t0 = time.perf_counter()
    res = prob.solve(raise_error=False)  # we check the status ourselves below
    solve_time = time.perf_counter() - t0

    status = res.info.status
    if status != "solved":
        raise RuntimeError(f"OSQP did not solve the min-curvature QP: status = {status!r}")

    alpha = np.asarray(res.x)
    return QPResult(
        alpha=alpha,
        points=st.xy + alpha[:, None] * st.normals,
        kappa=kappa_ref + M @ alpha,
        cost=curvature_cost(kappa_ref, M, alpha, lam),   # curvature (+ wiggle) only, without mu
        status=status,
        solve_time=solve_time,
        iterations=int(res.info.iter),
    )


# ---------------------------------------------------------------- step E: iterate

@dataclass
class IterationLog:
    iteration: int
    max_alpha: float       # m, largest correction this round
    peak_kappa: float      # 1/m, largest |kappa| predicted by the linear model
    cost: float
    solve_time: float      # s
    osqp_iterations: int
    mu: float              # step penalty used this round
    bending: float         # integral of kappa^2 ds of the NEW line (from T2's spline), 1/m
    line_peak_kappa: float # largest |kappa| of the NEW line (from T2's spline), 1/m


@dataclass
class IterateResult:
    points: np.ndarray               # (N, 2) racing-line points from the last round
    reference: QueryableTrack        # T2 curve through `points` (widths are placeholders: step F)
    stations: Stations               # stations of the last round
    qp: QPResult                     # QP of the last round
    walls: tuple[np.ndarray, np.ndarray]  # fixed (left_wall, right_wall)
    history: list[IterationLog]
    converged: bool                  # stopped because the line settled (moved / quality)
    stop_reason: str
    margin: float                    # the real margin (stations plan with margin + plan_buffer)
    plan_buffer: float


def line_quality(line: QueryableTrack, step: float = 0.05) -> tuple[float, float]:
    """(integral of kappa^2 ds, peak |kappa|) of a T2 line, sampled every `step` m."""
    k = line.sample_arrays(np.arange(0.0, line.length, step))[4]
    return float(np.sum(k**2) * step), float(np.max(np.abs(k)))


def iterate_min_curvature(track, spacing_target: float, margin: float,
                          kappa_max: float | None = None, lam: float = 0.0,
                          max_iterations: int = 30, convergence_tol: float = 0.01,
                          osqp_settings: dict | None = None,
                          mu0: float = 1.0, mu_decay: float = 0.5,
                          plan_buffer: float = 0.01, quality_tol: float = 5e-3,
                          mu_settle: float = 0.01,
                          ) -> IterateResult:
    """Repeat steps A-D, each time around the previous round's racing line.

    The linear model kappa ~= kappa_ref + M alpha is only accurate for small alpha.
    Round 1 moves the line a lot (up to ~1 m); later rounds only correct what is
    left. The walls are built once from `track` and never change.

    mu0, mu_decay:  step penalty mu = mu0 * mu_decay**(round-1): cautious early
                    rounds (stable on long straights), free late rounds (fast finish).
    plan_buffer:    stations plan with margin + plan_buffer, so the smooth curve
                    between stations (which can cut ~1 cm closer at a hairpin tip)
                    still stays outside the real margin.
    Stops when the line moves < convergence_tol, or its quality (integral of
    kappa^2 ds and peak |kappa|) changes < quality_tol (relative) two rounds in a
    row -- on straights the line can keep sliding sideways without getting any
    better -- or after max_iterations (then a warning). Both stop rules are only
    checked once mu <= mu_settle: while mu is large a small move or a small
    improvement means "held back by mu", not "arrived" (on a gentle circle round 1
    moves only ~8 mm with mu = 1).
    """
    walls = build_walls(track)
    reference = track
    history: list[IterationLog] = []
    converged, stop_reason, calm = False, f"{max_iterations} rounds", 0

    for k in range(1, max_iterations + 1):
        # A: stations on the current reference; the room on each side is measured by
        #    walking to the fixed walls in EVERY round (round 1 too: at folds/tips the
        #    track's own widths are larger than the real room).
        st = make_stations(reference, spacing_target, margin + plan_buffer, walls=walls,
                           start_tol=plan_buffer + 2e-3)
        # B-D: linear model + QP around this reference.
        mu = mu0 * mu_decay ** (k - 1)
        qp = solve_min_curvature_qp(st, kappa_max, lam, osqp_settings, mu=mu)

        # New reference = T2 curve through the new points. The widths are
        # placeholders (T2 needs them > 0): the next round re-measures them
        # from the walls, and step F re-measures them after the last round.
        ones = np.ones(len(st))
        reference = track_from_points(qp.points[:, 0], qp.points[:, 1], ones, ones)

        max_alpha = float(np.max(np.abs(qp.alpha)))
        bending, peak = line_quality(reference)
        history.append(IterationLog(k, max_alpha, float(np.max(np.abs(qp.kappa))), qp.cost,
                                    qp.solve_time, qp.iterations, mu, bending, peak))

        if mu > mu_settle:                  # still held back by mu: don't judge yet
            continue
        if max_alpha < convergence_tol:
            converged, stop_reason = True, f"moved < {convergence_tol} m"
            break
        if len(history) > 1:
            before = history[-2]
            stable = (abs(bending - before.bending) <= quality_tol * before.bending
                      and abs(peak - before.line_peak_kappa) <= quality_tol * before.line_peak_kappa)
            calm = calm + 1 if stable else 0
            if calm >= 2:
                converged, stop_reason = True, "quality stable"
                break

    if not converged:
        warnings.warn(f"min-curvature iteration did not settle in {max_iterations} rounds: "
                      f"last max|alpha| = {history[-1].max_alpha:.3f} m "
                      f"(tol {convergence_tol} m)")

    return IterateResult(qp.points, reference, st, qp, walls, history, converged,
                         stop_reason, margin, plan_buffer)
