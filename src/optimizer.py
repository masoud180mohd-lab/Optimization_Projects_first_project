"""Numba-accelerated smooth-SAT + SciPy L-BFGS-B local optimizer for Squares in Squares."""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from collections.abc import Sequence

import numpy as np
from numba import njit
from scipy.optimize import minimize

from src.constructive import DEFAULT_SAFETY_GAP, Packing, normalize_packing
from src.validator import validate_puzzle


@njit(cache=True)
def _energy_and_grad(
    z: np.ndarray,
    L: float,
    pair_gap: float,
    smooth_eps: float,
    beta: float = 80.0,
    conservative: bool = True,
) -> tuple[float, np.ndarray]:
    """Compute smooth containment + soft-max SAT overlap penalty energy and exact gradient."""
    n = z.shape[0] // 3
    grad = np.zeros(3 * n, dtype=np.float64)
    energy = 0.0
    eps2 = smooth_eps * smooth_eps
    w_wall = 10.0
    ln4_over_beta = math.log(4.0) / beta if conservative else 0.0

    cos_t = np.empty(n, dtype=np.float64)
    sin_t = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = z[3 * i + 2]
        c = math.cos(t)
        s = math.sin(t)
        cos_t[i] = c
        sin_t[i] = s

        ac = math.sqrt(c * c + eps2)
        as_ = math.sqrt(s * s + eps2)
        r = 0.5 * (ac + as_)
        dr_dt = 0.5 * (-(c * s) / ac + (s * c) / as_)

        xi = z[3 * i]
        yi = z[3 * i + 1]

        v = r - xi
        if v > 0.0:
            energy += w_wall * v * v
            w = 2.0 * w_wall * v
            grad[3 * i] -= w
            grad[3 * i + 2] += w * dr_dt

        v = xi + r - L
        if v > 0.0:
            energy += w_wall * v * v
            w = 2.0 * w_wall * v
            grad[3 * i] += w
            grad[3 * i + 2] += w * dr_dt

        v = r - yi
        if v > 0.0:
            energy += w_wall * v * v
            w = 2.0 * w_wall * v
            grad[3 * i + 1] -= w
            grad[3 * i + 2] += w * dr_dt

        v = yi + r - L
        if v > 0.0:
            energy += w_wall * v * v
            w = 2.0 * w_wall * v
            grad[3 * i + 1] += w
            grad[3 * i + 2] += w * dr_dt

    # Pairwise smooth-SAT overlap penalty
    for i in range(n):
        xi = z[3 * i]
        yi = z[3 * i + 1]
        ti = z[3 * i + 2]
        ci = cos_t[i]
        si = sin_t[i]

        for j in range(i + 1, n):
            dx = z[3 * j] - xi
            dy = z[3 * j + 1] - yi
            if dx * dx + dy * dy > 2.1:
                continue

            cj = cos_t[j]
            sj = sin_t[j]
            dt = z[3 * j + 2] - ti
            cd = math.cos(dt)
            sd = math.sin(dt)
            acd = math.sqrt(cd * cd + eps2)
            asd = math.sqrt(sd * sd + eps2)
            H = 0.5 + 0.5 * (acd + asd) + pair_gap

            p0 = dx * ci + dy * si
            p1 = -dx * si + dy * ci
            p2 = dx * cj + dy * sj
            p3 = -dx * sj + dy * cj

            q0 = math.sqrt(p0 * p0 + eps2)
            q1 = math.sqrt(p1 * p1 + eps2)
            q2 = math.sqrt(p2 * p2 + eps2)
            q3 = math.sqrt(p3 * p3 + eps2)

            m = q0
            if q1 > m:
                m = q1
            if q2 > m:
                m = q2
            if q3 > m:
                m = q3

            e0 = math.exp(beta * (q0 - m))
            e1 = math.exp(beta * (q1 - m))
            e2 = math.exp(beta * (q2 - m))
            e3 = math.exp(beta * (q3 - m))
            sum_e = e0 + e1 + e2 + e3

            smax = m + math.log(sum_e) / beta - ln4_over_beta
            pen = H - smax
            if pen <= 0.0:
                continue

            energy += pen * pen
            w = 2.0 * pen
            inv_sum = 1.0 / sum_e

            dH_ddt = 0.5 * (-(cd * sd) / acd + (sd * cd) / asd)
            grad[3 * i + 2] -= w * dH_ddt
            grad[3 * j + 2] += w * dH_ddt

            c0 = -w * (e0 * inv_sum) * (p0 / q0)
            c1 = -w * (e1 * inv_sum) * (p1 / q1)
            c2 = -w * (e2 * inv_sum) * (p2 / q2)
            c3 = -w * (e3 * inv_sum) * (p3 / q3)

            ddx = c0 * ci - c1 * si + c2 * cj - c3 * sj
            ddy = c0 * si + c1 * ci + c2 * sj + c3 * cj

            grad[3 * j] += ddx
            grad[3 * j + 1] += ddy
            grad[3 * i] -= ddx
            grad[3 * i + 1] -= ddy

            grad[3 * i + 2] += c0 * p1 - c1 * p0
            grad[3 * j + 2] += c2 * p3 - c3 * p2

    return energy, grad


def center_scale_certify(
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
    safety_gap: float = DEFAULT_SAFETY_GAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Scale centers slightly outward from centroid to guarantee ``safety_gap`` separation."""
    n = len(xs)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    alpha = 1.0
    for i in range(n):
        xi, yi = xs[i], ys[i]
        ci, si = cos_t[i], sin_t[i]
        ti = angles[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            dy = ys[j] - yi
            if dx * dx + dy * dy > 2.1:
                continue

            cj, sj = cos_t[j], sin_t[j]
            dt = angles[j] - ti
            h_exact = 0.5 + 0.5 * (abs(math.cos(dt)) + abs(math.sin(dt)))
            m_ij = max(
                abs(dx * ci + dy * si),
                abs(dx * (-si) + dy * ci),
                abs(dx * cj + dy * sj),
                abs(dx * (-sj) + dy * cj),
            )
            if h_exact - m_ij > 0.05 or m_ij < 0.45:
                return None
            req = (h_exact + safety_gap) / m_ij
            if req > alpha:
                alpha = req

    if alpha > 1.0:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        xs_out = cx + alpha * (xs - cx)
        ys_out = cy + alpha * (ys - cy)
    else:
        xs_out = xs
        ys_out = ys

    return normalize_packing(xs_out, ys_out, angles)


def _run_lbfgs_once(
    z0: np.ndarray,
    L_target: float,
    pair_gap: float,
    smooth_eps: float = 1e-8,
    beta: float = 80.0,
    conservative: bool = True,
    maxiter: int = 200,
) -> tuple[np.ndarray, float]:
    def obj(z: np.ndarray) -> tuple[float, np.ndarray]:
        return _energy_and_grad(z, L_target, pair_gap, smooth_eps, beta, conservative)

    res = minimize(
        obj,
        z0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9},
    )
    return res.x, float(res.fun)


def _untangle_and_polish(
    z_init: np.ndarray,
    L_target: float,
    safety_gap: float,
) -> tuple[np.ndarray, float]:
    """Run 3-stage conservative high-beta smooth-SAT polishing."""
    z1, e1 = _run_lbfgs_once(
        z_init,
        L_target,
        pair_gap=1e-6,
        smooth_eps=1e-6,
        beta=80.0,
        conservative=True,
        maxiter=220,
    )
    if e1 >= 0.05:
        return z1, e1
    z2, e2 = _run_lbfgs_once(
        z1,
        L_target,
        pair_gap=2.0 * safety_gap,
        smooth_eps=1e-9,
        beta=500.0,
        conservative=True,
        maxiter=220,
    )
    if e2 >= 0.01:
        return z2, e2
    return _run_lbfgs_once(
        z2,
        L_target,
        pair_gap=2.0 * safety_gap,
        smooth_eps=1e-10,
        beta=3000.0,
        conservative=True,
        maxiter=200,
    )


def optimize_puzzle_lbfgs(
    n: int,
    current_best: Packing,
    time_budget_s: float = 1.5,
    seed: int = 42,
    safety_gap: float = DEFAULT_SAFETY_GAP,
    prev_packing: Packing | None = None,
) -> Packing:
    """Optimize a single puzzle using rounds of restarts and subset basin-hopping."""
    if n <= 1:
        return current_best

    # Warm up Numba JIT once before timing
    dummy_z = np.zeros(3, dtype=np.float64)
    _energy_and_grad(dummy_z, 2.0, safety_gap, 1e-8, 80.0, True)

    rng = np.random.default_rng(seed + n * 1009)
    best = current_best
    t_start = time.perf_counter()

    def _try_accept(z_sol: np.ndarray, tag: str) -> bool:
        nonlocal best
        xs_s = z_sol[0::3]
        ys_s = z_sol[1::3]
        ang_s = z_sol[2::3]
        certified = center_scale_certify(xs_s, ys_s, ang_s, safety_gap=safety_gap)
        if certified is None:
            return False
        xs_c, ys_c, ang_c, side_c = certified
        if side_c >= best.side_length - 1e-9:
            return False
        vres = validate_puzzle(n, xs_c, ys_c, ang_c)
        if vres.valid and vres.side_length is not None and vres.side_length < best.side_length - 1e-9:
            best = Packing(
                n=n,
                xs=xs_c,
                ys=ys_c,
                angles=ang_c,
                side_length=vres.side_length,
                source=tag,
            )
            return True
        return False

    has_prev = (
        prev_packing is not None
        and prev_packing.n == n - 1
        and prev_packing.side_length < current_best.side_length - 1e-4
    )

    margins = (0.010, 0.018, 0.028)
    k_sub = max(2, min(5, n // 4))
    round_idx = 0

    while time.perf_counter() - t_start < time_budget_s:
        margin = margins[round_idx % len(margins)]
        L_target = max(math.sqrt(n) + 0.01, best.side_length * (1.0 - margin))
        round_best_z: np.ndarray | None = None
        round_best_e = float("inf")
        improved_this_round = False

        for trial in range(10):
            if time.perf_counter() - t_start >= time_budget_s:
                break

            z_init = np.empty(3 * n, dtype=np.float64)
            if trial == 0:
                scale_xy = L_target / best.side_length
                z_init[0::3] = best.xs * scale_xy
                z_init[1::3] = best.ys * scale_xy
                z_init[2::3] = best.angles
                tag = f"lbfgs_scale(r={round_idx})"
            elif trial == 1 and has_prev and prev_packing is not None and prev_packing.side_length < L_target:
                void_rng = np.random.default_rng(seed + n * 7919 + round_idx)
                scale_p = L_target / max(prev_packing.side_length, L_target)
                xs_p = prev_packing.xs * scale_p
                ys_p = prev_packing.ys * scale_p
                cands = void_rng.uniform(0.55, max(0.57, L_target - 0.55), size=(64, 2))
                d2 = np.min(
                    (cands[:, None, 0] - xs_p[None, :]) ** 2
                    + (cands[:, None, 1] - ys_p[None, :]) ** 2,
                    axis=1,
                )
                void_pt = cands[int(np.argmax(d2))]
                z_init[0 : 3 * (n - 1) : 3] = xs_p
                z_init[1 : 3 * (n - 1) : 3] = ys_p
                z_init[2 : 3 * (n - 1) : 3] = prev_packing.angles
                z_init[3 * (n - 1)] = void_pt[0]
                z_init[3 * (n - 1) + 1] = void_pt[1]
                z_init[3 * (n - 1) + 2] = void_rng.uniform(-0.4 * math.pi, 0.4 * math.pi)
                tag = f"lbfgs_inc({n - 1}+1)"
            elif trial < 5 or round_best_z is None:
                z_init[0::3] = rng.uniform(0.5, max(0.52, L_target - 0.5), size=n)
                z_init[1::3] = rng.uniform(0.5, max(0.52, L_target - 0.5), size=n)
                z_init[2::3] = rng.uniform(-0.4 * math.pi, 0.4 * math.pi, size=n)
                tag = f"lbfgs_rand(r={round_idx},t={trial})"
            else:
                z_init[:] = round_best_z
                chosen = rng.choice(n, size=min(k_sub, n), replace=False)
                z_init[3 * chosen] = np.clip(
                    z_init[3 * chosen] + rng.normal(0.0, 0.25, len(chosen)),
                    0.45,
                    max(0.46, L_target - 0.45),
                )
                z_init[3 * chosen + 1] = np.clip(
                    z_init[3 * chosen + 1] + rng.normal(0.0, 0.25, len(chosen)),
                    0.45,
                    max(0.46, L_target - 0.45),
                )
                z_init[3 * chosen + 2] += rng.uniform(-0.6, 0.6, len(chosen))
                tag = f"lbfgs_hop(r={round_idx},t={trial})"

            z_sol, e_sol = _untangle_and_polish(z_init, L_target, safety_gap)
            if e_sol < round_best_e:
                round_best_e = e_sol
                round_best_z = z_sol
            if e_sol < 2e-3 and _try_accept(z_sol, tag):
                improved_this_round = True
                break

        if not improved_this_round:
            round_idx += 1

    return best


def _optimize_worker(args: tuple[int, Packing, float, int, float, Packing | None]) -> Packing:
    n, packing, time_budget_s, seed, safety_gap, prev_packing = args
    return optimize_puzzle_lbfgs(
        n=n,
        current_best=packing,
        time_budget_s=time_budget_s,
        seed=seed,
        safety_gap=safety_gap,
        prev_packing=prev_packing,
    )


def optimize_puzzles_parallel(
    packings: dict[int, Packing],
    target_ns: Sequence[int],
    time_budget_s: float = 1.8,
    seed: int = 42,
    safety_gap: float = DEFAULT_SAFETY_GAP,
    workers: int | None = None,
) -> dict[int, Packing]:
    """Run ``optimize_puzzle_lbfgs`` in parallel over ``target_ns``."""
    out = dict(packings)
    work = [
        (n, out[n], time_budget_s, seed, safety_gap, out.get(n - 1))
        for n in target_ns
        if n in out and abs(math.isqrt(n) ** 2 - n) > 0
    ]
    if not work:
        return out

    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(work) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for res in pool.map(_optimize_worker, work):
                if res.side_length < out[res.n].side_length - 1e-12:
                    out[res.n] = res
    else:
        for item in work:
            res = _optimize_worker(item)
            if res.side_length < out[res.n].side_length - 1e-12:
                out[res.n] = res

    return out
