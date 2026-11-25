"""Extended v2 Numba-accelerated geometric packers for Squares in Squares.

Implements three first-principles constructive search engines (all enforcing
``safety_gap = 2e-8`` so float64 SAT certifies every pair without ambiguity):
1. ``search_gobel_unrestricted``: unrestricted 4-corner 45-degree staircase + band search.
2. ``search_offcenter_block_shells``: off-center tilted p x q block + 4-corner greedy shells.
3. ``search_4corner_slack_bands``: 4-corner exact-AABB + slack-transfer tilted bands.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from collections.abc import Sequence

import numpy as np
from numba import njit

from src.constructive import (
    DEFAULT_SAFETY_GAP,
    MAX_N,
    Packing,
    _build_gobel45,
    normalize_packing,
)


# ---------------------------------------------------------------------------
# Engine A: Unrestricted 4-corner 45-degree Gobel staircase + diagonal band
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gobel45_geometry_numba(
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    p: int,
    safety_gap: float,
) -> tuple[float, int]:
    step = 1.0 + safety_gap
    eps = safety_gap
    s2 = math.sqrt(2.0)
    inv_s2 = 1.0 / s2

    d1 = (2.0 + (a1 - 1) * step) * inv_s2 if a1 > 0 else 0.0
    d2 = (2.0 + (a2 - 1) * step) * inv_s2 if a2 > 0 else 0.0

    l_band = (d1 + d2 + 2.0 * eps + 1.0 + (p - 1) * step + 1e-12) * inv_s2

    bl_span = 1.0 + (a1 - 1) * step if a1 > 0 else 0.0
    tr_span = 1.0 + (a2 - 1) * step if a2 > 0 else 0.0
    br_span = 1.0 + (b1 - 1) * step if b1 > 0 else 0.0
    tl_span = 1.0 + (b2 - 1) * step if b2 > 0 else 0.0

    min_l = l_band
    if bl_span > 0.0 and br_span > 0.0:
        v = bl_span + eps + br_span
        if v > min_l:
            min_l = v
    elif bl_span + br_span > min_l:
        min_l = bl_span + br_span

    if tl_span > 0.0 and tr_span > 0.0:
        v = tl_span + eps + tr_span
        if v > min_l:
            min_l = v
    elif tl_span + tr_span > min_l:
        min_l = tl_span + tr_span

    if bl_span > 0.0 and tl_span > 0.0:
        v = bl_span + eps + tl_span
        if v > min_l:
            min_l = v
    elif bl_span + tl_span > min_l:
        min_l = bl_span + tl_span

    if br_span > 0.0 and tr_span > 0.0:
        v = br_span + eps + tr_span
        if v > min_l:
            min_l = v
    elif br_span + tr_span > min_l:
        min_l = br_span + tr_span

    L = min_l
    if b1 > 0 and b2 > 0:
        b1_v = (-L + 2.0 + (b1 - 1) * step) * inv_s2
        b2_v = (L - 2.0 - (b2 - 1) * step) * inv_s2
        if b1_v + eps > b2_v:
            v = (4.0 + (b1 + b2 - 2) * step + s2 * eps) * 0.5 + 1e-12
            if v > L:
                L = v

    if L > 32.05:
        return L, 0

    count = (
        a1 * (a1 + 1) // 2
        + a2 * (a2 + 1) // 2
        + b1 * (b1 + 1) // 2
        + b2 * (b2 + 1) // 2
    )

    b1_bound = (-L + 2.0 + (b1 - 1) * step) * inv_s2 + eps + 0.5 if b1 > 0 else -1e18
    b2_bound = (L - 2.0 - (b2 - 1) * step) * inv_s2 - eps - 0.5 if b2 > 0 else 1e18

    for r in range(p):
        u_r = d1 + eps + 0.5 + r * step
        v1 = 1.0 - u_r
        v2 = u_r - L * s2 + 1.0
        v_min = v1 if v1 > v2 else v2
        if b1_bound > v_min:
            v_min = b1_bound

        v3 = u_r - 1.0
        v4 = L * s2 - 1.0 - u_r
        v_max = v3 if v3 < v4 else v4
        if b2_bound < v_max:
            v_max = b2_bound

        if v_max >= v_min - 1e-13:
            v_span = v_max - v_min
            if v_span < 0.0:
                v_span = 0.0
            count += int(math.floor(v_span / step + 1e-12)) + 1

    return L, count


@njit(cache=True)
def _scan_gobel_unrestricted_numba(safety_gap: float) -> tuple[np.ndarray, np.ndarray]:
    best_L = np.empty(MAX_N + 1, dtype=np.float64)
    best_params = np.full((MAX_N + 1, 5), -1, dtype=np.int64)
    for n in range(MAX_N + 1):
        best_L[n] = float(math.ceil(math.sqrt(n))) if n >= 1 else 0.0

    for a1 in range(0, 32):
        c_a1 = a1 * (a1 + 1) // 2
        for a2 in range(a1, 32):
            if a1 == 0 and a2 == 0:
                continue
            c_a12 = c_a1 + a2 * (a2 + 1) // 2
            if c_a12 > 1050:
                break
            for b1 in range(0, 32):
                c_a12b1 = c_a12 + b1 * (b1 + 1) // 2
                if c_a12b1 > 1050:
                    break
                for b2 in range(b1, 32):
                    base_cnt = c_a12b1 + b2 * (b2 + 1) // 2
                    if base_cnt > 1050:
                        break
                    for p in range(1, 17):
                        L, count = _gobel45_geometry_numba(a1, a2, b1, b2, p, safety_gap)
                        if L > 32.05 or count < 2:
                            if L > 32.05:
                                break
                            continue
                        n_max = count if count < MAX_N else MAX_N
                        for n in range(n_max, 1, -1):
                            ceil_n = math.ceil(math.sqrt(n))
                            if L >= ceil_n - 1e-12:
                                break
                            if L < best_L[n] - 1e-12:
                                best_L[n] = L
                                best_params[n, 0] = a1
                                best_params[n, 1] = a2
                                best_params[n, 2] = b1
                                best_params[n, 3] = b2
                                best_params[n, 4] = p
    return best_L, best_params


def search_gobel_unrestricted(safety_gap: float = DEFAULT_SAFETY_GAP) -> dict[int, Packing]:
    """Unrestricted 4-corner 45-degree staircase + diagonal band search."""
    _, best_params = _scan_gobel_unrestricted_numba(safety_gap)
    results: dict[int, Packing] = {}
    for n in range(2, MAX_N + 1):
        a1 = int(best_params[n, 0])
        if a1 >= 0:
            a2 = int(best_params[n, 1])
            b1 = int(best_params[n, 2])
            b2 = int(best_params[n, 3])
            p = int(best_params[n, 4])
            results[n] = _build_gobel45(n, a1, a2, b1, b2, p, safety_gap=safety_gap)
    return results


# ---------------------------------------------------------------------------
# Engine B: Off-center tilted p x q block + 4-corner greedy shells
# ---------------------------------------------------------------------------


@njit(cache=True)
def _count_block_shells_numba(
    L: float,
    c: float,
    s: float,
    p: int,
    q: int,
    cx: float,
    cy: float,
    eps: float,
) -> int:
    """Count p*q tilted block squares + 4-corner shell uprights using an O(1) spatial grid."""
    step = 1.0 + eps
    W = p + (p - 1) * eps
    H = q + (q - 1) * eps
    rx = 0.5 * (W * c + H * s)
    ry = 0.5 * (W * s + H * c)
    if cx - rx < 0.0 or cx + rx > L or cy - ry < 0.0 or cy + ry > L:
        return 0

    lim_cs = 0.5 * W + 0.5 * (c + s) + eps
    lim_sc = 0.5 * H + 0.5 * (c + s) + eps
    lim_x = 0.5 + rx + eps
    lim_y = 0.5 + ry + eps
    tol_step = step - 1e-13

    max_m = int((L - 1.0) / step) + 1
    if max_m > 34:
        max_m = 34

    cell_cnt = np.zeros((35, 35), dtype=np.int32)
    cell_x = np.empty((35, 35, 4), dtype=np.float64)
    cell_y = np.empty((35, 35, 4), dtype=np.float64)
    placed = 0

    for shell in range(2 * max_m):
        for cid_idx in range(4):
            if cid_idx == 0:
                cid = 0
            elif cid_idx == 1:
                cid = 2
            elif cid_idx == 2:
                cid = 1
            else:
                cid = 3

            i_start = shell - (max_m - 1)
            if i_start < 0:
                i_start = 0
            i_end = shell
            if i_end >= max_m:
                i_end = max_m - 1

            for i in range(i_start, i_end + 1):
                j = shell - i
                sx = (0.5 + i * step) if (cid == 0 or cid == 3) else (L - 0.5 - i * step)
                sy = (0.5 + j * step) if (cid == 0 or cid == 1) else (L - 0.5 - j * step)
                if sx < 0.5 or sx > L - 0.5 or sy < 0.5 or sy > L - 0.5:
                    continue

                # SAT separation against tilted W x H block
                dx = sx - cx
                dy = sy - cy
                sep_block = (
                    abs(dx) >= lim_x
                    or abs(dy) >= lim_y
                    or abs(dx * c + dy * s) >= lim_cs
                    or abs(-dx * s + dy * c) >= lim_sc
                )
                if not sep_block:
                    continue

                # O(1) spatial hash check against already-placed uprights
                bx = int(sx)
                by = int(sy)
                if bx < 0:
                    bx = 0
                elif bx > 34:
                    bx = 34
                if by < 0:
                    by = 0
                elif by > 34:
                    by = 34

                ok = True
                x_lo = bx - 1 if bx > 0 else 0
                x_hi = bx + 1 if bx < 34 else 34
                y_lo = by - 1 if by > 0 else 0
                y_hi = by + 1 if by < 34 else 34
                for ux_i in range(x_lo, x_hi + 1):
                    for uy_i in range(y_lo, y_hi + 1):
                        cnt_c = cell_cnt[ux_i, uy_i]
                        for k in range(cnt_c):
                            if abs(sx - cell_x[ux_i, uy_i, k]) < tol_step and abs(sy - cell_y[ux_i, uy_i, k]) < tol_step:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break

                if ok:
                    k_here = cell_cnt[bx, by]
                    if k_here < 4:
                        cell_x[bx, by, k_here] = sx
                        cell_y[bx, by, k_here] = sy
                        cell_cnt[bx, by] = k_here + 1
                        placed += 1
                        if p * q + placed >= 1050:
                            return 1050

    return p * q + placed


@njit(cache=True)
def _search_45_blocks_numba(
    best_L: np.ndarray,
    best_cfg: np.ndarray,
    eps: float,
) -> None:
    s2 = math.sqrt(2.0)
    c = math.cos(math.pi / 4.0)
    s = math.sin(math.pi / 4.0)
    offsets = np.array([-0.5, 0.0, 0.5], dtype=np.float64)

    for A in range(2, 34):
        for B in range(1, 19):
            L = 0.5 * (A + B * s2) + 8.0 * eps
            if L < 2.5 or L > 16.0:
                continue
            p_max = 7 if B > 7 else B
            q_max = int(L * 1.3) + 1
            if q_max > 13:
                q_max = 13

            for p in range(1, p_max + 1):
                for q in range(p, q_max + 1):
                    for idx_dx in range(3):
                        dx = offsets[idx_dx]
                        for idx_dy in range(3):
                            dy = offsets[idx_dy]
                            cnt = _count_block_shells_numba(L, c, s, p, q, 0.5 * L + dx, 0.5 * L + dy, eps)
                            if cnt < 2:
                                continue
                            n_max = cnt if cnt < MAX_N else MAX_N
                            for n in range(n_max, 1, -1):
                                ceil_n = math.ceil(math.sqrt(n))
                                if L >= ceil_n - 1e-12:
                                    break
                                if L < best_L[n] - 1e-12:
                                    best_L[n] = L
                                    best_cfg[n, 0] = L
                                    best_cfg[n, 1] = math.pi / 4.0
                                    best_cfg[n, 2] = float(p)
                                    best_cfg[n, 3] = float(q)
                                    best_cfg[n, 4] = 0.5 * L + dx
                                    best_cfg[n, 5] = 0.5 * L + dy


@njit(cache=True)
def _search_rational_blocks_numba(
    thetas: np.ndarray,
    best_L: np.ndarray,
    best_cfg: np.ndarray,
    eps: float,
) -> None:
    offsets = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    L_step = 0.15
    n_steps = int((11.0 - 3.0) / L_step) + 1

    for idx_th in range(thetas.shape[0]):
        th = thetas[idx_th]
        c = math.cos(th)
        s = math.sin(th)
        for p in range(1, 6):
            for q in range(p, 10):
                for idx_dx in range(3):
                    dx = offsets[idx_dx]
                    for idx_dy in range(3):
                        dy = offsets[idx_dy]
                        prev_cnt = 0
                        prev_L = 3.0
                        for step_k in range(n_steps):
                            L = 3.0 + step_k * L_step
                            cnt = _count_block_shells_numba(L, c, s, p, q, 0.5 * L + dx, 0.5 * L + dy, eps)
                            if cnt > prev_cnt and cnt >= 2:
                                lo_base = prev_L if step_k > 0 else 2.5
                                hi_base = L
                                n_start = prev_cnt + 1 if prev_cnt + 1 >= 2 else 2
                                n_end = cnt if cnt <= MAX_N else MAX_N
                                for n in range(n_start, n_end + 1):
                                    ceil_n = math.ceil(math.sqrt(n))
                                    if lo_base >= best_L[n] or lo_base >= ceil_n:
                                        continue
                                    lo = lo_base
                                    hi = hi_base
                                    for _ in range(32):
                                        mid = 0.5 * (lo + hi)
                                        if _count_block_shells_numba(mid, c, s, p, q, 0.5 * mid + dx, 0.5 * mid + dy, eps) >= n:
                                            hi = mid
                                        else:
                                            lo = mid
                                    if hi < best_L[n] - 1e-12 and hi < ceil_n - 1e-12:
                                        best_L[n] = hi
                                        best_cfg[n, 0] = hi
                                        best_cfg[n, 1] = th
                                        best_cfg[n, 2] = float(p)
                                        best_cfg[n, 3] = float(q)
                                        best_cfg[n, 4] = 0.5 * hi + dx
                                        best_cfg[n, 5] = 0.5 * hi + dy
                            prev_cnt = cnt
                            prev_L = L


def _build_block_shells(
    n_target: int,
    L: float,
    theta: float,
    p: int,
    q: int,
    cx: float,
    cy: float,
    safety_gap: float,
) -> Packing:
    eps = safety_gap
    step = 1.0 + eps
    c, s = math.cos(theta), math.sin(theta)
    W = p + (p - 1) * eps
    H = q + (q - 1) * eps
    rx = 0.5 * (W * c + H * s)
    ry = 0.5 * (W * s + H * c)

    xs: list[float] = []
    ys: list[float] = []
    angs: list[float] = []

    for i in range(p):
        u = (i - 0.5 * (p - 1)) * step
        for j in range(q):
            v = (j - 0.5 * (q - 1)) * step
            xs.append(cx + u * c - v * s)
            ys.append(cy + u * s + v * c)
            angs.append(theta)

    max_m = min(34, int((L - 1.0) / step) + 1)
    uprights: list[tuple[float, float]] = []
    tol_step = step - 1e-13

    for shell in range(2 * max_m):
        if len(xs) >= n_target:
            break
        for cid in (0, 2, 1, 3):
            if len(xs) >= n_target:
                break
            i_start = max(0, shell - (max_m - 1))
            i_end = min(shell, max_m - 1)
            for i in range(i_start, i_end + 1):
                if len(xs) >= n_target:
                    break
                j = shell - i
                sx = (0.5 + i * step) if cid in (0, 3) else (L - 0.5 - i * step)
                sy = (0.5 + j * step) if cid in (0, 1) else (L - 0.5 - j * step)
                if not (0.5 <= sx <= L - 0.5 and 0.5 <= sy <= L - 0.5):
                    continue
                dx = sx - cx
                dy = sy - cy
                sep_block = (
                    abs(dx) >= 0.5 + rx + eps
                    or abs(dy) >= 0.5 + ry + eps
                    or abs(dx * c + dy * s) >= 0.5 * W + 0.5 * (c + s) + eps
                    or abs(-dx * s + dy * c) >= 0.5 * H + 0.5 * (c + s) + eps
                )
                if not sep_block:
                    continue
                ok = True
                for ux, uy in uprights:
                    if abs(sx - ux) < tol_step and abs(sy - uy) < tol_step:
                        ok = False
                        break
                if ok:
                    uprights.append((sx, sy))
                    xs.append(sx)
                    ys.append(sy)
                    angs.append(0.0)

    xs_arr = np.array(xs[:n_target], dtype=np.float64)
    ys_arr = np.array(ys[:n_target], dtype=np.float64)
    angs_arr = np.array(angs[:n_target], dtype=np.float64)
    xs_n, ys_n, angs_n, side = normalize_packing(xs_arr, ys_arr, angs_arr)
    return Packing(
        n=n_target,
        xs=xs_n,
        ys=ys_n,
        angles=angs_n,
        side_length=side,
        source=f"block_shell(th={theta:.4f},{p}x{q})",
    )


def search_offcenter_block_shells(safety_gap: float = DEFAULT_SAFETY_GAP) -> dict[int, Packing]:
    """Search off-center tilted p x q block + 4-corner upright shells."""
    best_L = np.array([float(math.ceil(math.sqrt(n))) for n in range(MAX_N + 1)], dtype=np.float64)
    best_cfg = np.full((MAX_N + 1, 6), -1.0, dtype=np.float64)

    _search_45_blocks_numba(best_L, best_cfg, safety_gap)

    rational_pairs = (
        (1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (5, 12),
    )
    thetas = np.array([math.atan2(qa, ma) for qa, ma in rational_pairs], dtype=np.float64)
    _search_rational_blocks_numba(thetas, best_L, best_cfg, safety_gap)

    results: dict[int, Packing] = {}
    for n in range(2, MAX_N + 1):
        L_val = float(best_cfg[n, 0])
        if L_val > 0.0:
            th = float(best_cfg[n, 1])
            p = int(best_cfg[n, 2])
            q = int(best_cfg[n, 3])
            cx = float(best_cfg[n, 4])
            cy = float(best_cfg[n, 5])
            results[n] = _build_block_shells(
                n, L_val + 2e-10, th, p, q, cx + 1e-10, cy + 1e-10, safety_gap
            )
    return results


# ---------------------------------------------------------------------------
# Engine C: 4-corner exact-AABB + slack-transfer tilted bands
# ---------------------------------------------------------------------------


@njit(cache=True)
def _count_4corner_slack_numba(
    L: float,
    c: float,
    s: float,
    p: int,
    d1_frac: float,
    v_cut_frac: float,
    safety_gap: float,
) -> int:
    step = 1.0 + safety_gap
    eps = safety_gap
    bw = 1.0 + (p - 1) * step + 2.0 * eps
    diag = L * (c + s)
    if bw > diag:
        return 0

    d1_raw = d1_frac * (diag - bw)
    max_i = int((L - 1.0) / step)
    if max_i < 0 or max_i > 38:
        return 0

    bl_row_xmax = np.full(40, -1.0, dtype=np.float64)
    d1_actual = 0.0
    bl_cnt = 0
    for j in range(max_i + 1):
        yt = 1.0 + j * step
        rem = d1_raw - yt * s
        if rem < c - 1e-13:
            break
        ilim = int((rem - c) / (c * step) + 1e-12)
        if ilim > max_i:
            ilim = max_i
        while ilim >= 0 and (1.0 + ilim * step) * c + yt * s > d1_raw:
            ilim -= 1
        if ilim >= 0:
            bl_cnt += ilim + 1
            bl_row_xmax[j] = 1.0 + ilim * step
            u_val = (1.0 + ilim * step) * c + yt * s
            if u_val > d1_actual:
                d1_actual = u_val

    # Transfer unused BL slack to TR
    d2 = diag - bw - d1_actual
    if d2 < 0.0:
        return 0

    tr_row_xmin = np.full(40, L + 10.0, dtype=np.float64)
    tr_cnt = 0
    for j in range(max_i + 1):
        ysp = 1.0 + j * step
        rem = d2 - ysp * s
        if rem < c - 1e-13:
            break
        ilim = int((rem - c) / (c * step) + 1e-12)
        if ilim > max_i:
            ilim = max_i
        while ilim >= 0 and (1.0 + ilim * step) * c + ysp * s > d2:
            ilim -= 1
        if ilim >= 0:
            tr_cnt += ilim + 1
            tr_row_xmin[j] = L - (1.0 + ilim * step)

    br_cnt = 0
    tl_cnt = 0
    b_lo_lim_t = -1e18
    b_hi_lim_t = 1e18

    if v_cut_frac > 0.0:
        t_span = L * (c + s)
        b_lo_cut = -L * s + v_cut_frac * t_span
        b_hi_cut = L * c - v_cut_frac * t_span
        b_lo = -1e18
        for j in range(max_i + 1):
            ylo = j * step
            yhi = 1.0 + j * step
            if yhi > L:
                break
            for i in range(max_i + 1):
                xhi = L - i * step
                xlo = xhi - 1.0
                if xlo < 0.0:
                    break
                tv = -xlo * s + yhi * c
                if tv > b_lo_cut:
                    break
                if bl_row_xmax[j] >= 0.0 and xlo < bl_row_xmax[j] + eps:
                    break
                hit_tr = False
                jt_min = int((L - yhi - eps) / step) - 1
                if jt_min < 0:
                    jt_min = 0
                jt_max = int((L - ylo + eps) / step) + 1
                if jt_max > max_i:
                    jt_max = max_i
                for jt in range(jt_min, jt_max + 1):
                    ty_lo = L - 1.0 - jt * step
                    ty_hi = ty_lo + 1.0
                    if ty_lo < yhi + eps and ty_hi > ylo - eps:
                        if xhi > tr_row_xmin[jt] - eps:
                            hit_tr = True
                            break
                if not hit_tr:
                    br_cnt += 1
                    if tv > b_lo:
                        b_lo = tv

        b_hi = 1e18
        for j in range(max_i + 1):
            yhi = L - j * step
            ylo = yhi - 1.0
            if ylo < 0.0:
                break
            for i in range(max_i + 1):
                xlo = i * step
                xhi = 1.0 + i * step
                if xhi > L:
                    break
                tv = -xhi * s + ylo * c
                if tv < b_hi_cut:
                    break
                if tr_row_xmin[j] <= L and xhi > tr_row_xmin[j] - eps:
                    break
                if br_cnt > 0 and b_lo + eps > tv:
                    break
                hit_bl = False
                jb_min = int((ylo - eps - 1.0) / step)
                if jb_min < 0:
                    jb_min = 0
                jb_max = int((yhi + eps) / step) + 1
                if jb_max > max_i:
                    jb_max = max_i
                for jb in range(jb_min, jb_max + 1):
                    by_lo = jb * step
                    by_hi = by_lo + 1.0
                    if by_lo < yhi + eps and by_hi > ylo - eps:
                        if bl_row_xmax[jb] >= 0.0 and xlo < bl_row_xmax[jb] + eps:
                            hit_bl = True
                            break
                if not hit_bl:
                    tl_cnt += 1
                    if tv < b_hi:
                        b_hi = tv

        if br_cnt > 0:
            b_lo_lim_t = b_lo + eps + 0.5
        if tl_cnt > 0:
            b_hi_lim_t = b_hi - eps - 0.5

    h_t = 0.5 * (c + s)
    tilted = 0
    for r in range(p):
        ur = d1_actual + eps + 0.5 + r * step
        v1 = (ur * c - (L - h_t)) / s
        v2 = (h_t - ur * s) / c
        vmin = v1 if v1 > v2 else v2
        if b_lo_lim_t > vmin:
            vmin = b_lo_lim_t

        v3 = (ur * c - h_t) / s
        v4 = (L - h_t - ur * s) / c
        vmax = v3 if v3 < v4 else v4
        if b_hi_lim_t < vmax:
            vmax = b_hi_lim_t

        if vmax >= vmin:
            tilted += int((vmax - vmin) / step + 1e-12) + 1

    return bl_cnt + tr_cnt + br_cnt + tl_cnt + tilted


@njit(cache=True)
def _sweep_4corner_angle_numba(
    theta: float,
    safety_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(theta)
    s = math.sin(theta)
    best_L = np.array([float(math.ceil(math.sqrt(n))) if n >= 1 else 0.0 for n in range(MAX_N + 1)], dtype=np.float64)
    best_params = np.full((MAX_N + 1, 5), -1.0, dtype=np.float64)

    d1_fracs = np.array([0.35, 0.42, 0.46, 0.50, 0.54, 0.58], dtype=np.float64)
    v_cuts = np.array([0.0, 0.12, 0.22, 0.34], dtype=np.float64)
    L_step = 0.15
    n_steps = int((32.0 - 2.5) / L_step) + 1

    for p in range(1, 18):
        for fi in range(d1_fracs.shape[0]):
            d1_frac = d1_fracs[fi]
            for vi in range(v_cuts.shape[0]):
                vc = v_cuts[vi]
                prev_cnt = 0
                prev_L = 2.5
                for k in range(n_steps):
                    L = 2.5 + k * L_step
                    if L * (c + s) < 1.0 + (p - 1):
                        continue
                    cnt = _count_4corner_slack_numba(L, c, s, p, d1_frac, vc, safety_gap)
                    if cnt > prev_cnt and cnt >= 2:
                        lo_base = prev_L if k > 0 else 1.5
                        hi_base = L
                        n_start = prev_cnt + 1 if prev_cnt + 1 >= 2 else 2
                        n_end = cnt if cnt <= MAX_N else MAX_N
                        for n in range(n_start, n_end + 1):
                            ceil_n = math.ceil(math.sqrt(n))
                            if lo_base >= best_L[n] or lo_base >= ceil_n:
                                continue
                            lo = lo_base
                            hi = hi_base
                            for _ in range(34):
                                mid = 0.5 * (lo + hi)
                                if _count_4corner_slack_numba(mid, c, s, p, d1_frac, vc, safety_gap) >= n:
                                    hi = mid
                                else:
                                    lo = mid
                            if hi < best_L[n] - 1e-12 and hi < ceil_n - 1e-12:
                                best_L[n] = hi
                                best_params[n, 0] = theta
                                best_params[n, 1] = float(p)
                                best_params[n, 2] = d1_frac
                                best_params[n, 3] = vc
                                best_params[n, 4] = hi
                    prev_cnt = cnt
                    prev_L = L
    return best_L, best_params


def _build_4corner_slack_band(
    n_target: int,
    L: float,
    theta: float,
    p: int,
    d1_frac: float,
    v_cut_frac: float,
    safety_gap: float,
) -> Packing:
    step = 1.0 + safety_gap
    eps = safety_gap
    c, s = math.cos(theta), math.sin(theta)
    bw = 1.0 + (p - 1) * step + 2.0 * eps
    diag = L * (c + s)
    d1_raw = d1_frac * (diag - bw)
    max_i = min(38, int((L - 1.0) / step))

    xs: list[float] = []
    ys: list[float] = []
    angs: list[float] = []

    bl_row_xmax = [-1.0] * 40
    d1_actual = 0.0
    for j in range(max_i + 1):
        yt = 1.0 + j * step
        rem = d1_raw - yt * s
        if rem < c - 1e-13:
            break
        ilim = min(max_i, int((rem - c) / (c * step) + 1e-12))
        while ilim >= 0 and (1.0 + ilim * step) * c + yt * s > d1_raw:
            ilim -= 1
        if ilim >= 0:
            bl_row_xmax[j] = 1.0 + ilim * step
            d1_actual = max(d1_actual, (1.0 + ilim * step) * c + yt * s)
            for i in range(ilim + 1):
                xs.append(0.5 + i * step)
                ys.append(0.5 + j * step)
                angs.append(0.0)

    d2 = diag - bw - d1_actual
    tr_row_xmin = [L + 10.0] * 40
    for j in range(max_i + 1):
        ysp = 1.0 + j * step
        rem = d2 - ysp * s
        if rem < c - 1e-13:
            break
        ilim = min(max_i, int((rem - c) / (c * step) + 1e-12))
        while ilim >= 0 and (1.0 + ilim * step) * c + ysp * s > d2:
            ilim -= 1
        if ilim >= 0:
            tr_row_xmin[j] = L - (1.0 + ilim * step)
            for i in range(ilim + 1):
                xs.append(L - 0.5 - i * step)
                ys.append(L - 0.5 - j * step)
                angs.append(0.0)

    br_cnt = 0
    tl_cnt = 0
    b_lo_lim_t = -1e18
    b_hi_lim_t = 1e18
    if v_cut_frac > 0.0:
        t_span = L * (c + s)
        b_lo_cut = -L * s + v_cut_frac * t_span
        b_hi_cut = L * c - v_cut_frac * t_span
        b_lo = -1e18
        for j in range(max_i + 1):
            ylo = j * step
            yhi = 1.0 + j * step
            if yhi > L:
                break
            for i in range(max_i + 1):
                xhi = L - i * step
                xlo = xhi - 1.0
                if xlo < 0.0:
                    break
                tv = -xlo * s + yhi * c
                if tv > b_lo_cut:
                    break
                if bl_row_xmax[j] >= 0.0 and xlo < bl_row_xmax[j] + eps:
                    break
                hit_tr = False
                for jt in range(max(0, int((L - yhi - eps) / step) - 1), min(max_i, int((L - ylo + eps) / step) + 1) + 1):
                    ty_lo = L - 1.0 - jt * step
                    ty_hi = ty_lo + 1.0
                    if ty_lo < yhi + eps and ty_hi > ylo - eps and xhi > tr_row_xmin[jt] - eps:
                        hit_tr = True
                        break
                if not hit_tr:
                    br_cnt += 1
                    b_lo = max(b_lo, tv)
                    xs.append(xlo + 0.5)
                    ys.append(ylo + 0.5)
                    angs.append(0.0)

        b_hi = 1e18
        for j in range(max_i + 1):
            yhi = L - j * step
            ylo = yhi - 1.0
            if ylo < 0.0:
                break
            for i in range(max_i + 1):
                xlo = i * step
                xhi = 1.0 + i * step
                if xhi > L:
                    break
                tv = -xhi * s + ylo * c
                if tv < b_hi_cut:
                    break
                if tr_row_xmin[j] <= L and xhi > tr_row_xmin[j] - eps:
                    break
                if br_cnt > 0 and b_lo + eps > tv:
                    break
                hit_bl = False
                for jb in range(max(0, int((ylo - eps - 1.0) / step)), min(max_i, int((yhi + eps) / step) + 1) + 1):
                    by_lo = jb * step
                    by_hi = by_lo + 1.0
                    if by_lo < yhi + eps and by_hi > ylo - eps and bl_row_xmax[jb] >= 0.0 and xlo < bl_row_xmax[jb] + eps:
                        hit_bl = True
                        break
                if not hit_bl:
                    tl_cnt += 1
                    b_hi = min(b_hi, tv)
                    xs.append(xlo + 0.5)
                    ys.append(ylo + 0.5)
                    angs.append(0.0)

        if br_cnt > 0:
            b_lo_lim_t = b_lo + eps + 0.5
        if tl_cnt > 0:
            b_hi_lim_t = b_hi - eps - 0.5

    h_t = 0.5 * (c + s)
    for r in range(p):
        ur = d1_actual + eps + 0.5 + r * step
        vmin = max((ur * c - (L - h_t)) / s, (h_t - ur * s) / c, b_lo_lim_t)
        vmax = min((ur * c - h_t) / s, (L - h_t - ur * s) / c, b_hi_lim_t)
        if vmax >= vmin:
            k_row = int((vmax - vmin) / step + 1e-12) + 1
            for m in range(k_row):
                if len(xs) >= n_target:
                    break
                v = vmin + m * step
                xs.append(ur * c - v * s)
                ys.append(ur * s + v * c)
                angs.append(theta)
        if len(xs) >= n_target:
            break

    xs_arr = np.array(xs[:n_target], dtype=np.float64)
    ys_arr = np.array(ys[:n_target], dtype=np.float64)
    angs_arr = np.array(angs[:n_target], dtype=np.float64)
    xs_n, ys_n, angs_n, side = normalize_packing(xs_arr, ys_arr, angs_arr)
    return Packing(
        n=n_target,
        xs=xs_n,
        ys=ys_n,
        angles=angs_n,
        side_length=side,
        source=f"slack_band(th={theta:.4f},p={p},vc={v_cut_frac:.2f})",
    )


def _generate_4corner_angles() -> list[float]:
    angles: set[float] = set()
    for m in range(1, 14):
        for q in range(1, m + 1):
            if math.gcd(q, m) == 1:
                angles.add(round(math.atan2(q, m), 12))
    return sorted(angles)


def _4corner_angle_worker(args: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    theta, safety_gap = args
    return _sweep_4corner_angle_numba(theta, safety_gap)


def search_4corner_slack_bands(
    safety_gap: float = DEFAULT_SAFETY_GAP,
    workers: int | None = None,
) -> dict[int, Packing]:
    """Search 4-corner exact-AABB + slack-transfer tilted bands in parallel."""
    angles = _generate_4corner_angles()
    best_L = np.array([float(math.ceil(math.sqrt(n))) if n >= 1 else 0.0 for n in range(MAX_N + 1)], dtype=np.float64)
    best_params = np.full((MAX_N + 1, 5), -1.0, dtype=np.float64)

    # Warm up Numba JIT before spawning pool
    _sweep_4corner_angle_numba(angles[0], safety_gap)

    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    tasks = [(th, safety_gap) for th in angles]
    if num_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            outputs = pool.map(_4corner_angle_worker, tasks)
    else:
        outputs = [_4corner_angle_worker(t) for t in tasks]

    for w_L, w_params in outputs:
        for n in range(2, MAX_N + 1):
            if w_params[n, 0] > 0.0 and w_L[n] < best_L[n] - 1e-12:
                best_L[n] = w_L[n]
                best_params[n] = w_params[n]

    results: dict[int, Packing] = {}
    for n in range(2, MAX_N + 1):
        th = float(best_params[n, 0])
        if th > 0.0:
            p = int(best_params[n, 1])
            d1_frac = float(best_params[n, 2])
            vc = float(best_params[n, 3])
            L_val = float(best_params[n, 4])
            results[n] = _build_4corner_slack_band(
                n, L_val + 2e-10, th, p, d1_frac, vc, safety_gap
            )
    return results
