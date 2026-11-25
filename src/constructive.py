"""Constructive geometric packers and closure dynamic programming for Squares in Squares.

Implements three complementary constructive methods (all enforcing an explicit
``safety_gap`` so float64 SAT certifies every pair without boundary ambiguity):
1. ``search_gobel_45``: 4-corner axis-aligned staircases with a 45-degree tilted band.
2. ``search_tilted_bands``: general shallow-angle tilted fault-strip packings.
3. ``apply_closures``: L-border addition (``m -> m'``) and monotone deletion (``n+1 -> n``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

DEFAULT_SAFETY_GAP = 2e-8
MAX_N = 1000


@dataclass
class Packing:
    """Candidate packing of n unit squares."""

    n: int
    xs: np.ndarray
    ys: np.ndarray
    angles: np.ndarray
    side_length: float
    source: str
    use_raw_baseline: bool = False


def compute_side_length(
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
) -> float:
    """Compute translation-invariant axis-aligned bounding square side length."""
    half_ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    span_x = float(np.max(xs + half_ext) - np.min(xs - half_ext))
    span_y = float(np.max(ys + half_ext) - np.min(ys - half_ext))
    return max(span_x, span_y)


def normalize_packing(
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shift bounding box bottom-left corner to (0, 0) and wrap angles to [0, pi/2)."""
    xs_arr = np.asarray(xs, dtype=np.float64).copy()
    ys_arr = np.asarray(ys, dtype=np.float64).copy()
    ang_arr = np.mod(np.asarray(angles, dtype=np.float64), math.pi / 2.0)

    half_ext = 0.5 * (np.abs(np.cos(ang_arr)) + np.abs(np.sin(ang_arr)))
    min_x = float(np.min(xs_arr - half_ext))
    min_y = float(np.min(ys_arr - half_ext))
    xs_arr -= min_x
    ys_arr -= min_y

    span_x = float(np.max(xs_arr + half_ext))
    span_y = float(np.max(ys_arr + half_ext))
    return xs_arr, ys_arr, ang_arr, max(span_x, span_y)


def grid_packing(n: int) -> Packing:
    """Return the trivial ceil(sqrt(n)) x ceil(sqrt(n)) axis-aligned grid packing."""
    k = math.ceil(math.sqrt(n))
    idx = np.arange(n, dtype=np.float64)
    xs = 0.5 + (idx % k)
    ys = 0.5 + np.floor(idx / k)
    angles = np.zeros(n, dtype=np.float64)
    return Packing(
        n=n,
        xs=xs,
        ys=ys,
        angles=angles,
        side_length=float(k),
        source="baseline_grid",
        use_raw_baseline=True,
    )


# ---------------------------------------------------------------------------
# Method 1: General 4-corner 45-degree Gobel staircase + diagonal band
# ---------------------------------------------------------------------------


def _gobel45_geometry(
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    p: int,
    safety_gap: float,
) -> tuple[float, int, float, float]:
    """Compute ``(L, total_count, d1, d2)`` for a 4-corner 45-degree configuration."""
    step = 1.0 + safety_gap
    eps = safety_gap
    s2 = math.sqrt(2.0)
    inv_s2 = 1.0 / s2

    d1 = (2.0 + (a1 - 1) * step) * inv_s2 if a1 > 0 else 0.0
    d2 = (2.0 + (a2 - 1) * step) * inv_s2 if a2 > 0 else 0.0

    # Tight side length needed for BL, p tilted rows, and TR along u = (x + y)/sqrt(2)
    l_band = (d1 + d2 + 2.0 * eps + 1.0 + (p - 1) * step + 1e-12) * inv_s2

    # Axis-aligned non-overlap constraints along all 4 outer walls
    bl_span = 1.0 + (a1 - 1) * step if a1 > 0 else 0.0
    tr_span = 1.0 + (a2 - 1) * step if a2 > 0 else 0.0
    br_span = 1.0 + (b1 - 1) * step if b1 > 0 else 0.0
    tl_span = 1.0 + (b2 - 1) * step if b2 > 0 else 0.0

    min_l = l_band
    for s_left, s_right in (
        (bl_span, br_span),
        (tl_span, tr_span),
        (bl_span, tl_span),
        (br_span, tr_span),
    ):
        if s_left > 0.0 and s_right > 0.0:
            min_l = max(min_l, s_left + eps + s_right)
        elif s_left > 0.0 or s_right > 0.0:
            min_l = max(min_l, s_left + s_right)

    L = min_l

    # Also ensure BR and TL don't overlap each other along v = (-x + y)/sqrt(2) if both active
    if b1 > 0 and b2 > 0:
        b1_v = (-L + 2.0 + (b1 - 1) * step) * inv_s2
        b2_v = (L - 2.0 - (b2 - 1) * step) * inv_s2
        if b1_v + eps > b2_v:
            # Increase L enough for BR and TL diagonal separation
            L = max(L, (4.0 + (b1 + b2 - 2) * step + s2 * eps) * 0.5 + 1e-12)

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
        v_min = max(1.0 - u_r, u_r - L * s2 + 1.0, b1_bound)
        v_max = min(u_r - 1.0, L * s2 - 1.0 - u_r, b2_bound)
        if v_max >= v_min - 1e-13:
            v_span = max(0.0, v_max - v_min)
            count += int(math.floor(v_span / step + 1e-12)) + 1

    return L, count, d1, d2


def _build_gobel45(
    n_target: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    p: int,
    safety_gap: float,
) -> Packing:
    L, _, d1, _ = _gobel45_geometry(a1, a2, b1, b2, p, safety_gap)
    step = 1.0 + safety_gap
    eps = safety_gap
    s2 = math.sqrt(2.0)
    inv_s2 = 1.0 / s2
    pi_4 = math.pi / 4.0

    xs: list[float] = []
    ys: list[float] = []
    angs: list[float] = []

    # BL(a1)
    for j in range(a1):
        for i in range(a1 - j):
            xs.append(0.5 + i * step)
            ys.append(0.5 + j * step)
            angs.append(0.0)

    # TR(a2)
    for j in range(a2):
        for i in range(a2 - j):
            xs.append(L - 0.5 - i * step)
            ys.append(L - 0.5 - j * step)
            angs.append(0.0)

    # BR(b1)
    for j in range(b1):
        for i in range(b1 - j):
            xs.append(L - 0.5 - i * step)
            ys.append(0.5 + j * step)
            angs.append(0.0)

    # TL(b2)
    for j in range(b2):
        for i in range(b2 - j):
            xs.append(0.5 + i * step)
            ys.append(L - 0.5 - j * step)
            angs.append(0.0)

    b1_bound = (-L + 2.0 + (b1 - 1) * step) * inv_s2 + eps + 0.5 if b1 > 0 else -1e18
    b2_bound = (L - 2.0 - (b2 - 1) * step) * inv_s2 - eps - 0.5 if b2 > 0 else 1e18

    tilted_pts: list[tuple[float, float]] = []
    for r in range(p):
        u_r = d1 + eps + 0.5 + r * step
        v_min = max(1.0 - u_r, u_r - L * s2 + 1.0, b1_bound)
        v_max = min(u_r - 1.0, L * s2 - 1.0 - u_r, b2_bound)
        if v_max >= v_min - 1e-13:
            v_span = max(0.0, v_max - v_min)
            k_row = int(math.floor(v_span / step + 1e-12)) + 1
            slack = max(0.0, v_span - (k_row - 1) * step)
            v_start = v_min + 0.5 * slack
            for m in range(k_row):
                v = v_start + m * step
                if k_row == 1:
                    v = 0.5 * (v_min + v_max)
                xc = (u_r - v) * inv_s2
                yc = (u_r + v) * inv_s2
                tilted_pts.append((xc, yc))

    # Keep n_target squares (drop excess tilted squares, or corner squares if needed)
    for xc, yc in tilted_pts:
        if len(xs) >= n_target:
            break
        xs.append(xc)
        ys.append(yc)
        angs.append(pi_4)

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
        source=f"gobel45(a={a1},{a2};b={b1},{b2};p={p})",
    )


def search_gobel_45(safety_gap: float = DEFAULT_SAFETY_GAP) -> dict[int, Packing]:
    """Search 4-corner 45-degree staircase + diagonal band packings for n=1..1000."""
    best_params: dict[int, tuple[float, tuple[int, int, int, int, int]]] = {}

    for a1 in range(0, 32):
        for a2 in range(max(0, a1 - 2), min(32, a1 + 3)):
            if a1 == 0 and a2 == 0:
                continue
            for b1 in range(0, 32):
                for b2 in range(max(0, b1 - 2), min(32, b1 + 3)):
                    # Symmetric prune when total base corners already exceed max needed
                    base_cnt = (
                        a1 * (a1 + 1) // 2
                        + a2 * (a2 + 1) // 2
                        + b1 * (b1 + 1) // 2
                        + b2 * (b2 + 1) // 2
                    )
                    if base_cnt > MAX_N + 50:
                        continue
                    for p in range(1, 13):
                        L, count, _, _ = _gobel45_geometry(a1, a2, b1, b2, p, safety_gap)
                        if L > 32.2 or count < 1:
                            continue
                        n_max = min(MAX_N, count)
                        # Only non-trivial when beating trivial grid ceil(sqrt(n))
                        # Walk downward from n_max while L < ceil(sqrt(n))
                        for n in range(n_max, 0, -1):
                            ceil_n = math.ceil(math.sqrt(n))
                            if L >= ceil_n - 1e-12:
                                break
                            prev = best_params.get(n)
                            if prev is None or L < prev[0] - 1e-12:
                                best_params[n] = (L, (a1, a2, b1, b2, p))

    results: dict[int, Packing] = {}
    for n, (_, params) in best_params.items():
        results[n] = _build_gobel45(n, *params, safety_gap=safety_gap)
    return results


# ---------------------------------------------------------------------------
# Method 2: General tilted separating-hyperplane fault bands
# ---------------------------------------------------------------------------


@njit(cache=True)
def _count_tilted_band_numba(
    L: float,
    c: float,
    s: float,
    p: int,
    delta: float,
    b_cut: int,
    safety_gap: float,
) -> int:
    """Fast O(L + p) square counter for a general tilted strip construction."""
    step = 1.0 + safety_gap
    eps = safety_gap
    band_width = 1.0 + (p - 1) * step + 2.0 * eps
    diag_span = L * (c + s)
    if band_width > diag_span:
        return 0

    d1 = 0.5 * (diag_span - band_width) + delta
    d2 = diag_span - band_width - d1
    if d1 < 0.0 or d2 < 0.0:
        return 0

    max_idx = int(math.floor((L - 1.0) / step + 1e-12))
    if max_idx < 0:
        return 0

    # BL region
    bl_count = 0
    bl_max_x = -1.0
    bl_max_y = -1.0
    for j in range(max_idx + 1):
        y_top = 1.0 + j * step
        rem = d1 - y_top * s
        if rem < c - 1e-13:
            break
        i_lim = int(math.floor((rem - c) / (c * step) + 1e-12))
        if i_lim > max_idx:
            i_lim = max_idx
        if i_lim >= 0:
            # Verify exact inequality for edge cell due to floor eps
            while i_lim >= 0 and (1.0 + i_lim * step) * c + y_top * s > d1:
                i_lim -= 1
            if i_lim >= 0:
                bl_count += i_lim + 1
                x_top = 1.0 + i_lim * step
                if x_top > bl_max_x:
                    bl_max_x = x_top
                if y_top > bl_max_y:
                    bl_max_y = y_top

    # TR region (symmetric from (L, L))
    tr_count = 0
    tr_min_x = L + 1.0
    tr_min_y = L + 1.0
    for j in range(max_idx + 1):
        y_span = 1.0 + j * step
        rem = d2 - y_span * s
        if rem < c - 1e-13:
            break
        i_lim = int(math.floor((rem - c) / (c * step) + 1e-12))
        if i_lim > max_idx:
            i_lim = max_idx
        if i_lim >= 0:
            while i_lim >= 0 and (1.0 + i_lim * step) * c + y_span * s > d2:
                i_lim -= 1
            if i_lim >= 0:
                tr_count += i_lim + 1
                x_left = L - (1.0 + i_lim * step)
                y_bot = L - y_span
                if x_left < tr_min_x:
                    tr_min_x = x_left
                if y_bot < tr_min_y:
                    tr_min_y = y_bot

    # Optional BR and TL corner cuts with exact AABB + diagonal-strip separation
    br_count = 0
    tl_count = 0
    b_low = -1e18
    b_high = 1e18

    if b_cut > 0:
        for j in range(b_cut):
            y_lo = j * step
            y_hi = 1.0 + j * step
            if y_hi > L:
                break
            for i in range(b_cut - j):
                x_hi = L - i * step
                x_lo = x_hi - 1.0
                if x_lo < 0.0:
                    break
                # Must be disjoint from BL (which occupies <= d1 along (c,s)) and TR
                # Check exact containment in BL/TR half-planes or bounding boxes:
                # For BL: max (x*c + y*s) <= d1 -> if min (x*c + y*s) of BR cell >= d1 + eps, separated!
                # Also if x_lo >= bl_max_x + eps (when BL exists), separated!
                bl_sep = (bl_count == 0) or (x_lo * c + y_lo * s >= d1 + eps) or (x_lo >= bl_max_x + eps)
                # For TR: min (x*c + y*s) >= diag_span - d2 -> separated if max(x*c + y*s) <= diag_span - d2 - eps or y_hi <= tr_min_y - eps
                tr_sep = (tr_count == 0) or (x_hi * c + y_hi * s <= diag_span - d2 - eps) or (y_hi <= tr_min_y - eps)
                if bl_sep and tr_sep:
                    br_count += 1
                    t_max = -x_lo * s + y_hi * c
                    if t_max > b_low:
                        b_low = t_max

        for j in range(b_cut):
            y_hi = L - j * step
            y_lo = y_hi - 1.0
            if y_lo < 0.0:
                break
            for i in range(b_cut - j):
                x_lo = i * step
                x_hi = 1.0 + i * step
                if x_hi > L:
                    break
                bl_sep = (bl_count == 0) or (x_lo * c + y_lo * s >= d1 + eps) or (y_lo >= bl_max_y + eps)
                tr_sep = (tr_count == 0) or (x_hi * c + y_hi * s <= diag_span - d2 - eps) or (x_hi <= tr_min_x - eps)
                if bl_sep and tr_sep:
                    tl_count += 1
                    t_min = -x_hi * s + y_lo * c
                    if t_min < b_high:
                        b_high = t_min

        # Check BR vs TL separation if both placed
        if br_count > 0 and tl_count > 0 and b_low + eps > b_high:
            return 0

    h_t = 0.5 * (c + s)
    if L < 2.0 * h_t:
        return bl_count + tr_count + br_count + tl_count

    b_lo_lim = b_low + eps + 0.5 if br_count > 0 else -1e18
    b_hi_lim = b_high - eps - 0.5 if tl_count > 0 else 1e18

    tilted_count = 0
    for r in range(p):
        u_r = d1 + eps + 0.5 + r * step
        v1 = (u_r * c - (L - h_t)) / s
        v2 = (h_t - u_r * s) / c
        v_min = v1 if v1 > v2 else v2
        if b_lo_lim > v_min:
            v_min = b_lo_lim

        v3 = (u_r * c - h_t) / s
        v4 = (L - h_t - u_r * s) / c
        v_max = v3 if v3 < v4 else v4
        if b_hi_lim < v_max:
            v_max = b_hi_lim

        if v_max >= v_min:
            tilted_count += int(math.floor((v_max - v_min) / step + 1e-12)) + 1

    return bl_count + tr_count + br_count + tl_count + tilted_count


@njit(cache=True)
def _sweep_tilted_band_params(
    c: float,
    s: float,
    p: int,
    delta: float,
    b_cut: int,
    safety_gap: float,
    best_L: np.ndarray,
    best_params: np.ndarray,
    theta_idx: int,
) -> None:
    """Coarse sweep + bisection over L in [2.0, 32.0] updating ``best_L`` and ``best_params``."""
    L_step = 0.025
    n_steps = int((32.0 - 2.0) / L_step) + 1
    prev_count = 0
    prev_L = 2.0

    for k in range(n_steps):
        L = 2.0 + k * L_step
        cnt = _count_tilted_band_numba(L, c, s, p, delta, b_cut, safety_gap)
        if cnt > prev_count and cnt >= 2:
            lo_base = prev_L if k > 0 else 1.0
            hi_base = L
            n_start = max(2, prev_count + 1)
            n_end = min(MAX_N, cnt)
            for n in range(n_start, n_end + 1):
                ceil_n = math.ceil(math.sqrt(n))
                if lo_base >= best_L[n] or lo_base >= ceil_n:
                    continue
                lo = lo_base
                hi = hi_base
                # Check if hi already beats best_L[n]
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    if _count_tilted_band_numba(mid, c, s, p, delta, b_cut, safety_gap) >= n:
                        hi = mid
                    else:
                        lo = mid
                if hi < best_L[n] - 1e-12 and hi < ceil_n - 1e-12:
                    best_L[n] = hi
                    best_params[n, 0] = float(theta_idx)
                    best_params[n, 1] = float(p)
                    best_params[n, 2] = delta
                    best_params[n, 3] = float(b_cut)
                    best_params[n, 4] = hi
        prev_count = cnt
        prev_L = L


def _build_tilted_band(
    n_target: int,
    L: float,
    theta: float,
    p: int,
    delta: float,
    b_cut: int,
    safety_gap: float,
) -> Packing:
    """Reconstruct coordinates for a winning tilted-band configuration."""
    step = 1.0 + safety_gap
    eps = safety_gap
    c = math.cos(theta)
    s = math.sin(theta)
    band_width = 1.0 + (p - 1) * step + 2.0 * eps
    diag_span = L * (c + s)
    d1 = 0.5 * (diag_span - band_width) + delta
    d2 = diag_span - band_width - d1
    max_idx = int(math.floor((L - 1.0) / step + 1e-12))

    xs: list[float] = []
    ys: list[float] = []
    angs: list[float] = []

    bl_max_x = -1.0
    bl_max_y = -1.0
    bl_count = 0
    for j in range(max_idx + 1):
        y_top = 1.0 + j * step
        rem = d1 - y_top * s
        if rem < c - 1e-13:
            break
        i_lim = min(max_idx, int(math.floor((rem - c) / (c * step) + 1e-12)))
        while i_lim >= 0 and (1.0 + i_lim * step) * c + y_top * s > d1:
            i_lim -= 1
        if i_lim >= 0:
            bl_count += i_lim + 1
            bl_max_x = max(bl_max_x, 1.0 + i_lim * step)
            bl_max_y = max(bl_max_y, y_top)
            for i in range(i_lim + 1):
                xs.append(0.5 + i * step)
                ys.append(0.5 + j * step)
                angs.append(0.0)

    tr_min_x = L + 1.0
    tr_min_y = L + 1.0
    tr_count = 0
    for j in range(max_idx + 1):
        y_span = 1.0 + j * step
        rem = d2 - y_span * s
        if rem < c - 1e-13:
            break
        i_lim = min(max_idx, int(math.floor((rem - c) / (c * step) + 1e-12)))
        while i_lim >= 0 and (1.0 + i_lim * step) * c + y_span * s > d2:
            i_lim -= 1
        if i_lim >= 0:
            tr_count += i_lim + 1
            tr_min_x = min(tr_min_x, L - (1.0 + i_lim * step))
            tr_min_y = min(tr_min_y, L - y_span)
            for i in range(i_lim + 1):
                xs.append(L - 0.5 - i * step)
                ys.append(L - 0.5 - j * step)
                angs.append(0.0)

    br_count = 0
    tl_count = 0
    b_low = -1e18
    b_high = 1e18
    if b_cut > 0:
        for j in range(b_cut):
            y_lo = j * step
            y_hi = 1.0 + j * step
            if y_hi > L:
                break
            for i in range(b_cut - j):
                x_hi = L - i * step
                x_lo = x_hi - 1.0
                if x_lo < 0.0:
                    break
                bl_sep = (bl_count == 0) or (x_lo * c + y_lo * s >= d1 + eps) or (x_lo >= bl_max_x + eps)
                tr_sep = (tr_count == 0) or (x_hi * c + y_hi * s <= diag_span - d2 - eps) or (y_hi <= tr_min_y - eps)
                if bl_sep and tr_sep:
                    br_count += 1
                    b_low = max(b_low, -x_lo * s + y_hi * c)
                    xs.append(x_lo + 0.5)
                    ys.append(y_lo + 0.5)
                    angs.append(0.0)

        for j in range(b_cut):
            y_hi = L - j * step
            y_lo = y_hi - 1.0
            if y_lo < 0.0:
                break
            for i in range(b_cut - j):
                x_lo = i * step
                x_hi = 1.0 + i * step
                if x_hi > L:
                    break
                bl_sep = (bl_count == 0) or (x_lo * c + y_lo * s >= d1 + eps) or (y_lo >= bl_max_y + eps)
                tr_sep = (tr_count == 0) or (x_hi * c + y_hi * s <= diag_span - d2 - eps) or (x_hi <= tr_min_x - eps)
                if bl_sep and tr_sep:
                    tl_count += 1
                    b_high = min(b_high, -x_hi * s + y_lo * c)
                    xs.append(x_lo + 0.5)
                    ys.append(y_lo + 0.5)
                    angs.append(0.0)

    h_t = 0.5 * (c + s)
    b_lo_lim = b_low + eps + 0.5 if br_count > 0 else -1e18
    b_hi_lim = b_high - eps - 0.5 if tl_count > 0 else 1e18

    for r in range(p):
        u_r = d1 + eps + 0.5 + r * step
        v_min = max((u_r * c - (L - h_t)) / s, (h_t - u_r * s) / c, b_lo_lim)
        v_max = min((u_r * c - h_t) / s, (L - h_t - u_r * s) / c, b_hi_lim)
        if v_max >= v_min:
            k_row = int(math.floor((v_max - v_min) / step + 1e-12)) + 1
            for m in range(k_row):
                if len(xs) >= n_target:
                    break
                v = v_min + m * step
                xc = u_r * c - v * s
                yc = u_r * s + v * c
                xs.append(xc)
                ys.append(yc)
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
        source=f"tilted_band(th={theta:.4f},p={p},b={b_cut})",
    )


def _generate_candidate_angles() -> list[float]:
    angles: set[float] = set()
    for m in range(1, 11):
        for q in range(1, m + 1):
            if math.gcd(q, m) == 1:
                angles.add(round(math.atan2(q, m), 12))
    for k in range(1, 20):
        angles.add(round(0.06 + (math.pi / 4.0 - 0.06) * (k / 20.0), 12))
    return sorted(angles)


def search_tilted_bands(safety_gap: float = DEFAULT_SAFETY_GAP) -> dict[int, Packing]:
    """Search general shallow-angle tilted fault-strip packings for n=1..1000."""
    angles = _generate_candidate_angles()
    best_L = np.array([float(math.ceil(math.sqrt(n))) for n in range(MAX_N + 1)], dtype=np.float64)
    best_params = np.full((MAX_N + 1, 5), -1.0, dtype=np.float64)

    deltas = (-0.4, -0.2, 0.0, 0.2, 0.4)
    b_cuts = (0, 1, 2, 3)

    for idx, theta in enumerate(angles):
        c = math.cos(theta)
        s = math.sin(theta)
        for p in range(1, 8):
            for delta in deltas:
                for b_cut in b_cuts:
                    _sweep_tilted_band_params(
                        c, s, p, delta, b_cut, safety_gap, best_L, best_params, idx
                    )

    results: dict[int, Packing] = {}
    for n in range(2, MAX_N + 1):
        theta_idx = int(best_params[n, 0])
        if theta_idx >= 0:
            p = int(best_params[n, 1])
            delta = float(best_params[n, 2])
            b_cut = int(best_params[n, 3])
            L_val = float(best_params[n, 4])
            # Add tiny safety nudge so boundary cells survive float reconstruction cleanly
            results[n] = _build_tilted_band(
                n, L_val + 1e-11, angles[theta_idx], p, delta, b_cut, safety_gap
            )
    return results


# ---------------------------------------------------------------------------
# Method 3: L-border addition and monotone deletion closures
# ---------------------------------------------------------------------------


def apply_closures(
    packings: dict[int, Packing],
    safety_gap: float = DEFAULT_SAFETY_GAP,
    max_passes: int = 5,
) -> dict[int, Packing]:
    """Propagate record side lengths via L-border addition and monotone deletion."""
    step = 1.0 + safety_gap
    out = dict(packings)

    for _ in range(max_passes):
        changed = False

        # 1. L-border addition: extend a packing of size m by t strips along right & top
        snapshot = sorted(out.keys())
        for m in snapshot:
            base = out[m]
            L_m = base.side_length
            for t in range(1, 32):
                L_new = L_m + safety_gap + 1.0 + (t - 1) * step
                if L_new >= 32.2:
                    break
                # Columns to the right of [0, L_m]^2 covering y in [0, L_new]
                ny_col = int(math.floor((L_new - 1.0) / step + 1e-12)) + 1
                # Rows above [0, L_m]^2 covering x in [0, L_m]
                nx_row = int(math.floor((L_m - 1.0) / step + 1e-12)) + 1
                added = t * ny_col + t * nx_row
                m_new = m + added
                if m_new > MAX_N:
                    break
                if L_new < out[m_new].side_length - 1e-11:
                    new_xs: list[float] = []
                    new_ys: list[float] = []
                    for a in range(t):
                        xc = L_m + safety_gap + 0.5 + a * step
                        for b in range(ny_col):
                            new_xs.append(xc)
                            new_ys.append(0.5 + b * step)
                    for b in range(t):
                        yc = L_m + safety_gap + 0.5 + b * step
                        for a in range(nx_row):
                            new_xs.append(0.5 + a * step)
                            new_ys.append(yc)

                    xs_all = np.concatenate([base.xs, np.array(new_xs, dtype=np.float64)])
                    ys_all = np.concatenate([base.ys, np.array(new_ys, dtype=np.float64)])
                    angs_all = np.concatenate([base.angles, np.zeros(added, dtype=np.float64)])
                    xs_n, ys_n, angs_n, side = normalize_packing(xs_all, ys_all, angs_all)
                    if side < out[m_new].side_length - 1e-11:
                        out[m_new] = Packing(
                            n=m_new,
                            xs=xs_n,
                            ys=ys_n,
                            angles=angs_n,
                            side_length=side,
                            source=f"l_border({m}+{added})",
                        )
                        changed = True

            # 1b. 4-fold pinwheel border: wrap 4 symmetric arms of thickness a around core m
            for a in range(1, 16):
                arm_w = 1.0 + (a - 1) * step
                L_new = L_m + 2.0 * (arm_w + safety_gap)
                if L_new >= 32.2:
                    break
                arm_len = int(math.floor((L_new - arm_w - safety_gap - 1.0) / step + 1e-12)) + 1
                added = 4 * a * arm_len
                m_new = m + added
                if m_new > MAX_N:
                    break
                if L_new < out[m_new].side_length - 1e-11:
                    shift = arm_w + safety_gap
                    p_xs: list[float] = []
                    p_ys: list[float] = []
                    for i in range(arm_len):
                        for j in range(a):
                            p_xs.append(0.5 + i * step)
                            p_ys.append(0.5 + j * step)
                            p_xs.append(L_new - 0.5 - j * step)
                            p_ys.append(0.5 + i * step)
                            p_xs.append(L_new - 0.5 - i * step)
                            p_ys.append(L_new - 0.5 - j * step)
                            p_xs.append(0.5 + j * step)
                            p_ys.append(L_new - 0.5 - i * step)

                    xs_all = np.concatenate([base.xs + shift, np.array(p_xs, dtype=np.float64)])
                    ys_all = np.concatenate([base.ys + shift, np.array(p_ys, dtype=np.float64)])
                    angs_all = np.concatenate([base.angles, np.zeros(added, dtype=np.float64)])
                    xs_n, ys_n, angs_n, side = normalize_packing(xs_all, ys_all, angs_all)
                    if side < out[m_new].side_length - 1e-11:
                        out[m_new] = Packing(
                            n=m_new,
                            xs=xs_n,
                            ys=ys_n,
                            angles=angs_n,
                            side_length=side,
                            source=f"pinwheel({m}+4x{a}x{arm_len})",
                        )
                        changed = True

        # 2. Monotone deletion: if packing(n+1) is tighter than packing(n), drop a boundary square
        for n in range(MAX_N - 1, 0, -1):
            nxt = out[n + 1]
            cur = out[n]
            if nxt.side_length >= cur.side_length - 1e-12:
                continue

            half_ext = 0.5 * (np.abs(np.cos(nxt.angles)) + np.abs(np.sin(nxt.angles)))
            x_lo = nxt.xs - half_ext
            x_hi = nxt.xs + half_ext
            y_lo = nxt.ys - half_ext
            y_hi = nxt.ys + half_ext

            candidates = {n}
            candidates.update(int(i) for i in np.argsort(x_lo)[:3])
            candidates.update(int(i) for i in np.argsort(x_hi)[-3:])
            candidates.update(int(i) for i in np.argsort(y_lo)[:3])
            candidates.update(int(i) for i in np.argsort(y_hi)[-3:])
            best_sub: Packing | None = None
            for drop_idx in candidates:
                mask = np.ones(n + 1, dtype=bool)
                mask[drop_idx] = False
                xs_n, ys_n, angs_n, side = normalize_packing(
                    nxt.xs[mask], nxt.ys[mask], nxt.angles[mask]
                )
                if best_sub is None or side < best_sub.side_length:
                    best_sub = Packing(
                        n=n,
                        xs=xs_n,
                        ys=ys_n,
                        angles=angs_n,
                        side_length=side,
                        source=f"del({nxt.source})",
                    )
            if best_sub is not None and best_sub.side_length < cur.side_length - 1e-12:
                out[n] = best_sub
                changed = True

        if not changed:
            break

    return out

