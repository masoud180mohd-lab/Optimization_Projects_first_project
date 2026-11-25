"""Local validator and scorer for the Kaggle Squares in Squares competition.

Mirrors the official Kaggle evaluation metric (``official_metric/squares-in-squares.ipynb``):
- Preserves exact decimal literals from CSV (avoiding float64 truncation).
- Performs a fast Numba-accelerated broad-phase AABB sort-and-sweep and float64 SAT check.
- Certifies any undecided / near-contact pairs using escalating-precision
  ``mpmath.iv`` interval arithmetic (50, 100, 200, 500 digits).
- Measures translation-invariant bounding side ``L`` at 60-digit precision and
  computes ``score(n) = max(0.0, L / sqrt(n) - 1.0)``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mpmath
import numpy as np
import pandas as pd
from numba import njit

# ---------------------------------------------------------------------------
# Constants matching the official metric
# ---------------------------------------------------------------------------

MAX_N = 1000
L_DPS = 60
L_DECIMALS = 50
ESCALATION_SEQUENCE = (50, 100, 200, 500)

MAX_LITERAL_CHARS = 100
MAX_COORD = 1e4
MAX_ANGLE = 100.0
AABB_PAD = 1e-7
SAT_UNDECIDED_TOL = 1e-8

REQUIRED_COLUMNS = ("id", "n", "square_id", "x", "y", "angle")
_DECIMAL_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

IVal = mpmath.ctx_iv.ivmpf


# ---------------------------------------------------------------------------
# Public result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PuzzleResult:
    """Validation and scoring result for a single puzzle n."""

    n: int
    valid: bool
    side_length: float | None
    score: float
    error: str | None = None
    overlap_pair: tuple[int, int] | None = None
    ambiguous_pairs_checked: int = 0
    side_length_str: str | None = None


@dataclass
class ValidationReport:
    """Aggregated validation and scoring report across a submission."""

    valid: bool
    total_score: float
    num_puzzles: int
    invalid_puzzles: list[int] = field(default_factory=list)
    puzzle_results: dict[int, PuzzleResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    schema_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_puzzle(n: int, side_length: float) -> float:
    """Compute normalized excess side length ``L / sqrt(n) - 1`` (lower is better)."""
    return max(0.0, side_length / math.sqrt(n) - 1.0)


def infeasible_score(n: int) -> float:
    """Grid-ceiling penalty assigned to missing or invalid puzzles."""
    return math.ceil(math.sqrt(n)) / math.sqrt(n) - 1.0


# ---------------------------------------------------------------------------
# Decimal / interval arithmetic helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def working_dps(dps: int):
    """Temporarily set both ``mpmath.iv.dps`` and ``mpmath.mp.dps``."""
    old_iv_dps = mpmath.iv.dps
    old_mp_dps = mpmath.mp.dps
    try:
        mpmath.iv.dps = dps
        mpmath.mp.dps = dps
        yield
    finally:
        mpmath.iv.dps = old_iv_dps
        mpmath.mp.dps = old_mp_dps


def check_decimal(s: str) -> str:
    """Return stripped decimal literal or raise ValueError if malformed."""
    stripped = s.strip()
    if len(stripped) > MAX_LITERAL_CHARS or not _DECIMAL_RE.match(stripped):
        shown = stripped if len(stripped) <= 40 else stripped[:40] + "..."
        raise ValueError(f"not a decimal number: {shown!r}")
    return stripped


def parse_value_float(s: str) -> float:
    """Parse a decimal literal to a finite float, matching the official metric."""
    cleaned = check_decimal(s)
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError(f"value out of range for a double: {cleaned!r}")
    return value


def parse_value_mp(s: str, dps: int = L_DPS) -> mpmath.mpf:
    """Parse a decimal literal into an ``mpmath.mpf`` at ``dps`` digits."""
    cleaned = check_decimal(s)
    old_dps = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps
        return mpmath.mpf(cleaned)
    finally:
        mpmath.mp.dps = old_dps


def _iv_le(a: IVal, b: IVal) -> bool | None:
    """Return True if a <= b, False if a > b, or None if intervals overlap."""
    try:
        return a <= b
    except ValueError:
        return None


def _iv_min(values: Sequence[IVal]) -> IVal:
    return mpmath.iv.mpf([min(v.a for v in values), min(v.b for v in values)])


def _iv_max(values: Sequence[IVal]) -> IVal:
    return mpmath.iv.mpf([max(v.a for v in values), max(v.b for v in values)])


def _compute_corners_interval(
    cx_iv: IVal,
    cy_iv: IVal,
    c_iv: IVal,
    s_iv: IVal,
) -> list[tuple[IVal, IVal]]:
    h = mpmath.iv.mpf("0.5")
    dx1, dy1 = h * c_iv - h * s_iv, h * s_iv + h * c_iv
    dx2, dy2 = -h * c_iv - h * s_iv, -h * s_iv + h * c_iv
    return [
        (cx_iv + dx1, cy_iv + dy1),
        (cx_iv + dx2, cy_iv + dy2),
        (cx_iv - dx1, cy_iv - dy1),
        (cx_iv - dx2, cy_iv - dy2),
    ]


def _sat_check_interval(
    corners1: list[tuple[IVal, IVal]],
    corners2: list[tuple[IVal, IVal]],
    axes: tuple[tuple[IVal, IVal], ...],
) -> bool | None:
    """Interval-arithmetic SAT test.

    Returns True if certified separated/touching, False if certified overlapping,
    or None if undecided at the current precision.
    """
    zero_iv = mpmath.iv.mpf("0")
    has_undecided = False
    for ax, ay in axes:
        projs1 = [ax * cx + ay * cy for cx, cy in corners1]
        projs2 = [ax * cx + ay * cy for cx, cy in corners2]
        min1, max1 = _iv_min(projs1), _iv_max(projs1)
        min2, max2 = _iv_min(projs2), _iv_max(projs2)
        overlap = _iv_min((max1, max2)) - _iv_max((min1, min2))
        le_result = _iv_le(overlap, zero_iv)
        if le_result is True:
            return True
        if le_result is None:
            has_undecided = True
    return None if has_undecided else False


# ---------------------------------------------------------------------------
# Fast Numba-JIT broad-phase and float64 SAT
# ---------------------------------------------------------------------------


@njit(cache=True)
def _compute_corners_and_aabbs(
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
    pad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = xs.shape[0]
    corners_x = np.empty((n, 4), dtype=np.float64)
    corners_y = np.empty((n, 4), dtype=np.float64)
    axes_cs = np.empty((n, 2), dtype=np.float64)
    aabbs = np.empty((n, 4), dtype=np.float64)

    h = 0.5
    for i in range(n):
        cx = xs[i]
        cy = ys[i]
        ang = angles[i]
        c = math.cos(ang)
        s = math.sin(ang)
        axes_cs[i, 0] = c
        axes_cs[i, 1] = s

        dx1 = h * c - h * s
        dy1 = h * s + h * c
        dx2 = -h * c - h * s
        dy2 = -h * s + h * c

        x0, y0 = cx + dx1, cy + dy1
        x1, y1 = cx + dx2, cy + dy2
        x2, y2 = cx - dx1, cy - dy1
        x3, y3 = cx - dx2, cy - dy2

        corners_x[i, 0] = x0
        corners_x[i, 1] = x1
        corners_x[i, 2] = x2
        corners_x[i, 3] = x3
        corners_y[i, 0] = y0
        corners_y[i, 1] = y1
        corners_y[i, 2] = y2
        corners_y[i, 3] = y3

        min_x = min(min(x0, x1), min(x2, x3))
        max_x = max(max(x0, x1), max(x2, x3))
        min_y = min(min(y0, y1), min(y2, y3))
        max_y = max(max(y0, y1), max(y2, y3))

        aabbs[i, 0] = min_x - pad
        aabbs[i, 1] = min_y - pad
        aabbs[i, 2] = max_x + pad
        aabbs[i, 3] = max_y + pad

    return corners_x, corners_y, axes_cs, aabbs


@njit(cache=True)
def _sat_pair_float(
    i: int,
    j: int,
    corners_x: np.ndarray,
    corners_y: np.ndarray,
    axes_cs: np.ndarray,
    undecided_tol: float,
) -> int:
    """Return 0 (separated), 1 (undecided), or -1 (overlapping)."""
    c1, s1 = axes_cs[i, 0], axes_cs[i, 1]
    c2, s2 = axes_cs[j, 0], axes_cs[j, 1]

    min_overlap = 1e300
    for k in range(4):
        if k == 0:
            ax, ay = c1, s1
        elif k == 1:
            ax, ay = -s1, c1
        elif k == 2:
            ax, ay = c2, s2
        else:
            ax, ay = -s2, c2

        min1 = 1e300
        max1 = -1e300
        min2 = 1e300
        max2 = -1e300
        for v in range(4):
            p1 = ax * corners_x[i, v] + ay * corners_y[i, v]
            if p1 < min1:
                min1 = p1
            if p1 > max1:
                max1 = p1
            p2 = ax * corners_x[j, v] + ay * corners_y[j, v]
            if p2 < min2:
                min2 = p2
            if p2 > max2:
                max2 = p2

        ov = min(max1, max2) - max(min1, min2)
        if ov < -undecided_tol:
            return 0
        if ov < min_overlap:
            min_overlap = ov

    if min_overlap > undecided_tol:
        return -1
    return 1


@njit(cache=True)
def _sweep_and_check_float(
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
    pad: float,
    undecided_tol: float,
) -> tuple[int, int, np.ndarray]:
    """Run broad-phase AABB sweep + float SAT.

    Returns ``(ov_i, ov_j, undecided_pairs)``. If ``ov_i >= 0``, an overlap
    was detected in float64 precision at ``(ov_i, ov_j)``.
    """
    n = xs.shape[0]
    if n <= 1:
        return -1, -1, np.empty((0, 2), dtype=np.int64)

    corners_x, corners_y, axes_cs, aabbs = _compute_corners_and_aabbs(xs, ys, angles, pad)
    order = np.argsort(aabbs[:, 0], kind="mergesort")

    undecided_i = []
    undecided_j = []
    # Unit squares have circumradius sqrt(2)/2; center distance^2 > 2 + eps cannot overlap.
    dist_cutoff = 2.0 + 4.0 * pad + 1e-6

    for idx_a in range(n):
        i = order[idx_a]
        ax0, ay0, ax1, ay1 = aabbs[i, 0], aabbs[i, 1], aabbs[i, 2], aabbs[i, 3]
        xi, yi = xs[i], ys[i]

        for idx_b in range(idx_a + 1, n):
            j = order[idx_b]
            if aabbs[j, 0] > ax1:
                break
            if aabbs[j, 2] < ax0 or aabbs[j, 3] < ay0 or aabbs[j, 1] > ay1:
                continue

            dx = xs[j] - xi
            dy = ys[j] - yi
            if dx * dx + dy * dy > dist_cutoff:
                continue

            status = _sat_pair_float(i, j, corners_x, corners_y, axes_cs, undecided_tol)
            if status == -1:
                p_i, p_j = (i, j) if i < j else (j, i)
                return p_i, p_j, np.empty((0, 2), dtype=np.int64)
            if status == 1:
                p_i, p_j = (i, j) if i < j else (j, i)
                undecided_i.append(p_i)
                undecided_j.append(p_j)

    m = len(undecided_i)
    out = np.empty((m, 2), dtype=np.int64)
    for k in range(m):
        out[k, 0] = undecided_i[k]
        out[k, 1] = undecided_j[k]
    return -1, -1, out


# ---------------------------------------------------------------------------
# High-precision side measurement & interval certification
# ---------------------------------------------------------------------------


def _measure_side_from_strings(
    x_strs: Sequence[str],
    y_strs: Sequence[str],
    angle_strs: Sequence[str],
    dps: int = L_DPS,
) -> tuple[mpmath.mpf, np.ndarray, np.ndarray, np.ndarray]:
    """Compute translation-invariant bounding side ``L`` at ``dps`` digits and
    return ``(L_mp, dx_float, dy_float, angle_reduced_float)`` for SAT.
    """
    n = len(x_strs)
    dx_f = np.empty(n, dtype=np.float64)
    dy_f = np.empty(n, dtype=np.float64)
    ang_f = np.empty(n, dtype=np.float64)

    old_dps = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps
        two_pi = 2 * mpmath.pi
        half = mpmath.mpf(1) / 2
        x0_mp = mpmath.mpf(x_strs[0]) if n else mpmath.mpf(0)
        y0_mp = mpmath.mpf(y_strs[0]) if n else mpmath.mpf(0)

        min_x = None
        max_x = None
        min_y = None
        max_y = None

        for i in range(n):
            xi_rel = mpmath.mpf(x_strs[i]) - x0_mp
            yi_rel = mpmath.mpf(y_strs[i]) - y0_mp
            ai_mp = mpmath.mpf(angle_strs[i])

            dx_f[i] = float(xi_rel)
            dy_f[i] = float(yi_rel)
            ang_f[i] = float(ai_mp % two_pi)

            c = mpmath.cos(ai_mp)
            s = mpmath.sin(ai_mp)
            dx1, dy1 = half * c - half * s, half * s + half * c
            dx2, dy2 = -half * c - half * s, -half * s + half * c

            xs_c = (xi_rel + dx1, xi_rel + dx2, xi_rel - dx1, xi_rel - dx2)
            ys_c = (yi_rel + dy1, yi_rel + dy2, yi_rel - dy1, yi_rel - dy2)

            c_min_x = min(xs_c)
            c_max_x = max(xs_c)
            c_min_y = min(ys_c)
            c_max_y = max(ys_c)

            if min_x is None or c_min_x < min_x:
                min_x = c_min_x
            if max_x is None or c_max_x > max_x:
                max_x = c_max_x
            if min_y is None or c_min_y < min_y:
                min_y = c_min_y
            if max_y is None or c_max_y > max_y:
                max_y = c_max_y

        if min_x is None:
            side_mp = mpmath.mpf(0)
        else:
            side_mp = max(max_x - min_x, max_y - min_y)
        return side_mp, dx_f, dy_f, ang_f
    finally:
        mpmath.mp.dps = old_dps


def _certify_undecided_pairs(
    undecided_pairs: list[tuple[int, int]],
    x_strs: Sequence[str],
    y_strs: Sequence[str],
    angle_strs: Sequence[str],
) -> tuple[bool, tuple[int, int] | None, str]:
    """Escalate ``mpmath.iv`` precision across ``ESCALATION_SEQUENCE`` to certify pairs."""
    remaining = list(undecided_pairs)

    for dps in ESCALATION_SEQUENCE:
        with working_dps(dps):
            cache: dict[int, tuple[list[tuple[IVal, IVal]], IVal, IVal]] = {}
            still_undecided: list[tuple[int, int]] = []

            for i, j in remaining:
                for idx in (i, j):
                    if idx not in cache:
                        cx_iv = mpmath.iv.mpf(x_strs[idx])
                        cy_iv = mpmath.iv.mpf(y_strs[idx])
                        a_iv = mpmath.iv.mpf(angle_strs[idx])
                        c_iv = mpmath.iv.cos(a_iv)
                        s_iv = mpmath.iv.sin(a_iv)
                        corners = _compute_corners_interval(cx_iv, cy_iv, c_iv, s_iv)
                        cache[idx] = (corners, c_iv, s_iv)

                corners_i, c1_iv, s1_iv = cache[i]
                corners_j, c2_iv, s2_iv = cache[j]
                axes = (
                    (c1_iv, s1_iv),
                    (-s1_iv, c1_iv),
                    (c2_iv, s2_iv),
                    (-s2_iv, c2_iv),
                )
                res = _sat_check_interval(corners_i, corners_j, axes)
                if res is False:
                    return False, (i, j), f"Squares {i} and {j} overlap (certified at {dps} dps)"
                if res is None:
                    still_undecided.append((i, j))

            if not still_undecided:
                return True, None, "feasible"
            remaining = still_undecided

    return False, remaining[0], "Overlap undecidable at max precision"


# ---------------------------------------------------------------------------
# Puzzle & Submission validation APIs
# ---------------------------------------------------------------------------


def validate_puzzle(
    n: int,
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    angles: Sequence[float] | np.ndarray,
    raw_strings: tuple[Sequence[str], Sequence[str], Sequence[str]] | None = None,
) -> PuzzleResult:
    """Validate and score a single puzzle ``n``.

    If ``raw_strings`` ``(x_strs, y_strs, angle_strs)`` is provided, exact decimal
    text literals are parsed directly into ``mpmath`` just as Kaggle's metric does.
    """
    if n < 1 or n > MAX_N:
        return PuzzleResult(
            n=n,
            valid=False,
            side_length=None,
            score=infeasible_score(max(1, n)),
            error=f"n={n} out of range [1, {MAX_N}]",
        )

    if len(xs) != n or len(ys) != n or len(angles) != n:
        return PuzzleResult(
            n=n,
            valid=False,
            side_length=None,
            score=infeasible_score(n),
            error=f"n={n}: expected {n} squares, got ({len(xs)}, {len(ys)}, {len(angles)})",
        )

    if raw_strings is not None:
        x_strs_raw, y_strs_raw, a_strs_raw = raw_strings
        if len(x_strs_raw) != n or len(y_strs_raw) != n or len(a_strs_raw) != n:
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: raw_strings length mismatch",
            )
        x_strs: list[str] = []
        y_strs: list[str] = []
        a_strs: list[str] = []
        try:
            for sid in range(n):
                xs_clean = check_decimal(x_strs_raw[sid])
                ys_clean = check_decimal(y_strs_raw[sid])
                as_clean = check_decimal(a_strs_raw[sid])
                xf = parse_value_float(xs_clean)
                yf = parse_value_float(ys_clean)
                af = parse_value_float(as_clean)
                if abs(xf) > MAX_COORD or abs(yf) > MAX_COORD:
                    raise ValueError(
                        f"coordinate out of bounds (|x|, |y| <= {MAX_COORD:g})"
                    )
                if abs(af) > MAX_ANGLE:
                    raise ValueError(
                        f"angle out of bounds (|angle| <= {MAX_ANGLE:g} radians)"
                    )
                x_strs.append(xs_clean)
                y_strs.append(ys_clean)
                a_strs.append(as_clean)
        except ValueError as exc:
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: {exc}",
            )
    else:
        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        ang_arr = np.asarray(angles, dtype=np.float64)
        if not (
            np.all(np.isfinite(xs_arr))
            and np.all(np.isfinite(ys_arr))
            and np.all(np.isfinite(ang_arr))
        ):
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: non-finite coordinates or angles",
            )
        if np.any(np.abs(xs_arr) > MAX_COORD) or np.any(np.abs(ys_arr) > MAX_COORD):
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: coordinate out of bounds (|x|, |y| <= {MAX_COORD:g})",
            )
        if np.any(np.abs(ang_arr) > MAX_ANGLE):
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: angle out of bounds (|angle| <= {MAX_ANGLE:g} radians)",
            )
        x_strs = [repr(float(v)) for v in xs_arr]
        y_strs = [repr(float(v)) for v in ys_arr]
        a_strs = [repr(float(v)) for v in ang_arr]

    side_mp, dx_f, dy_f, ang_f = _measure_side_from_strings(x_strs, y_strs, a_strs, dps=L_DPS)

    ov_i, ov_j, undecided_arr = _sweep_and_check_float(
        dx_f, dy_f, ang_f, AABB_PAD, SAT_UNDECIDED_TOL
    )
    if ov_i >= 0:
        return PuzzleResult(
            n=n,
            valid=False,
            side_length=None,
            score=infeasible_score(n),
            error=f"Squares {ov_i} and {ov_j} overlap (float check)",
            overlap_pair=(int(ov_i), int(ov_j)),
        )

    ambiguous_count = int(undecided_arr.shape[0])
    if ambiguous_count > 0:
        pairs = [(int(undecided_arr[k, 0]), int(undecided_arr[k, 1])) for k in range(ambiguous_count)]
        ok, bad_pair, reason = _certify_undecided_pairs(pairs, x_strs, y_strs, a_strs)
        if not ok:
            return PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=reason,
                overlap_pair=bad_pair,
                ambiguous_pairs_checked=ambiguous_count,
            )

    side_f = float(side_mp)
    lower_bound = math.sqrt(n)
    if side_f < lower_bound - 1e-7:
        return PuzzleResult(
            n=n,
            valid=False,
            side_length=None,
            score=infeasible_score(n),
            error=f"L={side_f:.6f} is below area lower bound sqrt({n})={lower_bound:.6f}",
            ambiguous_pairs_checked=ambiguous_count,
        )

    return PuzzleResult(
        n=n,
        valid=True,
        side_length=side_f,
        score=score_puzzle(n, side_f),
        error=None,
        overlap_pair=None,
        ambiguous_pairs_checked=ambiguous_count,
        side_length_str=mpmath.nstr(side_mp, n=L_DECIMALS + 5),
    )


def validate_schema(
    df: pd.DataFrame,
    expected_puzzles: Sequence[int] | None = None,
) -> list[str]:
    """Validate submission DataFrame schema, IDs, row counts, and literal formats."""
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors

    if list(df.columns) != list(REQUIRED_COLUMNS):
        errors.append(f"Column order mismatch: expected {list(REQUIRED_COLUMNS)}, got {list(df.columns)}")

    if df.isna().any().any():
        bad_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
        errors.append(f"NaN / missing values found in columns: {bad_cols}")
        return errors

    try:
        ids = pd.to_numeric(df["id"], errors="raise").to_numpy()
        ns = pd.to_numeric(df["n"], errors="raise").to_numpy()
        sids = pd.to_numeric(df["square_id"], errors="raise").to_numpy()
    except Exception as exc:
        errors.append(f"Non-integer id/n/square_id values: {exc}")
        return errors

    if not (np.issubdtype(ids.dtype, np.integer) and np.issubdtype(ns.dtype, np.integer) and np.issubdtype(sids.dtype, np.integer)):
        errors.append("Columns id, n, and square_id must contain integers")
        return errors

    target_ns = list(range(1, MAX_N + 1)) if expected_puzzles is None else sorted(set(expected_puzzles))
    expected_rows = sum(target_ns)
    if len(df) != expected_rows:
        errors.append(f"Row count mismatch: expected {expected_rows}, got {len(df)}")

    if expected_puzzles is None:
        if len(ids) == expected_rows and not np.array_equal(ids, np.arange(expected_rows)):
            errors.append("id column is not contiguous 0..N-1")
    elif len(np.unique(ids)) != len(ids):
        errors.append("Duplicate id values found")

    if np.any(ns < 1) or np.any(ns > MAX_N):
        errors.append(f"n values out of range [1, {MAX_N}]")

    counts = np.bincount(ns, minlength=MAX_N + 1)
    for n in target_ns:
        if counts[n] != n:
            errors.append(f"n={n}: expected {n} rows, got {int(counts[n])}")

    # Validate literal format on x, y, angle strings
    for col in ("x", "y", "angle"):
        series = df[col].astype(str).str.strip()
        bad_mask = (series.str.len() > MAX_LITERAL_CHARS) | (~series.str.match(_DECIMAL_RE))
        if bad_mask.any():
            first_idx = int(np.flatnonzero(bad_mask.to_numpy())[0])
            errors.append(f"Invalid decimal literal in column {col!r} at row {first_idx}: {series.iloc[first_idx]!r}")

    return errors


def _validate_puzzle_worker(args: tuple[int, list[str], list[str], list[str]]) -> PuzzleResult:
    n, x_strs, y_strs, a_strs = args
    dummy = np.empty(n, dtype=np.float64)
    return validate_puzzle(n, dummy, dummy, dummy, raw_strings=(x_strs, y_strs, a_strs))


def validate_submission(
    csv_path: str | Path,
    puzzles: Sequence[int] | None = None,
    show_progress: bool = False,
    workers: int | None = None,
) -> ValidationReport:
    """Load a CSV submission (preserving exact strings) and validate all target puzzles."""
    t0 = time.perf_counter()
    path = Path(csv_path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    target_ns = list(range(1, MAX_N + 1)) if puzzles is None else sorted(set(puzzles))
    if puzzles is not None:
        n_int = pd.to_numeric(df["n"], errors="coerce")
        df_target = df[n_int.isin(target_ns)].copy()
    else:
        df_target = df

    schema_errors = validate_schema(df_target, expected_puzzles=target_ns)

    # Group rows by puzzle n and square_id order
    grouped: dict[int, tuple[list[str], list[str], list[str]]] = {}
    if not {"n", "square_id", "x", "y", "angle"}.issubset(df_target.columns):
        return ValidationReport(
            valid=False,
            total_score=sum(infeasible_score(n) for n in target_ns),
            num_puzzles=len(target_ns),
            invalid_puzzles=list(target_ns),
            puzzle_results={},
            elapsed_seconds=time.perf_counter() - t0,
            schema_errors=schema_errors,
        )

    df_work = df_target.copy()
    df_work["_n_int"] = pd.to_numeric(df_work["n"], errors="coerce")
    df_work["_sid_int"] = pd.to_numeric(df_work["square_id"], errors="coerce")

    work_items: list[tuple[int, list[str], list[str], list[str]]] = []
    puzzle_results: dict[int, PuzzleResult] = {}

    for n_val, sub_df in df_work.groupby("_n_int", sort=True):
        if pd.isna(n_val):
            continue
        n = int(n_val)
        if n not in target_ns:
            continue
        if len(sub_df) != n or sub_df["_sid_int"].isna().any():
            puzzle_results[n] = PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: malformed row count or square_id",
            )
            continue

        sub_sorted = sub_df.sort_values("_sid_int", kind="mergesort")
        sids = sub_sorted["_sid_int"].to_numpy(dtype=np.int64)
        if not np.array_equal(sids, np.arange(n)):
            puzzle_results[n] = PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error=f"n={n}: square_ids must be 0..{n - 1}",
            )
            continue

        grouped[n] = (
            sub_sorted["x"].tolist(),
            sub_sorted["y"].tolist(),
            sub_sorted["angle"].tolist(),
        )
        work_items.append((n, *grouped[n]))

    # Missing puzzles
    for n in target_ns:
        if n not in grouped and n not in puzzle_results:
            puzzle_results[n] = PuzzleResult(
                n=n,
                valid=False,
                side_length=None,
                score=infeasible_score(n),
                error="missing from submission",
            )

    if work_items:
        # Warm up Numba JIT in the main process before spawning workers or timing
        dummy = np.array([0.0], dtype=np.float64)
        _sweep_and_check_float(dummy, dummy, dummy, AABB_PAD, SAT_UNDECIDED_TOL)

        num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
        if num_workers > 1 and len(work_items) >= 16:
            with multiprocessing.Pool(processes=num_workers) as pool:
                for idx, res in enumerate(pool.imap(_validate_puzzle_worker, work_items, chunksize=16), start=1):
                    puzzle_results[res.n] = res
                    if show_progress and (idx % 100 == 0 or idx == len(work_items)):
                        print(f"Validated {idx}/{len(work_items)} puzzles...", file=sys.stderr)
        else:
            for idx, item in enumerate(work_items, start=1):
                res = _validate_puzzle_worker(item)
                puzzle_results[res.n] = res
                if show_progress and (idx % 100 == 0 or idx == len(work_items)):
                    print(f"Validated {idx}/{len(work_items)} puzzles...", file=sys.stderr)

    invalid_puzzles = sorted(n for n, res in puzzle_results.items() if not res.valid)
    total_score = sum(puzzle_results[n].score for n in target_ns)
    all_valid = (len(schema_errors) == 0) and (len(invalid_puzzles) == 0)

    return ValidationReport(
        valid=all_valid,
        total_score=total_score,
        num_puzzles=len(target_ns),
        invalid_puzzles=invalid_puzzles,
        puzzle_results=puzzle_results,
        elapsed_seconds=time.perf_counter() - t0,
        schema_errors=schema_errors,
    )


# ---------------------------------------------------------------------------
# Parity check against official Kaggle metric notebook
# ---------------------------------------------------------------------------


def run_official_metric(
    csv_path: str | Path,
    puzzles: Sequence[int] | None = None,
    notebook_path: str | Path = Path("official_metric/squares-in-squares.ipynb"),
) -> tuple[float, dict[int, tuple[float, str | None, str]]]:
    """Run the pulled Kaggle official metric script on ``csv_path`` for parity testing.

    Returns ``(total_score, {n: (puzzle_score, L_str_or_None, reason)})``.
    """
    nb_path = Path(notebook_path)
    if not nb_path.is_file():
        raise FileNotFoundError(f"Official metric notebook not found: {nb_path}")

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    code = "\n".join(cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    target_ns = set(range(1, MAX_N + 1)) if puzzles is None else set(puzzles)
    if puzzles is not None:
        df = df[pd.to_numeric(df["n"], errors="coerce").isin(target_ns)].copy()

    sol = pd.DataFrame({"id": list(range(len(target_ns))), "n": sorted(target_ns)})

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_name = "_kaggle_official_metric_tmp"
        mod_file = Path(tmpdir) / f"{mod_name}.py"
        mod_file.write_text(code, encoding="utf-8")

        old_sys_path = list(sys.path)
        old_pythonpath = os.environ.get("PYTHONPATH")
        old_cwd = os.getcwd()
        try:
            sys.path.insert(0, tmpdir)
            os.environ["PYTHONPATH"] = (
                f"{tmpdir}{os.pathsep}{old_pythonpath}" if old_pythonpath else tmpdir
            )
            os.chdir(tmpdir)
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)

            # Evaluate directly via official helpers so we get both total and per-puzzle details
            sub = df.drop(columns=["id"]) if "id" in df.columns else df
            text = sub.to_csv(index=False)
            parsed = mod._parse_submission_csv(text)
            total, details = mod._evaluate_puzzles_parallel(parsed, target_ns)
            detail_map = {n: (sc, l_txt, reason) for n, sc, l_txt, reason in details}
            return float(total), detail_map
        finally:
            sys.path[:] = old_sys_path
            sys.modules.pop(mod_name, None)
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
            os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_puzzles_arg(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and score a Squares in Squares CSV submission.")
    parser.add_argument("csv_path", help="Path to submission CSV file")
    parser.add_argument("--puzzles", default=None, help="Optional puzzle filter, e.g. '1-20,50'")
    parser.add_argument("--top-worst", type=int, default=10, help="Show top N highest-score puzzles")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--progress", action="store_true", help="Print progress to stderr")
    parser.add_argument(
        "--compare-official",
        action="store_true",
        help="Also run official_metric/squares-in-squares.ipynb and verify exact parity",
    )
    args = parser.parse_args(argv)

    puzzles = _parse_puzzles_arg(args.puzzles)
    report = validate_submission(
        args.csv_path,
        puzzles=puzzles,
        show_progress=args.progress,
        workers=args.workers,
    )

    status = "VALID" if report.valid else "INVALID"
    print(f"Status: {status}")
    print(f"Puzzles checked: {report.num_puzzles}")
    print(f"Invalid puzzles: {len(report.invalid_puzzles)}")
    if report.schema_errors:
        print("Schema errors:")
        for err in report.schema_errors[:10]:
            print(f"  - {err}")
    if report.invalid_puzzles:
        print("First invalid puzzles:")
        for n in report.invalid_puzzles[:10]:
            print(f"  - n={n}: {report.puzzle_results[n].error}")
    print(f"Total score: {report.total_score:.12f}")
    print(f"Elapsed: {report.elapsed_seconds:.2f}s")

    if args.top_worst > 0 and report.puzzle_results:
        worst = sorted(report.puzzle_results.values(), key=lambda r: r.score, reverse=True)[: args.top_worst]
        print(f"\nTop {len(worst)} highest-cost puzzles:")
        for r in worst:
            side_disp = f"{r.side_length:.10f}" if r.side_length is not None else "INVALID"
            print(f"  n={r.n:4d} | score={r.score:.10f} | L={side_disp} | ambig_pairs={r.ambiguous_pairs_checked}")

    if args.compare_official:
        t_off = time.perf_counter()
        official_total, official_details = run_official_metric(args.csv_path, puzzles=puzzles)
        off_elapsed = time.perf_counter() - t_off
        diff = abs(report.total_score - official_total)
        mismatches = []
        for n, res in report.puzzle_results.items():
            off_sc, off_l, off_reason = official_details[n]
            off_valid = off_l is not None
            if res.valid != off_valid or abs(res.score - off_sc) > 1e-12:
                mismatches.append((n, res.valid, off_valid, res.score, off_sc, off_reason))
        print(f"\nOfficial metric score: {official_total:.12f} (in {off_elapsed:.2f}s)")
        print(f"Score difference: {diff:.3e}")
        print(f"Per-puzzle mismatches: {len(mismatches)}")
        if mismatches or diff > 1e-12:
            for m in mismatches[:10]:
                print(f"  Mismatch n={m[0]}: local_valid={m[1]} vs off_valid={m[2]}, scores=({m[3]}, {m[4]}), reason={m[5]}")
            return 1

    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
