"""Construct a diagonal strip of independently sliding 45-degree diamond columns.

Uses rotated coordinates ``(u, v) = ((x + y) / sqrt(2), (y - x) / sqrt(2))``.
A 45-degree unit square is an axis-aligned unit box in ``(u, v)``. Columns
one unit apart in ``v`` slide independently in ``u`` between two opposite
axis-aligned corner staircases while retaining surviving single cells in the
other two corners via weighted interval scheduling DP.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from src.constructive import DEFAULT_SAFETY_GAP, MAX_N, Packing, normalize_packing
from src.validator import validate_puzzle

ROOT2 = math.sqrt(2.0)
RADIUS = (1.0 + ROOT2) / 2.0
# Work with unit contact distances, then dilate centers while keeping squares
# unit-sized. Every separating projection is at least one unit at contact.
DILATION = 1.0 + DEFAULT_SAFETY_GAP + 2e-10
TARGETS = (20, 29, 54, 55, 70, 71, 235, 500, 600, 700)


@njit(cache=True)
def _tip(height: int, v: float) -> float:
    """Farthest forbidden u for a diamond at v opposite an upright staircase."""
    if height == 0:
        return -math.inf
    # The last diagonal of the upright staircase has tips sqrt(2) apart.
    # The forbidden center region of one upright/diamond pair is an octagon.
    first = -(height - 1) / ROOT2
    index = min(height - 1, max(0, round((v - first) / ROOT2)))
    distance = abs(v - first - index * ROOT2)
    if distance >= RADIUS:
        return -math.inf
    return height / ROOT2 + min(RADIUS, ROOT2 * RADIUS - distance)


@njit(cache=True)
def _cap(v: float, side: float, limit: int) -> int:
    return max(0, min(limit, int(math.floor(side + ROOT2 * (v - 0.5) - 1 + 1e-12))))


@njit(cache=True)
def _sweep(
    side: float,
    resolution: int,
    phase: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Weighted interval scheduling on column centers, including both end caps."""
    k = int(side)
    origin = 1 - side / ROOT2 + phase / resolution
    size = max(1, int((side / ROOT2 - 1 - origin) * resolution) + 1)
    extent = np.empty((k + 1, size))
    for h in range(k + 1):
        for j in range(size):
            v = origin + j / resolution
            extent[h, j] = max(1 + abs(v), _tip(h, v))
    best_count = 0
    best = np.zeros(4, dtype=np.int64)
    best_columns = np.empty(size, dtype=np.int64)
    best_length = 0
    score = np.zeros(size, dtype=np.int64)
    take = np.empty(size, dtype=np.bool_)
    cap = np.empty(2 * k, dtype=np.int64)
    # a and d count the SW and NE diagonals. A half-turn exchanges them.
    for a in range(k + 1):
        for d in range(a, k + 1):
            if a and d and a + d + 2 > 2 * side:
                continue
            limit = 2 * k - 1
            # A BR cell (i, j) meets the BL grid first at (k-1-i, j)
            # and the TR grid at (i, k-1-j). Count the surviving cells
            # diagonal by diagonal. TL has the same counts by reflection.
            cap[0] = 0
            for level in range(limit):
                lower = max(0, level - k + 1, (level - k + d + 2) // 2)
                upper = min(level, k - 1, (level + k - 1 - a) // 2)
                cap[level + 1] = cap[level] + max(0, upper - lower + 1)
            base = (a * (a + 1) + d * (d + 1)) // 2
            for j in range(size):
                count = (
                    int(math.floor(ROOT2 * side - extent[a, j] - extent[d, j] + 1e-12))
                    + 1
                )
                v = origin + j / resolution
                b = _cap(v, side, limit)
                prefix = cap[b]
                # Start after the SE cap, or extend a previous schedule
                # whose last column is at least one full unit away.
                if j >= resolution:
                    prefix = max(prefix, score[j - resolution])
                use = prefix + count
                skip = score[j - 1] if j else 0
                take[j] = count > 0 and use > skip
                score[j] = use if take[j] else skip
                e = _cap(-v, side, limit)
                total = base + score[j] + cap[e]
                if take[j] and total > best_count:
                    length = 0
                    t = j
                    while t >= 0:
                        if not take[t]:
                            t -= 1
                            continue
                        best_columns[length] = t
                        length += 1
                        b = _cap(origin + t / resolution, side, limit)
                        if t < resolution or cap[b] >= score[t - resolution]:
                            break
                        t -= resolution
                    best_count = total
                    best[:] = (a, d, b, e)
                    best_length = length
    return best_count, best, origin + best_columns[:best_length] / resolution


def _build(side: float, heights: np.ndarray, columns: np.ndarray) -> np.ndarray:
    a, d, b, e = heights
    k = int(side)
    poses: list[tuple[float, float, float]] = []
    for height, right, top in (
        (a, False, False),
        (d, True, True),
        (b, True, False),
        (e, False, True),
    ):
        for j in range(height):
            for i in range(height - j):
                if i >= k or j >= k:
                    continue
                if right != top:
                    bl_level = k - 1 - i + j if right else i + k - 1 - j
                    tr_level = i + k - 1 - j if right else k - 1 - i + j
                    if bl_level < a or tr_level < d:
                        continue
                x, y = i + 0.5, j + 0.5
                poses.append((side - x if right else x, side - y if top else y, 0.0))
    for v in columns:
        lo = max(1 + abs(v), _tip(int(a), float(v)))
        hi = ROOT2 * side - max(1 + abs(v), _tip(int(d), float(v)))
        count = math.floor(hi - lo + 1e-12) + 1
        start = (hi + lo - count + 1) / 2
        for row in range(count):
            u = start + row
            poses.append(((u - v) / ROOT2, (u + v) / ROOT2, math.pi / 4))
    return np.asarray(poses, dtype=np.float64)


def _packing(poses: np.ndarray, n: int, dilation: float = DILATION) -> Packing:
    xs, ys, angles = poses[:n].T
    x, y, ang_n, side = normalize_packing(xs * dilation, ys * dilation, angles)
    return Packing(n, x, y, ang_n, side, "sliding_diamond_columns")


def _certify(packing: Packing, strict: bool = False) -> Packing | None:
    coords = (packing.xs, packing.ys, packing.angles)
    raw = tuple([format(float(z), ".16f") for z in column] for column in coords)
    result = validate_puzzle(packing.n, *coords, raw_strings=raw)
    if not result.valid or result.ambiguous_pairs_checked or result.side_length is None:
        if strict:
            raise ArithmeticError(f"n={packing.n}: {result}")
        print(
            f"[WARN] sliding_diamond_columns n={packing.n} failed certification: {result.error}"
        )
        return None
    packing.side_length = result.side_length
    return packing


def _search_sliding_single(
    args: tuple[int, float, float, int, int, int, bool],
) -> tuple[int, Packing | None]:
    """Top-level Windows-safe worker running bisection over sliding diamond columns for one n."""
    n, bound, dilation, resolution, phases, iterations, strict = args
    hi = bound / dilation - 1e-8
    lo = math.sqrt(n)
    winner: Packing | None = None
    for it in range(iterations + 1):
        side = hi if it == 0 else 0.5 * (lo + hi)
        for p in range(phases):
            count, heights, columns = _sweep(side, resolution, p / phases)
            if count >= n:
                poses = _build(side, heights, columns)
                if len(poses) != count:
                    raise ArithmeticError(
                        "Column schedule and reconstructed layout disagree"
                    )
                candidate = _packing(poses, n, dilation=dilation)
                if winner is None or candidate.side_length < winner.side_length:
                    winner = candidate
                hi = side
                break
        else:
            if it == 0:
                break
            lo = side
    if winner is not None and winner.side_length < bound - 1e-9:
        certified = _certify(winner, strict=strict)
        if certified is not None and certified.side_length < bound - 1e-9:
            return n, certified
    return n, None


def search_sliding_diamond_columns(
    bounds: Mapping[int, float],
    safety_gap: float = DEFAULT_SAFETY_GAP,
    resolution: int = 64,
    phases: int = 4,
    iterations: int = 16,
    workers: int | None = None,
    strict: bool = False,
) -> dict[int, Packing]:
    """Return certified sliding-diamond-column improvements for requested ``n -> bound``."""
    if resolution < 1 or phases < 1 or iterations < 0:
        raise ValueError(
            "Require positive resolution and phases, and nonnegative iterations"
        )

    dilation = 1.0 + safety_gap + 2e-10
    tasks: list[tuple[int, float, float, int, int, int, bool]] = []
    for n, bound in bounds.items():
        if not 1 <= n <= MAX_N or not math.isfinite(bound) or bound < math.sqrt(n) - 1e-9:
            raise ValueError(
                f"Invalid puzzle size or side bound: n={n}, side={bound}"
            )
        if math.isqrt(n) ** 2 == n:
            continue
        tasks.append((n, float(bound), dilation, resolution, phases, iterations, strict))

    if not tasks:
        return {}

    # Warm up Numba JIT cache in the main process before spawning worker processes.
    _sweep(3.8, 16, 0.0)

    tasks.sort(key=lambda t: t[0], reverse=True)
    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(tasks) >= 16:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_search_sliding_single, tasks, chunksize=4)
    else:
        results = [_search_sliding_single(t) for t in tasks]

    wins = {n: p for n, p in results if p is not None}
    return {n: wins[n] for n in sorted(wins)}


def generate(
    bounds: Mapping[int, float],
    resolution: int = 64,
    phases: int = 4,
    iterations: int = 12,
) -> dict[int, Packing]:
    """Return certified improvements for the requested n -> incumbent L mapping."""
    return search_sliding_diamond_columns(
        bounds,
        resolution=resolution,
        phases=phases,
        iterations=iterations,
        workers=1,
        strict=True,
    )


def baseline_sides(path: Path) -> dict[int, float]:
    """Load scalar side lengths of each incumbent puzzle from a CSV."""
    sides: dict[int, float] = {}
    df = pd.read_csv(path, usecols=["n", "x", "y", "angle"])
    for n, group in df.groupby("n"):
        _, _, _, sides[int(n)] = normalize_packing(group.x, group.y, group.angle)
    return sides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=Path("submissions/submission_v2.csv"))
    parser.add_argument("--n", nargs="+", type=int, default=TARGETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    sides = baseline_sides(args.v2)
    bounds = sides if args.all else {n: sides[n] for n in args.n}
    wins = search_sliding_diamond_columns(
        bounds,
        resolution=args.resolution,
        phases=args.phases,
        workers=args.workers,
    )
    total = 0.0
    print("   n       v2 side       v3 side    score gain")
    for n, pack in wins.items():
        gain = (bounds[n] - pack.side_length) / math.sqrt(n)
        print(f"{n:4d}  {bounds[n]:12.8f}  {pack.side_length:12.8f}  {gain:.9f}")
        total += gain
    print(f"{len(wins)} wins across {len(bounds)} puzzles; score reduction {total:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
