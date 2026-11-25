"""Construct square packings from oblique columns and four upright corner grids.

In coordinates ``u = x*cos(theta) + y*sin(theta)``, ``v = -x*sin(theta) +
y*cos(theta)``, tilted unit squares occupy unit boxes. A staircase with
height ``h`` keeps upright grid cells with ``i + slope*j < h`` (up to the
roundoff guard in ``_levels``). Its slope is independent of the tilt angle.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from collections.abc import Mapping
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from numba import njit

from src.constructive import DEFAULT_SAFETY_GAP, MAX_N, Packing, normalize_packing
from src.validator import validate_puzzle

# Dilation separates unit squares by the requested gap on their separating
# axis. The small allowance also covers reconstruction and CSV rounding.
SCALE = 1.0 + DEFAULT_SAFETY_GAP + 2e-10
PRIORITY = (20, 55, 71, 500, 600, 700)


@njit(cache=True)
def _levels(i: int, j: int, slope: float) -> int:
    # First staircase height containing grid cell (i, j).
    return int(math.floor(i + slope * j + 1e-10)) + 1


@njit(cache=True)
def _profiles(
    side: float,
    theta: float,
    slope: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """SW lower endpoint for every staircase and column; the NE case is its reflection."""
    c, s = math.cos(theta), math.sin(theta)
    k = int(side)
    wall = (c + s) / 2
    support = 0.5 + wall
    vmin = -s * side + wall * (c + s)
    vmax = c * side - wall * (c + s)
    size = max(1, int(math.floor((vmax - vmin) * resolution)) + 1)
    origin = (vmin + vmax - (size - 1) / resolution) / 2
    lower = np.full((k + 1, size), -np.inf)
    counts = np.zeros(k + 1, dtype=np.int64)
    for t in range(size):
        v = origin + t / resolution
        lower[0, t] = max((wall + s * v) / c, (wall - c * v) / s)
    for i in range(k):
        for j in range(k):
            h = _levels(i, j, slope)
            if h > k:
                continue
            counts[h] += 1
            u0 = (i + 0.5) * c + (j + 0.5) * s
            v0 = -(i + 0.5) * s + (j + 0.5) * c
            left = max(0, int(math.floor((v0 - support - origin) * resolution)))
            right = min(size, int(math.ceil((v0 + support - origin) * resolution)) + 1)
            for t in range(left, right):
                dv = origin + t / resolution - v0
                if abs(dv) < support:
                    # The upright and tilted square exclude an octagon of
                    # relative center positions. Intersect its u, x and y
                    # upper bounds at this v to locate the last forbidden u.
                    du = min(support, (support + s * dv) / c, (support - c * dv) / s)
                    lower[h, t] = max(lower[h, t], u0 + du)
    for h in range(1, k + 1):
        counts[h] += counts[h - 1]
        for t in range(size):
            lower[h, t] = max(lower[h, t], lower[h - 1, t])
    return lower, counts, origin


@njit(cache=True)
def _cap_events(
    k: int,
    side: float,
    origin: float,
    c: float,
    s: float,
    slope: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First column clearing each SE cell, and the SW/NE heights excluding it."""
    first = np.empty(k * k, dtype=np.int64)
    sw = np.empty(k * k, dtype=np.int64)
    ne = np.empty(k * k, dtype=np.int64)
    for i in range(k):
        for j in range(k):
            index = i * k + j
            cut = -s * side + (i + 1) * s + (j + 1) * c + 0.5
            first[index] = max(0, math.ceil((cut - origin) * resolution - 1e-11))
            sw[index] = _levels(k - 1 - i, j, slope)
            ne[index] = _levels(i, k - 1 - j, slope)
    return first, sw, ne


@njit(cache=True)
def _sweep(
    side: float,
    theta: float,
    slope: float,
    resolution: int,
) -> tuple[int, int, int, np.ndarray]:
    """Best schedule on this mesh. The mesh is symmetric under a half-turn."""
    c, s = math.cos(theta), math.sin(theta)
    lower, counts, origin = _profiles(side, theta, slope, resolution)
    k, size = lower.shape[0] - 1, lower.shape[1]
    first, sw, ne = _cap_events(k, side, origin, c, s, slope, resolution)
    # Each entry gives the first NE height colliding with the SW staircase.
    ne_limit = np.full(k + 1, k + 1, dtype=np.int64)
    for i in range(k):
        for j in range(k):
            h = _levels(i, j, slope)
            if h <= k:
                ne_limit[h] = min(ne_limit[h], _levels(k - 1 - i, k - 1 - j, slope))
    for h in range(1, k + 1):
        ne_limit[h] = min(ne_limit[h], ne_limit[h - 1])
    left = np.empty(size, dtype=np.int64)
    right = np.empty(size, dtype=np.int64)
    scores = np.empty(size, dtype=np.int64)
    parent = np.empty(size, dtype=np.int64)
    best_cols = np.empty(size, dtype=np.int64)
    best_count, best_a, best_d, length = 0, 0, 0, 0
    for a in range(k + 1):
        for d in range(a, ne_limit[a]):
            left[:] = 0
            right[:] = 0
            for cell in range(k * k):
                t = first[cell]
                if t < size:
                    if a < sw[cell] and d < ne[cell]:
                        left[t] += 1
                    if d < sw[cell] and a < ne[cell]:
                        right[t] += 1
            for t in range(1, size):
                left[t] += left[t - 1]
                right[t] += right[t - 1]
            for t in range(size):
                count = (
                    1
                    + math.floor(
                        side * (c + s) - lower[a, t] - lower[d, size - 1 - t] + 1e-12
                    )
                )
                prefix = left[t]
                # Start at the SE cap or extend a schedule at least one unit
                # behind t. The NW cap is the half-turn of the SE lookup.
                previous = -1
                if t >= resolution and scores[t - resolution] > prefix:
                    prefix = scores[t - resolution]
                    previous = t - resolution
                skip = scores[t - 1] if t else 0
                if count <= 0 or prefix + count <= skip:
                    scores[t] = skip
                    parent[t] = -2
                    continue
                scores[t] = prefix + count
                parent[t] = previous
                total = counts[a] + counts[d] + scores[t] + right[size - 1 - t]
                if total > best_count:
                    best_count, best_a, best_d = total, a, d
                    length = 0
                    pos = t
                    while pos >= 0:
                        if parent[pos] == -2:
                            pos -= 1
                        else:
                            best_cols[length] = pos
                            length += 1
                            pos = parent[pos]
    return best_count, best_a, best_d, best_cols[:length].copy()


def _layout(
    side: float,
    degrees: float,
    slope: float,
    resolution: int,
    recipe: tuple[int, int, int, np.ndarray],
) -> np.ndarray:
    count, a, d, cols = recipe
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    lower, _, origin = _profiles(side, theta, slope, resolution)
    k, size = lower.shape[0] - 1, lower.shape[1]
    centers: list[tuple[float, float, float]] = []
    for t in cols:
        v = origin + t / resolution
        lo = lower[a, t]
        hi = side * (c + s) - lower[d, size - 1 - t]
        m = 1 + math.floor(hi - lo + 1e-12)
        start = (hi + lo - m + 1) / 2
        for row in range(m):
            u = start + row
            centers.append((u * c - v * s, u * s + v * c, theta))
    first, sw, ne = _cap_events(k, side, origin, c, s, slope, resolution)
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            x, y = i + 0.5, j + 0.5
            level = _levels(i, j, slope)
            if level <= a:
                centers.append((x, y, 0.0))
            if level <= d:
                centers.append((side - x, side - y, 0.0))
            if first[idx] <= cols[-1] and a < sw[idx] and d < ne[idx]:
                centers.append((side - x, y, 0.0))
            if first[idx] <= size - 1 - cols[0] and d < sw[idx] and a < ne[idx]:
                centers.append((x, side - y, 0.0))
    if len(centers) != count:
        raise ArithmeticError("Reconstructed packing disagrees with column count")
    return np.asarray(centers, dtype=np.float64)


@njit(cache=True)
def _minimum_near_gap(x: np.ndarray, y: np.ndarray, angles: np.ndarray) -> float:
    """Smallest separating-axis clearance among pairs close enough to touch."""
    c, s = np.cos(angles), np.sin(angles)
    gap = np.inf
    for i in range(len(x)):
        for j in range(i):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dx + dy * dy > 2.1:
                continue
            support = 0.5 * (
                1.0
                + abs(c[i] * c[j] + s[i] * s[j])
                + abs(s[i] * c[j] - c[i] * s[j])
            )
            distance = max(
                abs(dx * c[i] + dy * s[i]),
                abs(dy * c[i] - dx * s[i]),
                abs(dx * c[j] + dy * s[j]),
                abs(dy * c[j] - dx * s[j]),
            )
            gap = min(gap, distance - support)
    return gap


def _certify(
    n: int,
    poses: np.ndarray,
    degrees: float,
    safety_gap: float = DEFAULT_SAFETY_GAP,
    scale: float = SCALE,
    strict: bool = False,
) -> Packing | None:
    x, y, ang = poses[:n].T
    x, y, ang, _ = normalize_packing(scale * x, scale * y, ang)
    raw = tuple([format(float(z), ".16f") for z in col] for col in (x, y, ang))
    gap = float(_minimum_near_gap(*(np.asarray(col, dtype=float) for col in raw)))
    if gap < safety_gap:
        if strict:
            raise ArithmeticError(f"Serialized clearance {gap:.12g} at n={n}")
        print(f"[WARN] oblique_columns n={n} clearance below gap: {gap:.12g}")
        return None
    result = validate_puzzle(n, x, y, ang, raw_strings=raw)
    if not result.valid or result.ambiguous_pairs_checked or result.side_length is None:
        if strict:
            raise ArithmeticError(f"Failed certification at n={n}: {result}")
        print(f"[WARN] oblique_columns n={n} failed certification: {result.error}")
        return None
    return Packing(
        n, x, y, ang, result.side_length, f"oblique_columns({degrees:.6f}deg)"
    )


def _settings(k: int):
    """Screen both angle-aligned and independently diagonal upright fronts."""
    for degree in (9.0, 18.0, 26.565051177, 33.690067526, 40.0):
        yield degree, math.tan(math.radians(degree))
    for degree in (5.0, 9.0, 13.0, 17.0, 21.0, 26.0, 31.0, 35.0, 38.0, 41.0, 43.0, 44.5):
        yield degree, 1.0
    # Roughly one or two units of transverse drift across k squares.
    for drift in (1.0, 2.0):
        if drift < k:
            yield math.degrees(math.atan(drift / k)), 1.0


def _tighten(
    n: int,
    bound: float,
    degree: float,
    slope: float,
    resolution: int,
    sweep,
    scale: float = SCALE,
):
    """Keep a witness while tightening L; sampled feasibility can be nonmonotone."""
    hi = math.floor((bound / scale - 1e-7) * 10000) / 10000
    lo = math.sqrt(n)
    if hi <= lo:
        return None
    recipe = sweep(hi)
    if recipe[0] < n:
        return None
    for _ in range(12):
        mid = (lo + hi) / 2
        trial = sweep(mid)
        if trial[0] >= n:
            hi, recipe = mid, trial
        else:
            lo = mid
    return hi, degree, slope, resolution, recipe


def _tune_angle(n: int, record, scale: float = SCALE):
    """Probe both sides of a winning angle, retaining each feasible improvement."""
    for step in (2.0, 0.7, 0.24, 0.08):
        _, center, slope, resolution, _ = record
        mesh = max(32, resolution)
        for degree in (center - step, center + step):
            if not 0 < degree < 45:
                continue
            theta = math.radians(degree)
            sweep = partial(_sweep, theta=theta, slope=slope, resolution=mesh)
            trial = _tighten(
                n, record[0] * scale, degree, slope, mesh, sweep, scale=scale
            )
            if trial is not None:
                record = trial
    return record


def _search_oblique_k_group(
    args: tuple[int, list[tuple[int, float]], int, float, float, bool],
) -> list[tuple[int, Packing]]:
    """Top-level Windows-safe worker searching oblique-column packings for one k-group."""
    k, group_bounds, resolution, safety_gap, screen_margin, strict = args
    scale = 1.0 + safety_gap + 2e-10
    true_bounds = dict(group_bounds)
    targets = [n for n, _ in group_bounds]
    configs = list(_settings(k))
    priority = [n for n in targets if n in PRIORITY]
    extra = (
        [(d / 2.0, slope) for d in range(1, 90) for slope in (0.5, 1.0)]
        if priority
        else []
    )

    records = {}
    for index, (degree, slope) in enumerate(configs + extra):
        fine = index >= len(configs)
        mesh = max(resolution, 40) if fine else resolution
        theta = math.radians(degree)
        sweep = lru_cache(maxsize=256)(
            partial(_sweep, theta=theta, slope=slope, resolution=mesh)
        )
        for n in priority if fine else targets:
            if n in records:
                bound = records[n][0] * scale
            else:
                tb = true_bounds[n]
                eff_margin = 0.0 if tb >= float(k + 1) - 1e-9 else screen_margin
                bound = min(float(k + 1) - 1e-6, tb + eff_margin)
            record = _tighten(n, bound, degree, slope, mesh, sweep, scale=scale)
            if record is not None:
                records[n] = record
        sweep.cache_clear()

    wins: list[tuple[int, Packing]] = []
    for n, record in sorted(records.items()):
        side, degree, slope, mesh, recipe = _tune_angle(n, record, scale=scale)
        if side * scale >= true_bounds[n] + 1e-5:
            continue
        p = _certify(
            n,
            _layout(side, degree, slope, mesh, recipe),
            degree,
            safety_gap=safety_gap,
            scale=scale,
            strict=strict,
        )
        if p is not None and p.side_length < true_bounds[n] - 1e-9:
            wins.append((n, p))
    return wins


def search_oblique_columns(
    bounds: Mapping[int, float],
    safety_gap: float = DEFAULT_SAFETY_GAP,
    resolution: int = 24,
    workers: int | None = None,
    strict: bool = False,
) -> dict[int, Packing]:
    """Return certified oblique-column improvements from scalar side bounds."""
    if resolution < 1:
        raise ValueError("resolution must be positive")

    groups: dict[int, list[tuple[int, float]]] = {}
    for n, bound in sorted(bounds.items()):
        if not 1 <= n <= MAX_N or not math.isfinite(bound) or bound < math.sqrt(n) - 1e-9:
            raise ValueError(f"Invalid n or side bound: {n}, {bound}")
        k = math.isqrt(n)
        if k * k != n:
            groups.setdefault(k, []).append((n, float(bound)))

    if not groups:
        return {}

    # Warm up Numba JIT cache in the main process before spawning worker processes.
    _sweep(3.8, 0.3, 1.0, 16)
    _minimum_near_gap(
        np.array([0.5, 2.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )

    tasks = [
        (
            k,
            group_bounds,
            resolution,
            safety_gap,
            0.025 if k <= 16 else 0.008,
            strict,
        )
        for k, group_bounds in sorted(groups.items(), reverse=True)
    ]
    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(tasks) >= 4:
        with multiprocessing.Pool(processes=num_workers) as pool:
            group_results = pool.map(_search_oblique_k_group, tasks, chunksize=1)
    else:
        group_results = [_search_oblique_k_group(t) for t in tasks]

    wins: dict[int, Packing] = {}
    for pairs in group_results:
        for n, p in pairs:
            wins[n] = p
    return {n: wins[n] for n in sorted(wins)}


def generate(bounds: Mapping[int, float], resolution: int = 24) -> dict[int, Packing]:
    """Generate from scalar side bounds; return only certified strict improvements."""
    return search_oblique_columns(
        bounds,
        resolution=resolution,
        workers=1,
        strict=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=Path("submissions/submission_v2.csv"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--n", type=int, nargs="+", help="Benchmark the listed puzzle sizes")
    selection.add_argument("--all", action="store_true", help="Benchmark all 1000 sizes (default)")
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    from src import constructive_v3 as v3

    bounds = v3.baseline_sides(args.v2)
    if args.n:
        bounds = {n: bounds[n] for n in args.n}
    elif set(bounds) != set(range(1, 1001)):
        parser.error("A full benchmark requires all 1000 puzzles in the v2 CSV")
    for n, p in v3.search_sliding_diamond_columns(bounds, workers=args.workers).items():
        bounds[n] = p.side_length
    wins = search_oblique_columns(
        bounds, resolution=args.resolution, workers=args.workers
    )
    print("   n    incumbent L          v4 L    score gain    degrees")
    for n, p in wins.items():
        print(
            f"{n:4d}  {bounds[n]:13.9f}  {p.side_length:12.9f}  "
            f"{(bounds[n] - p.side_length) / math.sqrt(n):12.9f}  {np.rad2deg(p.angles.max()):9.5f}"
        )
    before = sum(max(0.0, L / math.sqrt(n) - 1.0) for n, L in bounds.items())
    gains = {n: (bounds[n] - p.side_length) / math.sqrt(n) for n, p in wins.items()}
    gain = math.fsum(gains.values())
    mid = [
        n
        for n in wins
        if 0.25 <= (n - math.isqrt(n) ** 2) / (2 * math.isqrt(n) + 1) <= 0.50
    ]
    print(f"Mid-remainder band: {len(mid)} wins, gain {math.fsum(gains[n] for n in mid):.9f}")
    print(f"All {len(wins)} replacements passed CSV-precision validation and the clearance check.")
    print(f"{len(wins)} wins over {len(bounds)} sizes; {before:.9f} -> {before - gain:.9f}; gain {gain:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
