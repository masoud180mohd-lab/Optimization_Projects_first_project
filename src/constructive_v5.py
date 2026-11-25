"""Construct unit-square packings with a grid core and independently tilted strips.

The two longest strips may overlap at a corner. Their pairwise conflicts
form a bipartite graph. A maximum matching determines the largest compatible
subset of the two strip layouts. All coordinates are generated locally.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numba import njit

from src.constructive import DEFAULT_SAFETY_GAP, MAX_N, Packing, normalize_packing
from src.validator import validate_puzzle

EPS = 1e-11
# Scale the centers only. Every contact has at least unit center separation
# on its separating axis, including pairs taken from different strip layouts.
SCALE = 1.0 + DEFAULT_SAFETY_GAP + 2e-10


@njit(cache=True)
def _end_count(width: float, height: float, c: float, s: float, v: float) -> int:
    """Uprights below v - 1/2, in columns flush with the right wall."""
    total = 0
    for i in range(int(width + EPS)):
        y_limit = (v - 0.5 + s * (width - 1 - i)) / c
        total += max(0, min(int(height + EPS), math.floor(y_limit + EPS)))
    return total


@njit(cache=True)
def _lane(
    width: float,
    height: float,
    c: float,
    s: float,
    v: float,
    i: int,
    mid: float,
    m: int,
) -> int:
    x = width - 0.5 - i
    base = max(
        0,
        min(
            int(height + EPS),
            math.floor((v - 0.5 + s * (x - 0.5)) / c + EPS),
        ),
    )
    limit = min(
        int(height + EPS),
        math.floor((v + 0.5 + s * (x - 0.5)) / c + EPS),
    )
    for j in range(base, limit):
        y = j + 0.5
        # One column of m touching squares is an oriented m by 1 rectangle.
        # Test exactly its four separating axes against this upright cell.
        free = (
            abs(c * x + s * y - mid) >= (m + c + s) / 2 - EPS
            or abs(-s * x + c * y - v) >= (1 + c + s) / 2 - EPS
            or abs(x - (c * mid - s * v)) >= (1 + m * c + s) / 2 - EPS
            or abs(y - (s * mid + c * v)) >= (1 + m * s + c) / 2 - EPS
        )
        if not free:
            break
        base += 1
    return base


@njit(cache=True)
def _end(
    width: float,
    height: float,
    c: float,
    s: float,
    v: float,
    m: int,
) -> tuple[int, float]:
    e = (c + s) / 2
    lo = max((e + s * v) / c, (e - c * v) / s) + (m - 1) / 2
    hi = min((width - e + s * v) / c, (height - e - c * v) / s) - (m - 1) / 2
    best = -1
    best_mid = (lo + hi) / 2
    for mid in (best_mid, lo, hi):
        count = 0
        for i in range(int(width + EPS)):
            count += _lane(width, height, c, s, v, i, mid, m)
        if count > best:
            best, best_mid = count, mid
    return best, best_mid


@njit(cache=True)
def _angle(
    width: float,
    height: float,
    theta: float,
    exact: bool = False,
) -> tuple[int, float, float, int, int, bool]:
    c, s = math.cos(theta), math.sin(theta)
    e = (c + s) / 2
    max_row = 1 + math.floor(min((width - 2 * e) / c, (height - 2 * e) / s) + EPS)
    best = (0, theta, 0.0, 0, 0, False)
    if max_row <= 0:
        return best
    vmin = -s * (width - e) + c * e
    vmax = c * (height - e) - s * e
    mirror = c * height - s * width
    # Changes of row capacity and of the fractional pitch at either end.
    phases = (
        0.0,
        (max_row - 1) * c * s,
        (vmax - vmin) - (max_row - 1) * c * s,
        (vmax - vmin),
        0.25,
        0.5,
        0.75,
    )
    trials = len(phases) + (2 * max_row + 12 if exact else 0)
    for idx in range(trials):
        if idx < len(phases):
            offset = phases[idx]
        elif idx < len(phases) + max_row:
            offset = (idx - len(phases)) * c * s
        elif idx < len(phases) + 2 * max_row:
            offset = vmax - vmin - (idx - len(phases) - max_row) * c * s
        else:
            offset = (idx - len(phases) - 2 * max_row) / 12
        origin = vmin + offset - math.floor(offset + EPS)
        last = int(math.floor(vmax - origin + EPS))
        prefix = 0
        start = 0
        advantage = -1000000
        for j in range(last + 1):
            v = origin + j
            # A column of m squares is feasible exactly on the interval
            # [vmin+(m-1)cs, vmax-(m-1)cs] when m fits between the walls.
            count = 1 + math.floor(min(v - vmin, vmax - v) / (c * s) + EPS)
            count = max(0, min(max_row, count))
            if count == 0:
                continue
            if exact:
                left, _ = _end(width, height, c, s, v, count)
                right, _ = _end(width, height, c, s, mirror - v, count)
            else:
                left = _end_count(width, height, c, s, v)
                right = _end_count(width, height, c, s, mirror - v)
            # With at least two columns the expanded caps are separated by
            # the plane halfway between their nearest column centers.
            total = prefix + count + advantage + right
            if total > best[0]:
                best = (total, theta, origin, start, j, exact)
            if exact:
                single = (
                    count
                    + _end_count(width, height, c, s, v)
                    + _end_count(width, height, c, s, mirror - v)
                )
            else:
                single = count + left + right
            if single > best[0]:
                best = (single, theta, origin, j, j, False)
            if left - prefix > advantage:
                advantage = left - prefix
                start = j
            prefix += count
    return best


@njit(cache=True)
def _rectangle(
    width: float,
    height: float,
    fine: bool = False,
) -> tuple[int, float, float, int, int, bool]:
    best = (int(width + EPS) * int(height + EPS), 0.0, 0.0, 0, 0, False)
    if min(width, height) < 1 or best[0] == math.floor(width * height + EPS):
        return best
    increment = 1 if fine else 4
    for degrees in range(increment, 90, increment):
        trial = _angle(width, height, math.radians(degrees))
        if trial[0] > best[0]:
            best = trial
    # Exact contact angles where another square first fits across a strip.
    for swapped in range(2):
        span = width if swapped == 0 else height
        for m in range(max(2, int(span)), int(math.hypot(width, height)) + 1):
            ratio = span / math.sqrt(m * m + 1.0)
            if ratio > 1:
                continue
            theta = math.atan2(1.0, m) + math.acos(ratio)
            if swapped:
                theta = math.pi / 2 - theta
            if 1e-6 < theta < math.pi / 2 - 1e-6:
                trial = _angle(width, height, theta)
                if trial[0] > best[0]:
                    best = trial
    if fine and best[1] > 0:
        trial = _angle(width, height, best[1], True)
        if trial[0] > best[0]:
            best = trial
    return best


def _rectangle_poses(
    width: float,
    height: float,
    fine: bool,
) -> list[tuple[float, float, float]]:
    count, theta, origin, start, last, exact = _rectangle(width, height, fine)
    if theta == 0:
        return [
            (i + 0.5, j + 0.5, 0.0)
            for i in range(int(width + EPS))
            for j in range(int(height + EPS))
        ]
    c, s = math.cos(theta), math.sin(theta)
    e = (c + s) / 2
    points: list[tuple[float, float, float]] = []
    for j in range(start, last + 1):
        v = origin + j
        low = max((e + s * v) / c, (e - c * v) / s)
        high = min((width - e + s * v) / c, (height - e - c * v) / s)
        m = math.floor(high - low + EPS) + 1
        mid = (low + high) / 2
        if exact and j == start:
            _, mid = _end(width, height, c, s, v, m)
        if exact and j == last:
            _, other = _end(width, height, c, s, c * height - s * width - v, m)
            mid = width * c + height * s - other
        for i in range(max(0, m)):
            u = mid - (m - 1) / 2 + i
            points.append((c * u - s * v, s * u + c * v, theta))
    for top, v in (
        (False, origin + start),
        (True, c * height - s * width - origin - last),
    ):
        if exact:
            lo = max((e + s * v) / c, (e - c * v) / s)
            hi = min((width - e + s * v) / c, (height - e - c * v) / s)
            m = math.floor(hi - lo + EPS) + 1
            _, mid = _end(width, height, c, s, v, m)
        for i in range(int(width + EPS)):
            rows = max(
                0,
                min(
                    int(height + EPS),
                    math.floor((v - 0.5 + s * (width - 1 - i)) / c + EPS),
                ),
            )
            if exact:
                rows = _lane(width, height, c, s, v, i, mid, m)
            for j in range(rows):
                x, y = width - 0.5 - i, 0.5 + j
                points.append((width - x, height - y, 0.0) if top else (x, y, 0.0))
    if len(points) != count:
        raise ArithmeticError(
            f"Rectangle reconstruction: {width} x {height}, {len(points)} vs {count}"
        )
    return points


def _local_rectangle(
    width: float,
    height: float,
    fine: bool = False,
) -> list[tuple[float, float, float]]:
    """Return local (x, y, angle) poses for a geometric rectangle before final dilation."""
    if width <= height:
        return _rectangle_poses(width, height, fine)
    return [
        (width - y, x, theta) for x, y, theta in _rectangle_poses(height, width, fine)
    ]


@njit(cache=True)
def _capacity(w: float, h: float, fine: bool = False) -> int:
    return _rectangle(min(w, h), max(w, h), fine)[0]


@njit(cache=True)
def _decompose(
    side: float,
    max_width: int,
    fine: bool = False,
) -> tuple[int, int, int, int]:
    k = int(side)
    best = (k * k, 0, 0, 0)
    for m in range(min(k, max_width) + 1):
        a = k - m
        w = side - a
        full = _capacity(w, side, fine)
        two = full + _capacity(a, side, fine)
        if two > best[0]:
            best = (two, 3, m, 0)
        for j in range(min(k, max_width) + 1):
            b = k - j
            top = _capacity(side - b, a, fine)
            total = full + a * b + top
            if total > best[0]:
                best = (total, 1, m, j)
            split = a * b + top + _capacity(w, b, fine) + _capacity(w, side - b, fine)
            if split > best[0]:
                best = (split, 4, m, j)
        # Four congruent arms around a central upright square.
        if 2 * w < side:
            total = 4 * _capacity(w, side - w, fine) + int(side - 2 * w + EPS) ** 2
            if total > best[0]:
                best = (total, 2, m, 0)
    return best


def _poses(
    side: float,
    recipe: tuple[int, int, int, int],
    fine: bool = False,
) -> np.ndarray:
    count, kind, m, j = recipe
    k = int(side)
    a, b = k - m, k - j
    w = side - a
    if kind in (1, 4):
        if kind == 1:
            points = [(x + a, y, t) for x, y, t in _local_rectangle(w, side, fine)]
        else:
            points = [(x + a, y, t) for x, y, t in _local_rectangle(w, b, fine)]
            points += [
                (x + a, y + b, t) for x, y, t in _local_rectangle(w, side - b, fine)
            ]
        points += [(x, y + b, t) for x, y, t in _local_rectangle(a, side - b, fine)]
        points += [(i + 0.5, h + 0.5, 0.0) for i in range(a) for h in range(b)]
    elif kind == 3:
        points = _local_rectangle(a, side, fine)
        points += [(a + x, y, t) for x, y, t in _local_rectangle(w, side, fine)]
    elif kind == 2:
        points = []
        arm = _local_rectangle(side - w, w, fine)
        for x, y, t in arm:
            points.extend(
                (
                    (x, y, t),
                    (side - y, x, t),
                    (side - x, side - y, t),
                    (y, side - x, t),
                )
            )
        core = int(side - 2 * w + EPS)
        points += [(w + i + 0.5, w + h + 0.5, 0.0) for i in range(core) for h in range(core)]
    else:
        points = [(i + 0.5, h + 0.5, 0.0) for i in range(k) for h in range(k)]
    if len(points) != count:
        raise ArithmeticError("Decomposition and reconstruction counts differ")
    return np.array(points, dtype=np.float64)


@njit(cache=True)
def _compatible_union(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum independent set of two feasible layouts, via bipartite matching."""
    nl, nr = len(left), len(right)
    cl, sl = np.cos(left[:, 2]), np.sin(left[:, 2])
    cr, sr = np.cos(right[:, 2]), np.sin(right[:, 2])
    edges = np.empty((nl, 12), dtype=np.int32)
    deg = np.zeros(nl, dtype=np.int32)
    # The corner graph is sparse. Grow the adjacency buffer when needed.
    for i in range(nl):
        for j in range(nr):
            dx, dy = left[i, 0] - right[j, 0], left[i, 1] - right[j, 1]
            if dx * dx + dy * dy >= 2.0 + 1e-10:
                continue
            h = (
                1.0
                + abs(cl[i] * cr[j] + sl[i] * sr[j])
                + abs(sl[i] * cr[j] - cl[i] * sr[j])
            ) / 2.0
            if (
                max(
                    abs(dx * cl[i] + dy * sl[i]),
                    abs(dy * cl[i] - dx * sl[i]),
                    abs(dx * cr[j] + dy * sr[j]),
                    abs(dy * cr[j] - dx * sr[j]),
                )
                < h - 1e-11
            ):
                if deg[i] == edges.shape[1]:
                    larger = np.empty((nl, 2 * edges.shape[1]), dtype=np.int32)
                    larger[:, : edges.shape[1]] = edges
                    edges = larger
                edges[i, deg[i]] = j
                deg[i] += 1
    lm = np.full(nl, -1, dtype=np.int32)
    rm = np.full(nr, -1, dtype=np.int32)
    pred = np.empty(nr, dtype=np.int32)
    seen = np.full(nl, -1, dtype=np.int32)
    visited = np.full(nr, -1, dtype=np.int32)
    queue = np.empty(nl, dtype=np.int32)
    for root in range(nl):
        if deg[root] == 0:
            continue
        queue[0] = root
        head, tail = 0, 1
        seen[root] = root
        end = -1
        while head < tail and end < 0:
            i = queue[head]
            head += 1
            for e in range(deg[i]):
                j = edges[i, e]
                if visited[j] == root:
                    continue
                visited[j] = root
                pred[j] = i
                nxt = rm[j]
                if nxt < 0:
                    end = j
                    break
                if seen[nxt] != root:
                    seen[nxt] = root
                    queue[tail] = nxt
                    tail += 1
        while end >= 0:
            i = pred[end]
            old = lm[i]
            lm[i] = end
            rm[end] = i
            end = old
    # Alternating paths from unmatched left vertices describe a minimum cover.
    keep_left = lm < 0
    keep_right = np.ones(nr, dtype=np.bool_)
    tail = 0
    for i in range(nl):
        if keep_left[i]:
            queue[tail] = i
            tail += 1
    head = 0
    while head < tail:
        i = queue[head]
        head += 1
        for e in range(deg[i]):
            j = edges[i, e]
            if lm[i] == j or not keep_right[j]:
                continue
            keep_right[j] = False
            nxt = rm[j]
            if nxt >= 0 and not keep_left[nxt]:
                keep_left[nxt] = True
                queue[tail] = nxt
                tail += 1
    return keep_left, keep_right


def _variants(width: float, side: float, fine: bool) -> list[np.ndarray]:
    """Four wall orientations of one strip, all generated from its rectangle recipe."""
    template = np.asarray(_local_rectangle(width, side, fine), dtype=np.float64).reshape(-1, 3)
    variants: list[np.ndarray] = []
    for mask in range(4):
        points = template.copy()
        if mask & 1:
            points[:, 0] = width - points[:, 0]
        if mask & 2:
            points[:, 1] = side - points[:, 1]
        if mask in (1, 2):
            points[:, 2] = np.mod(-points[:, 2], math.pi / 2)
        variants.append(points)
    return variants


def _wall_strips(side: float, m: int, fine: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    a = int(side) - m
    vertical, horizontal = [], []
    for points in _variants(side - a, side, fine):
        right = points.copy()
        right[:, 0] += a
        top = points.copy()
        top[:, 0] = side - points[:, 1]
        top[:, 1] = a + points[:, 0]
        vertical.append(right)
        horizontal.append(top)
    return vertical, horizontal


def _corner_poses(
    side: float,
    m: int,
    j: int,
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    a, b = int(side) - m, int(side) - j
    # A unit square's half-extent on either world axis is at most sqrt(2)/2.
    # Include all centers that could cross the shared corner boundary.
    near_p = p[:, 1] >= b - math.sqrt(0.5)
    near_q = q[:, 0] >= a - math.sqrt(0.5)
    keep_p, keep_q = _compatible_union(p[near_p], q[near_q])
    core = np.array(
        [(i + 0.5, h + 0.5, 0.0) for i in range(a) for h in range(b)],
        dtype=np.float64,
    ).reshape(-1, 3)
    return np.vstack(
        (core, p[~near_p], p[near_p][keep_p], q[~near_q], q[near_q][keep_q])
    )


def _best_reflections(
    a: int,
    b: int,
    m: int,
    j: int,
    vertical: list[np.ndarray],
    horizontal: list[np.ndarray],
) -> tuple[int, int, int, int, int]:
    # Every cross-strip collision has both centers in these narrow bands.
    left = [p[p[:, 1] >= b - math.sqrt(0.5)] for p in vertical]
    right = [q[q[:, 0] >= a - math.sqrt(0.5)] for q in horizontal]
    total = a * b + len(vertical[0]) + len(horizontal[0])
    best = (0, 0, 0, 0, 0)
    for u, p in enumerate(left):
        for v, q in enumerate(right):
            lp, rq = _compatible_union(p, q)
            count = total - int(np.count_nonzero(~lp)) - int(np.count_nonzero(~rq))
            if count > best[0]:
                best = (count, m, j, u, v)
    return best


def _best_seam(
    side: float,
    max_width: int,
    fine: bool,
) -> tuple[int, int, int, int, int]:
    k = int(side)
    strips = [_wall_strips(side, m, fine) for m in range(1, min(k, max_width) + 1)]
    best = (0, 0, 0, 0, 0)
    for m, (vertical, _) in enumerate(strips, 1):
        a = k - m
        for j in range(1, m + 1):
            b = k - j
            horizontal = strips[j - 1][1]
            trial = _best_reflections(a, b, m, j, vertical, horizontal)
            if trial[0] > best[0]:
                best = trial
    return best


def _seam_at(
    side: float,
    recipe: tuple[int, int, int, int, int],
    fine: bool,
    build: bool = False,
):
    _, m, j, u, v = recipe
    if int(side) < max(m, j):
        return (0, m, j, u, v)
    vertical, horizontal = _wall_strips(side, m, fine)
    if m != j:
        _, horizontal = _wall_strips(side, j, fine)
    if build:
        return _corner_poses(side, m, j, vertical[u], horizontal[v])
    return _best_reflections(int(side) - m, int(side) - j, m, j, vertical, horizontal)


@njit(cache=True)
def _gap(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    c, s = np.cos(t), np.sin(t)
    smallest = np.inf
    for i in range(len(x)):
        for j in range(i):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dx + dy * dy > 2.1:
                continue
            h = (
                1.0
                + abs(c[i] * c[j] + s[i] * s[j])
                + abs(s[i] * c[j] - c[i] * s[j])
            ) / 2.0
            distance = max(
                abs(dx * c[i] + dy * s[i]),
                abs(dy * c[i] - dx * s[i]),
                abs(dx * c[j] + dy * s[j]),
                abs(dy * c[j] - dx * s[j]),
            )
            smallest = min(smallest, distance - h)
    return smallest


def _certify(
    n: int,
    points: np.ndarray,
    source: str = "v5_rectangular_strips",
    safety_gap: float = DEFAULT_SAFETY_GAP,
    scale: float = SCALE,
    strict: bool = False,
) -> Packing | None:
    x, y, t = points[:n].T
    x, y, t, _ = normalize_packing(x * scale, y * scale, t)
    raw = tuple([format(float(z), ".16f") for z in col] for col in (x, y, t))
    gap = float(_gap(*(np.asarray(col, dtype=float) for col in raw)))
    if gap < safety_gap:
        if strict:
            raise ArithmeticError(f"Insufficient serialized clearance {gap:.12g} at n={n}")
        print(f"[WARN] v5_rectangular_strips n={n} clearance below gap: {gap:.12g}")
        return None
    res = validate_puzzle(n, x, y, t, raw_strings=raw)
    if not res.valid or res.ambiguous_pairs_checked or res.side_length is None:
        if strict:
            raise ArithmeticError(f"n={n}: {res}")
        print(f"[WARN] v5_rectangular_strips n={n} failed certification: {res.error}")
        return None
    return Packing(n, x, y, t, res.side_length, source)


@njit(cache=True)
def _fixed(side: float, kind: int, m: int, j: int, fine: bool) -> int:
    k = int(side)
    if m > k or j > k:
        return 0
    a, b = k - m, k - j
    w = side - a
    if kind == 1:
        return a * b + _capacity(w, side, fine) + _capacity(a, side - b, fine)
    if kind == 2:
        if 2 * w > side:
            return 0
        return 4 * _capacity(w, side - w, fine) + int(side - 2 * w + EPS) ** 2
    if kind == 3:
        return _capacity(a, side, fine) + _capacity(w, side, fine)
    if kind == 4:
        return (
            a * b
            + _capacity(a, side - b, fine)
            + _capacity(w, b, fine)
            + _capacity(w, side - b, fine)
        )
    return k * k


def pack_rectangle(width: float, height: float, fine: bool = True) -> np.ndarray:
    """Return gap-protected (x, y, angle in radians) rows inside the requested box."""
    if not (math.isfinite(width) and math.isfinite(height) and min(width, height) > 0):
        raise ValueError("Rectangle dimensions must be finite and positive")
    if min(width, height) < 1:
        return np.empty((0, 3), dtype=np.float64)
    # Dilation about (1/2, 1/2) preserves exact wall contact for a single
    # upright file in a rectangle whose short dimension is exactly one.
    inner_width = 1.0 + (width - 1.0) / SCALE
    inner_height = 1.0 + (height - 1.0) / SCALE
    points = np.asarray(
        _local_rectangle(inner_width, inner_height, fine), dtype=float
    ).reshape(-1, 3)
    points[:, :2] = 0.5 + SCALE * (points[:, :2] - 0.5)
    return points


def _search_v5_k_group(
    args: tuple[int, list[tuple[int, float]], int, bool, float, bool],
) -> list[tuple[int, Packing]]:
    """Top-level Windows-safe worker searching v5 rectangular strip layouts for one k-group."""
    _, group_bounds, max_width, fine, safety_gap, strict = args
    scale = 1.0 + safety_gap + 2e-10
    cache: dict[tuple[float, int, bool], tuple] = {}
    wins: list[tuple[int, Packing]] = []

    for n, bound in group_bounds:
        priority = n in (600, 700)
        width = max(max_width, 16) if priority else max_width
        use_fine = True if priority else fine
        hi = math.floor((bound / scale - 1e-7) * 100000) / 100000
        key = (hi, width, use_fine)
        if key not in cache:
            cache[key] = (
                _decompose(hi, width, use_fine),
                _best_seam(hi, width, use_fine),
            )
        direct, seam = cache[key]
        candidates = [(False, hi, direct), (True, hi, seam)]
        if priority:
            # Re-screen sharper side targets to expose alternative cuts.
            probes = [hi - delta for delta in (0.01, 0.025, 0.05, 0.075)]
            probes.append(math.floor(hi * 20) / 20 - 1e-5)
            for side in probes:
                candidates.append((True, side, _best_seam(side, width, True)))
        record = None
        for joined, start, recipe in candidates:
            if recipe[0] < n:
                continue
            lo, upper = math.sqrt(n), start
            for _ in range(15):
                mid = (lo + upper) / 2
                trial = (
                    _seam_at(mid, recipe, use_fine)
                    if joined
                    else (_fixed(mid, *recipe[1:], use_fine), *recipe[1:])
                )
                if trial[0] >= n:
                    upper = mid
                    recipe = trial
                else:
                    lo = mid
            if record is None or upper < record[0]:
                record = upper, joined, recipe
        if record is not None:
            side, joined, recipe = record
            points = (
                _seam_at(side, recipe, use_fine, True)
                if joined
                else _poses(side, recipe, use_fine)
            )
            if len(points) != recipe[0]:
                if strict:
                    raise ArithmeticError(
                        "Corner reconstruction count differs from the search"
                    )
                print(
                    f"[WARN] v5_rectangular_strips n={n} count mismatch: {len(points)} vs {recipe[0]}"
                )
                continue
            source = ("v5_matched_strips" if joined else "v5_partition") + str(
                recipe[1:]
            )
            pack = _certify(
                n,
                points,
                source=source,
                safety_gap=safety_gap,
                scale=scale,
                strict=strict,
            )
            if pack is not None and pack.side_length < bound - 1e-9:
                wins.append((n, pack))

    return wins


def search_matched_rectangular_strips(
    bounds: Mapping[int, float],
    safety_gap: float = DEFAULT_SAFETY_GAP,
    max_width: int = 10,
    fine: bool = True,
    workers: int | None = None,
    strict: bool = False,
) -> dict[int, Packing]:
    """Construct fresh layouts from scalar side bounds and return certified improvements."""
    if max_width < 1:
        raise ValueError("max_width must be positive")

    groups: dict[int, list[tuple[int, float]]] = {}
    for n, bound in sorted(bounds.items()):
        if not 1 <= n <= MAX_N or not math.isfinite(bound) or bound < math.sqrt(n) - EPS:
            raise ValueError(f"Invalid puzzle or bound: {n}, {bound}")
        k = math.isqrt(n)
        if k * k != n:
            groups.setdefault(k, []).append((n, float(bound)))

    if not groups:
        return {}

    # Warm up ALL 4 Numba entry graphs in the main process before spawning workers.
    _decompose(5.9, 3, True)
    _fixed(5.9, 1, 2, 2, True)
    _compatible_union(
        np.array([[0.5, 0.5, 0.0]], dtype=np.float64),
        np.array([[0.5, 0.5, 0.0]], dtype=np.float64),
    )
    _gap(
        np.array([0.5, 2.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )

    tasks = [
        (k, group_bounds, max_width, fine, safety_gap, strict)
        for k, group_bounds in sorted(groups.items(), reverse=True)
    ]
    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(tasks) >= 4:
        with multiprocessing.Pool(processes=num_workers) as pool:
            group_results = pool.map(_search_v5_k_group, tasks, chunksize=1)
    else:
        group_results = [_search_v5_k_group(t) for t in tasks]

    wins: dict[int, Packing] = {}
    for pairs in group_results:
        for n, p in pairs:
            wins[n] = p
    return {n: wins[n] for n in sorted(wins)}


def generate(bounds: Mapping[int, float], max_width: int = 10) -> dict[int, Packing]:
    """Construct fresh layouts from scalar side bounds and return certified improvements."""
    return search_matched_rectangular_strips(
        bounds,
        max_width=max_width,
        fine=False,
        workers=1,
        strict=True,
    )


def report(bounds: Mapping[int, float], wins: Mapping[int, Packing]) -> None:
    print("   n      incumbent L           v5 L       score gain")
    for n, p in wins.items():
        print(
            f"{n:4d}   {bounds[n]:13.9f}  {p.side_length:13.9f}   "
            f"{(bounds[n] - p.side_length) / math.sqrt(n):12.9f}"
        )
    before = math.fsum(max(0.0, L / math.sqrt(n) - 1.0) for n, L in bounds.items())
    gain = math.fsum((bounds[n] - p.side_length) / math.sqrt(n) for n, p in wins.items())
    band = [
        n
        for n in wins
        if n >= 100
        and 0.35 <= (n - math.isqrt(n) ** 2) / (2 * math.isqrt(n) + 1) <= 0.90
    ]
    band_gain = math.fsum(
        (bounds[n] - wins[n].side_length) / math.sqrt(n) for n in band
    )
    grids = sum(abs(bounds[n] - math.ceil(math.sqrt(n))) < 1e-9 for n in wins)
    print(f"n >= 100 in the 0.35-0.90 remainder band: {len(band)} wins, gain {band_gain:.9f}")
    print(f"Improvements over grid-ceiling side lengths: {grids}")
    print(
        f"All {len(wins)} replacements passed CSV-precision validation and the {DEFAULT_SAFETY_GAP:g} gap check."
    )
    print(f"{len(wins)} wins over {len(bounds)} sizes; {before:.9f} -> {before - gain:.9f}; gain {gain:.9f}")


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--v2",
        type=Path,
        help="Rebuild the constructive v2 + v3 + v4 baseline from this v2 CSV",
    )
    reference.add_argument(
        "--baseline",
        type=Path,
        help="Benchmark an existing submission CSV directly",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--n", nargs="+", type=int, help="Benchmark selected sizes")
    selection.add_argument(
        "--all", action="store_true", help="Benchmark all 1000 sizes (default)"
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.baseline is None and args.v2 is None:
        args.baseline = Path("submissions/submission_v4.csv")
    path = args.baseline or args.v2
    frame = pd.read_csv(path, usecols=["n", "x", "y", "angle"])
    bounds = {
        int(n): normalize_packing(group.x, group.y, group.angle)[3]
        for n, group in frame.groupby("n")
    }
    if args.n:
        bounds = {n: bounds[n] for n in args.n}
    elif set(bounds) != set(range(1, 1001)):
        parser.error("Full benchmark requires sizes 1 through 1000")
    if args.baseline is None:
        from src import constructive_v3 as v3
        from src import constructive_v4 as v4

        for stage_fn in (
            lambda b: v3.search_sliding_diamond_columns(b, workers=args.workers),
            lambda b: v4.search_oblique_columns(b, workers=args.workers),
        ):
            for n, p in stage_fn(bounds).items():
                bounds[n] = p.side_length
    wins = search_matched_rectangular_strips(bounds, workers=args.workers)
    print(f"Baseline: {path}" if args.baseline else f"Baseline: {path} plus constructive v3 and v4")
    report(bounds, wins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
