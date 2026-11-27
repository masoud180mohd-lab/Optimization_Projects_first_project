"""Four tilted boundary strips around an upright grid for Squares in Squares.

The four wall states are chosen together. A bipartite maximum matching then
selects compatible squares from both opposing pairs of walls in one graph.
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
from src.constructive_v5 import (
    SCALE,
    _angle,
    _certify,
    _compatible_union,
    _gap as v5_gap,
    _variants,
)

EPS = 1e-11


def _rotate(points: np.ndarray, side: float, quarter_turns: int) -> np.ndarray:
    result = points.copy()
    for _ in range(quarter_turns):
        x = result[:, 0].copy()
        result[:, 0] = side - result[:, 1]
        result[:, 1] = x
    return result


def _reflections(
    points: np.ndarray,
    width: float,
    side: float,
) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for mask in range(4):
        p = points.copy()
        if mask & 1:
            p[:, 0] = width - p[:, 0]
        if mask & 2:
            p[:, 1] = side - p[:, 1]
        if mask in (1, 2):
            p[:, 2] = np.mod(-p[:, 2], math.pi / 2)
        variants.append(p)
    return variants


def _sheared(
    width: float,
    side: float,
    angle: float,
    phase: float,
) -> np.ndarray:
    """World-vertical files with staggered y phases."""
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    radius = (c + s) / 2
    files = 1 + math.floor((width - 2 * radius) / c + EPS)
    pitch = 1 / c
    y_slack = side - 2 * radius - (math.floor(side) - 1) * pitch
    out: list[tuple[float, float, float]] = []
    for i in range(files):
        y = radius + phase * y_slack + i * s
        y -= math.floor((y - radius + EPS) / pitch) * pitch
        for j in range(max(0, 1 + math.floor((side - radius - y) / pitch + EPS))):
            out.append((radius + i * c, y + j * pitch, theta))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


@njit(cache=True)
def _band(
    width: float,
    side: float,
    recipe: tuple[int, float, float, int, int, bool],
    align: float,
) -> np.ndarray:
    """Slide columns along u. End caps clear every column by separation in v."""
    count, theta, origin, first, last, _ = recipe
    c, s = math.cos(theta), math.sin(theta)
    e = (c + s) / 2
    points = np.empty((count, 3))
    used = 0
    for j in range(first, last + 1):
        v = origin + j
        low = max((e + s * v) / c, (e - c * v) / s)
        high = min((width - e + s * v) / c, (side - e - c * v) / s)
        rows = 1 + math.floor(high - low + EPS)
        start = low + align * max(0.0, high - low - rows + 1)
        for i in range(rows):
            u = start + i
            points[used, 0], points[used, 1], points[used, 2] = (
                c * u - s * v,
                s * u + c * v,
                theta,
            )
            used += 1
    for top in range(2):
        v = origin + first if top == 0 else c * side - s * width - origin - last
        for i in range(int(width + EPS)):
            rows = max(
                0,
                min(
                    int(side + EPS),
                    math.floor((v - 0.5 + s * (width - 1 - i)) / c + EPS),
                ),
            )
            for j in range(rows):
                x, y = width - 0.5 - i, 0.5 + j
                points[used, 0] = width - x if top else x
                points[used, 1] = side - y if top else y
                points[used, 2] = 0.0
                used += 1
    if used != count:
        raise ArithmeticError(
            "Tilted band count differs from its geometric construction"
        )
    return points


def _wall_library(
    side: float,
    transfer: float,
    integers,
    primary: bool,
    rich: bool,
):
    layouts, widths, labels = [], [], []
    fraction = side - math.floor(side)
    for m in integers:
        w = m + (fraction - transfer if primary else transfer)
        if w < 1:
            continue
        if primary:
            variants = _variants(w, side, True)
            if np.any(variants[0][:, 2] > 1e-8):
                layouts.append(variants)
                widths.append(w)
                labels.append(m)
            if rich:
                q = math.ceil(w)
                contact = math.atan2(1.0, q) + math.acos(w / math.hypot(q, 1))
                angles = [
                    contact + d * math.pi / 180 for d in (0.0, 0.15, 0.45, 1.0, 2.5)
                ]
                angles += [math.radians(d) for d in range(5, 86, 4)]
                candidates = sorted(
                    (_angle(w, side, t, False) for t in angles if 0 < t < math.pi / 2),
                    reverse=True,
                )
                selected: list[float] = []
                for recipe in candidates:
                    if recipe[0] < len(variants[0]) - 1:
                        break
                    if any(abs(recipe[1] - t) < 0.002 for t in selected):
                        continue
                    selected.append(recipe[1])
                    for align in (0.0, 1.0):
                        p = _band(w, side, recipe, align)
                        if len(p) and np.any(p[:, 2] > 1e-8):
                            layouts.append(_reflections(p, w, side))
                            widths.append(w)
                            labels.append(m)
                    if len(selected) >= 2:
                        break
        else:
            variants = _variants(w, side, True)
            if np.any(variants[0][:, 2] > 1e-8):
                layouts.append(variants)
                widths.append(w)
                labels.append(m)
            root = math.atan2(1.0, m)
            if w < math.hypot(m, 1):
                root -= math.acos(w / math.hypot(m, 1))
            angles = [min(0.2, math.degrees(root) * 0.8)]
            if rich:
                angles.append(min(12.0, math.degrees(root)) * 0.999)
            for deg in angles:
                for phase in (0.0, 0.5) if rich else (0.0,):
                    p = _sheared(w, side, deg, phase)
                    if len(p):
                        layouts.append(_reflections(p, w, side))
                        widths.append(w)
                        labels.append(m)
    return layouts, widths, labels


def _pack_arrays(layouts):
    sizes = np.array([len(a[0]) for a in layouts], dtype=np.int64)
    data = np.zeros((len(sizes), 4, max(sizes, default=0), 3), dtype=np.float64)
    for i, a in enumerate(layouts):
        data[i, :, : sizes[i]] = a
    return data, sizes


@njit(cache=True)
def _losses(
    side: float,
    widths: np.ndarray,
    sizes: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """Canonical left/bottom corner. A diagonal reflection transposes its cost."""
    length = len(widths)
    loss = np.zeros((length, length, 4, 4), dtype=np.int64)
    r = math.sqrt(0.5)
    for i in range(length):
        for j in range(i, length):
            for u in range(4):
                p = data[i, u, : sizes[i]]
                p = p[p[:, 1] <= widths[j] + r]
                for v in range(4):
                    q = data[j, v, : sizes[j]]
                    corner = q[q[:, 1] >= side - widths[i] - r].copy()
                    for h in range(len(corner)):
                        x, y = corner[h, 0], corner[h, 1]
                        corner[h, 0], corner[h, 1] = side - y, x
                    kp, kq = _compatible_union(p, corner)
                    cost = len(p) + len(corner) - kp.sum() - kq.sum()
                    loss[i, j, u, v] = cost
                    loss[j, i, v ^ 2, u ^ 2] = cost
    return loss


@njit(cache=True)
def _cycle(
    side: float,
    widths: np.ndarray,
    sizes: np.ndarray,
    loss: np.ndarray,
    nreturn: int,
) -> tuple[int, np.ndarray]:
    """Optimize the four states as a cycle, conditioning on opposite walls."""
    best = 0
    choices = np.zeros(8, dtype=np.int64)
    # Left and bottom are the sheared strips; right and top use deeper bands.
    for left in range(nreturn):
        for right in range(nreturn, len(widths)):
            xgap = side - widths[left] - widths[right]
            nx = round(xgap)
            if nx < 2 or abs(nx - xgap) > 1e-9:
                continue
            for bottom in range(nreturn):
                for top in range(nreturn, len(widths)):
                    ygap = side - widths[bottom] - widths[top]
                    ny = round(ygap)
                    if ny < 2 or abs(ny - ygap) > 1e-9:
                        continue
                    gross = (
                        nx * ny
                        + sizes[left]
                        + sizes[bottom]
                        + sizes[right]
                        + sizes[top]
                    )
                    for u in range(4):
                        for w in range(4):
                            cb, ct = gross, gross
                            vb, vt = 0, 0
                            for v in range(4):
                                cost = (
                                    loss[left, bottom, u, v] + loss[bottom, right, v, w]
                                )
                                if cost < cb:
                                    cb, vb = cost, v
                                cost = loss[right, top, w, v] + loss[top, left, v, u]
                                if cost < ct:
                                    ct, vt = cost, v
                            if gross - cb - ct > best:
                                best = gross - cb - ct
                                choices[:] = (left, bottom, right, top, u, vb, w, vt)
    return best, choices


def _search(side: float, transfer: float, returns, primary, rich: bool = False):
    small, sw, sm = _wall_library(side, transfer, returns, False, rich)
    large, lw, lm = _wall_library(side, transfer, primary, True, rich)
    if not small or not large:
        return 0, None
    widths = np.asarray(sw + lw, dtype=np.float64)
    data, sizes = _pack_arrays(small + large)
    losses = _losses(side, widths, sizes, data)
    count, ix = _cycle(side, widths, sizes, losses, len(small))
    if count == 0:
        return 0, None
    strips = [
        _rotate(data[i, r, : sizes[i]], side, t)
        for t, (i, r) in enumerate(zip(ix[:4], ix[4:]))
    ]
    labels = tuple((sm + lm)[i] for i in ix[:4])
    return count, (strips, widths[ix[:4]], labels)


@njit(cache=True)
def _line_schedule(
    side: float,
    width: float,
    angle: float,
    fixed: np.ndarray,
    mesh: int,
) -> np.ndarray:
    """Pack tilted columns against fixed neighboring squares and retained cap squares."""
    c, s = math.cos(angle), math.sin(angle)
    e = (c + s) / 2
    vmin = -s * (width - e) + c * e
    vmax = c * (side - e) - s * e
    cells = math.floor((vmax - vmin) * mesh + EPS) + 1
    if cells < 1 or min(width, side) < 2 * e - EPS:
        return np.empty((0, 3))
    origin = (vmin + vmax - (cells - 1) / mesh) / 2
    us = c * fixed[:, 0] + s * fixed[:, 1]
    vs = -s * fixed[:, 0] + c * fixed[:, 1]
    cc = np.cos(fixed[:, 2] - angle)
    ss = np.sin(fixed[:, 2] - angle)
    support = (1 + np.abs(cc) + np.abs(ss)) / 2
    gaps = np.empty((len(fixed) + 1, 2))
    starts = np.empty((cells, int(math.hypot(width, side)) + 2))
    counts = np.zeros(cells, dtype=np.int64)
    scores = np.zeros(cells + 1, dtype=np.int64)
    take = np.zeros(cells, dtype=np.bool_)
    for j in range(cells):
        v = origin + j / mesh
        low = max((e + s * v) / c, (e - c * v) / s)
        high = min((width - e + s * v) / c, (side - e - c * v) / s)
        sz = 0
        for i in range(len(fixed)):
            dv = v - vs[i]
            h = support[i]
            if abs(dv) >= h - EPS:
                continue
            left, right = -h, h
            for axis in range(2):
                a = cc[i] if axis == 0 else -ss[i]
                b = ss[i] if axis == 0 else cc[i]
                if abs(a) < 1e-13:
                    if abs(b * dv) >= h - EPS:
                        right = left
                    continue
                p, q = (-h - b * dv) / a, (h - b * dv) / a
                left = max(left, min(p, q))
                right = min(right, max(p, q))
            left += us[i]
            right += us[i]
            if left + EPS < right and right > low + EPS and left < high + EPS:
                gaps[sz, 0], gaps[sz, 1] = left, right
                sz += 1
        gaps[sz, 0], gaps[sz, 1] = high, high + 2
        order = np.argsort(gaps[: sz + 1, 0])
        pos = low
        for entry in order:
            while pos <= min(high, gaps[entry, 0]) + EPS:
                starts[j, counts[j]] = pos
                counts[j] += 1
                pos += 1
            pos = max(pos, gaps[entry, 1])
        predecessor = max(0, j - mesh + 1)
        if scores[predecessor] + counts[j] > scores[j]:
            take[j] = True
            scores[j + 1] = scores[predecessor] + counts[j]
        else:
            scores[j + 1] = scores[j]
    poses = np.empty((scores[-1], 3))
    written = 0
    j = cells - 1
    while j >= 0:
        if not take[j]:
            j -= 1
            continue
        v = origin + j / mesh
        for i in range(counts[j]):
            u = starts[j, i]
            poses[written, 0], poses[written, 1], poses[written, 2] = (
                c * u - s * v,
                s * u + c * v,
                angle,
            )
            written += 1
        j -= mesh
    return poses


def _joint(strips: list[np.ndarray]) -> list[np.ndarray]:
    v = np.vstack((strips[0], strips[2]))
    h = np.vstack((strips[1], strips[3]))
    kv, kh = _compatible_union(v, h)
    return [
        strips[0][kv[: len(strips[0])]],
        strips[1][kh[: len(strips[1])]],
        strips[2][kv[len(strips[0]) :]],
        strips[3][kh[len(strips[1]) :]],
    ]


def _repack(side: float, config, mesh: int = 16):
    strips, widths, labels = config
    strips = list(strips)
    kept = _joint(strips)
    for wall in (2, 3, 0, 1, 2, 3):
        width = float(widths[wall])
        turn = (-wall) % 4
        current = _rotate(kept[wall], side, turn)
        others = _rotate(
            np.vstack((kept[(wall - 1) % 4], kept[(wall + 1) % 4])), side, turn
        )
        others = others[others[:, 0] <= width + math.sqrt(0.5)]
        caps = current[current[:, 2] < 1e-9]
        angles = current[current[:, 2] > 1e-9, 2]
        if not len(angles):
            continue
        theta = float(angles[0])
        q = math.ceil(width)
        contact = math.atan2(1.0, q) + math.acos(width / math.hypot(q, 1))
        inclinations = {theta, math.pi / 2 - theta, contact, math.pi / 2 - contact}
        for d in (-2.0, -0.5, 0.5, 2.0):
            inclinations.add(theta + math.radians(d))
        if wall < 2:
            inclinations.add(math.radians(0.5))
        best = None
        target = len(current)
        for preserve in (True, False) if len(caps) else (False,):
            cap = caps if preserve else caps[:0]
            fixed = np.vstack((cap, others))
            for t in sorted(inclinations):
                if not 1e-5 < t < math.pi / 2 - 1e-5:
                    continue
                pts = _line_schedule(side, width, t, fixed, mesh)
                if len(pts) >= 2 and len(pts) + len(cap) > target:
                    target = len(pts) + len(cap)
                    best = np.vstack((pts, cap))
        if best is not None:
            strips[wall] = _rotate(best, side, wall)
            kept = _joint(strips)
    nx = round(side - widths[0] - widths[2])
    ny = round(side - widths[1] - widths[3])
    return nx * ny + sum(map(len, kept)), (strips, widths, labels)


def _materialize(
    n: int,
    side: float,
    config,
    expected: int,
    safety_gap: float = DEFAULT_SAFETY_GAP,
    scale: float = SCALE,
    strict: bool = False,
) -> Packing | None:
    strips, widths, labels = config
    # Final selection is ONE maximum independent set on all four walls.
    groups = _joint(strips)
    nx = round(side - widths[0] - widths[2])
    ny = round(side - widths[1] - widths[3])
    core = np.array(
        [
            (widths[0] + i + 0.5, widths[1] + j + 0.5, 0.0)
            for i in range(nx)
            for j in range(ny)
        ],
        dtype=np.float64,
    )
    if len(core) + sum(map(len, groups)) != expected:
        if strict:
            raise ArithmeticError(
                "Global matching disagrees with the four-corner search"
            )
        print(
            f"[WARN] v6_four_strips n={n} count mismatch: {len(core) + sum(map(len, groups))} vs {expected}"
        )
        return None
    anchors, rest = [], []
    for wall in groups:
        kept = np.flatnonzero(wall[:, 2] > 1e-8)
        if len(kept) < 2:
            return None
        anchors.extend(wall[kept[:2]])
        rest.append(np.delete(wall, kept[:2], axis=0))
    poses = np.vstack((anchors, core, *rest))[:n]
    return _certify(
        n,
        poses,
        f"v6_four_strips{labels}",
        safety_gap=safety_gap,
        scale=scale,
        strict=strict,
    )


def _search_v6_k_group(
    args: tuple[int, list[tuple[int, float]], int, float, bool],
) -> list[tuple[int, Packing]]:
    """Top-level Windows-safe worker searching 4-wall matched strips for one k-group."""
    k, group_bounds, max_width, safety_gap, strict = args
    scale = 1.0 + safety_gap + 2e-10
    primary = range(1, min(max_width, k - 3) + 1)
    cache: dict[tuple[float, float], tuple[int, object]] = {}
    wins: list[tuple[int, Packing]] = []

    for n, bound in group_bounds:
        hi = math.floor((bound / scale - 2e-6) * 1000) / 1000
        record = None
        for transfer in (0.012, 0.035):
            key = (hi, transfer)
            if key not in cache:
                cache[key] = _search(hi, transfer, (1, 2, 3), primary)
            count, config = cache[key]
            if config is None or count < n - 2:
                continue
            labels = config[2]
            returns = sorted(set(labels[:2]))
            main_widths = sorted(set(labels[2:]))
            if count < n:
                trial, other = _search(hi, transfer, returns, main_widths, True)
                if trial >= count:
                    count, config = trial, other
            if count < n:
                count, config = _repack(hi, config)
            if count >= n:
                record = hi, count, config, transfer
                break
        if record is None:
            continue
        side, count, config, transfer = record
        returns = sorted(set(config[2][:2]))
        main_widths = sorted(set(config[2][2:]))
        low = max(math.sqrt(n), side - 0.12)
        for _ in range(13):
            mid = (low + side) / 2
            trial, new_config = _search(mid, transfer, returns, main_widths, True)
            if new_config is not None and n - 2 <= trial < n:
                trial, new_config = _repack(mid, new_config)
            if trial >= n:
                side, count, config = mid, trial, new_config
            else:
                low = mid
        p = _materialize(
            n,
            side,
            config,
            count,
            safety_gap=safety_gap,
            scale=scale,
            strict=strict,
        )
        if p is not None and p.side_length < bound - 1e-9:
            wins.append((n, p))

    return wins


def search_four_wall_strips(
    bounds: Mapping[int, float],
    safety_gap: float = DEFAULT_SAFETY_GAP,
    max_width: int = 14,
    workers: int | None = None,
    strict: bool = False,
) -> dict[int, Packing]:
    """Return certified four-strip improvements using only incumbent side lengths."""
    if max_width < 2:
        raise ValueError("max_width must be at least two")

    groups: dict[int, list[tuple[int, float]]] = {}
    for n, bound in sorted(bounds.items()):
        if (
            not 1 <= n <= MAX_N
            or not math.isfinite(bound)
            or bound < math.sqrt(n) - 1e-10
        ):
            raise ValueError(f"Invalid size or side bound: {n}, {bound}")
        k = math.isqrt(n)
        if k >= 5 and n != k * k:
            groups.setdefault(k, []).append((n, float(bound)))

    if not groups:
        return {}

    # Warm up ALL Numba entry graphs in the main process before spawning workers.
    _search(6.9, 0.012, (1,), range(1, 3), True)
    _line_schedule(
        6.9,
        2.9,
        0.3,
        np.array([[0.5, 0.5, 0.0]], dtype=np.float64),
        8,
    )
    v5_gap(
        np.array([0.5, 2.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )

    tasks = [
        (k, group_bounds, max_width, safety_gap, strict)
        for k, group_bounds in sorted(groups.items(), reverse=True)
    ]
    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(tasks) >= 4:
        with multiprocessing.Pool(processes=num_workers) as pool:
            group_results = pool.map(_search_v6_k_group, tasks, chunksize=1)
    else:
        group_results = [_search_v6_k_group(t) for t in tasks]

    wins: dict[int, Packing] = {}
    for pairs in group_results:
        for n, p in pairs:
            wins[n] = p
    return {n: wins[n] for n in sorted(wins)}


def generate(bounds: Mapping[int, float], max_width: int = 14) -> dict[int, Packing]:
    """Return certified four-strip improvements using only incumbent side lengths."""
    return search_four_wall_strips(
        bounds,
        max_width=max_width,
        workers=1,
        strict=True,
    )


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=Path("submissions/submission_v5.csv")
    )
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--n", type=int, nargs="+")
    subset.add_argument(
        "--all", action="store_true", help="Benchmark every size (default)"
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.baseline, usecols=["n", "x", "y", "angle"])
    bounds = {
        int(n): normalize_packing(g.x, g.y, g.angle)[3] for n, group in frame.groupby("n")
    }
    if args.n:
        bounds = {n: bounds[n] for n in sorted(set(args.n))}
    elif set(bounds) != set(range(1, 1001)):
        parser.error("A full benchmark requires all 1000 puzzle sizes")
    wins = search_four_wall_strips(bounds, workers=args.workers)
    print(f"Baseline {args.baseline.name}")
    print("   n       v5 side       v6 side      score gain")
    for n, p in sorted(wins.items()):
        print(
            f"{n:4d}  {bounds[n]:12.9f}  {p.side_length:12.9f}  "
            f"{(bounds[n] - p.side_length) / math.sqrt(n):12.9f}"
        )
    initial = math.fsum(max(0.0, L / math.sqrt(n) - 1) for n, L in bounds.items())
    gain = math.fsum(
        (bounds[n] - p.side_length) / math.sqrt(n) for n, p in wins.items()
    )
    print(
        f"Certified replacements: {len(wins)}; required clearance: 2e-8; ambiguous pairs: 0"
    )
    print(
        f"{len(wins)} wins over {len(bounds)} sizes; {initial:.9f} -> {initial - gain:.9f}; gain {gain:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
