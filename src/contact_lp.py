"""Sparse contact-constrained Sequential Linear Programming refiner."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from src.constructive import DEFAULT_SAFETY_GAP, MAX_N, Packing, normalize_packing
from src.validator import validate_puzzle

GAP = DEFAULT_SAFETY_GAP
# Covers floating-point arithmetic, the LP feasibility tolerance and CSV rounding.
PAD = 1e-10
CORNERS = 0.5 * np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
TARGETS = (11, 17, 18, 19, 20, 29, 54, 55, 69, 70, 71, 235, 500, 600, 700)


def _perp(v: np.ndarray) -> np.ndarray:
    return np.stack((-v[..., 1], v[..., 0]), axis=-1)


def _geometry(
    points: np.ndarray,
    angles: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = np.column_stack((np.cos(angles), np.sin(angles)))
    axes = np.stack((u, _perp(u)), axis=1)
    corners = CORNERS @ axes
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.empty(0, dtype=np.int64)
        return (
            corners,
            empty,
            empty,
            np.empty((0, 4, 2), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    i, j = pairs.T
    normals = np.concatenate((axes[i], axes[j]), axis=1)
    projection = np.einsum("pka,pa->pk", normals, points[j] - points[i])
    relative = angles[i] - angles[j]
    support = 0.5 * (1.0 + np.abs(np.cos(relative)) + np.abs(np.sin(relative)))
    return corners, i, j, normals, projection, support


def _certifiable(
    points: np.ndarray,
    angles: np.ndarray,
    gap: float = GAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Exact SAT center dilation for this trial. Distant pairs only move farther apart."""
    _, i, _, _, projection, support = _geometry(points, angles, math.sqrt(2.0) + 1e-6)
    if len(i) > 0:
        separation = np.max(np.abs(projection), axis=1)
        if np.any(separation < 1e-8):
            return None
        scale = float(np.max((support + gap + PAD) / separation, initial=1.0))
        points = (points - points.mean(axis=0)) * scale
    return normalize_packing(points[:, 0], points[:, 1], angles)


def _step(
    points: np.ndarray,
    angles: np.ndarray,
    side: float,
    trust: float,
    rng: np.random.Generator,
    gap: float = GAP,
) -> np.ndarray | None:
    """Minimize L with four linearized corner constraints per wall and contact."""
    n = len(angles)
    # Each center moves at most sqrt(2)*trust. The circumradius is angle independent.
    corners, i, j, normals, projection, _ = _geometry(
        points, angles, math.sqrt(2.0) * (1.0 + 2.0 * trust) + gap + PAD
    )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    bounds: list[np.ndarray] = []
    row_count = 0

    def block(rhs: np.ndarray, terms: list[tuple[np.ndarray, np.ndarray | float]]) -> None:
        nonlocal row_count
        shape = rhs.shape
        row = np.arange(row_count, row_count + rhs.size)
        for column, coefficient in terms:
            rows.append(row)
            cols.append(np.broadcast_to(column, shape).ravel())
            vals.append(np.broadcast_to(coefficient, shape).ravel())
        bounds.append(rhs.ravel())
        row_count += rhs.size

    if len(i) > 0:
        # Select a separating face. Break geometric ties reproducibly on each call.
        axes = np.argmax(
            np.abs(projection) + rng.uniform(0.0, 1e-11, projection.shape), axis=1
        )
        index = np.arange(len(i))
        reverse = axes >= 2
        owner, other = np.where(reverse, j, i), np.where(reverse, i, j)
        direction = np.where(projection[index, axes] >= 0, 1.0, -1.0) * np.where(
            reverse, -1.0, 1.0
        )
        normal = normals[index, axes] * direction[:, None]
        tangent = _perp(normal)
        offsets = points[other, None] - points[owner, None] + corners[other]
        gaps = np.einsum("pa,pka->pk", normal, offsets) - 0.5

        # -d(gap) <= gap - required clearance. The face rotates with its owner.
        p, q = (3 * owner)[:, None], (3 * other)[:, None]
        block(
            gaps - gap - PAD,
            [
                (p, normal[:, 0, None]),
                (p + 1, normal[:, 1, None]),
                (q, -normal[:, 0, None]),
                (q + 1, -normal[:, 1, None]),
                (p + 2, -np.einsum("pa,pka->pk", tangent, offsets)),
                (q + 2, -np.einsum("pa,pka->pk", normal, _perp(corners[other]))),
            ],
        )

    p_idx = 3 * np.arange(n)[:, None]
    for dim in range(2):
        coordinate = points[:, None, dim] + corners[:, :, dim]
        velocity = _perp(corners)[:, :, dim]
        block(coordinate, [(p_idx + dim, -1.0), (p_idx + 2, -velocity)])
        block(
            side - coordinate,
            [(p_idx + dim, 1.0), (p_idx + 2, velocity), (np.array(3 * n), -1.0)],
        )

    matrix = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row_count, 3 * n + 1),
    ).tocsr()
    objective = np.zeros(3 * n + 1)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.concatenate(bounds),
        bounds=[(-trust, trust)] * (3 * n) + [(math.sqrt(n) - side, None)],
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9},
    )
    if result.success:
        return result.x[:-1].reshape(n, 3)
    return None


def refine(
    incumbent: Packing,
    starts: Sequence[Packing] = (),
    steps: int = 30,
    seed: int = 42,
    safety_gap: float = GAP,
) -> Packing | None:
    """Return only a certified strict improvement. Starts are same-n original packings."""
    n = incumbent.n
    if math.isqrt(n) ** 2 == n:
        return None
    best: Packing | None = None
    threshold = incumbent.side_length
    rng = np.random.default_rng(seed + n)
    for start in (incumbent, *starts):
        if start.n != n:
            raise ValueError("All starts must contain the target number of squares")
        state = _certifiable(
            np.column_stack((start.xs, start.ys)), start.angles, gap=safety_gap
        )
        if state is None:
            continue
        trust = 0.06
        for _ in range(steps):
            x, y, angles, side = state
            points = np.column_stack((x, y))
            delta = _step(points, angles, side, trust, rng, gap=safety_gap)
            previous = side
            if delta is not None:
                for fraction in (1.0, 0.5, 0.25):
                    trial = _certifiable(
                        points + fraction * delta[:, :2],
                        angles + fraction * delta[:, 2],
                        gap=safety_gap,
                    )
                    if trial is not None and trial[3] < side - 1e-10:
                        state = trial
                        break
            if state[3] == previous:
                trust *= 0.5
                if trust < 0.0002:
                    break
            else:
                trust = min(0.12, 1.2 * trust)
        x, y, angles, side = state
        if side < threshold - 1e-9:
            literals = tuple(
                [format(float(z), ".16f") for z in arr] for arr in (x, y, angles)
            )
            result = validate_puzzle(n, x, y, angles, raw_strings=literals)
            if (
                result.valid
                and result.ambiguous_pairs_checked == 0
                and result.side_length is not None
                and result.side_length < threshold - 1e-9
            ):
                threshold = result.side_length
                best = Packing(n, x, y, angles, threshold, "v3_contact_lp")
    return best


def _refine_packing_worker(
    args: tuple[Packing, list[Packing], int, int, float],
) -> tuple[int, Packing | None]:
    incumbent, starts, steps, seed, safety_gap = args
    return incumbent.n, refine(
        incumbent,
        starts=starts,
        steps=steps,
        seed=seed,
        safety_gap=safety_gap,
    )


def refine_packings_parallel(
    packings: dict[int, Packing],
    target_ns: Sequence[int],
    steps: int = 80,
    donors: int = 2,
    seed: int = 42,
    safety_gap: float = DEFAULT_SAFETY_GAP,
    workers: int | None = None,
) -> tuple[dict[int, Packing], int]:
    """Refine selected non-square puzzles with sparse Contact-LP in parallel."""
    out = dict(packings)
    seen: set[int] = set()
    clean_ns: list[int] = []
    for raw_n in target_ns:
        n = int(raw_n)
        if n in seen or n < 2 or n > MAX_N or n not in out:
            continue
        if math.isqrt(n) ** 2 == n:
            continue
        seen.add(n)
        clean_ns.append(n)

    if not clean_ns or steps < 1:
        return out, 0

    tasks: list[tuple[Packing, list[Packing], int, int, float]] = []
    for n in sorted(clean_ns, reverse=True):
        starts: list[Packing] = []
        if donors > 0:
            for m in range(n + 1, min(MAX_N, n + donors) + 1):
                donor = out.get(m)
                if donor is None or not np.any(np.abs(donor.angles) > 1e-6):
                    continue
                order = np.argsort(
                    donor.xs + (1 if (m - n) % 2 else -1) * donor.ys, kind="stable"
                )[:n]
                x, y, a, side = normalize_packing(
                    donor.xs[order], donor.ys[order], donor.angles[order]
                )
                starts.append(Packing(n, x, y, a, side, "contact_lp_donor"))
        tasks.append((out[n], starts, steps, seed, safety_gap))

    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(tasks) >= 4:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_refine_packing_worker, tasks, chunksize=1)
    else:
        results = [_refine_packing_worker(t) for t in tasks]

    improved = 0
    for n, res in results:
        if res is not None and res.side_length < out[n].side_length - 1e-9:
            out[n] = res
            improved += 1

    return out, improved


def _load_original(path: Path, needed: set[int] | Sequence[int]) -> dict[int, Packing]:
    needed_set = set(needed)
    packings: dict[int, list[pd.DataFrame]] = {}
    coordinate_types = {col: str for col in ("x", "y", "angle")}
    for frame in pd.read_csv(path, chunksize=50000, dtype=coordinate_types):
        for n, group in frame[frame.n.isin(needed_set)].groupby("n"):
            packings.setdefault(int(n), []).append(group)
    out: dict[int, Packing] = {}
    for n in needed_set:
        if n not in packings:
            raise ValueError(f"Missing puzzle {n} in {path}")
        group = pd.concat(packings[n]).sort_values("square_id")
        if not np.array_equal(group.square_id, np.arange(n)):
            raise ValueError(f"Bad square IDs for n={n}")
        x, y, a, side = normalize_packing(group.x, group.y, group.angle)
        out[n] = Packing(n, x, y, a, side, "own_v2")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path("submissions/submission_v2.csv"),
        help="Your own algorithm-generated incumbent CSV",
    )
    parser.add_argument("--n", type=int, nargs="+", default=TARGETS)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--donors",
        type=int,
        default=2,
        help="Try one deletion start from each of n+1 through n+donors",
    )
    args = parser.parse_args()
    targets = list(dict.fromkeys(args.n))
    if args.steps < 1 or args.donors < 0 or any(n < 1 or n > 1000 for n in targets):
        parser.error("Require 1..1000 squares, positive steps, and nonnegative donors")
    needed = {m for n in targets for m in range(n, min(1000, n + args.donors) + 1)}
    original = _load_original(args.v2, needed)
    total = 0.0
    print("n     v2 side     best side    score gain     seconds", flush=True)
    for n in targets:
        started = time.perf_counter()
        starts = []
        for m in range(n + 1, min(1000, n + args.donors) + 1):
            donor = original[m]
            # Peel squares from one extremal corner of the larger original layout.
            order = np.argsort(
                donor.xs + (1 if (m - n) % 2 else -1) * donor.ys, kind="stable"
            )[:n]
            x, y, a, side = normalize_packing(
                donor.xs[order], donor.ys[order], donor.angles[order]
            )
            starts.append(Packing(n, x, y, a, side, "own_v2_deletion"))
        best = refine(original[n], starts, args.steps)
        old = original[n].side_length
        side = best.side_length if best else old
        gain = (old - side) / math.sqrt(n)
        total += gain
        print(
            f"{n:3d} {old:12.8f} {side:12.8f} {gain:13.9f} {time.perf_counter()-started:9.2f}",
            flush=True,
        )
    print(f"Target-only score reduction: {total:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
