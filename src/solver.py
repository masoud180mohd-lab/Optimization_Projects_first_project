"""End-to-end multi-stage v2 solver pipeline for the Squares in Squares competition."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.constructive import (
    DEFAULT_SAFETY_GAP,
    MAX_N,
    Packing,
    apply_closures,
    grid_packing,
    normalize_packing,
    search_gobel_45,
    search_tilted_bands,
)
from src.constructive_v2 import (
    search_4corner_slack_bands,
    search_gobel_unrestricted,
    search_offcenter_block_shells,
)
from src.optimizer import optimize_puzzles_parallel
from src.validator import score_puzzle, validate_puzzle


def total_packings_score(packings: dict[int, Packing]) -> float:
    """Compute total normalized competition score across puzzles 1..1000."""
    return sum(score_puzzle(n, packings[n].side_length) for n in range(1, MAX_N + 1))


def _is_perfect_square(n: int) -> bool:
    r = math.isqrt(n)
    return r * r == n


def _merge_if_better(
    packings: dict[int, Packing],
    candidates: dict[int, Packing],
) -> int:
    improved = 0
    for n, cand in candidates.items():
        if _is_perfect_square(n):
            continue
        if cand.side_length < packings[n].side_length - 1e-11:
            packings[n] = cand
            improved += 1
    return improved


def load_seed_csv(
    packings: dict[int, Packing],
    seed_csv: str | Path,
) -> int:
    """Load strictly improving packings from an existing verified submission CSV."""
    path = Path(seed_csv)
    if not path.is_file():
        return 0

    df = pd.read_csv(path, usecols=["n", "square_id", "x", "y", "angle"], dtype=float)
    loaded = 0
    for n_val, sub_df in df.groupby("n", sort=True):
        n = int(n_val)
        if n < 1 or n > MAX_N or len(sub_df) != n or _is_perfect_square(n):
            continue
        sub_sorted = sub_df.sort_values("square_id", kind="mergesort")
        xs = sub_sorted["x"].to_numpy(dtype=np.float64)
        ys = sub_sorted["y"].to_numpy(dtype=np.float64)
        angles = sub_sorted["angle"].to_numpy(dtype=np.float64)
        xs_n, ys_n, angs_n, side = normalize_packing(xs, ys, angles)
        if side < packings[n].side_length - 1e-11:
            packings[n] = Packing(
                n=n,
                xs=xs_n,
                ys=ys_n,
                angles=angs_n,
                side_length=side,
                source="seed_csv",
            )
            loaded += 1
    return loaded


def _verify_puzzle_worker(
    args: tuple[int, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, bool, float | None, str | None]:
    n, xs, ys, angles = args
    res = validate_puzzle(n, xs, ys, angles)
    return n, res.valid, res.side_length, res.error


def verify_all_improved(
    packings: dict[int, Packing],
    workers: int | None = None,
) -> tuple[dict[int, Packing], int]:
    """Validate every non-baseline puzzle using ``src.validator.validate_puzzle``."""
    out = dict(packings)
    to_check = [
        (n, p.xs, p.ys, p.angles)
        for n, p in sorted(out.items())
        if not p.use_raw_baseline
    ]
    if not to_check:
        return out, 0

    reverted = 0
    num_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if num_workers > 1 and len(to_check) >= 16:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_verify_puzzle_worker, to_check, chunksize=16)
    else:
        results = [_verify_puzzle_worker(item) for item in to_check]

    for n, ok, side, err in results:
        if not ok or side is None:
            print(f"[WARN] Puzzle n={n} failed verification ({err}); reverting to baseline grid.")
            out[n] = grid_packing(n)
            reverted += 1
        else:
            out[n].side_length = side

    return out, reverted


def write_submission_csv(
    packings: dict[int, Packing],
    baseline_csv: str | Path = Path("data/sample_submission.csv"),
    output_csv: str | Path = Path("submissions/submission_v2.csv"),
) -> Path:
    """Export final packings to CSV preserving exact row/ID layout from ``baseline_csv``."""
    in_path = Path(baseline_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            row_id = row[0]
            n = int(row[1])
            sid = int(row[2])
            p = packings[n]
            if p.use_raw_baseline:
                writer.writerow(row)
            else:
                writer.writerow(
                    [
                        row_id,
                        row[1],
                        row[2],
                        f"{float(p.xs[sid]):.16f}",
                        f"{float(p.ys[sid]):.16f}",
                        f"{float(p.angles[sid]):.16f}",
                    ]
                )
    return out_path


def run_pipeline(
    baseline_csv: str | Path = Path("data/sample_submission.csv"),
    output_csv: str | Path = Path("submissions/submission_v2.csv"),
    seed_csv: str | Path | None = Path("submissions/submission_v1.csv"),
    lbfgs_budget: float = 2.0,
    workers: int | None = None,
    skip_lbfgs: bool = False,
    safety_gap: float = DEFAULT_SAFETY_GAP,
) -> dict[int, Packing]:
    """Run the full multi-stage v2 solver pipeline and export a verified CSV submission."""
    t_all = time.perf_counter()

    # 1. Baseline initialization (+ optional verified seed floor)
    packings: dict[int, Packing] = {n: grid_packing(n) for n in range(1, MAX_N + 1)}
    baseline_scores = {n: score_puzzle(n, packings[n].side_length) for n in range(1, MAX_N + 1)}
    score_base = total_packings_score(packings)
    print(f"Stage 0  [Baseline Grid]:          total_score = {score_base:.6f}")

    if seed_csv:
        loaded = load_seed_csv(packings, seed_csv)
        if loaded > 0:
            print(
                f"Stage 0b [Seed CSV Floor]:         total_score = {total_packings_score(packings):.6f} "
                f"(loaded={loaded})"
            )

    # Stage 1A: Gobel 45-degree 4-corner staircase + diagonal band
    t0 = time.perf_counter()
    imp_g = _merge_if_better(packings, search_gobel_45(safety_gap=safety_gap))
    print(
        f"Stage 1A [Gobel 45-deg]:           total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_g}, dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 1B: General shallow-angle tilted bands
    t0 = time.perf_counter()
    imp_b = _merge_if_better(packings, search_tilted_bands(safety_gap=safety_gap))
    print(
        f"Stage 1B [Tilted Bands]:           total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_b}, dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 1D: Unrestricted 4-corner Gobel 45-deg search
    t0 = time.perf_counter()
    imp_gu = _merge_if_better(packings, search_gobel_unrestricted(safety_gap=safety_gap))
    print(
        f"Stage 1D [Gobel Unrestricted]:     total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_gu}, dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 1E: Off-center tilted block + 4-corner greedy shells
    t0 = time.perf_counter()
    imp_blk = _merge_if_better(packings, search_offcenter_block_shells(safety_gap=safety_gap))
    print(
        f"Stage 1E [Offcenter Block Shells]: total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_blk}, dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 1F: 4-corner exact-AABB + slack-transfer tilted bands
    t0 = time.perf_counter()
    imp_sb = _merge_if_better(
        packings, search_4corner_slack_bands(safety_gap=safety_gap, workers=workers)
    )
    print(
        f"Stage 1F [4-Corner Slack Bands]:   total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_sb}, dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 1G: Closure pass 1 (L-border + 4-fold pinwheel + monotone deletion)
    t0 = time.perf_counter()
    packings = apply_closures(packings, safety_gap=safety_gap)
    print(
        f"Stage 1G [Closure Pass 1]:         total_score = {total_packings_score(packings):.6f} "
        f"(dt={time.perf_counter() - t0:.2f}s)"
    )

    # Stage 2: Two-wave parallel L-BFGS-B local optimization
    if not skip_lbfgs and lbfgs_budget > 0.0:
        t0 = time.perf_counter()
        wave1_ns = [
            11, 17, 18, 19, 20, 26, 27, 28, 29, 30,
            37, 38, 39, 40, 41, 50, 51, 52, 53, 54, 55, 56,
            65, 66, 67, 68, 69, 70, 71, 72,
        ]
        packings = optimize_puzzles_parallel(
            packings,
            wave1_ns,
            time_budget_s=2.0 * lbfgs_budget,
            safety_gap=safety_gap,
            workers=workers,
        )
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 2A [L-BFGS Wave 1 + Clos]:   total_score = {total_packings_score(packings):.6f} "
            f"(dt={time.perf_counter() - t0:.2f}s)"
        )

        t0 = time.perf_counter()
        wave1_set = set(wave1_ns)
        small_rest = [
            n for n in range(4, 91) if not _is_perfect_square(n) and n not in wave1_set
        ]
        mid_cands = [
            k * k + r
            for k in range(10, 21)
            for r in range(1, k + 1)
            if k * k + r <= MAX_N
            and (k * k + r) not in wave1_set
            and packings[k * k + r].side_length > k + 0.85
        ]
        mid_cands.sort(
            key=lambda n: score_puzzle(n, packings[n].side_length),
            reverse=True,
        )
        wave2_ns = (small_rest + mid_cands)[:70]
        packings = optimize_puzzles_parallel(
            packings,
            wave2_ns,
            time_budget_s=lbfgs_budget,
            safety_gap=safety_gap,
            workers=workers,
        )
        print(
            f"Stage 2B [L-BFGS Wave 2]:          total_score = {total_packings_score(packings):.6f} "
            f"(n_puzzles={len(wave2_ns)}, dt={time.perf_counter() - t0:.2f}s)"
        )

        t0 = time.perf_counter()
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 3  [Closure Pass 2]:         total_score = {total_packings_score(packings):.6f} "
            f"(dt={time.perf_counter() - t0:.2f}s)"
        )

    # Stage 4: Final per-puzzle certification & safe fallback
    t0 = time.perf_counter()
    packings, reverted = verify_all_improved(packings, workers=workers)
    final_score = total_packings_score(packings)
    improved_total = sum(1 for n in range(1, MAX_N + 1) if not packings[n].use_raw_baseline)
    print(
        f"Stage 4  [Certified Final]:        total_score = {final_score:.6f} "
        f"(improved_puzzles={improved_total}/1000, reverted={reverted}, dt={time.perf_counter() - t0:.2f}s)"
    )

    out_path = write_submission_csv(packings, baseline_csv=baseline_csv, output_csv=output_csv)
    print(f"Exported submission to {out_path} (total pipeline time: {time.perf_counter() - t_all:.2f}s)")

    deltas = []
    for n in range(1, MAX_N + 1):
        new_sc = score_puzzle(n, packings[n].side_length)
        diff = baseline_scores[n] - new_sc
        if diff > 1e-9:
            deltas.append((diff, n, packings[n].side_length, math.ceil(math.sqrt(n)), packings[n].source))
    deltas.sort(reverse=True)
    print("\nTop 15 biggest puzzle score reductions (vs baseline grid):")
    for diff, n, side_new, side_base, src in deltas[:15]:
        print(f"  n={n:4d} | L: {side_base} -> {side_new:.8f} | saved={diff:.8f} | {src}")

    return packings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Squares in Squares v2 optimization solver.")
    parser.add_argument("--baseline", default="data/sample_submission.csv", help="Path to sample_submission.csv")
    parser.add_argument("--output", default="submissions/submission_v2.csv", help="Output submission CSV path")
    parser.add_argument(
        "--seed-csv",
        default="submissions/submission_v1.csv",
        help="Optional existing verified submission CSV used as initial floor (pass '' to disable)",
    )
    parser.add_argument("--lbfgs-budget", type=float, default=2.0, help="Base seconds per puzzle for L-BFGS-B")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--skip-lbfgs", action="store_true", help="Skip Stage 2 L-BFGS-B optimization")
    args = parser.parse_args(argv)

    run_pipeline(
        baseline_csv=args.baseline,
        output_csv=args.output,
        seed_csv=args.seed_csv if args.seed_csv else None,
        lbfgs_budget=args.lbfgs_budget,
        workers=args.workers,
        skip_lbfgs=args.skip_lbfgs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
