"""End-to-end multi-stage v5 solver pipeline for the Squares in Squares competition."""

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
from src.constructive_v3 import search_sliding_diamond_columns
from src.constructive_v4 import search_oblique_columns
from src.constructive_v5 import search_matched_rectangular_strips
from src.contact_lp import refine_packings_parallel
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


def _has_rotation(p: Packing) -> bool:
    return bool(np.any(np.abs(p.angles) > 1e-6))


def _verify_puzzle_worker(
    args: tuple[int, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, bool, float | None, str | None]:
    n, xs, ys, angles = args
    raw_strings = tuple(
        [format(float(z), ".16f") for z in arr] for arr in (xs, ys, angles)
    )
    res = validate_puzzle(n, xs, ys, angles, raw_strings=raw_strings)
    ok = bool(res.valid and res.ambiguous_pairs_checked == 0)
    err = res.error or (
        f"ambiguous={res.ambiguous_pairs_checked}"
        if res.ambiguous_pairs_checked
        else None
    )
    return n, ok, res.side_length, err


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
    output_csv: str | Path = Path("submissions/submission_v5.csv"),
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
    output_csv: str | Path = Path("submissions/submission_v5.csv"),
    seed_csv: str | Path | None = Path("submissions/submission_v4.csv"),
    lbfgs_budget: float = 2.0,
    workers: int | None = None,
    skip_lbfgs: bool = False,
    skip_v3_constructive: bool = False,
    skip_v4_constructive: bool = False,
    skip_v5_constructive: bool = False,
    skip_contact_lp: bool = False,
    contact_lp_steps: int = 80,
    safety_gap: float = DEFAULT_SAFETY_GAP,
) -> dict[int, Packing]:
    """Run the full multi-stage v5 solver pipeline and export a verified CSV submission."""
    t_all = time.perf_counter()

    # 1. Baseline initialization (+ optional verified seed floor)
    packings: dict[int, Packing] = {n: grid_packing(n) for n in range(1, MAX_N + 1)}
    baseline_scores = {n: score_puzzle(n, packings[n].side_length) for n in range(1, MAX_N + 1)}
    score_base = total_packings_score(packings)
    print(f"Stage 0  [Baseline Grid]:          total_score = {score_base:.6f}", flush=True)

    if seed_csv:
        loaded = load_seed_csv(packings, seed_csv)
        if loaded > 0:
            print(
                f"Stage 0b [Seed CSV Floor]:         total_score = {total_packings_score(packings):.6f} "
                f"(loaded={loaded})",
                flush=True,
            )

    # Stage 1A: Gobel 45-degree 4-corner staircase + diagonal band
    t0 = time.perf_counter()
    imp_g = _merge_if_better(packings, search_gobel_45(safety_gap=safety_gap))
    print(
        f"Stage 1A [Gobel 45-deg]:           total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_g}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1B: General shallow-angle tilted bands
    t0 = time.perf_counter()
    imp_b = _merge_if_better(packings, search_tilted_bands(safety_gap=safety_gap))
    print(
        f"Stage 1B [Tilted Bands]:           total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_b}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1D: Unrestricted 4-corner Gobel 45-deg search
    t0 = time.perf_counter()
    imp_gu = _merge_if_better(packings, search_gobel_unrestricted(safety_gap=safety_gap))
    print(
        f"Stage 1D [Gobel Unrestricted]:     total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_gu}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1E: Off-center tilted block + 4-corner greedy shells
    t0 = time.perf_counter()
    imp_blk = _merge_if_better(packings, search_offcenter_block_shells(safety_gap=safety_gap))
    print(
        f"Stage 1E [Offcenter Block Shells]: total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_blk}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1F: 4-corner exact-AABB + slack-transfer tilted bands
    t0 = time.perf_counter()
    imp_sb = _merge_if_better(
        packings, search_4corner_slack_bands(safety_gap=safety_gap, workers=workers)
    )
    print(
        f"Stage 1F [4-Corner Slack Bands]:   total_score = {total_packings_score(packings):.6f} "
        f"(improved={imp_sb}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1G: Closure pass 1 (L-border + 4-fold pinwheel + monotone deletion)
    t0 = time.perf_counter()
    packings = apply_closures(packings, safety_gap=safety_gap)
    print(
        f"Stage 1G [Closure Pass 1]:         total_score = {total_packings_score(packings):.6f} "
        f"(dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Stage 1H: Sliding 45-degree diamond columns + individual SE/TL corner cells (v3)
    sliding_won_ns: set[int] = set()
    if not skip_v3_constructive:
        t0 = time.perf_counter()
        bounds = {n: p.side_length for n, p in packings.items()}
        sliding_wins = search_sliding_diamond_columns(
            bounds,
            safety_gap=safety_gap,
            resolution=64,
            phases=4,
            iterations=16,
            workers=workers,
        )
        imp_sl = _merge_if_better(packings, sliding_wins)
        sliding_won_ns = set(sliding_wins.keys())
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 1H [Sliding Diamonds + Clos]:total_score = {total_packings_score(packings):.6f} "
            f"(improved={imp_sl}, dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

    # Stage 1I: Oblique sliding columns + independent-slope upright staircases (v4)
    oblique_won_ns: set[int] = set()
    if not skip_v4_constructive:
        t0 = time.perf_counter()
        bounds = {n: p.side_length for n, p in packings.items()}
        oblique_wins = search_oblique_columns(
            bounds,
            safety_gap=safety_gap,
            resolution=24,
            workers=workers,
        )
        imp_ob = _merge_if_better(packings, oblique_wins)
        oblique_won_ns = set(oblique_wins.keys())
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 1I [Oblique Sliding + Clos]: total_score = {total_packings_score(packings):.6f} "
            f"(improved={imp_ob}, dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

    # Stage 1J: Tilted rectangular strips + bipartite-matched shared corners (v5)
    v5_won_ns: set[int] = set()
    if not skip_v5_constructive:
        t0 = time.perf_counter()
        v5_coarse = search_matched_rectangular_strips(
            {n: p.side_length for n, p in packings.items()},
            safety_gap=safety_gap,
            max_width=10,
            fine=False,
            workers=workers,
        )
        imp_v5_c = _merge_if_better(packings, v5_coarse)
        v5_fine = search_matched_rectangular_strips(
            {n: p.side_length for n, p in packings.items()},
            safety_gap=safety_gap,
            max_width=10,
            fine=True,
            workers=workers,
        )
        imp_v5_f = _merge_if_better(packings, v5_fine)
        v5_won_ns = set(v5_coarse.keys()) | set(v5_fine.keys())
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 1J [Matched Rect Strips]:    total_score = {total_packings_score(packings):.6f} "
            f"(improved={imp_v5_c}+{imp_v5_f}, dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
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
            f"(dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
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
            f"(n_puzzles={len(wave2_ns)}, dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

        t0 = time.perf_counter()
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 2D [Closure Pass 2]:         total_score = {total_packings_score(packings):.6f} "
            f"(dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

    # Stage 2C: Sparse Contact-LP local refinement (v3) + closures + micro-polish
    if not skip_contact_lp and contact_lp_steps > 0:
        t0 = time.perf_counter()
        t_t1 = time.perf_counter()
        tier1_ns = [
            n
            for n in range(2, min(95, MAX_N) + 1)
            if not _is_perfect_square(n)
            and (
                _has_rotation(packings[n])
                or any(
                    m in packings and _has_rotation(packings[m])
                    for m in (n + 1, n + 2)
                )
            )
        ]
        packings, w1 = refine_packings_parallel(
            packings,
            tier1_ns,
            steps=contact_lp_steps,
            donors=2,
            safety_gap=safety_gap,
            workers=workers,
        )
        dt1 = time.perf_counter() - t_t1

        t_t2 = time.perf_counter()
        tier2_ns = [
            n
            for n in range(96, min(185, MAX_N) + 1)
            if not _is_perfect_square(n) and _has_rotation(packings[n])
        ]
        packings, w2 = refine_packings_parallel(
            packings,
            tier2_ns,
            steps=min(40, contact_lp_steps),
            donors=0,
            safety_gap=safety_gap,
            workers=workers,
        )
        dt2 = time.perf_counter() - t_t2

        t_t3 = time.perf_counter()
        tier3_ns = [
            n
            for n in range(186, min(400, MAX_N) + 1)
            if not _is_perfect_square(n)
            and n in (sliding_won_ns | oblique_won_ns | v5_won_ns)
        ]
        packings, w3 = refine_packings_parallel(
            packings,
            tier3_ns,
            steps=min(25, contact_lp_steps),
            donors=0,
            safety_gap=safety_gap,
            workers=workers,
        )
        dt3 = time.perf_counter() - t_t3
        print(
            f"Stage 2C [Sparse Contact-LP]:      total_score = {total_packings_score(packings):.6f} "
            f"(wins={w1}/{len(tier1_ns)} [{dt1:.1f}s], {w2}/{len(tier2_ns)} [{dt2:.1f}s], "
            f"{w3}/{len(tier3_ns)} [{dt3:.1f}s], total={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

        t0 = time.perf_counter()
        pre_closure_sides = {n: p.side_length for n, p in packings.items()}
        packings = apply_closures(packings, safety_gap=safety_gap)
        print(
            f"Stage 3  [Closure Pass 3]:         total_score = {total_packings_score(packings):.6f} "
            f"(dt={time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

        changed_small = [
            n
            for n in range(4, 151)
            if not _is_perfect_square(n)
            and packings[n].side_length < pre_closure_sides[n] - 1e-9
            and _has_rotation(packings[n])
        ]
        if changed_small:
            t0 = time.perf_counter()
            packings, w4 = refine_packings_parallel(
                packings,
                changed_small,
                steps=min(40, contact_lp_steps),
                donors=0,
                safety_gap=safety_gap,
                workers=workers,
            )
            packings = apply_closures(packings, safety_gap=safety_gap)
            print(
                f"Stage 3B [Post-Clos Micro-Polish]: total_score = {total_packings_score(packings):.6f} "
                f"(wins={w4}/{len(changed_small)}, dt={time.perf_counter() - t0:.2f}s)",
                flush=True,
            )

    # Stage 4: Final per-puzzle certification & safe fallback
    t0 = time.perf_counter()
    packings, reverted = verify_all_improved(packings, workers=workers)
    final_score = total_packings_score(packings)
    improved_total = sum(1 for n in range(1, MAX_N + 1) if not packings[n].use_raw_baseline)
    print(
        f"Stage 4  [Certified Final]:        total_score = {final_score:.6f} "
        f"(improved_puzzles={improved_total}/1000, reverted={reverted}, dt={time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    out_path = write_submission_csv(packings, baseline_csv=baseline_csv, output_csv=output_csv)
    print(f"Exported submission to {out_path} (total pipeline time: {time.perf_counter() - t_all:.2f}s)", flush=True)

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
    parser = argparse.ArgumentParser(description="Run Squares in Squares v5 optimization solver.")
    parser.add_argument("--baseline", default="data/sample_submission.csv", help="Path to sample_submission.csv")
    parser.add_argument("--output", default="submissions/submission_v5.csv", help="Output submission CSV path")
    parser.add_argument(
        "--seed-csv",
        default="submissions/submission_v4.csv",
        help="Optional existing verified submission CSV used as initial floor (pass '' to disable)",
    )
    parser.add_argument("--lbfgs-budget", type=float, default=2.0, help="Base seconds per puzzle for L-BFGS-B")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--skip-lbfgs", action="store_true", help="Skip Stage 2 L-BFGS-B optimization")
    parser.add_argument(
        "--skip-v3-constructive",
        action="store_true",
        help="Skip Stage 1H sliding diamond column search",
    )
    parser.add_argument(
        "--skip-v4-constructive",
        action="store_true",
        help="Skip Stage 1I oblique sliding column search",
    )
    parser.add_argument(
        "--skip-v5-constructive",
        action="store_true",
        help="Skip Stage 1J matched rectangular strip search",
    )
    parser.add_argument(
        "--skip-contact-lp",
        action="store_true",
        help="Skip Stage 2C sparse Contact-LP refinement",
    )
    parser.add_argument(
        "--contact-lp-steps",
        type=int,
        default=80,
        help="Maximum Contact-LP steps per Tier-1 puzzle",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        baseline_csv=args.baseline,
        output_csv=args.output,
        seed_csv=args.seed_csv if args.seed_csv else None,
        lbfgs_budget=args.lbfgs_budget,
        workers=args.workers,
        skip_lbfgs=args.skip_lbfgs,
        skip_v3_constructive=args.skip_v3_constructive,
        skip_v4_constructive=args.skip_v4_constructive,
        skip_v5_constructive=args.skip_v5_constructive,
        skip_contact_lp=args.skip_contact_lp,
        contact_lp_steps=args.contact_lp_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
