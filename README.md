# Squares in Squares — Optimization Project

Multi-stage combinatorial geometry and continuous optimization solver for the Kaggle **Squares in Squares** competition (packing $n = 1 \dots 1000$ non-overlapping unit squares into the smallest axis-aligned square container).

## Project Structure

- `src/validator.py` — Local validator & scorer matching Kaggle's official metric (Numba broad-phase + float64 SAT + escalating `mpmath.iv` interval arithmetic).
- `src/constructive.py` — Axis-aligned baseline grid, 4-corner 45° Göbel staircases + diagonal band (`search_gobel_45`), shallow-angle tilted bands (`search_tilted_bands`), and closure DP (`apply_closures` with L-border, 4-fold pinwheel, and monotone deletion).
- `src/constructive_v2.py` — Unrestricted 4-corner Göbel search, off-center tilted $p \times q$ blocks + 4-corner shells (`search_offcenter_block_shells`), and 4-corner exact-AABB slack-transfer tilted bands (`search_4corner_slack_bands`).
- `src/constructive_v3.py` — Sliding 45° diamond columns + individual SE/TL corner-cell weighted interval scheduling DP (`search_sliding_diamond_columns`).
- `src/constructive_v4.py` — Arbitrary-angle oblique sliding columns + independent-slope upright corner staircases (`search_oblique_columns`).
- `src/optimizer.py` — Numba-accelerated smooth-SAT analytical gradients + `SciPy L-BFGS-B` continuation, largest-void insertion, and subset basin-hopping.
- `src/contact_lp.py` — Sparse contact-constrained Sequential Linear Programming refiner (`refine`, `refine_packings_parallel`).
- `src/solver.py` — End-to-end multi-stage pipeline and CSV submission exporter.
- `tests/` — Unit tests for the validator and solver components.
- `official_metric/` — Official Kaggle evaluation notebook (`metric/squares-in-squares`).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Download competition data (`sample_submission.csv`) into `data/`:

```powershell
.\.venv\Scripts\kaggle.exe competitions download -c squares-in-squares -p data
Expand-Archive -Path data\squares-in-squares.zip -DestinationPath data -Force
```

## Usage

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the v4 solver:

```powershell
.\.venv\Scripts\python.exe -m src.solver --output submissions/submission_v4.csv
```

Validate a submission CSV locally:

```powershell
.\.venv\Scripts\python.exe -m src.validator submissions/submission_v4.csv --top-worst 10 --progress
```

## Results Summary

- **Baseline (`data/sample_submission.csv`):** `28.937670`
- **v1 (`submission_v1.csv`):** `23.207727` (Kaggle public score: `23.2077270951`)
- **v2 (`submission_v2.csv`):** `23.082662` (Kaggle public score: `23.0826616410`)
- **v3 (`submission_v3.csv`):** `22.950716` (1,000/1,000 puzzles locally verified valid, `0` overlaps)
- **v4 (`submission_v4.csv`):** `22.827743` (1,000/1,000 puzzles locally verified valid, `0` overlaps)
