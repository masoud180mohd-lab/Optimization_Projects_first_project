"""Unit tests for src.validator and parity checks against Kaggle's official metric."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.validator import (
    run_official_metric,
    validate_puzzle,
    validate_schema,
    validate_submission,
)


class ValidatorTests(unittest.TestCase):
    def test_touching_edges_valid(self) -> None:
        res = validate_puzzle(
            2,
            [0.5, 1.5],
            [0.5, 0.5],
            [0.0, 0.0],
            raw_strings=(["0.5", "1.5"], ["0.5", "0.5"], ["0", "0"]),
        )
        self.assertTrue(res.valid, res.error)
        self.assertIsNotNone(res.side_length)
        self.assertAlmostEqual(res.side_length, 2.0, places=12)
        self.assertAlmostEqual(res.score, 2.0 / math.sqrt(2.0) - 1.0, places=12)
        self.assertEqual(res.ambiguous_pairs_checked, 1)

    def test_axis_aligned_overlap_invalid(self) -> None:
        res = validate_puzzle(
            2,
            [0.5, 1.499999],
            [0.5, 0.5],
            [0.0, 0.0],
            raw_strings=(["0.5", "1.499999"], ["0.5", "0.5"], ["0", "0"]),
        )
        self.assertFalse(res.valid)
        self.assertEqual(res.overlap_pair, (0, 1))

    def test_tiny_rotated_overlap_invalid(self) -> None:
        # Two 45-degree rotated unit squares have diagonal sqrt(2).
        # Placing their centers at distance sqrt(2) - 1e-9 along the x-axis creates a tiny interior overlap.
        angle_str = f"{math.pi / 4:.30f}"
        x2_str = f"{math.sqrt(2.0) - 1e-9:.30f}"
        res = validate_puzzle(
            2,
            [0.0, float(x2_str)],
            [0.0, 0.0],
            [math.pi / 4, math.pi / 4],
            raw_strings=(["0", x2_str], ["0", "0"], [angle_str, angle_str]),
        )
        self.assertFalse(res.valid)
        self.assertEqual(res.overlap_pair, (0, 1))

    def test_rotated_n5_valid(self) -> None:
        # Optimal n=5 configuration: 4 axis-aligned corner squares + 1 center square rotated by pi/4.
        s = 2.0 + math.sqrt(2.0) / 2.0 + 1e-12
        xs = [0.5, s - 0.5, 0.5, s - 0.5, s / 2.0]
        ys = [0.5, 0.5, s - 0.5, s - 0.5, s / 2.0]
        angles = [0.0, 0.0, 0.0, 0.0, math.pi / 4.0]
        res = validate_puzzle(5, xs, ys, angles)
        self.assertTrue(res.valid, res.error)
        self.assertIsNotNone(res.side_length)
        self.assertAlmostEqual(res.side_length, 2.0 + math.sqrt(2.0) / 2.0, places=10)

    def test_malformed_schema_rejected(self) -> None:
        # Missing column
        df_missing = pd.DataFrame(
            {"id": ["0"], "n": ["1"], "square_id": ["0"], "x": ["0.5"], "y": ["0.5"]}
        )
        self.assertTrue(len(validate_schema(df_missing, expected_puzzles=[1])) > 0)

        # NaN / empty value
        df_nan = pd.DataFrame(
            {
                "id": ["0"],
                "n": ["1"],
                "square_id": ["0"],
                "x": [""],
                "y": ["0.5"],
                "angle": ["0"],
            }
        )
        self.assertTrue(len(validate_schema(df_nan, expected_puzzles=[1])) > 0)

        # Wrong row count via CSV validation
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = Path(tmpdir) / "bad.csv"
            df_wrong_rows = pd.DataFrame(
                {
                    "id": ["0", "1"],
                    "n": ["2", "2"],
                    "square_id": ["0", "0"],
                    "x": ["0.5", "1.5"],
                    "y": ["0.5", "0.5"],
                    "angle": ["0", "0"],
                }
            )
            df_wrong_rows.to_csv(bad_csv, index=False)
            report = validate_submission(bad_csv, puzzles=[2], workers=1)
            self.assertFalse(report.valid)
            self.assertIn(2, report.invalid_puzzles)

    def test_sample_submission_subset_matches_official(self) -> None:
        sample_csv = Path("data/sample_submission.csv")
        official_nb = Path("official_metric/squares-in-squares.ipynb")
        if not (sample_csv.is_file() and official_nb.is_file()):
            self.skipTest("sample_submission.csv or official_metric notebook missing")

        puzzles = list(range(1, 26))
        report = validate_submission(sample_csv, puzzles=puzzles, workers=1)
        self.assertTrue(report.valid, f"Schema/puzzle errors: {report.schema_errors}, {report.invalid_puzzles}")

        official_total, official_details = run_official_metric(sample_csv, puzzles=puzzles)
        self.assertAlmostEqual(report.total_score, official_total, places=12)
        for n in puzzles:
            off_sc, off_l, off_reason = official_details[n]
            self.assertIsNotNone(off_l, f"Official metric rejected n={n}: {off_reason}")
            self.assertAlmostEqual(report.puzzle_results[n].score, off_sc, places=12)


if __name__ == "__main__":
    unittest.main()
