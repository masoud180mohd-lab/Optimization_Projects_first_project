"""Unit tests for constructive packers, L-BFGS-B optimizer, and closures."""

from __future__ import annotations

import unittest

import numpy as np

from src.constructive import (
    DEFAULT_SAFETY_GAP,
    apply_closures,
    grid_packing,
    search_gobel_45,
    search_tilted_bands,
)
from src.constructive_v2 import (
    search_4corner_slack_bands,
    search_offcenter_block_shells,
)
from src.constructive_v3 import search_sliding_diamond_columns
from src.constructive_v4 import search_oblique_columns
from src.constructive_v5 import pack_rectangle, search_matched_rectangular_strips
from src.constructive_v6 import search_four_wall_strips
from src.contact_lp import refine
from src.optimizer import _energy_and_grad, optimize_puzzle_lbfgs
from src.validator import validate_puzzle


class SolverTests(unittest.TestCase):
    def test_gobel_45_recovers_n5_and_n10(self) -> None:
        gobel = search_gobel_45(safety_gap=DEFAULT_SAFETY_GAP)
        self.assertIn(5, gobel)
        self.assertIn(10, gobel)

        for n, upper in ((5, 2.7072), (10, 3.7072)):
            p = gobel[n]
            self.assertLess(p.side_length, upper)
            res = validate_puzzle(n, p.xs, p.ys, p.angles)
            self.assertTrue(res.valid, f"n={n} invalid: {res.error}")
            self.assertIsNotNone(res.side_length)
            self.assertLess(res.side_length, upper)

    def test_tilted_bands_beats_gobel_large_n(self) -> None:
        bands = search_tilted_bands(safety_gap=DEFAULT_SAFETY_GAP)
        self.assertIn(962, bands)
        p = bands[962]
        self.assertLess(p.side_length, 31.65)
        res = validate_puzzle(962, p.xs, p.ys, p.angles)
        self.assertTrue(res.valid, f"n=962 invalid: {res.error}")
        self.assertIsNotNone(res.side_length)
        self.assertLess(res.side_length, 31.65)

    def test_energy_gradient_matches_finite_diff(self) -> None:
        z = np.array(
            [0.6, 0.6, 0.2, 1.35, 0.75, 0.5, 0.9, 1.4, 0.35],
            dtype=np.float64,
        )
        L = 2.1
        e0, grad = _energy_and_grad(z, L, 1e-6, 1e-9)
        self.assertGreater(e0, 0.0)
        eps = 1e-7
        num_grad = np.zeros_like(z)
        for i in range(len(z)):
            zp = z.copy()
            zm = z.copy()
            zp[i] += eps
            zm[i] -= eps
            ep, _ = _energy_and_grad(zp, L, 1e-6, 1e-9)
            em, _ = _energy_and_grad(zm, L, 1e-6, 1e-9)
            num_grad[i] = (ep - em) / (2.0 * eps)
        np.testing.assert_allclose(grad, num_grad, rtol=1e-5, atol=1e-5)

    def test_lbfgs_improves_n11_and_n19(self) -> None:
        gobel = search_gobel_45(safety_gap=DEFAULT_SAFETY_GAP)

        # n=11 via incremental (n=10)+1 insertion + L-BFGS-B
        p11 = optimize_puzzle_lbfgs(
            11,
            grid_packing(11),
            time_budget_s=3.5,
            seed=42,
            safety_gap=DEFAULT_SAFETY_GAP,
            prev_packing=gobel[10],
        )
        self.assertLess(p11.side_length, 3.99)
        res11 = validate_puzzle(11, p11.xs, p11.ys, p11.angles)
        self.assertTrue(res11.valid, f"n=11 invalid: {res11.error}")
        self.assertIsNotNone(res11.side_length)
        self.assertLess(res11.side_length, 3.99)

        # n=19 basin-hopping + polish below gobel45 (4.9142)
        p19 = optimize_puzzle_lbfgs(
            19,
            gobel[19],
            time_budget_s=2.0,
            seed=42,
            safety_gap=DEFAULT_SAFETY_GAP,
        )
        self.assertLess(p19.side_length, 4.905)
        res19 = validate_puzzle(19, p19.xs, p19.ys, p19.angles)
        self.assertTrue(res19.valid, f"n=19 invalid: {res19.error}")
        self.assertIsNotNone(res19.side_length)
        self.assertLess(res19.side_length, 4.905)

    def test_closures_preserve_validity(self) -> None:
        gobel = search_gobel_45(safety_gap=DEFAULT_SAFETY_GAP)
        packings = {n: gobel.get(n, grid_packing(n)) for n in range(1, 1001)}
        closed = apply_closures(packings, safety_gap=DEFAULT_SAFETY_GAP, max_passes=2)
        for n in range(1, 50):
            self.assertLessEqual(closed[n].side_length, closed[n + 1].side_length + 1e-9)
            if not closed[n].use_raw_baseline:
                res = validate_puzzle(n, closed[n].xs, closed[n].ys, closed[n].angles)
                self.assertTrue(res.valid, f"n={n} after closure invalid: {res.error}")

    def test_offcenter_block_hits_n26(self) -> None:
        blocks = search_offcenter_block_shells(safety_gap=DEFAULT_SAFETY_GAP)
        self.assertIn(26, blocks)
        p = blocks[26]
        self.assertLess(p.side_length, 5.622)
        res = validate_puzzle(26, p.xs, p.ys, p.angles)
        self.assertTrue(res.valid, f"n=26 invalid: {res.error}")
        self.assertIsNotNone(res.side_length)
        self.assertLess(res.side_length, 5.622)

    def test_4corner_slack_bands_improves_mid_n(self) -> None:
        bands = search_4corner_slack_bands(safety_gap=DEFAULT_SAFETY_GAP, workers=2)
        self.assertIn(53, bands)
        p = bands[53]
        self.assertLess(p.side_length, 7.86)
        res = validate_puzzle(53, p.xs, p.ys, p.angles)
        self.assertTrue(res.valid, f"n=53 invalid: {res.error}")
        self.assertIsNotNone(res.side_length)
        self.assertLess(res.side_length, 7.86)

    def test_sliding_diamond_columns_beats_v2_targets(self) -> None:
        wins = search_sliding_diamond_columns({29: 6.0, 54: 7.915, 70: 8.95}, workers=1)
        for n, upper in ((29, 5.995), (54, 7.895), (70, 8.92)):
            self.assertIn(n, wins)
            p = wins[n]
            self.assertLess(p.side_length, upper)
            res = validate_puzzle(n, p.xs, p.ys, p.angles)
            self.assertTrue(res.valid, f"n={n} invalid: {res.error}")
            self.assertEqual(res.ambiguous_pairs_checked, 0)

    def test_contact_lp_refines_sliding_diamond_n29(self) -> None:
        wins = search_sliding_diamond_columns({29: 6.0}, workers=1)
        self.assertIn(29, wins)
        refined = refine(wins[29], steps=40)
        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertLess(refined.side_length, 5.97)
        res = validate_puzzle(29, refined.xs, refined.ys, refined.angles)
        self.assertTrue(res.valid, f"n=29 refined invalid: {res.error}")
        self.assertEqual(res.ambiguous_pairs_checked, 0)

    def test_contact_lp_skips_perfect_square(self) -> None:
        self.assertIsNone(refine(grid_packing(16)))

    def test_oblique_columns_beats_v3_targets(self) -> None:
        wins = search_oblique_columns({29: 5.965, 55: 8.0, 71: 9.0}, workers=1)
        for n, upper in ((29, 5.961), (55, 7.98), (71, 8.97)):
            self.assertIn(n, wins)
            p = wins[n]
            self.assertLess(p.side_length, upper)
            res = validate_puzzle(n, p.xs, p.ys, p.angles)
            self.assertTrue(res.valid, f"n={n} invalid: {res.error}")
            self.assertEqual(res.ambiguous_pairs_checked, 0)

    def test_contact_lp_refines_oblique_n55(self) -> None:
        wins = search_oblique_columns({55: 8.0}, workers=1)
        self.assertIn(55, wins)
        p55 = wins[55]
        self.assertLess(p55.side_length, 7.98)
        refined = refine(p55, steps=30)
        if refined is not None:
            self.assertLess(refined.side_length, p55.side_length)
            res = validate_puzzle(55, refined.xs, refined.ys, refined.angles)
            self.assertTrue(res.valid, f"n=55 refined invalid: {res.error}")
            self.assertEqual(res.ambiguous_pairs_checked, 0)

    def test_pack_rectangle_beats_grid_strip(self) -> None:
        pts = pack_rectangle(6.95, 26.95, fine=True)
        self.assertGreaterEqual(len(pts), 170)
        xs, ys, angs = pts[:, 0], pts[:, 1], pts[:, 2]
        h = 0.5 * (np.abs(np.cos(angs)) + np.abs(np.sin(angs)))
        self.assertTrue(np.all(xs - h >= -1e-9))
        self.assertTrue(np.all(xs + h <= 6.95 + 1e-9))
        self.assertTrue(np.all(ys - h >= -1e-9))
        self.assertTrue(np.all(ys + h <= 26.95 + 1e-9))

    def test_matched_strips_v5_beats_n600_and_n700(self) -> None:
        wins = search_matched_rectangular_strips({600: 25.0, 700: 26.986}, workers=1)
        for n, upper in ((600, 24.985), (700, 26.95)):
            self.assertIn(n, wins)
            p = wins[n]
            self.assertLess(p.side_length, upper)
            res = validate_puzzle(n, p.xs, p.ys, p.angles)
            self.assertTrue(res.valid, f"n={n} invalid: {res.error}")
            self.assertEqual(res.ambiguous_pairs_checked, 0)

    def test_four_wall_strips_v6_beats_n757(self) -> None:
        wins = search_four_wall_strips({757: 27.9983}, workers=1)
        self.assertIn(757, wins)
        p = wins[757]
        self.assertLess(p.side_length, 27.975)
        res = validate_puzzle(757, p.xs, p.ys, p.angles)
        self.assertTrue(res.valid, f"n=757 invalid: {res.error}")
        self.assertEqual(res.ambiguous_pairs_checked, 0)


if __name__ == "__main__":
    unittest.main()
