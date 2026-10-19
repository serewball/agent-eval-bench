"""指标计算的单元测试：python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalbench.metrics import pass_at_k, pass_pow_k  # noqa: E402


class TestPassAtK(unittest.TestCase):
    def test_all_pass(self) -> None:
        self.assertEqual(pass_at_k(10, 10, 5), 1.0)

    def test_none_pass(self) -> None:
        self.assertEqual(pass_at_k(10, 0, 5), 0.0)

    def test_single_sample(self) -> None:
        self.assertAlmostEqual(pass_at_k(10, 6, 1), 0.6)

    def test_known_value(self) -> None:
        # n=10, c=2, k=2 -> 1 - C(8,2)/C(10,2) = 1 - 28/45
        self.assertAlmostEqual(pass_at_k(10, 2, 2), 1 - 28 / 45)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(pass_at_k(0, 0, 5), 0.0)
        self.assertEqual(pass_at_k(10, 1, 0), 0.0)


class TestPassPowK(unittest.TestCase):
    def test_all_pass(self) -> None:
        self.assertEqual(pass_pow_k(10, 10, 5), 1.0)

    def test_known_value(self) -> None:
        # n=10, c=7, k=5 -> C(7,5)/C(10,5) = 21/252
        self.assertAlmostEqual(pass_pow_k(10, 7, 5), 21 / 252)

    def test_not_enough_passes(self) -> None:
        self.assertEqual(pass_pow_k(10, 3, 5), 0.0)

    def test_empty(self) -> None:
        self.assertEqual(pass_pow_k(0, 0, 5), 0.0)


class TestReliabilityGap(unittest.TestCase):
    def test_gap_is_monotonic(self) -> None:
        """同一份数据下，Pass@k 不应低于 Pass^k。"""
        n, c, k = 10, 7, 5
        self.assertGreater(pass_at_k(n, c, k), pass_pow_k(n, c, k))


if __name__ == "__main__":
    unittest.main()
