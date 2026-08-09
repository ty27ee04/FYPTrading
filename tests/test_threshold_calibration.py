import tempfile
import unittest
from pathlib import Path

import numpy as np

from strategy_config import STRATEGY, load_gatekeeper_threshold
from threshold_calibration import (
    calibrate_gatekeeper_threshold,
    threshold_grid,
    wilson_lower_bound,
)


class ThresholdCalibrationTests(unittest.TestCase):
    def test_threshold_grid_is_inclusive_and_stable(self):
        np.testing.assert_array_equal(
            threshold_grid(0.40, 0.46, 0.02),
            np.array([0.40, 0.42, 0.44, 0.46]),
        )

    def test_wilson_bound_penalizes_unsupported_perfect_precision(self):
        self.assertGreater(wilson_lower_bound(80, 100), wilson_lower_bound(1, 1))

    def test_calibration_uses_only_eligible_signals_and_constraints(self):
        signals = np.array([1, 1, 2, 2, 0, 1])
        probabilities = np.array([0.90, 0.80, 0.70, 0.60, 0.99, 0.10])
        correctness = np.array([1, 1, 0, 0, 1, 1])

        result = calibrate_gatekeeper_threshold(
            signals,
            probabilities,
            correctness,
            [0.50, 0.70, 0.85],
            minimum_accepted_signals=2,
            minimum_signal_coverage=0.20,
        )

        self.assertEqual(result["eligible_signals"], 5)
        self.assertEqual(result["selected_threshold"], 0.70)
        self.assertEqual(result["selected_metrics"]["accepted_signals"], 3)

    def test_trading_metrics_do_not_change_statistical_selection(self):
        signals = np.array([1, 1, 2, 2])
        probabilities = np.array([0.90, 0.80, 0.70, 0.60])
        correctness = np.array([1, 1, 0, 0])

        result = calibrate_gatekeeper_threshold(
            signals,
            probabilities,
            correctness,
            [0.50, 0.70],
            minimum_accepted_signals=2,
            minimum_signal_coverage=0.20,
            trading_evaluator=lambda gated: {
                "net_profit_fixed": -999.0 if np.count_nonzero(gated) == 3 else 999.0
            },
        )

        self.assertEqual(result["selected_threshold"], 0.70)

    def test_loader_falls_back_and_reads_frozen_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "threshold_calibration.json"
            self.assertEqual(load_gatekeeper_threshold(path), STRATEGY.gatekeeper_threshold)
            path.write_text('{"selected_threshold": 0.64}', encoding="utf-8")
            self.assertEqual(load_gatekeeper_threshold(path), 0.64)


if __name__ == "__main__":
    unittest.main()
