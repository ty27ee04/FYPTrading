import unittest

import numpy as np
import pandas as pd

from execution_simulator import simulate_execution


def market(rows):
    frame = pd.DataFrame(rows)
    frame["time"] = pd.date_range("2026-01-01", periods=len(frame), freq="5min")
    frame["signal_atr"] = 1.0
    return frame


class ExecutionSimulatorTests(unittest.TestCase):
    def run_simulation(self, frame, signals, **kwargs):
        return simulate_execution(
            frame,
            np.array(signals),
            initial_equity=10000,
            lot_size_for_equity=lambda _: 0.1,
            spread_penalty=0.0,
            **kwargs,
        )

    def test_same_bar_tp_sl_conflict_chooses_stop(self):
        frame = market([{"open": 100, "high": 104, "low": 97, "close": 100}])
        trades, _ = self.run_simulation(frame, [1], use_break_even=False)
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Stop Loss (Same Bar Conflict)")
        self.assertEqual(trades.iloc[0]["Exit_Price"], 98)

    def test_opposite_signal_reverses_at_next_open(self):
        frame = market(
            [
                {"open": 100, "high": 101, "low": 99, "close": 100},
                {"open": 101, "high": 102, "low": 100, "close": 101},
            ]
        )
        trades, _ = self.run_simulation(frame, [1, 2], max_horizon=10)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Opposite Signal Reversal")
        self.assertEqual(trades.iloc[0]["Exit_Price"], 101)

    def test_break_even_activates_then_protects_next_bar(self):
        frame = market(
            [
                {"open": 100, "high": 102, "low": 99, "close": 101},
                {"open": 101, "high": 101.5, "low": 100, "close": 100.5},
            ]
        )
        trades, _ = self.run_simulation(frame, [1, 0])
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Break Even Stop")
        self.assertEqual(trades.iloc[0]["Exit_Price"], 100.05)

    def test_time_stop_closes_at_configured_horizon(self):
        frame = market(
            [
                {"open": 100, "high": 101, "low": 99, "close": 100},
                {"open": 100, "high": 101, "low": 99, "close": 100.5},
            ]
        )
        trades, _ = self.run_simulation(
            frame, [1, 0], max_horizon=1, use_break_even=False
        )
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Time Stop")
        self.assertEqual(trades.iloc[0]["Exit_Price"], 100.5)

    def test_same_direction_signal_does_not_pyramid(self):
        frame = market(
            [
                {"open": 100, "high": 101, "low": 99, "close": 100},
                {"open": 101, "high": 102, "low": 100, "close": 101},
            ]
        )
        trades, _ = self.run_simulation(frame, [1, 1], max_horizon=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Entry_Price"], 100)
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Time Stop")

    def test_spread_penalty_is_applied_once_per_round_trip(self):
        frame = market([{"open": 100, "high": 103, "low": 99, "close": 103}])
        trades, _ = simulate_execution(
            frame,
            np.array([1]),
            initial_equity=10000,
            lot_size_for_equity=lambda _: 0.1,
            spread_penalty=0.2,
            use_break_even=False,
        )
        self.assertAlmostEqual(trades.iloc[0]["Net_PnL"], 28.0)

    def test_recorded_spread_overrides_static_fallback(self):
        frame = market([{"open": 100, "high": 103, "low": 99, "close": 103}])
        frame["spread_price"] = 0.5
        trades, _ = simulate_execution(
            frame,
            np.array([1]),
            initial_equity=10000,
            lot_size_for_equity=lambda _: 0.1,
            spread_penalty=0.2,
            use_break_even=False,
        )
        self.assertAlmostEqual(trades.iloc[0]["Net_PnL"], 25.0)
        self.assertEqual(trades.iloc[0]["Spread_Price"], 0.5)


if __name__ == "__main__":
    unittest.main()
