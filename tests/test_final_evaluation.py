import unittest

import numpy as np
import pandas as pd

from final_evaluation import build_classification_metrics, build_trading_metrics


class FinalEvaluationHelperTests(unittest.TestCase):
    def test_classification_metrics_keep_all_three_classes(self):
        metrics = build_classification_metrics(
            np.array([0, 1, 2]), np.array([0, 1, 0])
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        self.assertIn("Sell", metrics["report"])

    def test_trading_metrics_close_open_position_at_evaluation_end(self):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=2, freq="5min"),
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.5],
                "signal_atr": [1.0, 1.0],
            }
        )
        metrics, trades, history = build_trading_metrics(frame, np.array([1, 0]))
        self.assertEqual(metrics["executed_trades"], 1)
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "End Of Evaluation")
        self.assertEqual(len(history), len(frame))


if __name__ == "__main__":
    unittest.main()
