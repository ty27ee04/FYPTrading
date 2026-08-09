import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_validation import validate_dataset_pair, validate_market_dataset


def valid_rows(start: str, periods: int = 4) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "time": times,
            "open": [2000.0] * periods,
            "high": [2001.0] * periods,
            "low": [1999.0] * periods,
            "close": [2000.5] * periods,
            "tick_volume": [100] * periods,
        }
    )


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.root / name
        frame.to_csv(path, index=False)
        return path

    def test_valid_non_overlapping_pair_passes(self):
        development = self.write("development.csv", valid_rows("2025-01-01"))
        test = self.write("test.csv", valid_rows("2025-02-01"))

        development_report, test_report = validate_dataset_pair(development, test)

        self.assertEqual(development_report.rows, 4)
        self.assertEqual(test_report.rows, 4)

    def test_overlap_is_rejected(self):
        development = self.write("development.csv", valid_rows("2025-01-01"))
        test = self.write("test.csv", valid_rows("2025-01-01 00:15"))

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_dataset_pair(development, test)

    def test_duplicate_timestamp_is_rejected(self):
        frame = valid_rows("2025-01-01")
        frame.loc[1, "time"] = frame.loc[0, "time"]
        path = self.write("duplicate.csv", frame)

        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            validate_market_dataset(path)

    def test_invalid_ohlc_is_rejected(self):
        frame = valid_rows("2025-01-01")
        frame.loc[0, "high"] = 1998.0
        path = self.write("invalid_ohlc.csv", frame)

        with self.assertRaisesRegex(ValueError, "invalid OHLC"):
            validate_market_dataset(path)


if __name__ == "__main__":
    unittest.main()
