"""Dataset integrity checks that run before any model training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
}


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    market_gaps: int


def validate_market_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"{path} contains no rows")

    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if parsed_time.isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    if parsed_time.duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    if not parsed_time.is_monotonic_increasing:
        raise ValueError(f"{path} is not in chronological order")

    numeric_columns = ["open", "high", "low", "close", "tick_volume"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{path} contains missing or non-finite market values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path} contains non-positive prices")
    if (numeric["tick_volume"] < 0).any():
        raise ValueError(f"{path} contains negative tick volume")

    invalid_high = numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    invalid_low = numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"{path} contains invalid OHLC relationships")

    intervals = parsed_time.diff().dropna()
    if (intervals < pd.Timedelta(minutes=5)).any():
        raise ValueError(f"{path} contains intervals shorter than five minutes")
    off_grid = intervals.dt.total_seconds().mod(300).ne(0)
    if off_grid.any():
        raise ValueError(f"{path} contains timestamps that are not on an M5 grid")

    df = df.copy()
    df["time"] = parsed_time
    report = DatasetReport(
        path=path,
        rows=len(df),
        start=parsed_time.iloc[0],
        end=parsed_time.iloc[-1],
        market_gaps=int((intervals > pd.Timedelta(minutes=5)).sum()),
    )
    return df, report


def validate_dataset_pair(
    development_path: str | Path,
    test_path: str | Path,
) -> tuple[DatasetReport, DatasetReport]:
    development, development_report = validate_market_dataset(development_path)
    test, test_report = validate_market_dataset(test_path)

    overlap = pd.Index(development["time"]).intersection(pd.Index(test["time"]))
    if len(overlap):
        raise ValueError(
            f"Development and test datasets overlap by {len(overlap)} timestamps "
            f"({overlap.min()} through {overlap.max()})"
        )
    if development_report.end >= test_report.start:
        raise ValueError(
            "The test dataset must begin strictly after the development dataset ends"
        )

    return development_report, test_report


if __name__ == "__main__":
    development_report, test_report = validate_dataset_pair(
        "XAUUSD_M5_2Year.csv",
        "XAUUSD_M5_6month.csv",
    )
    print(
        f"Development: {development_report.rows:,} rows, "
        f"{development_report.start} to {development_report.end}"
    )
    print(
        f"Test: {test_report.rows:,} rows, "
        f"{test_report.start} to {test_report.end}"
    )
    print("Dataset validation passed: chronological and non-overlapping.")
