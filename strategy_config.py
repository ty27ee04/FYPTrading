"""Authoritative configuration shared by training, backtesting, and live inference."""

import json
from dataclasses import dataclass
from pathlib import Path


FEATURE_COLUMNS = (
    "log_ret",
    "rsi_n",
    "mfi_n",
    "atr_p",
    "vol_filter",
    "sin_h",
    "cos_h",
    "h1_trend_slope",
    "rsi_h1",
)


@dataclass(frozen=True)
class ModelConfig:
    lookback: int = 60
    max_horizon: int = 24
    seed: int = 42
    model_a_train_fraction: float = 0.70
    model_a_validation_fraction: float = 0.15

    @property
    def purge_gap(self) -> int:
        """Conservative embargo covering both feature history and label horizon."""
        return self.lookback + self.max_horizon


@dataclass(frozen=True)
class StrategyConfig:
    take_profit_atr: float = 3.0
    stop_loss_atr: float = 2.0
    gatekeeper_threshold: float = 0.52
    spread_penalty: float = 0.20


@dataclass(frozen=True)
class ThresholdCalibrationConfig:
    """Rules fixed before looking at the locked final-test period."""

    # Model B scores rank signal reliability; they are not assumed to be
    # calibrated probabilities or centred on 0.50.
    minimum: float = 0.05
    maximum: float = 0.95
    step: float = 0.01
    minimum_accepted_signals: int = 100
    minimum_signal_coverage: float = 0.05
    wilson_z: float = 1.96
    model_b_train_fraction: float = 0.60
    model_b_validation_fraction: float = 0.20


def load_gatekeeper_threshold(path="threshold_calibration.json") -> float:
    """Load a frozen production cutoff, falling back to the configured default."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        return STRATEGY.gatekeeper_threshold

    with calibration_path.open(encoding="utf-8") as calibration_file:
        selected = float(json.load(calibration_file)["selected_threshold"])
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"Invalid calibrated gatekeeper threshold: {selected}")
    return selected


MODEL = ModelConfig()
STRATEGY = StrategyConfig()
CALIBRATION = ThresholdCalibrationConfig()
