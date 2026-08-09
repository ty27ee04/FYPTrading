"""Authoritative configuration shared by training, backtesting, and live inference."""

from dataclasses import dataclass


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


MODEL = ModelConfig()
STRATEGY = StrategyConfig()
