"""Leakage-safe selection of the Model B gatekeeper cutoff."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np


def threshold_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    """Return a stable inclusive threshold grid without float accumulation errors."""
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("Threshold bounds must satisfy 0 <= minimum <= maximum <= 1")
    if step <= 0:
        raise ValueError("Threshold step must be positive")
    count = int(round((maximum - minimum) / step))
    values = minimum + np.arange(count + 1) * step
    return np.round(values[values <= maximum + 1e-12], 10)


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Conservative lower confidence bound for accepted-signal precision."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return (centre - margin) / denominator


def calibrate_gatekeeper_threshold(
    model_a_signals: np.ndarray,
    model_b_probabilities: np.ndarray,
    correctness_labels: np.ndarray,
    thresholds: Iterable[float],
    *,
    minimum_accepted_signals: int,
    minimum_signal_coverage: float,
    wilson_z: float = 1.96,
    trading_evaluator: Callable[[np.ndarray], dict] | None = None,
) -> dict:
    """Select a cutoff on calibration-only observations.

    Selection maximizes the Wilson lower bound of precision. This rewards reliable
    accepted signals while penalizing thresholds supported by only a few examples.
    Backtest metrics can be attached for audit, but do not influence selection.
    """
    signals = np.asarray(model_a_signals)
    probabilities = np.asarray(model_b_probabilities, dtype=float)
    labels = np.asarray(correctness_labels)
    if not (signals.ndim == probabilities.ndim == labels.ndim == 1):
        raise ValueError("Calibration inputs must be one-dimensional")
    if not (len(signals) == len(probabilities) == len(labels)):
        raise ValueError("Calibration inputs must have equal lengths")
    if len(signals) == 0:
        raise ValueError("Calibration inputs cannot be empty")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model B probabilities must be finite values in [0, 1]")
    if not np.isin(signals, [0, 1, 2]).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Unexpected signal or correctness label")
    if not 0.0 <= minimum_signal_coverage <= 1.0:
        raise ValueError("Minimum signal coverage must be in [0, 1]")
    if minimum_accepted_signals < 1:
        raise ValueError("Minimum accepted signals must be positive")

    eligible = signals != 0
    eligible_count = int(eligible.sum())
    eligible_correct = int(labels[eligible].sum())
    if eligible_count == 0:
        raise ValueError("Model A produced no Buy/Sell signals in the calibration period")
    ungated_precision = eligible_correct / eligible_count

    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        accepted = eligible & (probabilities >= threshold)
        accepted_count = int(accepted.sum())
        correct_count = int(labels[accepted].sum())
        precision = correct_count / accepted_count if accepted_count else 0.0
        coverage = accepted_count / eligible_count
        recall = correct_count / eligible_correct if eligible_correct else 0.0
        gated_signals = np.where(accepted, signals, 0)
        row = {
            "threshold": threshold,
            "eligible_signals": eligible_count,
            "accepted_signals": accepted_count,
            "correct_accepted_signals": correct_count,
            "signal_coverage": coverage,
            "accepted_precision": precision,
            "precision_lift_over_ungated": precision - ungated_precision,
            "correct_signal_recall": recall,
            "precision_wilson_lower_95": wilson_lower_bound(
                correct_count, accepted_count, wilson_z
            ),
            "meets_constraints": (
                accepted_count >= minimum_accepted_signals
                and coverage >= minimum_signal_coverage
                and precision >= ungated_precision
            ),
        }
        if trading_evaluator is not None:
            row["trading"] = trading_evaluator(gated_signals)
        rows.append(row)

    candidates = [row for row in rows if row["meets_constraints"]]
    if not candidates:
        raise ValueError(
            "No threshold meets the pre-declared accepted-signal and coverage constraints"
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["precision_wilson_lower_95"],
            row["accepted_precision"],
            row["signal_coverage"],
            -row["threshold"],
        ),
    )
    return {
        "selected_threshold": selected["threshold"],
        "selection_rule": "maximum_precision_wilson_lower_95",
        "constraints": {
            "minimum_accepted_signals": minimum_accepted_signals,
            "minimum_signal_coverage": minimum_signal_coverage,
            "minimum_precision": "ungated_model_a_precision",
            "wilson_z": wilson_z,
        },
        "eligible_signals": eligible_count,
        "eligible_correct_signals": eligible_correct,
        "ungated_precision": ungated_precision,
        "ungated_precision_wilson_lower_95": wilson_lower_bound(
            eligible_correct, eligible_count, wilson_z
        ),
        "selected_metrics": selected,
        "threshold_results": rows,
    }
