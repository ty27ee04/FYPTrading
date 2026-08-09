"""Derive audit diagnostics from saved Phase 6 outputs without rerunning inference."""

import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/final_evaluation")


def main() -> None:
    predictions = pd.read_csv(
        OUTPUT_DIR / "final_predictions.csv", parse_dates=["time"]
    )
    trades = pd.read_csv(
        OUTPUT_DIR / "final_trade_log.csv", parse_dates=["Entry_Time", "Exit_Time"]
    )
    accepted = predictions["final_prediction"].ne(0)
    diagnostics = {
        "source": "saved_phase_6_outputs_only",
        "inference_rerun": False,
        "actual_distribution": predictions["actual_label"].value_counts().sort_index().to_dict(),
        "model_a_distribution": predictions["model_a_prediction"].value_counts().sort_index().to_dict(),
        "final_distribution": predictions["final_prediction"].value_counts().sort_index().to_dict(),
        "accepted_directional_precision": float(
            (
                predictions.loc[accepted, "actual_label"]
                == predictions.loc[accepted, "final_prediction"]
            ).mean()
        ),
        "model_b_probability_quantiles": {
            str(key): float(value)
            for key, value in predictions["model_b_probability"]
            .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            .items()
        },
        "by_direction": {},
        "monthly": {},
    }
    for direction, group in trades.groupby("Direction"):
        diagnostics["by_direction"][direction] = {
            "trades": len(group),
            "net_pnl": float(group["Net_PnL"].sum()),
            "win_rate_percent": float(group["Net_PnL"].gt(0).mean() * 100.0),
            "average_pnl": float(group["Net_PnL"].mean()),
        }

    trades["month"] = trades["Exit_Time"].dt.to_period("M").astype(str)
    for month, group in trades.groupby("month"):
        diagnostics["monthly"][month] = {
            "trades": len(group),
            "net_pnl": float(group["Net_PnL"].sum()),
            "win_rate_percent": float(group["Net_PnL"].gt(0).mean() * 100.0),
        }

    current_loss_run = 0
    maximum_loss_run = 0
    for is_loss in trades["Net_PnL"].lt(0):
        current_loss_run = current_loss_run + 1 if is_loss else 0
        maximum_loss_run = max(maximum_loss_run, current_loss_run)
    diagnostics["maximum_consecutive_losses"] = maximum_loss_run

    path = OUTPUT_DIR / "final_evaluation_diagnostics.json"
    path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
