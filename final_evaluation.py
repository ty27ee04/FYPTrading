"""One-shot locked evaluation of the frozen classifier on the final dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from execution_simulator import simulate_execution
from strategy_config import MODEL, load_gatekeeper_threshold


ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "final_evaluation_lock.json"
OUTPUT_DIR = ROOT / "outputs" / "final_evaluation"
ARTIFACT_PATHS = {
    "evaluator": ROOT / "final_evaluation.py",
    "training_pipeline": ROOT / "gemini-training.py",
    "execution_simulator": ROOT / "execution_simulator.py",
    "model_a": ROOT / "best_model_a.pth",
    "model_b": ROOT / "best_model_b.pth",
    "scaler": ROOT / "scaler.pkl",
    "threshold": ROOT / "threshold_calibration.json",
    "final_dataset": ROOT / "XAUUSD_M5_6month.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=["Hold", "Buy", "Sell"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "report": json_ready(report),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def build_trading_metrics(
    market: pd.DataFrame,
    signals: np.ndarray,
    *,
    initial_equity: float = 10000.0,
    fixed_lot: float = 0.01,
) -> tuple[dict, pd.DataFrame, list[float]]:
    trades, history = simulate_execution(
        market,
        signals,
        initial_equity=initial_equity,
        lot_size_for_equity=lambda _: fixed_lot,
        close_at_end=True,
    )
    equity = np.asarray([initial_equity, *history], dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / np.maximum(peaks, 1e-9)
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = 0.0
    if len(returns) and float(returns.std(ddof=0)) > 0:
        sharpe = float(
            returns.mean() / returns.std(ddof=0) * np.sqrt(288 * 252)
        )
    pnl = trades["Net_PnL"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    metrics = {
        "initial_equity": initial_equity,
        "final_equity": float(equity[-1]),
        "net_profit": float(equity[-1] - initial_equity),
        "return_percent": float((equity[-1] / initial_equity - 1.0) * 100.0),
        "maximum_drawdown_percent": float(drawdown.min() * 100.0),
        "sharpe_ratio": sharpe,
        "executed_trades": int(len(trades)),
        "winning_trades": int((pnl > 0).sum()),
        "losing_trades": int((pnl < 0).sum()),
        "win_rate_percent": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        "average_trade_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "exit_reasons": (
            {str(key): int(value) for key, value in trades["Exit_Reason"].value_counts().items()}
            if not trades.empty
            else {}
        ),
    }
    return metrics, trades, history


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "gemini_training_final", ROOT / "gemini-training.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the frozen training definitions")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lock(payload: dict) -> None:
    LOCK_PATH.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def main() -> int:
    if LOCK_PATH.exists():
        raise SystemExit(
            f"Final evaluation lock already exists at {LOCK_PATH}; refusing to run twice."
        )
    missing = [str(path) for path in ARTIFACT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen evaluation inputs are missing: {missing}")

    started_at = datetime.now().astimezone().isoformat()
    fingerprints = {name: sha256(path) for name, path in ARTIFACT_PATHS.items()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = {
        "state": "running",
        "started_at": started_at,
        "git_commit": commit,
        "fingerprints_sha256": fingerprints,
        "policy": {
            "single_evaluation_only": True,
            "retraining_allowed": False,
            "post_test_tuning_allowed": False,
        },
    }
    write_lock(lock)

    try:
        training = load_training_module()
        scaler = joblib.load(ARTIFACT_PATHS["scaler"])
        (
            _x_train,
            _y_train,
            _x_validation,
            _y_validation,
            _x_meta,
            _y_meta,
            _meta_market,
            x_final,
            y_final,
            final_market,
            _metadata,
        ) = training.preprocess_gold_data(
            ROOT / "XAUUSD_M5_2Year.csv",
            ARTIFACT_PATHS["final_dataset"],
            scaler_override=scaler,
            write_artifacts=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_a_state = torch.load(
            ARTIFACT_PATHS["model_a"], map_location=device, weights_only=True
        )
        hidden_dimension = int(model_a_state["lstm.weight_hh_l0"].shape[1])
        model_a = training.ModelA_Base(x_final.shape[2], hidden_dimension).to(device)
        model_b = training.ModelB_TCN(x_final.shape[2]).to(device)
        model_a.load_state_dict(model_a_state)
        model_b.load_state_dict(
            torch.load(
                ARTIFACT_PATHS["model_b"], map_location=device, weights_only=True
            )
        )
        model_a.eval()
        model_b.eval()

        model_a_predictions = []
        model_b_probabilities = []
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_final)), batch_size=512, shuffle=False
        )
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                model_a_predictions.extend(
                    torch.argmax(model_a(batch), dim=1).cpu().numpy()
                )
                model_b_probabilities.extend(
                    F.softmax(model_b(batch), dim=1)[:, 1].cpu().numpy()
                )

        model_a_predictions = np.asarray(model_a_predictions, dtype=int)
        model_b_probabilities = np.asarray(model_b_probabilities, dtype=float)
        threshold = load_gatekeeper_threshold(ARTIFACT_PATHS["threshold"])
        final_predictions = np.where(
            (model_a_predictions != 0) & (model_b_probabilities >= threshold),
            model_a_predictions,
            0,
        )

        model_a_metrics = build_classification_metrics(y_final, model_a_predictions)
        gated_metrics = build_classification_metrics(y_final, final_predictions)
        eligible = model_a_predictions != 0
        accepted = final_predictions != 0
        signal_metrics = {
            "total_sequences": int(len(y_final)),
            "model_a_signals": int(eligible.sum()),
            "model_a_signal_coverage_percent": float(eligible.mean() * 100.0),
            "accepted_signals": int(accepted.sum()),
            "accepted_signal_coverage_percent": float(accepted.mean() * 100.0),
            "gate_acceptance_percent_of_model_a_signals": (
                float(accepted.sum() / eligible.sum() * 100.0) if eligible.any() else 0.0
            ),
        }

        trading_metrics, trades, equity_history = build_trading_metrics(
            final_market, final_predictions
        )
        results = {
            "evaluation_name": "locked_final_six_month_classifier_evaluation",
            "started_at": started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "git_commit": commit,
            "fingerprints_sha256": fingerprints,
            "period": {
                "start": final_market["time"].iloc[0].isoformat(),
                "end": final_market["time"].iloc[-1].isoformat(),
                "sequences": len(final_market),
            },
            "frozen_threshold": threshold,
            "model_a_ungated_classification": model_a_metrics,
            "hierarchical_gated_classification": gated_metrics,
            "signal_metrics": signal_metrics,
            "fixed_001_lot_trading": trading_metrics,
            "policy": lock["policy"],
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_path = OUTPUT_DIR / "final_evaluation_results.json"
        results_path.write_text(
            json.dumps(json_ready(results), indent=2), encoding="utf-8"
        )
        predictions = pd.DataFrame(
            {
                "time": final_market["time"],
                "actual_label": y_final,
                "model_a_prediction": model_a_predictions,
                "model_b_probability": model_b_probabilities,
                "final_prediction": final_predictions,
            }
        )
        predictions.to_csv(OUTPUT_DIR / "final_predictions.csv", index=False)
        trades.to_csv(OUTPUT_DIR / "final_trade_log.csv", index=False)
        pd.DataFrame(
            gated_metrics["confusion_matrix"],
            index=["Actual Hold", "Actual Buy", "Actual Sell"],
            columns=["Predicted Hold", "Predicted Buy", "Predicted Sell"],
        ).to_csv(OUTPUT_DIR / "final_confusion_matrix.csv")
        pd.DataFrame(
            {"time": final_market["time"], "equity": equity_history}
        ).to_csv(OUTPUT_DIR / "final_equity_curve.csv", index=False)

        lock.update(
            {
                "state": "completed",
                "completed_at": results["completed_at"],
                "results_path": str(results_path.relative_to(ROOT)),
                "selected_summary": {
                    "threshold": threshold,
                    **signal_metrics,
                    **trading_metrics,
                },
            }
        )
        write_lock(lock)
        print(json.dumps(json_ready(lock["selected_summary"]), indent=2))
        print(f"Final evaluation completed once. Results: {results_path}")
        return 0
    except Exception as error:
        lock.update(
            {
                "state": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "error": repr(error),
                "manual_review_required_before_any_retry": True,
            }
        )
        write_lock(lock)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
