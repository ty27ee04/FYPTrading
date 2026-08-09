"""Deterministic candle-level execution simulator matching the live policy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from strategy_config import MODEL, STRATEGY


def simulate_execution(
    market: pd.DataFrame,
    signals: np.ndarray,
    *,
    initial_equity: float,
    lot_size_for_equity: Callable[[float], float],
    take_profit_atr: float = STRATEGY.take_profit_atr,
    stop_loss_atr: float = STRATEGY.stop_loss_atr,
    spread_penalty: float = STRATEGY.spread_penalty,
    max_horizon: int = MODEL.max_horizon,
    use_break_even: bool = STRATEGY.use_break_even,
    break_even_trigger: float = STRATEGY.break_even_trigger,
    break_even_buffer: float = STRATEGY.break_even_buffer,
    reverse_on_opposite: bool = STRATEGY.reverse_on_opposite_signal,
    contract_size: float = 100.0,
    use_recorded_spread: bool = STRATEGY.use_recorded_spread,
    close_at_end: bool = False,
) -> tuple[pd.DataFrame, list[float]]:
    """Simulate one-position execution using only information available per bar.

    Signals are decisions made from the previous completed candle and execute at
    the current row's open. `signal_atr` must therefore also come from that prior
    candle. Intrabar TP/SL ambiguity is resolved against the strategy.
    """
    required = {"time", "open", "high", "low", "close", "signal_atr"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"Market data missing execution columns: {sorted(missing)}")
    signals = np.asarray(signals)
    if len(market) != len(signals):
        raise ValueError("Market rows and signals must have equal lengths")
    if not np.isin(signals, [0, 1, 2]).all():
        raise ValueError("Signals must use 0=Hold, 1=Buy, 2=Sell")

    equity = float(initial_equity)
    equity_history = []
    trades = []
    position = None

    def spread_at(index: int) -> float:
        if use_recorded_spread and "spread_price" in market.columns:
            recorded = float(market["spread_price"].iloc[index])
            if np.isfinite(recorded) and recorded >= 0:
                return recorded
        return spread_penalty

    def close_position(index: int, price: float, reason: str) -> None:
        nonlocal equity, position
        direction = position["direction"]
        raw_difference = (price - position["entry_price"]) * direction
        paid_spread = (
            position["entry_spread"]
            if direction == 1
            else spread_at(index)
        )
        pnl = (raw_difference - paid_spread) * position["lot_size"] * contract_size
        equity += pnl
        trades.append(
            {
                "Entry_Time": position["entry_time"],
                "Exit_Time": market["time"].iloc[index],
                "Direction": "Long" if direction == 1 else "Short",
                "Entry_Price": position["entry_price"],
                "Exit_Price": float(price),
                "Exit_Reason": reason,
                "Bars_Held": index - position["entry_index"],
                "Break_Even_Activated": position["break_even_active"],
                "Lot_Size": position["lot_size"],
                "Spread_Price": paid_spread,
                "Net_PnL": pnl,
                "Running_Equity": equity,
            }
        )
        position = None

    def open_position(index: int, signal: int) -> None:
        nonlocal position
        direction = 1 if signal == 1 else -1
        entry = float(market["open"].iloc[index])
        atr = float(market["signal_atr"].iloc[index])
        if not np.isfinite(atr) or atr <= 0:
            raise ValueError(f"Invalid signal ATR at execution row {index}: {atr}")
        position = {
            "direction": direction,
            "entry_index": index,
            "entry_time": market["time"].iloc[index],
            "entry_price": entry,
            "atr": atr,
            "tp": entry + direction * take_profit_atr * atr,
            "original_sl": entry - direction * stop_loss_atr * atr,
            "break_even_sl": entry + direction * break_even_buffer,
            "break_even_active": False,
            "lot_size": float(lot_size_for_equity(equity)),
            # Historical OHLC is bid-based: a Buy pays spread on entry, while a
            # Sell pays the then-current spread when buying back at exit.
            "entry_spread": spread_at(index),
        }

    for index in range(len(market)):
        open_price = float(market["open"].iloc[index])
        high = float(market["high"].iloc[index])
        low = float(market["low"].iloc[index])
        close = float(market["close"].iloc[index])
        signal = int(signals[index])

        # Live evaluates the new signal near the bar open. Opposite signals close
        # the old position first; same-direction signals are ignored by the lock.
        if position is not None:
            active_sl = (
                position["break_even_sl"]
                if position["break_even_active"]
                else position["original_sl"]
            )
            direction = position["direction"]
            if (direction == 1 and open_price <= active_sl) or (
                direction == -1 and open_price >= active_sl
            ):
                close_position(index, open_price, "Stop Loss Gap")
            elif (direction == 1 and open_price >= position["tp"]) or (
                direction == -1 and open_price <= position["tp"]
            ):
                close_position(index, position["tp"], "Take Profit")

        if position is not None and signal in (1, 2):
            signal_direction = 1 if signal == 1 else -1
            if signal_direction != position["direction"] and reverse_on_opposite:
                close_position(index, open_price, "Opposite Signal Reversal")
                open_position(index, signal)
        elif position is None and signal in (1, 2):
            open_position(index, signal)

        if position is not None:
            direction = position["direction"]
            active_sl = (
                position["break_even_sl"]
                if position["break_even_active"]
                else position["original_sl"]
            )
            hit_tp = high >= position["tp"] if direction == 1 else low <= position["tp"]
            hit_sl = low <= active_sl if direction == 1 else high >= active_sl

            if hit_tp and hit_sl:
                close_position(index, active_sl, "Stop Loss (Same Bar Conflict)")
            elif hit_sl:
                reason = "Break Even Stop" if position["break_even_active"] else "Stop Loss"
                close_position(index, active_sl, reason)
            elif hit_tp:
                close_position(index, position["tp"], "Take Profit")
            elif use_break_even and not position["break_even_active"]:
                trigger_price = position["entry_price"] + (
                    direction
                    * take_profit_atr
                    * position["atr"]
                    * break_even_trigger
                )
                trigger_hit = high >= trigger_price if direction == 1 else low <= trigger_price
                if trigger_hit:
                    position["break_even_active"] = True

            if position is not None:
                bars_held = index - position["entry_index"]
                if bars_held >= max_horizon:
                    close_position(index, close, "Time Stop")

        if close_at_end and index == len(market) - 1 and position is not None:
            close_position(index, close, "End Of Evaluation")

        equity_history.append(equity)

    return pd.DataFrame(trades), equity_history
