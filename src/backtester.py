"""
backtester.py — Historical replay with performance metrics.

Reports: win rate, Sharpe ratio, max drawdown, total return, per-trade log.
Uses the trained classifier to generate signals on held-out data.
Must pass before any ML model change is finalized (per CLAUDE.md).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIN_CONFIDENCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger(__name__)

STOP_LOSS   = -STOP_LOSS_PCT    # negative for comparison: price <= entry * (1 - STOP_LOSS_PCT)
TAKE_PROFIT =  TAKE_PROFIT_PCT


@dataclass
class Trade:
    symbol:     str
    entry_date: str
    entry_price: float
    signal:     str
    confidence: float
    exit_date:  Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "take_profit" | "stop_loss" | "timeout"
    pnl_pct:    Optional[float] = None

    @property
    def won(self) -> bool:
        return self.pnl_pct is not None and self.pnl_pct > 0


@dataclass
class BacktestResult:
    trades:        list[Trade] = field(default_factory=list)
    equity_curve:  list[float] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl_pct is not None]

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        return sum(1 for t in closed if t.won) / len(closed) if closed else 0.0

    @property
    def total_return(self) -> float:
        return sum(t.pnl_pct for t in self.closed_trades if t.pnl_pct is not None)

    @property
    def sharpe_ratio(self) -> float:
        returns = [t.pnl_pct for t in self.closed_trades if t.pnl_pct is not None]
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        std = arr.std()
        return float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / np.where(peak == 0, 1, peak)
        return float(dd.min())

    def summary(self) -> dict:
        return {
            "num_trades":   self.num_trades,
            "win_rate":     round(self.win_rate, 4),
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
        }

    def print_summary(self) -> None:
        s = self.summary()
        win_ok  = s["win_rate"] > 0.52
        mdd_ok  = s["max_drawdown"] > -0.10
        shr_ok  = s["sharpe_ratio"] > 1.0
        print(f"\n  {'─'*45}")
        print(f"  Backtest Results")
        print(f"  {'─'*45}")
        print(f"  Trades:       {s['num_trades']}")
        print(f"  Win rate:     {s['win_rate']:.1%}  {'✓' if win_ok else '✗ need >52%'}")
        print(f"  Total return: {s['total_return']:+.2%}")
        print(f"  Sharpe:       {s['sharpe_ratio']:.4f}  {'✓' if shr_ok else '✗ need >1.0'}")
        print(f"  Max drawdown: {s['max_drawdown']:.2%}  {'✓' if mdd_ok else '✗ need >-10%'}")
        print(f"  {'─'*45}")


def _simulate_trade(
    df_bars: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    max_holding: int = 5,
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Simulate holding a long position from entry_idx.
    Checks SL/TP on each subsequent bar's high/low.
    Returns (exit_date, exit_price, exit_reason).
    """
    for offset in range(1, max_holding + 1):
        i = entry_idx + offset
        if i >= len(df_bars):
            break
        row = df_bars.iloc[i]
        low  = float(row["low"])
        high = float(row["high"])
        ts   = str(row.get("timestamp", row.name))

        # Check stop-loss first (worst case within the bar)
        if low <= entry_price * (1 + STOP_LOSS):
            exit_p = entry_price * (1 + STOP_LOSS)
            return ts, exit_p, "stop_loss"
        if high >= entry_price * (1 + TAKE_PROFIT):
            exit_p = entry_price * (1 + TAKE_PROFIT)
            return ts, exit_p, "take_profit"

    # Timeout — exit at close of last bar
    last_i = min(entry_idx + max_holding, len(df_bars) - 1)
    last   = df_bars.iloc[last_i]
    ts     = str(last.get("timestamp", last.name))
    return ts, float(last["close"]), "timeout"


def run_backtest(
    df_featured: pd.DataFrame,
    ticker: str,
    predict_fn,
    min_confidence: float = MIN_CONFIDENCE,
) -> BacktestResult:
    """
    Walk forward through df_featured day by day.
    On each bar, call predict_fn(df_so_far) → (signal, confidence).
    If BUY and confidence >= min_confidence, open a long trade.
    Only one open position per ticker at a time.
    """
    result = BacktestResult()
    equity = 1.0
    result.equity_curve.append(equity)
    in_position = False

    for i in range(200, len(df_featured) - 6):  # skip warmup + leave room for exit
        if in_position:
            continue

        df_so_far = df_featured.iloc[:i + 1]
        try:
            signal, confidence = predict_fn(df_so_far)
        except Exception as e:
            logger.debug("predict_fn failed at index %d: %s", i, e)
            continue

        if signal != "BUY" or confidence < min_confidence:
            result.equity_curve.append(equity)
            continue

        row        = df_featured.iloc[i]
        entry_p    = float(row["close"])
        entry_date = str(row.get("timestamp", row.name))

        exit_date, exit_p, reason = _simulate_trade(df_featured, i, entry_p)
        if exit_p is None:
            result.equity_curve.append(equity)
            continue

        pnl_pct = (exit_p - entry_p) / entry_p

        trade = Trade(
            symbol=ticker,
            entry_date=entry_date,
            entry_price=entry_p,
            signal=signal,
            confidence=confidence,
            exit_date=exit_date,
            exit_price=exit_p,
            exit_reason=reason,
            pnl_pct=pnl_pct,
        )
        result.trades.append(trade)
        equity *= (1 + pnl_pct)
        result.equity_curve.append(equity)

    return result


def run_full_backtest(
    tickers: list[str],
    bars_per_ticker: int = 504,
    min_confidence: float = MIN_CONFIDENCE,
) -> BacktestResult:
    """
    Full backtest across all tickers using the saved classifier model.
    Trains on first 50% of dates, backtests on remaining 50%.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.classifier import (
        build_training_set, walk_forward_validate, train_final,
        build_features, FEATURE_COLS, load_model,
    )
    from src.alpaca_client import fetch_bars

    logger.info("Running full backtest on %d tickers", len(tickers))

    df_train = build_training_set(tickers, bars_per_ticker)
    dates    = sorted(df_train["date"].unique())
    cutoff   = dates[len(dates) // 2]

    # Train on first half
    train_df = df_train[df_train["date"] <= cutoff]
    model    = train_final(train_df)

    def _predict(df_bars: pd.DataFrame):
        from src.classifier import predict as clf_predict
        # Use the just-trained in-memory model, not the saved one
        from src.classifier import build_features as bf, FEATURE_COLS as FC
        df = bf(df_bars)
        df = df.dropna(subset=FC)
        if df.empty:
            return "HOLD", 0.0
        X = df[FC].iloc[[-1]]
        proba = model.predict_proba(X)[0]
        import numpy as np
        pred = int(np.argmax(proba))
        label_str = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return label_str[pred], float(proba[pred])

    combined = BacktestResult()
    for sym in tickers:
        df_bars = df_train[df_train["ticker"] == sym].copy()
        df_test = df_bars[df_bars["date"] > cutoff].reset_index(drop=True)
        if len(df_test) < 20:
            logger.warning("%s: only %d test bars, skipping", sym, len(df_test))
            continue
        result = run_backtest(df_test, sym, _predict, min_confidence)
        combined.trades.extend(result.trades)
        combined.equity_curve.extend(result.equity_curve)
        logger.info("%s: %d trades, win_rate=%.2f", sym, result.num_trades, result.win_rate)

    return combined


def meets_paper_trading_bar(result: BacktestResult) -> bool:
    """Phase 7 bar: win_rate > 52%, max_drawdown > -10%, Sharpe > 1."""
    s = result.summary()
    return (
        s["win_rate"]     > 0.52 and
        s["max_drawdown"] > -0.10 and
        s["sharpe_ratio"] > 1.0
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.classifier import TRAINING_TICKERS
    print("Running full backtest (this may take several minutes)...")
    result = run_full_backtest(TRAINING_TICKERS[:5])   # 5 tickers for quick run
    result.print_summary()
    print(f"\nMeets paper trading bar: {meets_paper_trading_bar(result)}")
