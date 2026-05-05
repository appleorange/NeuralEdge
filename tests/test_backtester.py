"""Verification test for backtester.py."""
import logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from src.backtester import Trade, BacktestResult, _simulate_trade, run_backtest, meets_paper_trading_bar
from src.alpaca_client import fetch_bars
from src.classifier import build_features, load_model, FEATURE_COLS

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []

def check(label, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, cond))
    return cond

def section(t):
    print(f"\n{'─'*55}\n  {t}\n{'─'*55}")


# ── Test 1: BacktestResult metrics ────────────────────────────────────────────
section("TEST 1 — BacktestResult metrics")

r = BacktestResult()
r.trades = [
    Trade("AAPL","2024-01-01",100,"BUY",0.7,exit_price=105,exit_reason="take_profit",pnl_pct=0.05),
    Trade("AAPL","2024-01-06",105,"BUY",0.7,exit_price=102,exit_reason="stop_loss",pnl_pct=-0.0286),
    Trade("AAPL","2024-01-11",102,"BUY",0.7,exit_price=107,exit_reason="take_profit",pnl_pct=0.049),
    Trade("AAPL","2024-01-16",107,"BUY",0.7,exit_price=104,exit_reason="timeout",pnl_pct=-0.028),
    Trade("AAPL","2024-01-21",104,"BUY",0.7,exit_price=109,exit_reason="take_profit",pnl_pct=0.048),
]
r.equity_curve = [1.0, 1.05, 1.02, 1.07, 1.04, 1.09]

check("win_rate = 3/5 = 0.6", abs(r.win_rate - 0.6) < 0.01, f"got {r.win_rate:.4f}")
check("total_return = sum of pnl", abs(r.total_return - sum(t.pnl_pct for t in r.trades)) < 0.001)
check("sharpe computes without error", isinstance(r.sharpe_ratio, float))
check("max_drawdown <= 0", r.max_drawdown <= 0, f"got {r.max_drawdown:.4f}")
check("summary() has all keys", set(r.summary().keys()) == {"num_trades","win_rate","total_return","sharpe_ratio","max_drawdown"})
check("meets_paper_bar returns bool", isinstance(meets_paper_trading_bar(r), bool))


# ── Test 2: _simulate_trade ───────────────────────────────────────────────────
section("TEST 2 — _simulate_trade (SL/TP logic)")

df = fetch_bars("AAPL", limit=60)
entry_idx = 30
entry_p   = float(df.iloc[entry_idx]["close"])

exit_date, exit_p, reason = _simulate_trade(df, entry_idx, entry_p, max_holding=5)
check("Returns exit date", bool(exit_date))
check("Returns exit price > 0", exit_p is not None and exit_p > 0, f"exit_p={exit_p:.2f}")
check("Reason is valid", reason in {"take_profit","stop_loss","timeout"}, f"reason={reason}")

# Construct synthetic bars where TP is hit on day 2
tp_bars = pd.concat([df.iloc[:entry_idx+1]] + [df.iloc[[entry_idx]]] * 5, ignore_index=True)
tp_bars = tp_bars.copy()
high_col = tp_bars.columns.get_loc("high")
tp_bars.iloc[-4, high_col] = entry_p * 1.06   # TP triggered
_, _, tp_reason = _simulate_trade(tp_bars, entry_idx, entry_p)
check("Take-profit triggered when high > entry*1.05", tp_reason == "take_profit", f"got {tp_reason}")

# Construct synthetic bars where SL is hit
sl_bars = pd.concat([df.iloc[:entry_idx+1]] + [df.iloc[[entry_idx]]] * 5, ignore_index=True)
sl_bars = sl_bars.copy()
low_col = sl_bars.columns.get_loc("low")
sl_bars.iloc[-4, low_col] = entry_p * 0.96    # SL triggered
_, _, sl_reason = _simulate_trade(sl_bars, entry_idx, entry_p)
check("Stop-loss triggered when low < entry*0.97", sl_reason == "stop_loss", f"got {sl_reason}")


# ── Test 3: run_backtest end-to-end ───────────────────────────────────────────
section("TEST 3 — run_backtest() end-to-end")

df_live = fetch_bars("AAPL", limit=300)
df_feat = build_features(df_live)
model   = load_model()

def _predict(df_bars):
    df = build_features(df_bars)
    df = df.dropna(subset=FEATURE_COLS)
    if df.empty: return "HOLD", 0.0
    X = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    return {0:"SELL",1:"HOLD",2:"BUY"}[pred], float(proba[pred])

bt = run_backtest(df_feat, "AAPL", _predict, min_confidence=0.65)
check("run_backtest returns BacktestResult", isinstance(bt, BacktestResult))
check("equity_curve non-empty", len(bt.equity_curve) > 0)
check("all trade pnl_pct set", all(t.pnl_pct is not None for t in bt.trades))
check("all exit reasons valid", all(t.exit_reason in {"take_profit","stop_loss","timeout"} for t in bt.trades))
check("win_rate in [0,1]", 0.0 <= bt.win_rate <= 1.0, f"win_rate={bt.win_rate:.4f}")
print(f"\n  AAPL backtest: {bt.num_trades} trades  win_rate={bt.win_rate:.1%}  "
      f"return={bt.total_return:+.2%}  sharpe={bt.sharpe_ratio:.4f}  mdd={bt.max_drawdown:.2%}")
bt.print_summary()


# ── Test 4: Edge cases ────────────────────────────────────────────────────────
section("TEST 4 — Edge cases")

# predict always returns HOLD → no trades
bt_hold = run_backtest(df_feat, "AAPL", lambda df: ("HOLD", 0.9))
check("HOLD-only predictor → 0 trades", bt_hold.num_trades == 0)

# predict always BUY but low confidence → no trades
bt_low = run_backtest(df_feat, "AAPL", lambda df: ("BUY", 0.3), min_confidence=0.65)
check("Low confidence → 0 trades", bt_low.num_trades == 0)

# Empty trades → metrics return 0
empty_bt = BacktestResult()
check("Empty result: win_rate=0", empty_bt.win_rate == 0.0)
check("Empty result: sharpe=0", empty_bt.sharpe_ratio == 0.0)
check("Empty result: max_dd=0", empty_bt.max_drawdown == 0.0)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  RESULT: {passed}/{total} checks passed")
if passed == total:
    print("  backtester.py VERIFIED — Phase 3 complete")
else:
    print("  FAILURES:")
    for label, ok in results:
        if not ok: print(f"    • {label}")
print(f"{'═'*55}")
sys.exit(0 if passed == total else 1)
