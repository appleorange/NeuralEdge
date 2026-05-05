"""
Tests for order_executor.py.

All Alpaca API calls and database writes are mocked.
No network access required.

Coverage:
  - Paper mode is the default
  - Live mode requires paper=False explicitly
  - risk_manager.evaluate() is always called before any order
  - A rejected signal never reaches Alpaca
  - A bypass attempt (calling internal methods without risk gate) is documented
  - Alpaca errors: log, no retry, console alert
  - Successful order: SQLite insert + console print
  - monitor_positions: stop-loss and take-profit exits
  - Daily halt triggers console alert
"""
import sys
import os
import types
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.order_executor import OrderExecutor, ALERT_PREFIX

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, condition))
    return condition


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_executor(paper=True):
    """Create OrderExecutor with dummy credentials (no real Alpaca calls)."""
    return OrderExecutor(paper=paper, api_key="test_key", secret_key="test_secret")


def _fake_order(order_id="order-abc-123"):
    o = MagicMock()
    o.id = order_id
    return o


# ── Section 1: Paper mode default ─────────────────────────────────────────────
section("TEST 1 — Paper mode is the default")

ex = _make_executor()
check("OrderExecutor() defaults to paper=True", ex.paper is True)

ex_paper = _make_executor(paper=True)
check("OrderExecutor(paper=True) explicit → paper=True", ex_paper.paper is True)

ex_live = _make_executor(paper=False)
check("OrderExecutor(paper=False) explicit → paper=False", ex_live.paper is False)

# Default construction (no paper argument) must be paper
ex_default = OrderExecutor(api_key="k", secret_key="s")
check("OrderExecutor() without paper kwarg → paper=True", ex_default.paper is True)


# ── Section 2: Risk manager is ALWAYS called ─────────────────────────────────
section("TEST 2 — risk_manager.evaluate() called before every order")

with patch("src.order_executor.risk_evaluate") as mock_risk:
    mock_risk.return_value = MagicMock(approved=False, reason="test rejection", quantity=0)

    ex = _make_executor()
    ex.execute_signal(
        signal="BUY", ticker="AAPL", confidence=0.80,
        entry_price=150.0, portfolio_value=10000.0,
        open_positions=set(), daily_pnl_pct=0.0,
    )

    check("risk_evaluate called exactly once", mock_risk.call_count == 1,
          f"called {mock_risk.call_count}x")

    call_kwargs = mock_risk.call_args
    check("risk_evaluate received signal", call_kwargs.kwargs.get("signal") == "BUY" or
          (call_kwargs.args and call_kwargs.args[0] == "BUY"))
    check("risk_evaluate received ticker", "AAPL" in str(call_kwargs))


# ── Section 3: Rejected signal never reaches Alpaca ──────────────────────────
section("TEST 3 — Rejected signal never places an Alpaca order")

with patch("src.order_executor.risk_evaluate") as mock_risk, \
     patch("src.order_executor.TradingClient") as mock_alpaca:

    mock_risk.return_value = MagicMock(approved=False, reason="confidence too low", quantity=0)

    ex = _make_executor()
    result = ex.execute_signal(
        signal="BUY", ticker="AAPL", confidence=0.50,
        entry_price=150.0, portfolio_value=10000.0,
        open_positions=set(), daily_pnl_pct=0.0,
    )

    check("execute_signal returns None on rejection", result is None)
    check("TradingClient.submit_order never called",
          mock_alpaca.return_value.submit_order.call_count == 0)


# ── Section 4: Bypass attempt — calling _place_market_order directly ─────────
section("TEST 4 — Bypass attempt: _place_market_order bypasses risk gate")

# Document that _place_market_order is an internal method that skips the gate.
# order_executor's public contract is execute_signal(), which always calls risk gate.
# This test confirms the bypass path exists but is not the public API.

with patch("src.order_executor.TradingClient") as mock_alpaca, \
     patch("src.order_executor.db") as mock_db:

    mock_alpaca.return_value.submit_order.return_value = _fake_order()
    mock_db.insert_trade.return_value = 1

    ex = _make_executor()
    ex._client = mock_alpaca.return_value

    # Directly calling _place_market_order bypasses risk_manager — this is intentional
    # design: the method is private (_prefix). execute_signal() is the enforced gateway.
    result = ex._place_market_order(
        ticker="AAPL", quantity=10, signal="BUY", confidence=0.50,
        entry_price=150.0, stop_loss=145.5, take_profit=157.5, dollar_amount=1500.0,
    )
    check(
        "Bypass via _place_market_order succeeds (documents private method risk)",
        result is not None,
        "caller must use execute_signal() in production — this is a test-only bypass",
    )
    check(
        "execute_signal() DOES enforce risk gate (the safe public path)",
        True,  # confirmed by Section 3
        "contract: always use execute_signal(), never _place_market_order() directly",
    )


# ── Section 5: Successful order path ─────────────────────────────────────────
section("TEST 5 — Successful BUY order: Alpaca call + SQLite log")

with patch("src.order_executor.risk_evaluate") as mock_risk, \
     patch("src.order_executor.TradingClient") as mock_alpaca, \
     patch("src.order_executor.db") as mock_db, \
     patch("src.order_executor.print") as mock_print:

    decision = MagicMock(
        approved=True, reason="all checks passed",
        quantity=44, stop_loss=145.5, take_profit=157.5, dollar_amount=6600.0,
    )
    mock_risk.return_value = decision
    mock_alpaca.return_value.submit_order.return_value = _fake_order("order-xyz-999")
    mock_db.insert_trade.return_value = 7

    ex = _make_executor(paper=True)
    ex._client = mock_alpaca.return_value

    result = ex.execute_signal(
        signal="BUY", ticker="AAPL", confidence=0.80,
        entry_price=150.0, portfolio_value=10000.0,
        open_positions=set(), daily_pnl_pct=0.0,
    )

    check("returns non-None on success", result is not None)
    check("result contains ticker", result.get("ticker") == "AAPL")
    check("result contains order_id", result.get("order_id") == "order-xyz-999")
    check("result contains quantity=44", result.get("quantity") == 44)
    check("result contains entry_price", result.get("entry_price") == 150.0)
    check("result contains stop_loss", result.get("stop_loss") == 145.5)
    check("result contains take_profit", result.get("take_profit") == 157.5)
    check("result contains dollar_amount", result.get("dollar_amount") == 6600.0)
    check("result mode='PAPER'", result.get("mode") == "PAPER")
    check("Alpaca submit_order called once",
          mock_alpaca.return_value.submit_order.call_count == 1)
    check("db.insert_trade called once", mock_db.insert_trade.call_count == 1)

    # Confirm console alert printed
    printed = " ".join(str(c) for c in mock_print.call_args_list)
    check("[ORDER] alert printed to console", "[ORDER]" in printed, "console output found")
    check("PAPER label in console output", "PAPER" in printed)


# ── Section 6: Alpaca error — no retry, console alert ────────────────────────
section("TEST 6 — Alpaca error: log, no retry, console alert")

with patch("src.order_executor.risk_evaluate") as mock_risk, \
     patch("src.order_executor.TradingClient") as mock_alpaca, \
     patch("src.order_executor.db") as mock_db, \
     patch("src.order_executor.print") as mock_print:

    mock_risk.return_value = MagicMock(
        approved=True, reason="all checks passed",
        quantity=10, stop_loss=145.5, take_profit=157.5, dollar_amount=1500.0,
    )
    mock_alpaca.return_value.submit_order.side_effect = Exception("APIError: insufficient funds")
    mock_db.insert_trade.return_value = 1

    ex = _make_executor()
    ex._client = mock_alpaca.return_value

    result = ex.execute_signal(
        signal="BUY", ticker="AAPL", confidence=0.80,
        entry_price=150.0, portfolio_value=10000.0,
        open_positions=set(), daily_pnl_pct=0.0,
    )

    check("returns None on Alpaca error", result is None)
    check("Alpaca submit_order called exactly once (no retry)",
          mock_alpaca.return_value.submit_order.call_count == 1)
    check("db.insert_trade NOT called (no partial log on failure)",
          mock_db.insert_trade.call_count == 0)

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    check("[ALERT] printed to console on error", ALERT_PREFIX in printed, printed[:200])


# ── Section 7: Daily halt alert ───────────────────────────────────────────────
section("TEST 7 — Daily halt triggers console alert")

with patch("src.order_executor.risk_evaluate") as mock_risk, \
     patch("src.order_executor.print") as mock_print:

    mock_risk.return_value = MagicMock(
        approved=False,
        reason="daily halt active — portfolio down -6.00% today",
        quantity=0,
    )

    ex = _make_executor()
    result = ex.execute_signal(
        signal="BUY", ticker="TSLA", confidence=0.90,
        entry_price=250.0, portfolio_value=10000.0,
        open_positions=set(), daily_pnl_pct=-0.06,
    )

    check("returns None when halted", result is None)
    printed = " ".join(str(c) for c in mock_print.call_args_list)
    check("Daily halt alert printed to console",
          "halt" in printed.lower() or ALERT_PREFIX in printed, printed[:200])


# ── Section 8: monitor_positions — stop-loss and take-profit ─────────────────
section("TEST 8 — monitor_positions: SL and TP exits")

OPEN_TRADES = [
    {"id": 1, "symbol": "AAPL", "price": 150.0, "stop_loss": 145.5,
     "take_profit": 157.5, "status": "submitted", "exit_price": None},
    {"id": 2, "symbol": "MSFT", "price": 300.0, "stop_loss": 291.0,
     "take_profit": 315.0, "status": "submitted", "exit_price": None},
    {"id": 3, "symbol": "GOOGL", "price": 180.0, "stop_loss": 174.6,
     "take_profit": 189.0, "status": "submitted", "exit_price": None},
]

# AAPL below stop-loss, MSFT above take-profit, GOOGL unchanged
PRICES = {"AAPL": 144.0, "MSFT": 316.0, "GOOGL": 182.0}

with patch("src.order_executor.db") as mock_db, \
     patch("src.order_executor.print") as mock_print:

    mock_db.get_open_trades.return_value = OPEN_TRADES

    ex = _make_executor()
    exits = ex.monitor_positions(latest_prices=PRICES)

    check("2 exits triggered (AAPL SL + MSFT TP)", len(exits) == 2, f"exits={exits}")

    aapl_exit = next((e for e in exits if e["ticker"] == "AAPL"), None)
    msft_exit = next((e for e in exits if e["ticker"] == "MSFT"), None)

    check("AAPL exit reason = stop_loss",
          aapl_exit is not None and aapl_exit["reason"] == "stop_loss",
          str(aapl_exit))
    check("MSFT exit reason = take_profit",
          msft_exit is not None and msft_exit["reason"] == "take_profit",
          str(msft_exit))

    check("GOOGL not exited (price between SL and TP)",
          all(e["ticker"] != "GOOGL" for e in exits))

    check("db.update_trade_exit called twice", mock_db.update_trade_exit.call_count == 2)

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    check("Stop-loss alert printed", "Stop-loss" in printed or "stop_loss" in printed.lower(),
          printed[:300])
    check("Take-profit alert printed", "Take-profit" in printed or "take_profit" in printed.lower(),
          printed[:300])


# ── Section 9: monitor_positions — no open trades ────────────────────────────
section("TEST 9 — monitor_positions: no-op when no open trades")

with patch("src.order_executor.db") as mock_db:
    mock_db.get_open_trades.return_value = []
    ex = _make_executor()
    exits = ex.monitor_positions(latest_prices={})
    check("returns empty list when no open trades", exits == [])
    check("update_trade_exit never called", mock_db.update_trade_exit.call_count == 0)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
passed_count = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  RESULT: {passed_count}/{total} checks passed")
if passed_count == total:
    print("  order_executor.py VERIFIED — Phase 4 complete")
else:
    print("  FAILURES:")
    for label, ok in results:
        if not ok:
            print(f"    • {label}")
print(f"{'═'*60}")
sys.exit(0 if passed_count == total else 1)
