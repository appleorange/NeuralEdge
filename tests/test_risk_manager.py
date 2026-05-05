"""
Tests for risk_manager.py — pure logic, no API calls, no database.

Coverage:
  - Non-BUY signals rejected
  - Confidence gate (below, at, and above threshold)
  - Existing position blocked
  - Daily halt (at threshold, below threshold, just above)
  - Happy path: all checks pass
  - Position sizing formula
  - stop_loss_price and take_profit_price helpers
  - RiskDecision fields populated correctly
  - Bypass attempts: confirm gate cannot be short-circuited
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_manager import (
    evaluate,
    size_position,
    stop_loss_price,
    take_profit_price,
    RiskDecision,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_RISK_PCT,
    DAILY_HALT_PCT,
    MIN_CONFIDENCE,
)

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


# ── Fixtures ──────────────────────────────────────────────────────────────────

PORTFOLIO    = 10_000.0
ENTRY        = 150.0
TICKER       = "AAPL"
GOOD_CONF    = 0.75
EMPTY_POS    = set()
NORMAL_PNL   = 0.0   # flat day


# ── Section 1: Non-BUY signals ────────────────────────────────────────────────
section("TEST 1 — Non-BUY signals are rejected")

for sig in ("HOLD", "SELL", "hold", "sell", "", "UNKNOWN"):
    d = evaluate(sig, TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
    check(f"signal='{sig}' → rejected", not d.approved, f"reason: {d.reason}")
    check(f"signal='{sig}' → quantity=0", d.quantity == 0)


# ── Section 2: Confidence gate ────────────────────────────────────────────────
section("TEST 2 — Confidence gate")

d = evaluate("BUY", TICKER, 0.0, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check("confidence=0.0 → rejected", not d.approved, d.reason)

d = evaluate("BUY", TICKER, 0.64, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check("confidence=0.64 (one tick below MIN) → rejected", not d.approved, d.reason)

d = evaluate("BUY", TICKER, MIN_CONFIDENCE - 0.001, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check(f"confidence={MIN_CONFIDENCE - 0.001:.3f} → rejected", not d.approved, d.reason)

d = evaluate("BUY", TICKER, MIN_CONFIDENCE, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check(f"confidence=MIN_CONFIDENCE={MIN_CONFIDENCE} (exact) → approved", d.approved, d.reason)

d = evaluate("BUY", TICKER, 0.99, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check("confidence=0.99 → approved", d.approved, d.reason)


# ── Section 3: Existing position check ───────────────────────────────────────
section("TEST 3 — Existing position blocks re-entry")

open_with_ticker = {TICKER, "MSFT"}
d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, open_with_ticker, NORMAL_PNL)
check(f"ticker already in open_positions → rejected", not d.approved, d.reason)
check("reason mentions ticker", TICKER in d.reason)

# Different ticker is not blocked by another ticker's position
d = evaluate("BUY", "GOOGL", GOOD_CONF, ENTRY, PORTFOLIO, open_with_ticker, NORMAL_PNL)
check("different ticker not in open_positions → approved", d.approved, d.reason)

# Bypass attempt 1: pass empty set even though position logically exists
# (order_executor must always pass the real position set — test that evaluate
#  trusts the input and a wrong input leads to a wrong (but not crashed) outcome,
#  confirming evaluate() cannot self-verify; the caller must be honest)
d_bypass = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check(
    "Bypass attempt: passing empty set skips position check (caller responsibility)",
    d_bypass.approved,  # this PASSES — documents that evaluate() trusts its inputs
    "evaluate() is pure logic; order_executor must supply the real position set",
)


# ── Section 4: Daily halt ─────────────────────────────────────────────────────
section("TEST 4 — Daily halt")

at_threshold = -DAILY_HALT_PCT           # exactly -5%
below        = -DAILY_HALT_PCT - 0.01    # -6%
just_above   = -DAILY_HALT_PCT + 0.001   # -4.9%

d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, at_threshold)
check(f"daily_pnl_pct={at_threshold:.1%} (at threshold) → halted", not d.approved, d.reason)

d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, below)
check(f"daily_pnl_pct={below:.1%} (below threshold) → halted", not d.approved, d.reason)

d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, just_above)
check(f"daily_pnl_pct={just_above:.3%} (just above threshold) → NOT halted", d.approved, d.reason)

d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, 0.03)
check("daily_pnl_pct=+3% (profitable day) → not halted", d.approved, d.reason)

# Halt message contains pnl information
d_halt = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, -0.07)
check("halt reason mentions 'halt'", "halt" in d_halt.reason.lower(), d_halt.reason)


# ── Section 5: Happy path ─────────────────────────────────────────────────────
section("TEST 5 — Happy path: all checks pass")

d = evaluate("BUY", TICKER, GOOD_CONF, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check("approved=True", d.approved, d.reason)
check("quantity > 0", d.quantity > 0, f"qty={d.quantity}")
check("stop_loss < entry", d.stop_loss < ENTRY, f"sl={d.stop_loss}")
check("take_profit > entry", d.take_profit > ENTRY, f"tp={d.take_profit}")
check("dollar_amount > 0", d.dollar_amount > 0, f"$={d.dollar_amount}")
check("reason='all checks passed'", d.reason == "all checks passed")


# ── Section 6: Position sizing ────────────────────────────────────────────────
section("TEST 6 — Position sizing (size_position)")

# Formula: qty = floor((portfolio * MAX_RISK_PCT) / (entry * STOP_LOSS_PCT))
portfolio = 10_000.0
entry     = 100.0
expected  = int((portfolio * MAX_RISK_PCT) / (entry * STOP_LOSS_PCT))
check(
    f"size_position($10k, $100) = {expected}",
    size_position(portfolio, entry) == expected,
    f"got={size_position(portfolio, entry)}",
)

# Minimum of 1 share
check("minimum 1 share", size_position(100.0, 999.0) >= 1)

# Zero / negative inputs → 0 shares
check("portfolio=0 → 0 shares", size_position(0, 100.0) == 0)
check("entry=0 → 0 shares", size_position(10_000.0, 0) == 0)
check("negative entry → 0 shares", size_position(10_000.0, -50.0) == 0)

# Larger portfolio → proportionally more shares
qty_small = size_position(5_000.0, 100.0)
qty_large = size_position(20_000.0, 100.0)
check("larger portfolio → more shares", qty_large > qty_small,
      f"small={qty_small} large={qty_large}")

# Higher entry price → fewer shares
qty_cheap = size_position(10_000.0, 50.0)
qty_pricey = size_position(10_000.0, 500.0)
check("higher entry price → fewer shares", qty_cheap > qty_pricey,
      f"$50={qty_cheap} $500={qty_pricey}")


# ── Section 7: stop_loss_price and take_profit_price ─────────────────────────
section("TEST 7 — stop_loss_price / take_profit_price helpers")

entry = 200.0
sl = stop_loss_price(entry)
tp = take_profit_price(entry)

expected_sl = round(entry * (1 - STOP_LOSS_PCT), 4)
expected_tp = round(entry * (1 + TAKE_PROFIT_PCT), 4)

check(f"stop_loss_price({entry}) = {expected_sl}", abs(sl - expected_sl) < 0.0001,
      f"got={sl}")
check(f"take_profit_price({entry}) = {expected_tp}", abs(tp - expected_tp) < 0.0001,
      f"got={tp}")
check("stop_loss < entry", sl < entry)
check("take_profit > entry", tp > entry)
check("take_profit > stop_loss", tp > sl)


# ── Section 8: RiskDecision fields on approval ───────────────────────────────
section("TEST 8 — RiskDecision fields populated correctly on approval")

port  = 10_000.0
entry = 150.0
d = evaluate("BUY", "MSFT", 0.80, entry, port, set(), 0.0)

expected_qty    = size_position(port, entry)
expected_sl     = stop_loss_price(entry)
expected_tp     = take_profit_price(entry)
expected_dollar = round(expected_qty * entry, 2)

check("quantity matches size_position()", d.quantity == expected_qty,
      f"expected={expected_qty} got={d.quantity}")
check("stop_loss matches stop_loss_price()", abs(d.stop_loss - expected_sl) < 0.001,
      f"expected={expected_sl} got={d.stop_loss}")
check("take_profit matches take_profit_price()", abs(d.take_profit - expected_tp) < 0.001,
      f"expected={expected_tp} got={d.take_profit}")
check("dollar_amount = qty * entry", abs(d.dollar_amount - expected_dollar) < 0.01,
      f"expected={expected_dollar} got={d.dollar_amount}")


# ── Section 9: Bypass attempts ───────────────────────────────────────────────
section("TEST 9 — Bypass attempts confirm gate cannot be short-circuited")

# Attempt: high confidence + zero entry price (degenerate input)
d = evaluate("BUY", TICKER, 0.99, 0.0, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check("entry_price=0 → approved but qty=0 (degenerate trade blocked by sizing)", d.quantity == 0)

# Attempt: confidence exactly at threshold, ticker in open_positions — both rules apply
open_pos = {TICKER}
d = evaluate("BUY", TICKER, MIN_CONFIDENCE, ENTRY, PORTFOLIO, open_pos, NORMAL_PNL)
check(
    "confidence=ok BUT ticker already held → still rejected (existing position rule wins)",
    not d.approved,
    d.reason,
)

# Attempt: good confidence + no existing position + daily halt active
d = evaluate("BUY", TICKER, 0.99, ENTRY, PORTFOLIO, EMPTY_POS, -0.06)
check(
    "confidence=0.99 + no existing position BUT daily halt → still rejected",
    not d.approved,
    d.reason,
)

# Attempt: all rules pass EXCEPT signal — confirm BUY check cannot be bypassed
d = evaluate("SELL", TICKER, 0.99, ENTRY, PORTFOLIO, EMPTY_POS, NORMAL_PNL)
check(
    "SELL signal with perfect confidence + no other blocks → still rejected",
    not d.approved,
    d.reason,
)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
passed_count = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  RESULT: {passed_count}/{total} checks passed")
if passed_count == total:
    print("  risk_manager.py VERIFIED — ready for order_executor.py")
else:
    print("  FAILURES:")
    for label, ok in results:
        if not ok:
            print(f"    • {label}")
print(f"{'═'*60}")
sys.exit(0 if passed_count == total else 1)
