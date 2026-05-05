"""
risk_manager.py — Pure position-sizing and order-gate logic.

No imports from order_executor, Alpaca, or database.
All decisions are returned as RiskDecision; the caller logs rejections.

Rules (in evaluation order):
  1. Signal must be BUY
  2. Confidence >= MIN_CONFIDENCE
  3. No existing position in ticker
  4. Daily P&L has not tripped the halt threshold
  5. Compute position size, stop-loss, take-profit
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STOP_LOSS_PCT   = 0.03   # exit if position falls 3% below entry
TAKE_PROFIT_PCT = 0.05   # exit if position rises 5% above entry
MAX_RISK_PCT    = 0.02   # max 2% of portfolio at risk per trade (Kelly-style)
DAILY_HALT_PCT  = 0.05   # halt all trading if day's P&L drops 5%
MIN_CONFIDENCE  = 0.65   # minimum classifier confidence to allow an order


# ── Decision object ───────────────────────────────────────────────────────────

@dataclass
class RiskDecision:
    approved: bool
    reason: str
    quantity: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    dollar_amount: float = 0.0


# ── Core helpers ──────────────────────────────────────────────────────────────

def stop_loss_price(entry_price: float) -> float:
    return round(entry_price * (1 - STOP_LOSS_PCT), 4)


def take_profit_price(entry_price: float) -> float:
    return round(entry_price * (1 + TAKE_PROFIT_PCT), 4)


def size_position(portfolio_value: float, entry_price: float) -> int:
    """
    Kelly-style sizing: risk exactly MAX_RISK_PCT of portfolio at the
    STOP_LOSS_PCT distance.

    shares = (portfolio * 2%) / (entry * 3%)

    Returns at least 1 if entry_price > 0, else 0.
    """
    if entry_price <= 0 or portfolio_value <= 0:
        return 0
    dollar_risk    = portfolio_value * MAX_RISK_PCT
    loss_per_share = entry_price * STOP_LOSS_PCT
    return max(1, int(dollar_risk / loss_per_share))


# ── Single evaluation gate ────────────────────────────────────────────────────

def evaluate(
    signal: str,
    ticker: str,
    confidence: float,
    entry_price: float,
    portfolio_value: float,
    open_positions: set,
    daily_pnl_pct: float,
) -> RiskDecision:
    """
    Gate every candidate order through all risk rules.

    Parameters
    ----------
    signal          : classifier output — 'BUY' | 'SELL' | 'HOLD'
    ticker          : equity symbol, e.g. 'AAPL'
    confidence      : classifier probability of the predicted class (0–1)
    entry_price     : latest close price (dollars)
    portfolio_value : current total portfolio value (dollars)
    open_positions  : set of ticker strings currently held
    daily_pnl_pct   : today's realized P&L as a fraction of starting value
                      (negative = loss; e.g. -0.06 means down 6%)

    Returns
    -------
    RiskDecision with approved=True and sizing details, or approved=False
    with a human-readable rejection reason.
    """
    # Rule 1 — only BUY signals are actionable in v1
    if signal != "BUY":
        reason = f"signal={signal} is not actionable"
        logger.info("REJECTED %s %s: %s", signal, ticker, reason)
        return RiskDecision(False, reason)

    # Rule 2 — confidence gate
    if confidence < MIN_CONFIDENCE:
        reason = f"confidence={confidence:.3f} below minimum {MIN_CONFIDENCE}"
        logger.info("REJECTED %s %s: %s", signal, ticker, reason)
        return RiskDecision(False, reason)

    # Rule 3 — no double-entry in same ticker
    if ticker in open_positions:
        reason = f"already holding position in {ticker}"
        logger.info("REJECTED %s %s: %s", signal, ticker, reason)
        return RiskDecision(False, reason)

    # Rule 4 — daily halt
    if daily_pnl_pct <= -DAILY_HALT_PCT:
        reason = f"daily halt active — portfolio down {daily_pnl_pct:.2%} today"
        logger.info("REJECTED %s %s: %s", signal, ticker, reason)
        return RiskDecision(False, reason)

    # Rule 5 — compute sizing
    qty    = size_position(portfolio_value, entry_price)
    sl     = stop_loss_price(entry_price)
    tp     = take_profit_price(entry_price)
    dollar = round(qty * entry_price, 2)

    logger.info(
        "APPROVED BUY %s  qty=%d  entry=%.2f  sl=%.2f  tp=%.2f  $=%.2f",
        ticker, qty, entry_price, sl, tp, dollar,
    )
    return RiskDecision(
        approved=True,
        reason="all checks passed",
        quantity=qty,
        stop_loss=sl,
        take_profit=tp,
        dollar_amount=dollar,
    )
