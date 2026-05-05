"""
order_executor.py — Alpaca order placement with mandatory risk gating and trade logging.

Rules:
  - Paper trading by default; live only when paper=False is explicitly passed
  - Market orders only (v1)
  - risk_manager.evaluate() is called before EVERY order — no exceptions
  - Any Alpaca error: log, do NOT retry, alert console
  - Every order and rejection is logged to SQLite
  - Console alerts on: order placed, stop-loss hit, take-profit hit, daily halt
"""
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src import database as db
from src.risk_manager import evaluate as risk_evaluate, RiskDecision

logger = logging.getLogger(__name__)

ALERT_PREFIX = "[ALERT]"


# ── OrderExecutor ─────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Wraps all order placement, position monitoring, and trade logging.

    Parameters
    ----------
    paper : bool
        True  → Alpaca paper trading (default, safe)
        False → LIVE trading (requires explicit opt-in)
    api_key, secret_key : str
        Alpaca credentials. Defaults to config.py values.
    """

    def __init__(self, paper: bool = True, api_key: str = None, secret_key: str = None):
        self.paper = paper

        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        self._api_key    = api_key    or ALPACA_API_KEY
        self._secret_key = secret_key or ALPACA_SECRET_KEY

        if not self.paper:
            print(f"{ALERT_PREFIX} *** LIVE TRADING MODE ENABLED — real money at risk ***")
            logger.warning("OrderExecutor initialized in LIVE mode")

        self._client: TradingClient | None = None

    # ── Alpaca client (lazy init) ─────────────────────────────────────────────

    def _get_client(self) -> TradingClient:
        if self._client is None:
            self._client = TradingClient(
                self._api_key,
                self._secret_key,
                paper=self.paper,
            )
        return self._client

    # ── Portfolio helpers ─────────────────────────────────────────────────────

    def get_portfolio_value(self) -> float:
        """Return current portfolio value from Alpaca account."""
        try:
            account = self._get_client().get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error("Failed to fetch portfolio value: %s", e)
            print(f"{ALERT_PREFIX} Could not fetch portfolio value: {e}")
            return 0.0

    def get_open_positions(self) -> set[str]:
        """Return set of ticker symbols with an open Alpaca position."""
        try:
            positions = self._get_client().get_all_positions()
            return {p.symbol for p in positions}
        except Exception as e:
            logger.error("Failed to fetch open positions: %s", e)
            print(f"{ALERT_PREFIX} Could not fetch open positions: {e}")
            return set()

    def get_daily_pnl_pct(self, portfolio_value: float) -> float:
        """
        Compute today's realized P&L as a fraction of current portfolio value.
        Reads today's trades from SQLite.
        Returns 0.0 if no trades today or portfolio_value is 0.
        """
        if portfolio_value <= 0:
            return 0.0
        try:
            today = date.today().isoformat()
            open_trades = db.get_open_trades()
            today_pnl = sum(
                t.get("pnl") or 0.0
                for t in open_trades
                if t.get("timestamp", "").startswith(today) and t.get("pnl") is not None
            )
            return today_pnl / portfolio_value
        except Exception as e:
            logger.error("Failed to compute daily P&L: %s", e)
            return 0.0

    # ── Core: execute a signal ────────────────────────────────────────────────

    def execute_signal(
        self,
        signal: str,
        ticker: str,
        confidence: float,
        entry_price: float,
        portfolio_value: float = None,
        open_positions: set[str] = None,
        daily_pnl_pct: float = None,
        sentiment_score: float = None,
    ) -> dict | None:
        """
        Gate the signal through risk_manager, then place a market order.

        Returns the trade log dict on success, None on rejection or error.
        All rejections are logged to SQLite and alerted on console.
        """
        # Fetch live state if not provided (allows injection for tests)
        if portfolio_value is None:
            portfolio_value = self.get_portfolio_value()
        if open_positions is None:
            open_positions = self.get_open_positions()
        if daily_pnl_pct is None:
            daily_pnl_pct = self.get_daily_pnl_pct(portfolio_value)

        # ── Risk gate (MANDATORY — no order without this) ─────────────────────
        decision: RiskDecision = risk_evaluate(
            signal=signal,
            ticker=ticker,
            confidence=confidence,
            entry_price=entry_price,
            portfolio_value=portfolio_value,
            open_positions=open_positions,
            daily_pnl_pct=daily_pnl_pct,
        )

        if not decision.approved:
            self._log_rejection(ticker, signal, confidence, decision.reason)
            if "halt" in decision.reason.lower():
                print(f"{ALERT_PREFIX} Daily halt active — all new orders blocked ({ticker})")
            return None

        # ── Place market order ────────────────────────────────────────────────
        return self._place_market_order(
            ticker=ticker,
            quantity=decision.quantity,
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            dollar_amount=decision.dollar_amount,
            sentiment_score=sentiment_score,
        )

    def _place_market_order(
        self,
        ticker: str,
        quantity: int,
        signal: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        dollar_amount: float,
        sentiment_score: float = None,
    ) -> dict | None:
        """
        Submit a market order to Alpaca.
        On any error: log, print alert, return None — no retry.
        """
        try:
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self._get_client().submit_order(order_request)
            order_id = str(order.id)
        except Exception as e:
            logger.error("Alpaca order failed for %s: %s", ticker, e)
            print(f"{ALERT_PREFIX} Order placement failed for {ticker}: {e}")
            self._log_rejection(ticker, signal, confidence, f"Alpaca error: {e}")
            return None

        # ── Log to SQLite ─────────────────────────────────────────────────────
        trade_id = db.insert_trade(
            symbol=ticker,
            side="buy",
            qty=float(quantity),
            price=entry_price,
            order_id=order_id,
            strategy_signal=signal,
            sentiment_score=sentiment_score,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="submitted",
        )

        trade = {
            "trade_id":     trade_id,
            "order_id":     order_id,
            "ticker":       ticker,
            "signal":       signal,
            "confidence":   confidence,
            "entry_price":  entry_price,
            "quantity":     quantity,
            "dollar_amount": dollar_amount,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "timestamp":    datetime.utcnow().isoformat(),
            "mode":         "PAPER" if self.paper else "LIVE",
        }

        print(
            f"[ORDER] {'PAPER' if self.paper else 'LIVE'} BUY {ticker}  "
            f"qty={quantity}  entry=${entry_price:.2f}  $={dollar_amount:.0f}  "
            f"sl=${stop_loss:.2f}  tp=${take_profit:.2f}  conf={confidence:.3f}  "
            f"order_id={order_id}"
        )
        logger.info("Order placed: %s", trade)
        return trade

    # ── Position monitoring ───────────────────────────────────────────────────

    def monitor_positions(self, latest_prices: dict[str, float] = None) -> list[dict]:
        """
        Check every open trade in SQLite against latest prices.
        Triggers stop-loss or take-profit exits where conditions are met.

        Parameters
        ----------
        latest_prices : {ticker: current_price}
            If None, fetched from Alpaca. Allows injection for tests.

        Returns list of exit actions taken.
        """
        open_trades = db.get_open_trades()
        if not open_trades:
            return []

        if latest_prices is None:
            tickers = list({t["symbol"] for t in open_trades})
            try:
                from src.alpaca_client import fetch_latest_bars
                bars = fetch_latest_bars(tickers)
                latest_prices = {sym: data["close"] for sym, data in bars.items()}
            except Exception as e:
                logger.error("Failed to fetch latest prices: %s", e)
                print(f"{ALERT_PREFIX} Could not fetch latest prices for monitoring: {e}")
                return []

        exits = []
        for trade in open_trades:
            trade_id   = trade["id"]
            ticker     = trade["symbol"]
            entry_p    = float(trade["price"])
            sl         = float(trade["stop_loss"])  if trade["stop_loss"]   else None
            tp         = float(trade["take_profit"]) if trade["take_profit"] else None
            current_p  = latest_prices.get(ticker)

            if current_p is None:
                continue

            reason = None
            if sl is not None and current_p <= sl:
                reason = "stop_loss"
            elif tp is not None and current_p >= tp:
                reason = "take_profit"

            if reason:
                pnl = (current_p - entry_p) / entry_p
                db.update_trade_exit(trade_id, current_p, pnl, status="closed")
                exits.append({"trade_id": trade_id, "ticker": ticker,
                               "reason": reason, "exit_price": current_p, "pnl_pct": pnl})

                label = "Stop-loss" if reason == "stop_loss" else "Take-profit"
                print(
                    f"{ALERT_PREFIX} {label} triggered: {ticker}  "
                    f"entry=${entry_p:.2f}  exit=${current_p:.2f}  "
                    f"pnl={pnl:+.2%}  trade_id={trade_id}"
                )
                logger.info("%s triggered for trade #%d %s: pnl=%.4f", label, trade_id, ticker, pnl)

        return exits

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_rejection(self, ticker: str, signal: str, confidence: float, reason: str) -> None:
        logger.info("REJECTED %s %s (conf=%.3f): %s", signal, ticker, confidence, reason)
        try:
            db.insert_signal(
                symbol=ticker,
                signal=f"REJECTED:{signal}",
                confidence=confidence,
                features={"rejection_reason": reason},
            )
        except Exception as e:
            logger.warning("Failed to log rejection to DB: %s", e)
