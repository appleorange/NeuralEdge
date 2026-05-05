"""
main.py — NeuralEdge orchestration entry point.

Usage:
    python main.py           # paper trading — exits if market closed
    python main.py --wait    # paper trading — sleeps until market open, then starts
    python main.py --live    # live trading (requires explicit CONFIRM prompt)
    python main.py --wait --live  # live + auto-wait (CONFIRM required before sleeping)

Recommended background launch (run the night before):
    screen -dmS neuralEdge bash -c 'cd /path/to/NeuralEdge && python main.py --wait'

Schedule (once running):
    Every 15 min, Mon–Fri 9:30–16:00 ET  → trading_cycle()
    Daily at 16:05 ET                     → end_of_day_summary()
"""
import argparse
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DAILY_HALT_PCT,
    MODEL_PATH,
    SIGNAL_INTERVAL_MINUTES,
    SYMBOLS,
    should_retrain,
)
from src import database as db
from src.alpaca_client import fetch_bars
from src.classifier import predict
from src.news_client import fetch_headlines
from src.order_executor import OrderExecutor
from src.sentiment import aggregate_sentiment

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


# ── Startup checks ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuralEdge trading bot")
    parser.add_argument("--live", action="store_true",
                        help="Enable live trading (default: paper)")
    parser.add_argument("--wait", action="store_true",
                        help="If market is closed, sleep until open then start automatically")
    return parser.parse_args()


def _confirm_live_mode() -> bool:
    """
    Safety gate for live trading.
    Prints a warning and requires the user to type CONFIRM exactly.
    Returns True only on explicit confirmation; exits the process otherwise.
    """
    print("\n" + "!" * 60)
    print("  WARNING: You are about to enable LIVE trading with real money.")
    print("  All orders will be sent to your live brokerage account.")
    print("!" * 60)
    try:
        confirm = input("\nType CONFIRM to proceed (anything else exits): ").strip()
    except (EOFError, KeyboardInterrupt):
        confirm = ""
    if confirm != "CONFIRM":
        print("Confirmation not received — exiting safely.")
        sys.exit(0)
    print("[NeuralEdge] LIVE mode confirmed.\n")
    return True


def _check_market_open(paper: bool) -> tuple[bool, str]:
    """
    Calls Alpaca's clock endpoint.
    Returns (is_open, next_open_description).
    """
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    from alpaca.trading.client import TradingClient

    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=paper)
        clock = client.get_clock()
        if clock.is_open:
            return True, "NOW"
        next_open = clock.next_open.strftime("%Y-%m-%d %H:%M ET")
        return False, next_open
    except Exception as e:
        logger.error("Failed to query Alpaca market clock: %s", e)
        return False, "unknown (Alpaca error)"


# ── Wait-for-open loop ───────────────────────────────────────────────────────

def _wait_for_market_open(paper: bool) -> None:
    """
    Blocks until Alpaca reports the market is open.
    Polls every 60 s. Logs a countdown every 10 min so the log shows life.
    Handles holidays and early closes automatically — truth comes from Alpaca.
    """
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    from alpaca.trading.client import TradingClient

    POLL_SECS = 60
    LOG_SECS  = 600   # countdown line every 10 min

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=paper)
    last_logged: float = 0.0

    while True:
        try:
            clock = client.get_clock()
        except Exception as e:
            logger.warning("Clock poll failed: %s — retrying in %ds", e, POLL_SECS)
            time.sleep(POLL_SECS)
            continue

        if clock.is_open:
            msg = "[NeuralEdge] Market open — starting scheduler"
            print(msg)
            logger.info(msg)
            return

        # Compute seconds until next open
        next_open = clock.next_open
        if next_open.tzinfo is None:
            import pytz as _tz
            next_open = _tz.utc.localize(next_open)
        now_utc = datetime.now(next_open.tzinfo)
        secs_left = max(0.0, (next_open - now_utc).total_seconds())
        h = int(secs_left // 3600)
        m = int((secs_left % 3600) // 60)
        countdown = f"{h}h {m}m" if h else f"{m}m"

        mono = time.monotonic()
        if mono - last_logged >= LOG_SECS or last_logged == 0.0:
            msg = f"[NeuralEdge] Market opens in {countdown} — waiting..."
            print(msg)
            logger.info(msg)
            last_logged = mono

        time.sleep(POLL_SECS)


# ── Trading cycle ─────────────────────────────────────────────────────────────

def _run_trading_cycle(executor: OrderExecutor) -> None:
    """
    Runs once per 15-minute tick during market hours.

    Order of operations:
      1. Guard: skip if outside 9:30–16:00 ET (scheduler fires at :00/:15/:30/:45)
      2. Fetch portfolio state
      3. Check daily halt — skip cycle if triggered
      4. For each ticker: fetch bars → sentiment → predict → execute_signal
      5. Monitor open positions for SL/TP exits
      6. Log cycle completion

    Any unhandled exception is caught, logged with full traceback, and a console
    alert is printed. The scheduler continues to the next cycle regardless.
    """
    now_et = datetime.now(ET)

    # Guard: APScheduler fires at :00/:15/:30/:45 each hour 9–16.
    # Skip 9:00 and 9:15 (before market open) and anything after 16:00.
    if now_et.hour == 9 and now_et.minute < 30:
        return
    if now_et.hour > 16 or (now_et.hour == 16 and now_et.minute > 0):
        return

    try:
        portfolio_value  = executor.get_portfolio_value()
        open_positions   = executor.get_open_positions()
        daily_pnl_pct    = executor.get_daily_pnl_pct(portfolio_value)

        # Daily halt check — log and skip the full cycle
        if daily_pnl_pct <= -DAILY_HALT_PCT:
            msg = (f"[CYCLE] {now_et.strftime('%H:%M ET')} — daily halt active "
                   f"(portfolio down {daily_pnl_pct:.2%}). All orders blocked.")
            logger.info(msg)
            print(msg)
            return

        signals_evaluated = 0

        for ticker in SYMBOLS:
            try:
                df_bars = fetch_bars(ticker, limit=300)
                if df_bars.empty or len(df_bars) < 250:
                    logger.warning("%s: only %d bars, skipping", ticker, len(df_bars))
                    continue

                articles  = fetch_headlines(ticker)
                sentiment = aggregate_sentiment(articles)

                signal, confidence = predict(df_bars, sentiment)
                signals_evaluated += 1

                executor.execute_signal(
                    signal=signal,
                    ticker=ticker,
                    confidence=confidence,
                    entry_price=float(df_bars["close"].iloc[-1]),
                    portfolio_value=portfolio_value,
                    open_positions=open_positions,
                    daily_pnl_pct=daily_pnl_pct,
                    sentiment_score=sentiment.get("sentiment_today"),
                )

            except Exception as ticker_err:
                logger.error("Cycle error on %s: %s\n%s",
                             ticker, ticker_err, traceback.format_exc())
                print(f"[ALERT] Error processing {ticker}: {ticker_err}")

        # Monitor all open positions for SL/TP
        executor.monitor_positions()

        ts = now_et.strftime("%H:%M ET")
        logger.info("Cycle complete — %d signals evaluated at %s", signals_evaluated, ts)
        print(f"[CYCLE] {ts} — {signals_evaluated} signals evaluated, "
              f"{len(open_positions)} open positions monitored")

    except Exception as cycle_err:
        # Catch-all: log full traceback, alert, do NOT crash the scheduler
        logger.error("Unhandled exception in trading cycle: %s\n%s",
                     cycle_err, traceback.format_exc())
        print(f"[ALERT] Unhandled exception in trading cycle: {cycle_err}")


# ── End-of-day summary ────────────────────────────────────────────────────────

def _end_of_day_summary(executor: OrderExecutor) -> None:
    """
    Runs once at 16:05 ET.
    Queries today's closed trades from SQLite, computes performance metrics,
    upserts to the performance table, and prints a console summary.
    """
    try:
        today = date.today().isoformat()
        all_trades = db.get_open_trades()
        today_trades = [
            t for t in all_trades
            if t.get("timestamp", "").startswith(today) and t.get("pnl") is not None
        ]

        portfolio_value = executor.get_portfolio_value()

        print(f"\n{'═'*55}")
        print(f"  End-of-Day Summary — {today}")
        print(f"{'═'*55}")

        if not today_trades:
            print(f"  No closed trades today.")
            print(f"  Portfolio value: ${portfolio_value:,.2f}")
            print(f"{'═'*55}\n")
            return

        pnls     = [float(t["pnl"]) for t in today_trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate  = len(wins) / len(pnls)
        best      = max(pnls)
        worst     = min(pnls)

        print(f"  Trades:        {len(today_trades)}")
        print(f"  Wins / Losses: {len(wins)} / {len(losses)}")
        print(f"  Win rate:      {win_rate:.1%}")
        print(f"  Total P&L:     {total_pnl:+.4f} ({total_pnl / max(portfolio_value, 1):.2%})")
        print(f"  Best trade:    {best:+.4f}")
        print(f"  Worst trade:   {worst:+.4f}")
        print(f"  Portfolio:     ${portfolio_value:,.2f}")
        print(f"{'═'*55}\n")

        db.upsert_performance(
            date=today,
            total_pnl=total_pnl,
            realized_pnl=total_pnl,
            num_trades=len(today_trades),
            win_rate=win_rate,
            max_drawdown=worst,
            sharpe_ratio=0.0,
        )
        logger.info("EOD summary logged: %d trades, total_pnl=%.4f, win_rate=%.2f",
                    len(today_trades), total_pnl, win_rate)

    except Exception as e:
        logger.error("End-of-day summary failed: %s\n%s", e, traceback.format_exc())
        print(f"[ALERT] EOD summary error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    paper = not args.live

    # Live trading requires explicit confirmation
    if args.live:
        _confirm_live_mode()

    # ── Logging: console + rotating file ─────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "neuralEdge.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler: 5 MB per file, keep 7 rotations (~35 MB total)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info("Logging to %s", log_file)

    mode = "PAPER" if paper else "LIVE"
    print(f"[NeuralEdge] Starting — mode={mode}  watchlist={', '.join(SYMBOLS)}")

    # ── Pre-flight checks ─────────────────────────────────────────────────────

    # 1. Market check — exit, or wait, depending on --wait flag
    is_open, next_open = _check_market_open(paper)
    if not is_open:
        if args.wait:
            print(f"[NeuralEdge] Market opens at {next_open} — sleeping until then")
            logger.info("Market closed. Next open: %s — entering wait loop", next_open)
            try:
                _wait_for_market_open(paper)
            except KeyboardInterrupt:
                print("\n[NeuralEdge] Wait interrupted — exiting.")
                sys.exit(0)
        else:
            logger.info("Market is closed. Next open: %s", next_open)
            print(f"[NeuralEdge] Market is closed. Next open: {next_open}")
            print("[NeuralEdge] Exiting — use --wait to auto-start when market opens.")
            sys.exit(0)

    # 2. Trained model must exist
    if not MODEL_PATH.exists():
        logger.error("No trained model at %s — run: python src/classifier.py", MODEL_PATH)
        print(f"[NeuralEdge] ERROR: No model at {MODEL_PATH}. Run classifier.py first.")
        sys.exit(1)

    # 3. Retraining advisory
    if should_retrain():
        print("[NeuralEdge] WARNING: Model may be stale — consider retraining (see logs)")

    # 4. Database
    db.init_db()

    # ── Executor ──────────────────────────────────────────────────────────────

    executor = OrderExecutor(paper=paper)
    print(f"[NeuralEdge] OrderExecutor ready (paper={paper})")

    # ── Scheduler ────────────────────────────────────────────────────────────
    # Fires at :00, :15, :30, :45 for hours 9–16 on weekdays.
    # The cycle function itself guards the 9:00/9:15 pre-open slots.

    scheduler = BlockingScheduler(timezone=ET)

    scheduler.add_job(
        lambda: _run_trading_cycle(executor),
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="0,15,30,45",
            timezone=ET,
        ),
        id="trading_cycle",
        name=f"15-minute trading cycle ({', '.join(SYMBOLS)})",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        lambda: _end_of_day_summary(executor),
        CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=5,
            timezone=ET,
        ),
        id="eod_summary",
        name="End-of-day performance summary",
    )

    print(f"[NeuralEdge] Scheduler running — {SIGNAL_INTERVAL_MINUTES}-min cycles "
          f"9:30–16:00 ET, EOD summary at 16:05. Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        print("\n[NeuralEdge] Stopped.")


if __name__ == "__main__":
    main()
