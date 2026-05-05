"""
config.py — Single source of truth for all NeuralEdge settings.

Every threshold, rate, and constant used across the codebase is defined here.
Hardcoded numbers elsewhere in the codebase are a bug; import from this module.

All values can be overridden via environment variables.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Alpaca credentials ────────────────────────────────────────────────────────

ALPACA_API_KEY  = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER    = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ── News ──────────────────────────────────────────────────────────────────────

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "data/neuralEdge.db")

# ── Watchlist ─────────────────────────────────────────────────────────────────

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")

# ── Scheduler ─────────────────────────────────────────────────────────────────

SIGNAL_INTERVAL_MINUTES = int(os.getenv("SIGNAL_INTERVAL_MINUTES", "15"))

# ── Risk thresholds ───────────────────────────────────────────────────────────
# These are the live-trading gates used by risk_manager.py on every cycle.

MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",  "0.65"))   # classifier gate
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT",   "0.03"))   # 3% below entry
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))   # 5% above entry
MAX_RISK_PCT    = float(os.getenv("MAX_RISK_PCT",    "0.02"))   # 2% of portfolio at risk per trade
DAILY_HALT_PCT  = float(os.getenv("DAILY_HALT_PCT",  "0.05"))   # 5% daily loss triggers halt

# ── Model retraining ──────────────────────────────────────────────────────────

RETRAIN_AFTER_DAYS = int(os.getenv("RETRAIN_AFTER_DAYS", "30"))
MODEL_META_PATH    = Path(os.getenv("MODEL_META_PATH", "models/classifier_meta.json"))
MODEL_PATH         = Path(os.getenv("MODEL_PATH",      "models/classifier.joblib"))


def should_retrain(days: int = None) -> bool:
    """
    Returns True if the saved model is older than `days` calendar days,
    or if no model metadata exists.
    Logs a warning when retraining is due so the main loop can surface it.
    """
    threshold = days if days is not None else RETRAIN_AFTER_DAYS

    if not MODEL_META_PATH.exists():
        logger.warning("should_retrain: no metadata at %s — retraining recommended", MODEL_META_PATH)
        return True

    try:
        meta = json.loads(MODEL_META_PATH.read_text())
        saved_at = datetime.fromisoformat(meta["saved_at"])
        age_days = (datetime.utcnow() - saved_at).days
        if age_days >= threshold:
            logger.warning(
                "Model is %d days old (threshold: %d days) — retraining recommended",
                age_days, threshold,
            )
            return True
        logger.debug("Model age: %d days (threshold: %d) — OK", age_days, threshold)
        return False
    except Exception as e:
        logger.warning("should_retrain: failed to read metadata: %s", e)
        return True


# ── Legacy aliases (kept for backward compatibility) ──────────────────────────

MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "1000"))
DAILY_LOSS_LIMIT  = float(os.getenv("DAILY_LOSS_LIMIT",  "200"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
