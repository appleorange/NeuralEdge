import sqlite3
import json
import logging
import os
import sys
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    exit_price      REAL,
    order_id        TEXT,
    strategy_signal TEXT,
    sentiment_score REAL,
    confidence      REAL,
    stop_loss       REAL,
    take_profit     REAL,
    pnl             REAL,
    status          TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_data (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    timeframe TEXT,
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS news_headlines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    headline        TEXT NOT NULL,
    description     TEXT,
    source          TEXT,
    url             TEXT,
    published_at    TEXT,
    sentiment_score REAL,
    sentiment_label TEXT,
    fetched_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    signal        TEXT NOT NULL,
    confidence    REAL,
    features_json TEXT,
    model_version TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT NOT NULL UNIQUE,
    total_pnl    REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    num_trades   INTEGER DEFAULT 0,
    win_rate     REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add columns that were absent in earlier schema versions."""
    migrations = [
        ("trades", "exit_price",  "REAL"),
        ("trades", "stop_loss",   "REAL"),
        ("trades", "take_profit", "REAL"),
    ]
    existing = {
        (row[0], row[1])
        for table in ["trades", "price_data", "news_headlines", "signals", "performance"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for table, col, col_type in migrations:
        if (table, col) not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            logger.info("Migration: added %s.%s", table, col)


def init_db() -> bool:
    try:
        with get_connection() as conn:
            conn.executescript(SCHEMA)
            _migrate_db(conn)
        logger.info("DB initialized at %s", DB_PATH)
        print(f"[DB] Initialized at {DB_PATH}")
        return True
    except Exception as e:
        logger.error("DB init failed: %s", e)
        print(f"[DB] Init failed: {e}")
        return False


def verify_schema() -> bool:
    expected = {"trades", "price_data", "news_headlines", "signals", "performance"}
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        found = {r["name"] for r in rows}
        missing = expected - found
        if missing:
            print(f"[DB] Missing tables: {missing}")
            return False
        print(f"[DB] Schema OK — tables: {', '.join(sorted(found))}")
        return True
    except Exception as e:
        print(f"[DB] Verify failed: {e}")
        return False


# ── Price Data ────────────────────────────────────────────────────────────────

def insert_price_bars(symbol: str, df: pd.DataFrame, timeframe: str = "1D") -> int:
    rows = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        rows.append((symbol, str(ts), float(row["open"]), float(row["high"]),
                     float(row["low"]), float(row["close"]), float(row["volume"]), timeframe))

    sql = """
        INSERT OR IGNORE INTO price_data (symbol, timestamp, open, high, low, close, volume, timeframe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.executemany(sql, rows)
        inserted = cursor.rowcount

    logger.info("Inserted %d/%d bars for %s (%s)", inserted, len(rows), symbol, timeframe)
    return inserted


def get_price_bars(symbol: str, timeframe: str = "1D", limit: int = 300) -> pd.DataFrame:
    sql = """
        SELECT timestamp, open, high, low, close, volume
        FROM price_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, timeframe, limit)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── News Headlines ────────────────────────────────────────────────────────────

def insert_headlines(symbol: str, articles: list[dict]) -> int:
    rows = []
    for a in articles:
        rows.append((
            symbol,
            a.get("headline", ""),
            a.get("description", ""),
            a.get("source", ""),
            a.get("url", ""),
            a.get("published_at", ""),
            a.get("sentiment_score"),
            a.get("sentiment_label"),
        ))

    sql = """
        INSERT INTO news_headlines (symbol, headline, description, source, url, published_at,
                                    sentiment_score, sentiment_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.executemany(sql, rows)
        inserted = cursor.rowcount

    logger.info("Inserted %d headlines for %s", inserted, symbol)
    return inserted


def get_headlines(symbol: str, limit: int = 20) -> list[dict]:
    sql = """
        SELECT symbol, headline, description, source, url, published_at,
               sentiment_score, sentiment_label, fetched_at
        FROM news_headlines
        WHERE symbol = ?
        ORDER BY published_at DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Signals ───────────────────────────────────────────────────────────────────

def insert_signal(
    symbol: str,
    signal: str,
    confidence: float,
    features: dict = None,
    model_version: str = None,
) -> int:
    timestamp = datetime.utcnow().isoformat()
    features_json = json.dumps(features) if features else None
    sql = """
        INSERT INTO signals (symbol, timestamp, signal, confidence, features_json, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (symbol, timestamp, signal, confidence, features_json, model_version))
        return cursor.lastrowid


# ── Trades ────────────────────────────────────────────────────────────────────

def insert_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    order_id: str = None,
    strategy_signal: str = None,
    sentiment_score: float = None,
    confidence: float = None,
    stop_loss: float = None,
    take_profit: float = None,
    status: str = "pending",
) -> int:
    timestamp = datetime.utcnow().isoformat()
    sql = """
        INSERT INTO trades (timestamp, symbol, side, qty, price, order_id, strategy_signal,
                            sentiment_score, confidence, stop_loss, take_profit, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (timestamp, symbol, side, qty, price, order_id,
                                    strategy_signal, sentiment_score, confidence,
                                    stop_loss, take_profit, status))
        trade_id = cursor.lastrowid

    logger.info("Inserted trade #%d: %s %s %s @ %.2f", trade_id, side.upper(), qty, symbol, price)
    return trade_id


def update_trade_exit(trade_id: int, exit_price: float, pnl: float, status: str = "filled") -> None:
    sql = "UPDATE trades SET exit_price=?, pnl=?, status=? WHERE id=?"
    with get_connection() as conn:
        conn.execute(sql, (exit_price, pnl, status, trade_id))
    logger.info("Updated trade #%d exit: price=%.2f pnl=%.2f", trade_id, exit_price, pnl)


def get_open_trades() -> list[dict]:
    sql = "SELECT * FROM trades WHERE status IN ('pending', 'filled') AND exit_price IS NULL"
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ── Performance ───────────────────────────────────────────────────────────────

def upsert_performance(
    date: str,
    total_pnl: float,
    realized_pnl: float,
    num_trades: int,
    win_rate: float,
    max_drawdown: float,
    sharpe_ratio: float,
) -> None:
    sql = """
        INSERT INTO performance (date, total_pnl, realized_pnl, num_trades, win_rate, max_drawdown, sharpe_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_pnl=excluded.total_pnl,
            realized_pnl=excluded.realized_pnl,
            num_trades=excluded.num_trades,
            win_rate=excluded.win_rate,
            max_drawdown=excluded.max_drawdown,
            sharpe_ratio=excluded.sharpe_ratio
    """
    with get_connection() as conn:
        conn.execute(sql, (date, total_pnl, realized_pnl, num_trades, win_rate, max_drawdown, sharpe_ratio))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    verify_schema()
