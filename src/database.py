import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    order_id        TEXT,
    strategy_signal TEXT,
    sentiment_score REAL,
    confidence      REAL,
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


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> bool:
    try:
        with get_connection() as conn:
            conn.executescript(SCHEMA)
        print(f"[DB] Initialized at {DB_PATH}")
        return True
    except Exception as e:
        print(f"[DB] Init failed: {e}")
        return False


def verify_schema() -> bool:
    expected = {"trades", "price_data", "news_headlines", "signals", "performance"}
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
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


if __name__ == "__main__":
    init_db()
    verify_schema()
