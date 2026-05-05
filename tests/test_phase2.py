"""End-to-end smoke test for Phase 2 data pipeline."""
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

from src.alpaca_client import fetch_bars, fetch_latest_bars
from src.news_client import fetch_headlines
from src.indicators import compute_indicators, get_latest_indicators
from src.database import init_db, insert_price_bars, get_price_bars, insert_headlines, get_headlines

SYMBOL = "AAPL"
PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition


def test_alpaca_bars():
    print("\n[1] Alpaca OHLCV fetch")
    df = fetch_bars(SYMBOL, limit=250)
    check("Returns DataFrame", not df.empty, f"{len(df)} rows")
    check("Has OHLCV columns", all(c in df.columns for c in ["open", "high", "low", "close", "volume"]))
    check("Has 200+ rows for SMA200", len(df) >= 200, f"got {len(df)}")
    check("Sorted ascending", df["timestamp"].is_monotonic_increasing)
    return df


def test_latest_bars():
    print("\n[2] Alpaca latest bars")
    result = fetch_latest_bars(["AAPL", "MSFT"])
    check("Returns dict", isinstance(result, dict))
    check("Both symbols present", "AAPL" in result and "MSFT" in result)
    check("AAPL has close price", "close" in result.get("AAPL", {}))
    return result


def test_indicators(df):
    print("\n[3] Technical indicators")
    df_ind = compute_indicators(df)
    for col in ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "sma50", "sma200"]:
        check(f"Column: {col}", col in df_ind.columns)
    latest = get_latest_indicators(df)
    check("RSI in valid range", 0 <= latest["rsi"] <= 100, f"rsi={latest['rsi']}")
    check("SMA50 computed", latest["sma50"] > 0, f"sma50={latest['sma50']}")
    check("SMA200 computed", latest["sma200"] > 0, f"sma200={latest['sma200']}")
    print(f"    Latest close={latest['close']}  rsi={latest['rsi']}  sma50={latest['sma50']}  sma200={latest['sma200']}")
    return df_ind


def test_news():
    print("\n[4] NewsAPI headline fetch")
    articles = fetch_headlines(SYMBOL, max_articles=5)
    check("Returns list", isinstance(articles, list))
    check("Has articles", len(articles) > 0, f"{len(articles)} articles")
    if articles:
        a = articles[0]
        check("Has headline", bool(a.get("headline")))
        check("Has published_at", bool(a.get("published_at")))
        print(f"    Latest: {a['headline'][:80]}")
    return articles


def test_database(df, articles):
    print("\n[5] Database read/write")
    init_db()
    inserted = insert_price_bars(SYMBOL, df, timeframe="1D")
    check("Insert bars", inserted >= 0, f"{inserted} new rows")
    stored = get_price_bars(SYMBOL, timeframe="1D", limit=300)
    check("Retrieve bars", not stored.empty, f"{len(stored)} rows")
    check("Stored count matches", len(stored) >= min(len(df), 300))

    ins_news = insert_headlines(SYMBOL, articles)
    check("Insert headlines", ins_news >= 0, f"{ins_news} rows")
    stored_news = get_headlines(SYMBOL, limit=10)
    check("Retrieve headlines", len(stored_news) > 0, f"{len(stored_news)} rows")


def main():
    print("=" * 50)
    print("Phase 2 — Data Pipeline Test")
    print("=" * 50)
    df = test_alpaca_bars()
    test_latest_bars()
    test_indicators(df)
    articles = test_news()
    test_database(df, articles)
    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
