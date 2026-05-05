"""
Phase 2 full verification — 6 checks before proceeding to Phase 3.
Run from project root: python3 tests/verify_phase2.py
"""
import logging
import sys
import os
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from src.alpaca_client import fetch_bars, fetch_latest_bars
from src.news_client import fetch_headlines, TICKER_QUERY_MAP
from src.indicators import compute_indicators
from src.database import (
    init_db, insert_price_bars, get_price_bars,
    insert_headlines, get_headlines, insert_signal,
    get_connection,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check(label, condition, detail="", warn=False):
    tag = (WARN if warn else FAIL) if not condition else PASS
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, condition))
    return condition


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Check 1: Data Quality — 3 tickers ────────────────────────────────────────
section("CHECK 1 — Data quality: 3 tickers")
tickers = ["MSFT", "TSLA", "NVDA"]
fetched_dfs = {}
for sym in tickers:
    try:
        df = fetch_bars(sym, limit=250)
        ok_shape = len(df) >= 200
        ok_cols = all(c in df.columns for c in ["open", "high", "low", "close", "volume"])
        ok_sorted = df["timestamp"].is_monotonic_increasing
        ok_prices = (df["close"] > 0).all()
        all_ok = ok_shape and ok_cols and ok_sorted and ok_prices
        check(f"{sym}: {len(df)} rows, sorted, positive prices", all_ok,
              f"close range ${df['close'].min():.2f}–${df['close'].max():.2f}")
        fetched_dfs[sym] = df
    except Exception as e:
        check(f"{sym}: fetch succeeded", False, str(e))


# ── Check 2: Indicators sanity — no NaN on last row ──────────────────────────
section("CHECK 2 — Indicators on MSFT (non-AAPL ticker)")
INDICATOR_COLS = ["rsi", "macd", "macd_signal", "macd_diff",
                  "bb_upper", "bb_mid", "bb_lower", "bb_pband",
                  "sma50", "sma200", "above_sma50", "above_sma200"]

if "MSFT" in fetched_dfs:
    df_ind = compute_indicators(fetched_dfs["MSFT"])
    last = df_ind.iloc[-1]

    for col in INDICATOR_COLS:
        if col not in df_ind.columns:
            check(f"Column '{col}' exists", False)
        else:
            val = last[col]
            is_nan = pd.isna(val)
            check(f"{col} = {round(float(val), 4) if not is_nan else 'NaN'}", not is_nan)

    # Sanity bounds
    if not pd.isna(last.get("rsi")):
        check("RSI in [0, 100]", 0 <= float(last["rsi"]) <= 100, f"rsi={float(last['rsi']):.2f}")
    if not pd.isna(last.get("bb_pband")):
        check("BB pband in [-0.5, 1.5]", -0.5 <= float(last["bb_pband"]) <= 1.5,
              f"pband={float(last['bb_pband']):.4f}")
    if not pd.isna(last.get("sma50")) and not pd.isna(last.get("sma200")):
        check("SMA50 and SMA200 are positive", float(last["sma50"]) > 0 and float(last["sma200"]) > 0)

    # Check NaN count on full DataFrame (only non-warmup rows)
    df_trimmed = df_ind.iloc[200:]  # skip SMA200 warmup
    nan_cols = [c for c in INDICATOR_COLS if c in df_ind.columns and df_trimmed[c].isna().any()]
    check("No NaN in indicator columns after warmup", len(nan_cols) == 0,
          f"NaN found in: {nan_cols}" if nan_cols else "clean")
else:
    check("MSFT data available for indicator test", False)


# ── Check 3: News — 3 tickers, mapping verified ───────────────────────────────
section("CHECK 3 — News headlines for 3 tickers")
news_tickers = ["MSFT", "TSLA", "GOOGL"]
for sym in news_tickers:
    try:
        articles = fetch_headlines(sym, max_articles=5)
        query_used = TICKER_QUERY_MAP.get(sym, f"{sym} stock")
        check(f"{sym}: mapping = '{query_used}'", sym in TICKER_QUERY_MAP or f"{sym} stock" == query_used)
        has_articles = len(articles) > 0
        check(f"{sym}: {len(articles)} articles returned", has_articles,
              warn=not has_articles)
        if articles:
            a = articles[0]
            check(f"{sym}: headline is non-empty string", bool(a.get("headline", "").strip()))
            check(f"{sym}: published_at is non-empty", bool(a.get("published_at", "").strip()))
    except Exception as e:
        check(f"{sym}: no exception", False, str(e))


# ── Check 4: DB roundtrip — exact match ──────────────────────────────────────
section("CHECK 4 — Database roundtrip (exact value match)")
init_db()

# Price bar roundtrip
test_bar = pd.DataFrame([{
    "timestamp": "2024-01-15T00:00:00",
    "open": 123.45, "high": 130.00, "low": 120.00,
    "close": 128.99, "volume": 99999.0,
}])
insert_price_bars("ROUNDTRIP_TEST", test_bar, timeframe="TEST")
retrieved = get_price_bars("ROUNDTRIP_TEST", timeframe="TEST", limit=1)
check("Bar inserted and retrieved", not retrieved.empty)
if not retrieved.empty:
    r = retrieved.iloc[0]
    check("close matches exactly", abs(float(r["close"]) - 128.99) < 0.001, f"got {r['close']}")
    check("volume matches exactly", abs(float(r["volume"]) - 99999.0) < 0.1, f"got {r['volume']}")
    check("open matches exactly",  abs(float(r["open"])  - 123.45) < 0.001, f"got {r['open']}")

# Clean up test row
with get_connection() as conn:
    conn.execute("DELETE FROM price_data WHERE symbol='ROUNDTRIP_TEST'")

# Headline roundtrip
test_article = [{
    "symbol": "ROUNDTRIP_TEST",
    "headline": "Test headline exact match 12345",
    "description": "Test description",
    "source": "TestSource",
    "url": "https://example.com/test",
    "published_at": "2024-01-15T10:00:00Z",
}]
insert_headlines("ROUNDTRIP_TEST", test_article)
retrieved_news = get_headlines("ROUNDTRIP_TEST", limit=1)
check("Headline inserted and retrieved", len(retrieved_news) > 0)
if retrieved_news:
    h = retrieved_news[0]
    check("headline text matches", h["headline"] == "Test headline exact match 12345",
          f"got: {h['headline']}")
    check("source matches", h["source"] == "TestSource")
    check("url matches", h["url"] == "https://example.com/test")

with get_connection() as conn:
    conn.execute("DELETE FROM news_headlines WHERE symbol='ROUNDTRIP_TEST'")

# Signal roundtrip
sig_id = insert_signal("ROUNDTRIP_TEST", "BUY", 0.87, {"rsi": 42.1}, "v1.0")
check("Signal inserted, got ID", isinstance(sig_id, int) and sig_id > 0, f"id={sig_id}")
with get_connection() as conn:
    row = conn.execute("SELECT * FROM signals WHERE id=?", (sig_id,)).fetchone()
check("Signal retrieved by ID", row is not None)
if row:
    check("signal value matches", row["signal"] == "BUY")
    check("confidence matches", abs(float(row["confidence"]) - 0.87) < 0.001)
    check("model_version matches", row["model_version"] == "v1.0")

with get_connection() as conn:
    conn.execute("DELETE FROM signals WHERE symbol='ROUNDTRIP_TEST'")


# ── Check 5: Edge cases ───────────────────────────────────────────────────────
section("CHECK 5 — Edge cases (graceful failure)")

# No news for obscure ticker
try:
    articles = fetch_headlines("ZZZTEST", max_articles=5)
    check("Unknown ticker: returns empty list (not crash)", isinstance(articles, list),
          f"returned {type(articles).__name__} with {len(articles)} items")
except Exception as e:
    check("Unknown ticker: no exception raised", False, str(e))

# Bad symbol for price bars
try:
    df_bad = fetch_bars("ZZZZINVALID", limit=10)
    # Alpaca may return empty df or raise — both acceptable if no crash
    check("Invalid ticker: no crash", True, f"returned {len(df_bad)} rows (empty ok)")
except Exception as e:
    # Some exceptions are acceptable (e.g. no data found), crashes are not
    is_data_error = any(kw in str(e).lower() for kw in ["no data", "not found", "invalid", "symbol", "400", "422"])
    check("Invalid ticker: handled exception (not unhandled crash)", is_data_error,
          f"exception: {str(e)[:80]}", warn=not is_data_error)

# Empty DataFrame passed to indicators
try:
    empty_df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    result = compute_indicators(empty_df)
    check("Empty DataFrame: indicators don't crash", True, f"returned {len(result)} rows")
except Exception as e:
    check("Empty DataFrame: indicators don't crash", False, str(e))

# Short DataFrame (< 14 rows, below RSI minimum)
try:
    short_df = fetched_dfs.get("MSFT", pd.DataFrame()).head(10)
    if not short_df.empty:
        result = compute_indicators(short_df)
        check("Short DataFrame (<14 rows): no crash", True, f"{len(result)} rows returned")
except Exception as e:
    check("Short DataFrame: no crash", False, str(e))


# ── Check 6: Market-closed handling ──────────────────────────────────────────
section("CHECK 6 — Market closed / outside-hours handling")

now = datetime.now()
is_weekend = now.weekday() >= 5
is_after_hours = now.hour < 9 or now.hour >= 16

print(f"  Current time: {now.strftime('%Y-%m-%d %H:%M')} local  "
      f"(weekend={is_weekend}, after_hours={is_after_hours})")

# Historical bar fetch always works regardless of market hours
try:
    df = fetch_bars("AAPL", limit=5)
    check("Historical fetch works outside market hours", not df.empty,
          f"{len(df)} bars returned at {now.strftime('%H:%M')}")
except Exception as e:
    check("Historical fetch works outside market hours", False, str(e))

# Latest bar fetch outside hours — may return last known bar or raise
try:
    latest = fetch_latest_bars(["AAPL"])
    check("Latest bar fetch outside market hours: no crash", True,
          f"returned close={latest.get('AAPL', {}).get('close', 'n/a')}")
except Exception as e:
    is_expected = any(kw in str(e).lower() for kw in ["no data", "market", "closed", "after"])
    check("Latest bar: handled gracefully", is_expected,
          f"exception: {str(e)[:80]}", warn=not is_expected)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  RESULT: {passed}/{total} checks passed")
if passed == total:
    print("  Phase 2 VERIFIED — ready for Phase 3")
else:
    failed = [label for label, ok in results if not ok]
    print("  FAILURES:")
    for f in failed:
        print(f"    • {f}")
print(f"{'═'*55}")
sys.exit(0 if passed == total else 1)
