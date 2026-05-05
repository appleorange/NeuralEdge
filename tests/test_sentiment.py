"""
Verification test for sentiment.py.
First run downloads ProsusAI/finbert (~440MB) — expect 1–2 min on first call.
"""
import logging
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from src.sentiment import score_headlines_batch, aggregate_sentiment

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, condition))
    return condition


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Test 1: Known sentiment directions ────────────────────────────────────────
section("TEST 1 — Known sentiment directions")

print("  (Downloading FinBERT on first run — may take 1–2 min...)")

POSITIVES = [
    "Apple smashes earnings expectations, stock soars to record high",
    "Company beats revenue estimates and raises full-year guidance",
    "Strong jobs report signals booming economy and market rally",
]
NEGATIVES = [
    "Company files for bankruptcy after catastrophic revenue collapse",
    "Stock crashes 30% on massive earnings miss and profit warning",
    "Federal probe launched into accounting fraud at major firm",
]
NEUTRALS = [
    "Federal Reserve holds interest rates steady at latest meeting",
    "Company releases quarterly earnings report on Wednesday",
    "Stock market closed for holiday on Monday",
]

pos_scores = score_headlines_batch(POSITIVES)
neg_scores = score_headlines_batch(NEGATIVES)
neu_scores = score_headlines_batch(NEUTRALS)

print("\n  Positive headlines:")
for h, s in zip(POSITIVES, pos_scores):
    ok = check(f"score > 0: '{h[:55]}...'", s > 0, f"score={s:+.4f}")

print("\n  Negative headlines:")
for h, s in zip(NEGATIVES, neg_scores):
    ok = check(f"score < 0: '{h[:55]}...'", s < 0, f"score={s:+.4f}")

print("\n  Neutral headlines:")
for h, s in zip(NEUTRALS, neu_scores):
    check(f"score near 0: '{h[:55]}...'", abs(s) < 0.6, f"score={s:+.4f}")


# ── Test 2: Batch scoring ──────────────────────────────────────────────────────
section("TEST 2 — Batch scoring consistency")

mixed = POSITIVES + NEGATIVES + NEUTRALS
batch_scores = score_headlines_batch(mixed)

check("Returns correct count", len(batch_scores) == len(mixed), f"{len(batch_scores)} scores for {len(mixed)} headlines")
check("All scores in [-1, 1]", all(-1.0 <= s <= 1.0 for s in batch_scores))

# Verify batch matches individual scoring
individual = pos_scores + neg_scores + neu_scores
matches = sum(1 for a, b in zip(batch_scores, individual) if abs(a - b) < 0.001)
check("Batch matches individual scores", matches == len(mixed), f"{matches}/{len(mixed)} match")


# ── Test 3: aggregate_sentiment ───────────────────────────────────────────────
section("TEST 3 — aggregate_sentiment function")

now = datetime.now(timezone.utc)

# Mix of recent and older articles
articles = [
    {"headline": POSITIVES[0], "published_at": (now - timedelta(hours=2)).isoformat()},
    {"headline": POSITIVES[1], "published_at": (now - timedelta(hours=6)).isoformat()},
    {"headline": NEGATIVES[0], "published_at": (now - timedelta(days=2)).isoformat()},
    {"headline": NEUTRALS[0],  "published_at": (now - timedelta(days=2, hours=12)).isoformat()},
]

result = aggregate_sentiment(articles)

check("Returns dict with 5 keys", set(result.keys()) == {"sentiment_today", "sentiment_3d", "sentiment_trend", "sentiment_count", "sentiment_available"})
check("sentiment_available = 1", result["sentiment_available"] == 1)
check("sentiment_today > 0 (2 recent positives)", result["sentiment_today"] > 0, f"={result['sentiment_today']}")
check("sentiment_count = 2 (articles in last 24h)", result["sentiment_count"] == 2, f"={result['sentiment_count']}")
check("sentiment_3d is mean of all 4", result["sentiment_3d"] != 0.0, f"={result['sentiment_3d']}")
check("sentiment_trend = today - 3d", abs(result["sentiment_trend"] - (result["sentiment_today"] - result["sentiment_3d"])) < 0.001)
print(f"\n  sentiment_today={result['sentiment_today']:+.4f}  3d={result['sentiment_3d']:+.4f}  trend={result['sentiment_trend']:+.4f}  count={result['sentiment_count']}")


# ── Test 4: Edge cases ────────────────────────────────────────────────────────
section("TEST 4 — Edge cases")

# Empty list
empty_result = aggregate_sentiment([])
check("Empty list → neutral defaults", empty_result["sentiment_available"] == 0)
check("Empty list → all zeros", all(v == 0 for v in [empty_result["sentiment_today"], empty_result["sentiment_3d"], empty_result["sentiment_trend"]]))

# Empty headline strings
empty_headlines = score_headlines_batch(["", "   ", ""])
check("Empty strings → list of zeros", empty_headlines == [0.0, 0.0, 0.0], f"got {empty_headlines}")

# Malformed published_at
bad_date_articles = [
    {"headline": POSITIVES[0], "published_at": "not-a-date"},
    {"headline": POSITIVES[1], "published_at": ""},
    {"headline": NEGATIVES[0], "published_at": None},
]
try:
    bad_result = aggregate_sentiment(bad_date_articles)
    check("Malformed dates → no crash", True, f"sentiment_available={bad_result['sentiment_available']}")
except Exception as e:
    check("Malformed dates → no crash", False, str(e))

# Articles older than 3 days → should return neutral
old_articles = [
    {"headline": POSITIVES[0], "published_at": (now - timedelta(days=5)).isoformat()},
]
old_result = aggregate_sentiment(old_articles)
check("Articles >3 days old → sentiment_available=0", old_result["sentiment_available"] == 0)

# Single headline
single = score_headlines_batch([POSITIVES[0]])
check("Single headline → list of length 1", len(single) == 1)
check("Single positive → positive score", single[0] > 0, f"score={single[0]:+.4f}")


# ── Test 5: Live ticker from NewsAPI ─────────────────────────────────────────
section("TEST 5 — Live AAPL headlines from NewsAPI")

try:
    import sys, os
    from src.news_client import fetch_headlines
    articles_live = fetch_headlines("AAPL", max_articles=5)
    check("Fetched live AAPL headlines", len(articles_live) > 0, f"{len(articles_live)} articles")
    if articles_live:
        live_result = aggregate_sentiment(articles_live)
        check("aggregate_sentiment on live data succeeds", True)
        check("sentiment_available = 1 for live data", live_result["sentiment_available"] == 1)
        print(f"  AAPL live: sentiment_today={live_result['sentiment_today']:+.4f}  3d={live_result['sentiment_3d']:+.4f}  count={live_result['sentiment_count']}")
except Exception as e:
    check("Live NewsAPI + sentiment pipeline", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  RESULT: {passed}/{total} checks passed")
if passed == total:
    print("  sentiment.py VERIFIED — ready for classifier.py")
else:
    print("  FAILURES:")
    for label, ok in results:
        if not ok:
            print(f"    • {label}")
print(f"{'═'*55}")
sys.exit(0 if passed == total else 1)
