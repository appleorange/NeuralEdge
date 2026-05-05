"""Run this after filling in .env to confirm all connections are live."""
import sys
from src.alpaca_client import verify_connection as verify_alpaca
from src.news_client import verify_connection as verify_news
from src.database import init_db, verify_schema

results = {
    "Alpaca": verify_alpaca(),
    "NewsAPI": verify_news(),
    "SQLite": init_db() and verify_schema(),
}

print("\n--- Verification Summary ---")
all_ok = True
for name, ok in results.items():
    status = "OK" if ok else "FAIL"
    print(f"  {name:10s} {status}")
    if not ok:
        all_ok = False

sys.exit(0 if all_ok else 1)
