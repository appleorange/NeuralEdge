import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_API_KEY

logger = logging.getLogger(__name__)

# Maps ticker → search query for better headline relevance
TICKER_QUERY_MAP = {
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN": "Amazon stock",
    "TSLA": "Tesla stock",
    "NVDA": "Nvidia stock",
    "META": "Meta Facebook stock",
    "JPM": "JPMorgan stock",
    "V": "Visa stock",
    "JNJ": "Johnson Johnson stock",
}


def get_news_client() -> NewsApiClient:
    return NewsApiClient(api_key=NEWS_API_KEY)


def fetch_headlines(symbol: str, max_articles: int = 10, days_back: int = 3) -> list[dict]:
    query = TICKER_QUERY_MAP.get(symbol, f"{symbol} stock")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        client = get_news_client()
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            from_param=from_date,
            page_size=max_articles,
        )
        if response["status"] != "ok":
            logger.warning("NewsAPI returned status %s for %s", response["status"], symbol)
            return []

        articles = []
        for a in response["articles"]:
            articles.append({
                "symbol": symbol,
                "headline": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })

        logger.info("Fetched %d headlines for %s", len(articles), symbol)
        return articles

    except Exception as e:
        logger.error("Failed to fetch headlines for %s: %s", symbol, e)
        return []


def verify_connection() -> bool:
    try:
        client = get_news_client()
        response = client.get_top_headlines(category="business", language="en", page_size=1)
        if response["status"] == "ok":
            print(f"[NewsAPI] Connected — {response['totalResults']} business headlines available")
            return True
        print(f"[NewsAPI] Unexpected status: {response['status']}")
        return False
    except Exception as e:
        print(f"[NewsAPI] Connection failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_connection()
    articles = fetch_headlines("AAPL", max_articles=5)
    for a in articles:
        print(f"  [{a['published_at']}] {a['headline']}")
