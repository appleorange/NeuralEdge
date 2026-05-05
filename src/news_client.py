from newsapi import NewsApiClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_API_KEY


def get_news_client() -> NewsApiClient:
    return NewsApiClient(api_key=NEWS_API_KEY)


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
    verify_connection()
