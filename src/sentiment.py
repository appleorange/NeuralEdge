import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE = 16
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load FinBERT once and cache. First call downloads ~440MB."""
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device = 0  # MPS on Apple Silicon
    elif torch.cuda.is_available():
        device = 0  # CUDA
    else:
        device = -1  # CPU

    logger.info("Loading FinBERT (%s) on device=%s ...", FINBERT_MODEL, device)
    pipe = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        top_k=None,        # return all three class probabilities
        device=device,
    )
    logger.info("FinBERT ready")
    return pipe


def _result_to_float(result: list[dict]) -> float:
    """Convert [{label, score}, ...] → float in [-1, 1] via positive − negative."""
    scores = {r["label"].lower(): r["score"] for r in result}
    return round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 4)


def score_headlines_batch(headlines: list[str]) -> list[float]:
    """
    Score a batch of headlines in one forward pass.
    Returns one float per headline in [-1, 1].
    Filters empty strings before scoring; preserves output length.
    """
    if not headlines:
        return []

    pipe = _load_pipeline()
    output = []
    indices_scored = []
    clean = []

    for i, h in enumerate(headlines):
        if h and h.strip():
            clean.append(h[:MAX_LENGTH])
            indices_scored.append(i)

    if not clean:
        return [0.0] * len(headlines)

    results = pipe(clean, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
    scored = [_result_to_float(r) for r in results]

    # Re-expand to original length (empty headlines → 0.0)
    output = [0.0] * len(headlines)
    for idx, score in zip(indices_scored, scored):
        output[idx] = score

    return output


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Aggregate multiple articles into the 5 sentiment features used by the classifier.

    Each article dict must have 'headline' and 'published_at' (ISO 8601).
    Returns:
        sentiment_today    — mean score of headlines published in last 24 h
        sentiment_3d       — mean score of all headlines from last 3 days
        sentiment_trend    — sentiment_today − sentiment_3d
        sentiment_count    — # headlines scored today (capped at 10)
        sentiment_available — 1 if any real headlines were scored, 0 if neutral default
    """
    NEUTRAL = {
        "sentiment_today": 0.0,
        "sentiment_3d": 0.0,
        "sentiment_trend": 0.0,
        "sentiment_count": 0,
        "sentiment_available": 0,
    }

    if not articles:
        return NEUTRAL

    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_3d = now - timedelta(days=3)

    headlines = [a.get("headline", "") for a in articles]
    scores = score_headlines_batch(headlines)

    today_scores, scores_3d = [], []
    for article, score in zip(articles, scores):
        pub_str = article.get("published_at", "")
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            pub_dt = now  # assume recent if unparseable

        if pub_dt >= cutoff_3d:
            scores_3d.append(score)
        if pub_dt >= cutoff_24h:
            today_scores.append(score)

    if not scores_3d:
        return NEUTRAL

    sentiment_today = sum(today_scores) / len(today_scores) if today_scores else 0.0
    sentiment_3d = sum(scores_3d) / len(scores_3d)

    return {
        "sentiment_today": round(sentiment_today, 4),
        "sentiment_3d": round(sentiment_3d, 4),
        "sentiment_trend": round(sentiment_today - sentiment_3d, 4),
        "sentiment_count": min(len(today_scores), 10),
        "sentiment_available": 1,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_headlines = [
        "Apple smashes earnings expectations, stock soars to record high",
        "Company files for bankruptcy after catastrophic revenue miss",
        "Federal Reserve holds interest rates steady at latest meeting",
    ]
    scores = score_headlines_batch(test_headlines)
    for h, s in zip(test_headlines, scores):
        print(f"  {s:+.4f}  {h}")
