# NeuralEdge — Architecture

## System Overview
NeuralEdge is a Python-based algorithmic trading bot. It fetches price data
and news headlines, scores sentiment with FinBERT, computes technical
indicators, feeds all features into an XGBoost classifier, and executes
trades via Alpaca's paper trading API.

## Data Flow

```
Alpaca API (OHLCV) ───┐
                       ├──▶ Feature Builder ──▶ XGBoost Classifier ──▶ Signal + Confidence
NewsAPI (headlines) ───┤                                                        │
                       └──▶ FinBERT Sentiment                          Risk Manager
                                                                                │
                                                                        Order Executor
                                                                                │
                                                                      Alpaca Paper API
                                                                                │
                                                                       SQLite Trade Log
```

## Component Responsibilities
- alpaca_client.py — price data + order connection
- news_client.py — headline fetching per ticker
- indicators.py — RSI, MACD, Bollinger Bands, SMA
- database.py — SQLite schema + read/write
- sentiment.py — FinBERT scoring
- classifier.py — XGBoost training + inference
- backtester.py — historical replay + metrics
- risk_manager.py — position sizing, stop-loss, daily halt
- order_executor.py — place orders, log trades, alerts
- main.py — APScheduler loop, market hours logic
- dashboard/app.py — Streamlit read-only view
