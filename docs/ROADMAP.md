# NeuralEdge — Roadmap

## Status Key
- [ ] Not started
- [~] In progress
- [x] Complete

## Phase 1 — Foundation
- [x] Project scaffold (folder structure, requirements.txt, .env setup)
- [x] Alpaca API connection + paper trading account verified
- [x] NewsAPI connection verified
- [x] SQLite database schema designed and initialized

## Phase 2 — Data Pipeline
- [x] Historical OHLCV price data fetching (alpaca_client.py)
- [x] News headline fetching (news_client.py)
- [x] Technical indicators: RSI, MACD, Bollinger Bands, 50/200 SMA (indicators.py)
- [x] Data storage and retrieval layer (database.py)

## Phase 3 — ML Model
- [x] FinBERT sentiment analysis pipeline (sentiment.py)
- [x] Feature engineering (combine sentiment + indicators)
- [x] XGBoost/Random Forest classifier training (classifier.py)
- [x] Backtester with win rate, Sharpe ratio, max drawdown reporting (backtester.py)

## Phase 4 — Risk & Execution
- [x] Risk manager: position sizing, stop-loss, take-profit, daily loss limit (risk_manager.py)
- [x] Order executor with full trade logging (order_executor.py)
- [x] Discord/console alerts on trade events

## Phase 5 — Orchestration
- [ ] Main loop scheduler (market hours aware, 15-min intervals)
- [ ] End-of-day performance summary
- [ ] --live flag for switching to real money mode
- [ ] Add should_retrain(days=30) helper to classifier.py that reads saved_at from classifier_meta.json and triggers retrain when threshold is crossed

## Phase 6 — Dashboard
- [ ] Streamlit dashboard: positions, trades, P&L, win rate, confidence chart

## Phase 7 — Paper Trading Validation
- [ ] 30 days paper trading with consistent logging
- [ ] Win rate > 52%, max drawdown < 10%, Sharpe > 1
- [ ] All edge cases and bugs resolved

## Phase 8 — Live Trading
- [ ] Live trading enabled with small initial capital ($500–$1000)
- [ ] Monitoring and alerting confirmed working
- [ ] Performance matches paper trading baseline

## Lessons Learned
*(auto-updated by Claude after corrections)*

## Session Log
*(Claude appends a one-line summary after each session)*
