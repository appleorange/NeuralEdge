# NeuralEdge — Roadmap

## Status Key
- [ ] Not started
- [~] In progress
- [x] Complete

## Phase 1 — Foundation
- [x] Project scaffold (folder structure, requirements.txt, .env setup)
- [~] Alpaca API connection + paper trading account verified
- [~] NewsAPI connection verified
- [x] SQLite database schema designed and initialized

## Phase 2 — Data Pipeline
- [ ] Historical OHLCV price data fetching (alpaca_client.py)
- [ ] News headline fetching (news_client.py)
- [ ] Technical indicators: RSI, MACD, Bollinger Bands, 50/200 SMA (indicators.py)
- [ ] Data storage and retrieval layer (database.py)

## Phase 3 — ML Model
- [ ] FinBERT sentiment analysis pipeline (sentiment.py)
- [ ] Feature engineering (combine sentiment + indicators)
- [ ] XGBoost/Random Forest classifier training (classifier.py)
- [ ] Backtester with win rate, Sharpe ratio, max drawdown reporting (backtester.py)

## Phase 4 — Risk & Execution
- [ ] Risk manager: position sizing, stop-loss, take-profit, daily loss limit (risk_manager.py)
- [ ] Order executor with full trade logging (order_executor.py)
- [ ] Discord/console alerts on trade events

## Phase 5 — Orchestration
- [ ] Main loop scheduler (market hours aware, 15-min intervals)
- [ ] End-of-day performance summary
- [ ] --live flag for switching to real money mode

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
