# NeuralEdge — Project Status
**Last Updated:** 2026-05-05
**Current Phase:** Phase 6 — Dashboard
**Overall Status:** Phases 1–5 complete; bot is runnable in paper mode

## Milestones
- [x] Phase 1: Foundation & API connections
- [x] Phase 2: Data pipeline
- [x] Phase 3: ML model (XGBoost + FinBERT sentiment)
- [x] Phase 4: Risk & execution (risk_manager + order_executor)
- [x] Phase 5: Orchestration (main loop + scheduler + config consolidation)
- [ ] Phase 6: Dashboard (Streamlit)
- [ ] Phase 7: Paper trading validation (30 days)
- [ ] Phase 8: Live trading

## What's Running
- `python main.py` — starts the paper trading bot
  - APScheduler fires every 15 min, Mon–Fri 9:30–16:00 ET
  - Each cycle: fetch bars → sentiment → predict → risk gate → order → monitor SL/TP
  - EOD summary at 16:05 ET logs to SQLite and prints to console
  - `python main.py --live` requires typing CONFIRM (real-money safety gate)
- All thresholds centralized in config.py (env-var overridable)
- Model retraining advisory via should_retrain(days=30) in config.py

## Test Coverage
- test_risk_manager.py: 52/52 — pure logic, no external deps
- test_order_executor.py: 39/39 — fully mocked Alpaca + SQLite
- test_integration.py: 18/18 — real Alpaca paper API, real XGBoost model

## What's Next
- Streamlit dashboard: positions table, P&L chart, win rate, confidence histogram
- After 30 days paper trading: check win_rate > 52%, max_drawdown < 10%, Sharpe > 1
