# NeuralEdge — Changelog
All notable changes will be documented here.

## [2026-05-05] — Phase 5 Complete
- config.py: single source of truth for all thresholds (MIN_CONFIDENCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_RISK_PCT, DAILY_HALT_PCT, RETRAIN_AFTER_DAYS); should_retrain() helper reads saved_at from classifier_meta.json
- main.py: APScheduler BlockingScheduler, 15-min cycles Mon–Fri 9:30–16:00 ET; --live flag with CONFIRM gate; market-open check on startup; end-of-day summary at 16:05 ET; per-ticker and cycle-level exception isolation (never crashes silently)
- database.py: migration layer for exit_price/stop_loss/take_profit columns added to live DB
- risk_manager.py / backtester.py: thresholds now imported from config.py (no duplicated constants)
- apscheduler added to requirements.txt
- tests/test_integration.py: 18/18 checks — real Alpaca paper API, real XGBoost inference, full signal→risk→execute→monitor cycle verified

## [2026-05-05] — Phase 4 Complete
- risk_manager.py: pure logic gate (zero external deps) — 2% portfolio risk/trade, 3% SL, 5% TP, 65% confidence minimum, duplicate-position block, 5% daily halt. 52/52 tests.
- order_executor.py: Alpaca paper-mode default, mandatory risk gate, SQLite trade logging, SL/TP position monitor, console alerts on all events. 39/39 tests.

## [2026-05-04] — Phase 2 Complete
- alpaca_client.py: fetch_bars() (250-row OHLCV, multi-timeframe), fetch_latest_bars() for live ticks
- news_client.py: fetch_headlines() per ticker with ticker→query mapping
- indicators.py: RSI(14), MACD(12/26/9), Bollinger Bands(20), SMA50/200, golden/death cross signals
- database.py: full read/write layer — insert_price_bars, get_price_bars, insert_headlines, get_headlines, insert_trade, update_trade_exit, insert_signal, upsert_performance
- tests/test_phase2.py: 20/20 checks passing end-to-end

## [2026-05-04] — Phase 1 Complete
- Project initialized with full scaffold (folder structure, requirements.txt, .env)
- CLAUDE.md, ROADMAP.md, project_spec.md, docs/ folder created
- Alpaca paper trading API connected and verified ($100k paper account)
- NewsAPI connected and verified
- SQLite database initialized with 5-table schema (trades, price_data, news_headlines, signals, performance)
- verify.py all-green: Alpaca OK, NewsAPI OK, SQLite OK
