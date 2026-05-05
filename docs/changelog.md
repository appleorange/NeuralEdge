# NeuralEdge — Changelog
All notable changes will be documented here.

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
