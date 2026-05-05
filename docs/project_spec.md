# NeuralEdge — Project Spec

## Part 1: Product Requirements

### Who Is This For?
NeuralEdge is built for a single user (the developer) who wants to grow a
small investment account autonomously using AI-driven stock trading, without
needing to monitor markets manually or make emotional decisions.

### What Problems Does It Solve?
- Manual trading requires constant attention during market hours — NeuralEdge
  runs fully unattended from 9:30am–4:00pm ET every trading day
- Emotional trading leads to poor decisions — NeuralEdge uses only data-driven
  signals with no human override in v1
- Spotting trade opportunities across many stocks simultaneously is impossible
  for one person — NeuralEdge scans its entire watchlist every 15 minutes
- News moves markets faster than humans can react — NeuralEdge processes
  headlines in real time using FinBERT sentiment scoring

### What Does The Product Do? (Specific Behaviors)

**Signal Generation (every 15 minutes during market hours):**
- The bot wakes up on a schedule and loops through a configured watchlist of
  stock tickers (default: S&P 500 large caps)
- For each ticker, it fetches the latest OHLCV price data from Alpaca
- It fetches the most recent news headlines for that ticker from NewsAPI
- It scores each headline using FinBERT, producing a sentiment score between
  -1.0 (very negative) and +1.0 (very positive)
- It computes the following technical indicators from price data:
  RSI (14), MACD (12/26/9), Bollinger Bands (20), 50 SMA, 200 SMA
- All features (sentiment score + indicators) are combined into a single
  feature vector and passed to the trained XGBoost classifier
- The classifier outputs one of three signals: BUY, SELL, or HOLD, along
  with a confidence score between 0–100%

**Trade Execution:**
- If signal is BUY or SELL and confidence >= 65%, the risk manager is consulted
- Risk manager checks: daily loss limit not hit, position size within 2% of
  portfolio, no existing open position in the same ticker
- If risk manager approves, a market order is placed via Alpaca paper API
- If signal is HOLD or confidence < 65%, no action is taken
- Every decision (including HOLDs) is logged to SQLite with full context

**Trade Management:**
- Every open position is monitored every 15 minutes
- If a position drops 3% below entry price, a stop-loss market order is placed
  immediately and the position is closed
- If a position rises 5% above entry price, a take-profit market order is
  placed and the position is closed
- If the portfolio drops 5% in a single day, all trading halts until next
  market open and an alert is fired

**Logging & Alerts:**
- Every trade logged to SQLite with: timestamp, ticker, signal, confidence
  score, entry price, exit price, P&L, stop-loss level, take-profit level,
  and reason for trade
- Console alert printed on every order placed, stop-loss triggered, or daily
  halt activated
- Optional Discord webhook alert for the same events

**End of Day:**
- At 4:00pm ET, the bot generates a daily summary: total trades, win/loss
  count, P&L for the day, best trade, worst trade, and current portfolio value
- Summary is logged to SQLite and printed to console

**Dashboard:**
- A Streamlit app shows: current open positions, today's trades, cumulative
  P&L, win rate, Sharpe ratio, max drawdown, and a chart of model confidence
  scores over time
- Dashboard reads directly from SQLite — it is read-only and never places orders

### What The Product Does NOT Do (v1)
- Does not trade crypto, forex, or options — US equities only
- Does not allow manual trade overrides
- Does not support multiple users or accounts
- Does not place orders faster than the 15-minute signal interval
- Does not go live automatically — requires explicit --live flag

### Success Metrics
- Win rate > 52% over any 30-day window
- Max drawdown < 10% over any 30-day window
- Sharpe ratio > 1.0 over any 30-day window
- Bot runs unattended for a full trading day without crashing
- Every trade has a logged reason and confidence score
- 60 consecutive profitable paper trading days before live transition

---

## Part 2: Engineering Requirements

### Technical Architecture

#### Data Flow
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

#### Project Structure
```
neuraledge/
├── main.py                  # Entry point + APScheduler loop
├── config.py                # All thresholds, constants, watchlist
├── CLAUDE.md                # Claude Code session instructions
├── ROADMAP.md               # Phase-by-phase task tracking
├── project_spec.md          # This file
├── .env                     # API keys (never committed to git)
├── .env.example             # Key template for setup reference
├── src/
│   ├── alpaca_client.py     # Price data fetching + order connection
│   ├── news_client.py       # News headline fetching per ticker
│   ├── indicators.py        # RSI, MACD, Bollinger Bands, SMA
│   ├── database.py          # SQLite schema + read/write layer
│   ├── sentiment.py         # FinBERT pipeline + headline scoring
│   ├── classifier.py        # XGBoost training, inference, saving
│   ├── backtester.py        # Historical replay + performance metrics
│   ├── risk_manager.py      # Position sizing, stop-loss, daily halt
│   └── order_executor.py    # Place orders, log trades, fire alerts
├── dashboard/
│   └── app.py               # Streamlit read-only performance view
├── tasks/
│   └── lessons.md           # Claude self-improvement log
├── logs/                    # Runtime logs per session
├── data/                    # SQLite database
├── models/                  # Trained model artifacts
├── tests/
├── requirements.txt
└── README.md
```

### Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Brokerage | Alpaca (`alpaca-py`) |
| News data | NewsAPI |
| Sentiment model | FinBERT via HuggingFace `transformers` |
| ML classifier | XGBoost + scikit-learn |
| Indicators | `ta` library + pandas |
| Storage | SQLite via `sqlite3` |
| Scheduler | APScheduler |
| Dashboard | Streamlit |
| Config/env | `python-dotenv` |

### Engineering Rules
- API keys loaded from `.env` only — never hardcoded anywhere
- Paper trading is the default mode at all times
- Live trading requires the explicit `--live` flag passed at runtime
- `risk_manager.py` must be called before every single order — no exceptions
- No order is placed if confidence score is below the configured threshold
- The bot must handle gracefully: Alpaca API downtime, market closed hours,
  NewsAPI rate limits, missing or malformed data, and keyboard interrupts
- Every module must have structured logging (use Python `logging`, not print)
- Backtest must be run and pass before any change to the ML model is finalized
- Git commit after every completed, working module with semantic prefix:
  `feat:`, `fix:`, `refactor:`

### Risk Parameters (all configurable in config.py)
| Parameter | Default Value |
|---|---|
| Max risk per trade | 2% of portfolio |
| Stop-loss threshold | -3% from entry |
| Take-profit threshold | +5% from entry |
| Max daily portfolio loss | -5% |
| Minimum confidence to trade | 65% |
| Signal check interval | 15 minutes |
| Default watchlist | Top 20 S&P 500 by volume |

### Paper → Live Trading Checklist
- [ ] 60 consecutive profitable paper trading days
- [ ] Win rate > 52% over last 30 days
- [ ] Max drawdown < 10% over last 30 days
- [ ] Sharpe ratio > 1.0 over last 30 days
- [ ] Zero unhandled crashes in last 30 days
- [ ] All trade logs reviewed — no anomalies
- [ ] Discord alerts confirmed working
- [ ] Start live with $500–$1,000 maximum initial capital
