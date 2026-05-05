# NeuralEdge

AI/ML stock trading bot — FinBERT sentiment + XGBoost signals + Alpaca execution.

## Stack
Python · Alpaca API · FinBERT · XGBoost · SQLite · Streamlit (Phase 6)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env   # then fill in your keys

# 3. Train the model (one-time, ~3 min)
python src/classifier.py

# 4. Verify connections
python verify.py
```

## Running the bot

| Command | Behaviour |
|---------|-----------|
| `python main.py` | Paper trading. Exits immediately if market is closed. |
| `python main.py --wait` | Paper trading. **Sleeps until market open**, then starts automatically. |
| `python main.py --live` | Live trading. Requires typing `CONFIRM` at the prompt. |
| `python main.py --wait --live` | Live + auto-wait. `CONFIRM` required before sleeping. |

### Recommended: run the night before

```bash
screen -dmS neuralEdge bash -c \
  'cd /Users/sabellahan/Downloads/NeuralEdge && \
   /opt/anaconda3/bin/python3 main.py --wait'
```

The bot sleeps, printing a countdown every 10 minutes:

```
[NeuralEdge] Market opens in 13h 40m — waiting...
[NeuralEdge] Market opens in 13h 30m — waiting...
...
[NeuralEdge] Market open — starting scheduler
[NeuralEdge] Scheduler running — 15-min cycles 9:30–16:00 ET
```

### Managing the background session

```bash
screen -r neuralEdge          # reattach
# Ctrl+A then D               # detach (bot keeps running)
screen -ls                    # list sessions
tail -f logs/neuralEdge.log   # tail logs without reattaching
```

## Schedule (while running)

- **Every 15 min, 9:30–16:00 ET, Mon–Fri**: fetch bars → sentiment → predict → risk gate → order → monitor SL/TP
- **16:05 ET daily**: end-of-day summary (trades, P&L, win rate) logged to SQLite

## Key thresholds (all in `config.py`, overridable via `.env`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `MIN_CONFIDENCE` | 0.65 | Minimum classifier confidence to place an order |
| `STOP_LOSS_PCT` | 0.03 | Exit if position falls 3% below entry |
| `TAKE_PROFIT_PCT` | 0.05 | Exit if position rises 5% above entry |
| `MAX_RISK_PCT` | 0.02 | Risk at most 2% of portfolio per trade |
| `DAILY_HALT_PCT` | 0.05 | Halt all trading if portfolio drops 5% in a day |
| `RETRAIN_AFTER_DAYS` | 30 | Warn when model is older than 30 days |

## Project layout

```
main.py              Entry point + scheduler
config.py            All thresholds (single source of truth)
src/
  classifier.py      XGBoost signal model (train + predict)
  risk_manager.py    Pure position-sizing and order gate
  order_executor.py  Alpaca order placement + trade logging
  backtester.py      Historical replay with metrics
  sentiment.py       FinBERT headline scoring
  news_client.py     NewsAPI headline fetching
  alpaca_client.py   Bar fetching + account queries
  indicators.py      RSI, MACD, Bollinger Bands, SMA
  database.py        SQLite read/write layer
models/              Saved XGBoost model + metadata
data/                SQLite database
logs/                Rotating log files (neuralEdge.log)
docs/                ROADMAP, changelog, architecture
tests/               Unit + integration tests
```

## Tests

```bash
python tests/test_risk_manager.py    # 52/52 — pure logic
python tests/test_order_executor.py  # 39/39 — mocked Alpaca
python tests/test_integration.py     # 18/18 — real paper API
```
