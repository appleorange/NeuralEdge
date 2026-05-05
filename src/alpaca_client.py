import logging
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

logger = logging.getLogger(__name__)


def get_trading_client() -> TradingClient:
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_bars(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.Day,
    start: datetime = None,
    end: datetime = None,
    limit: int = 300,
) -> pd.DataFrame:
    if start is None:
        start = datetime.now() - timedelta(days=365)
    if end is None:
        end = datetime.now()

    client = get_data_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]

    df = df.reset_index()
    df = df.rename(columns={"index": "timestamp"})

    if "timestamp" not in df.columns and df.columns[0] != "timestamp":
        df.columns = ["timestamp"] + list(df.columns[1:])

    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Fetched %d bars for %s", len(df), symbol)
    return df


def fetch_latest_bars(symbols: list[str]) -> dict[str, dict]:
    client = get_data_client()
    request = StockLatestBarRequest(symbol_or_symbols=symbols)
    latest = client.get_stock_latest_bar(request)
    result = {}
    for sym, bar in latest.items():
        result[sym] = {
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
    logger.info("Fetched latest bars for %d symbols", len(result))
    return result


def verify_connection() -> bool:
    try:
        client = get_trading_client()
        account = client.get_account()
        mode = "PAPER" if ALPACA_PAPER else "LIVE"
        print(f"[Alpaca] Connected ({mode}) — Account #{account.account_number}")
        print(f"[Alpaca] Buying power: ${float(account.buying_power):,.2f}")
        print(f"[Alpaca] Portfolio value: ${float(account.portfolio_value):,.2f}")
        return True
    except Exception as e:
        print(f"[Alpaca] Connection failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_connection()
    df = fetch_bars("AAPL", limit=10)
    print(df.tail())
