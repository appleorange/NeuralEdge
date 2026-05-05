import logging
import pandas as pd
import ta

logger = logging.getLogger(__name__)

MIN_ROWS_FOR_SMA200 = 200
MIN_ROWS_FOR_SMA50 = 50


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 14:
        logger.warning("Only %d rows — RSI needs at least 14", len(df))
        return df

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI(14)
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD(12/26/9)
    macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_diff"] = macd_indicator.macd_diff()

    # Bollinger Bands(20, 2σ)
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pband"] = bb.bollinger_pband()   # 0=at lower, 1=at upper

    # SMA 50 / 200
    df["sma50"] = close.rolling(window=50).mean()
    df["sma200"] = close.rolling(window=200).mean()

    # Derived signal helpers used by feature builder
    df["above_sma50"] = (close > df["sma50"]).astype(int)
    df["above_sma200"] = (close > df["sma200"]).astype(int)
    df["golden_cross"] = ((df["sma50"] > df["sma200"]) & (df["sma50"].shift(1) <= df["sma200"].shift(1))).astype(int)
    df["death_cross"] = ((df["sma50"] < df["sma200"]) & (df["sma50"].shift(1) >= df["sma200"].shift(1))).astype(int)

    nan_count = df.isnull().sum().sum()
    logger.info("Computed indicators on %d rows (%d NaN cells from warmup)", len(df), nan_count)
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    df = compute_indicators(df)
    row = df.iloc[-1]
    return {
        "rsi": round(float(row.get("rsi", float("nan"))), 4),
        "macd": round(float(row.get("macd", float("nan"))), 4),
        "macd_signal": round(float(row.get("macd_signal", float("nan"))), 4),
        "macd_diff": round(float(row.get("macd_diff", float("nan"))), 4),
        "bb_upper": round(float(row.get("bb_upper", float("nan"))), 4),
        "bb_mid": round(float(row.get("bb_mid", float("nan"))), 4),
        "bb_lower": round(float(row.get("bb_lower", float("nan"))), 4),
        "bb_pband": round(float(row.get("bb_pband", float("nan"))), 4),
        "sma50": round(float(row.get("sma50", float("nan"))), 4),
        "sma200": round(float(row.get("sma200", float("nan"))), 4),
        "above_sma50": int(row.get("above_sma50", 0)),
        "above_sma200": int(row.get("above_sma200", 0)),
        "close": round(float(row.get("close", float("nan"))), 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.alpaca_client import fetch_bars
    df = fetch_bars("AAPL", limit=300)
    result = get_latest_indicators(df)
    print("\nLatest AAPL indicators:")
    for k, v in result.items():
        print(f"  {k:15s}: {v}")
