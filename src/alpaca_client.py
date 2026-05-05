from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.historical import StockHistoricalDataClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


def get_trading_client() -> TradingClient:
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


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
    verify_connection()
