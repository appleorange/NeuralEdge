import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

DB_PATH = os.getenv("DB_PATH", "data/neuralEdge.db")

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "1000"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "200"))

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
