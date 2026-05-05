import argparse
from config import ALPACA_PAPER


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralEdge trading bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.live:
        print("[NeuralEdge] LIVE trading mode — real money at risk")
    else:
        print("[NeuralEdge] Paper trading mode")

    # Phases 2-5 will wire in here
    print("[NeuralEdge] Bot not yet implemented — complete Phases 2-5 first")


if __name__ == "__main__":
    main()
