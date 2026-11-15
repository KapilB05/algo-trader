"""
Algo-Trader Platform — Live Paper Trader Launcher.

Initializes and starts the real-time paper trading engine using the optimized configuration.
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.live_paper_trader import LivePaperTrader


def main():
    trader = LivePaperTrader()
    # Runs the continuous loop (5-second polling interval)
    trader.run_loop()


if __name__ == "__main__":
    main()
