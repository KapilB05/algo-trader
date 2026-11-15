"""
Continuous Auto-Learner & Self-Optimization Master Launcher — Algo-Trader.

1. Executes one complete manual auto-learning optimization cycle on launch.
2. Starts the background scheduler to run every Sunday at 10:00 AM IST.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.auto_learner.scheduler import auto_learner_scheduler


def main():
    print("=" * 80)
    print("  🏛️ ALGO-TRADER: CONTINUOUS AUTO-LEARNER MASTER LAUNCHER")
    print("=" * 80)

    # 1. Run full cycle immediately once for manual testing/calibration
    results = auto_learner_scheduler.run_full_cycle()

    # 2. Enter continuous background scheduler (Sunday 10:00 AM IST)
    print("\n[INFO] Manual cycle completed. Starting background weekly scheduler...")
    try:
        auto_learner_scheduler.start_weekly_schedule()
    except KeyboardInterrupt:
        print("\n[INFO] Auto-Learner Scheduler stopped by user.")


if __name__ == "__main__":
    main()
