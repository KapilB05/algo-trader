"""
Entry Point — 9:00 AM Pre-Market Morning Bias & System Health Check Script.
Master Architecture & Requirements v2.1 (Hardened + Library Locked).

Run manually or via cron at 9:00 AM IST:
    python -m src.run_morning_prep

Exits with code 0 if all 5 health checks pass (SYSTEM_STATUS = "READY").
Exits with code 1 if any health check fails (SYSTEM_STATUS = "NOT_READY").
"""

from __future__ import annotations

import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.strategy.morning_bias import MorningPrepEngine


def main() -> None:
    engine = MorningPrepEngine()
    passed, report = engine.run_morning_prep()

    print("\n" + "=" * 70)
    print("  ALGO TRADER PRO — MODULE 1: 9:00 AM PRE-MARKET PREP & HEALTH CHECK")
    print("  Architecture: Master Spec v2.1 (Hardened + Zero-Cost + Library-Locked)")
    print("=" * 70)
    print(f"  Trace ID         : {report.trace_id}")
    print(f"  Timestamp (IST)  : {report.timestamp}")
    print(f"  System Status    : {'[READY]' if report.health_check.status.value == 'READY' else '[NOT_READY]'}")
    print(f"  Macro Bias       : {report.bias.value}")
    print(f"  Strategy Regime  : {report.strategy_regime.value}")
    print(f"  S&P 500 Change   : {report.sp500_change_pct:+.2f}%")
    print(f"  India VIX        : {report.india_vix:.2f}")
    print(f"  Nifty 50 Close   : ₹{report.nifty_prev_close:,.2f}")
    if report.gift_nifty > 0:
        print(f"  GIFT Nifty Proxy : ₹{report.gift_nifty:,.2f}")
    print("-" * 70)
    print("  Key Levels (Intraday Pivot Points):")
    print(f"    NIFTY 50    ->  P: ₹{report.key_levels.pivot:,.2f} | R1: ₹{report.key_levels.resistance_1:,.2f} | S1: ₹{report.key_levels.support_1:,.2f}")
    if report.banknifty_key_levels:
        print(f"    BANK NIFTY  ->  P: ₹{report.banknifty_key_levels.pivot:,.2f} | R1: ₹{report.banknifty_key_levels.resistance_1:,.2f} | S1: ₹{report.banknifty_key_levels.support_1:,.2f}")
    print("-" * 70)
    if report.active_params:
        print("  Calibrated Risk Parameters (Non-Negotiable):")
        print(f"    * Max Risk Per Trade : {report.active_params.max_risk_per_trade_pct:.1f}%")
        print(f"    * Min Reward:Risk    : 1:{report.active_params.min_reward_risk_ratio:.1f}")
        print(f"    * Daily Drawdown Cap : {report.active_params.daily_drawdown_lockout_pct:.1f}% Lockout")
        print(f"    * Open Risk Cap      : {report.active_params.max_open_risk_cap_pct:.1f}% Total")
        print(f"    * Official Lot Sizes : Nifty = {report.active_params.lot_size_nifty} | Bank Nifty = {report.active_params.lot_size_banknifty}")
        print(f"    * Allowed Strategies : {', '.join(report.active_params.allowed_strategies)}")
    print("-" * 70)
    print("  5-Point System Health Checks:")
    for item in report.health_check.items:
        status_icon = "[PASSED]" if item.passed else "[FAILED]"
        print(f"    {status_icon} | {item.name}: {item.details}")
        if item.error:
            print(f"                 └─ Error: {item.error}")
    print("=" * 70 + "\n")

    if not passed:
        print("[!] FAIL-CLOSED: System Health Check failed. Live Trading Engine boot BLOCKED.")
        sys.exit(1)
    else:
        print("[+] SUCCESS: All 5 health checks passed. System is READY for live execution.")
        sys.exit(0)


if __name__ == "__main__":
    main()
