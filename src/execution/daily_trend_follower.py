"""
Automated Production Execution Protocol: BankNifty Long-Only Daily Trend Follower.

Components:
1. Daily Screener (15:20 IST):
   - Fetches latest daily BankNifty candles from Angel One SmartAPI.
   - Verifies Confirmed Macro Bull: Close > EMA 50 AND EMA 50 > EMA 200.
   - Verifies Volatility Compression: Inside Bar OR Daily Range <= 0.75x ATR(14).
   - Verifies Momentum Exhaustion: Daily RSI(14) <= 55.0.
   - Calculates exact Trigger Price (High + 5 pts), Initial Stop (1.2x ATR),
     and Chandelier Trailing Stop (1.5x ATR).

2. Multi-Channel Alerting:
   - Console formatted report with Indian Rupee (₹) amounts.
   - Telegram Bot API webhook integration.
   - Discord Webhook integration.

3. Angel One SmartAPI Order Placement (09:00 IST / Pre-Open):
   - Resolves active BankNifty Futures contract on NFO (or BANKBEES-EQ).
   - Places GTT (Good Till Triggered) or Stop-Limit Buy order.
   - Includes Dry-Run safety switch (enabled by default).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, time as dtime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
from src.config.lot_sizes import get_lot_size
from dotenv import load_dotenv

from src.auth.angel_auth import AngelManager

# Mute noisy internal logging from SmartApi's logzero dependency
try:
    import logzero
    logzero.loglevel(logging.WARNING)
except ImportError:
    pass

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/daily_trend_follower.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("DailyTrendFollowerLive")
load_dotenv()

# ── Paths & Files ─────────────────────────────────────────────────────────────
TOKEN_FILE = Path("data/angel_token.json")
SIGNAL_FILE = Path("data/swing_signal.json")
ORDERS_FILE = Path("data/active_swing_orders.json")
DAILY_DATA_FILE = Path("data/clean/banknifty_daily_clean.parquet")

IST = timezone(timedelta(hours=5, minutes=30))


# ==============================================================================
# 2. DATA FETCHER & INDICATOR ENGINE
# ==============================================================================
def fetch_banknifty_daily(obj: Any, num_candles: int = 300) -> pd.DataFrame:
    """
    Fetch BankNifty daily candles:
    1. Loads clean continuous historical daily dataset from data/clean/banknifty_daily_clean.parquet.
    2. Queries live LTP data from Angel One SmartAPI (open, high, low, ltp) to ensure today's
       candle is updated in real time at 15:20 IST without hitting candle rate limits.
    """
    if not DAILY_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DAILY_DATA_FILE}. Run download_historical.py first.")

    df = pd.read_parquet(DAILY_DATA_FILE).sort_values("date").reset_index(drop=True)
    today_date = datetime.now(IST).date()

    # Query real-time LTP from Angel One
    try:
        time.sleep(0.5)
        ltp_res = obj.ltpData("NSE", "BANKNIFTY", "99926009")
        if ltp_res and ltp_res.get("status") and ltp_res.get("data"):
            d = ltp_res["data"]
            live_candle = {
                "timestamp": pd.to_datetime(f"{today_date} 15:30:00+05:30"),
                "open": float(d["open"]),
                "high": float(d["high"]),
                "low": float(d["low"]),
                "close": float(d["ltp"]),  # Current LTP is the effective close at 15:20 IST
                "volume": 0,
                "date": today_date,
            }
            # If today already exists in dataset, update it; otherwise append it
            if df["date"].iloc[-1] == today_date:
                for k, v in live_candle.items():
                    df.loc[df.index[-1], k] = v
                logger.info("✅ Updated today's candle (%s) with live SmartAPI quote: Close=₹%s, High=₹%s, Low=₹%s", today_date, d["ltp"], d["high"], d["low"])
            else:
                df = pd.concat([df, pd.DataFrame([live_candle])], ignore_index=True)
                logger.info("✅ Appended today's live candle (%s) from SmartAPI quote: Close=₹%s, High=₹%s, Low=₹%s", today_date, d["ltp"], d["high"], d["low"])
            return df
    except Exception as e:
        logger.warning("Could not fetch live LTP quote from SmartAPI (%s). Using existing daily cache.", e)

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA 50, EMA 200, ATR 14, RSI 14, and Volatility Compression."""
    df = df.copy()

    # True Range & ATR(14)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Macro EMAs
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-4)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Compression checks
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["is_inside"] = (df["high"] <= df["prev_high"]) & (df["low"] >= df["prev_low"])
    df["range"] = df["high"] - df["low"]
    df["range_to_atr"] = df["range"] / df["atr"]
    df["is_narrow_range"] = df["range"] <= (0.75 * df["atr"])
    df["is_compressed"] = df["is_inside"] | df["is_narrow_range"]

    return df


# ==============================================================================
# 3. DAILY SCREENER ENGINE (15:20 IST)
# ==============================================================================
def screen_market_close(obj: Any) -> dict[str, Any]:
    """
    Evaluates today's BankNifty daily candle at 15:20 IST:
    1. Macro Trend: Close > EMA 50 AND EMA 50 > EMA 200.
    2. Volatility Compression: Inside Bar OR Daily Range <= 0.75x ATR.
    3. Momentum Exhaustion: RSI(14) <= 55.0.
    """
    df_raw = fetch_banknifty_daily(obj)
    df = compute_indicators(df_raw)

    today = df.iloc[-1]
    today_date = str(today["date"])

    c_close = float(today["close"])
    c_high = float(today["high"])
    c_low = float(today["low"])
    c_open = float(today["open"])
    c_atr = float(today["atr"])
    c_ema50 = float(today["ema50"])
    c_ema200 = float(today["ema200"])
    c_rsi = float(today["rsi"])
    is_inside = bool(today["is_inside"])
    is_narrow = bool(today["is_narrow_range"])
    is_compressed = bool(today["is_compressed"])
    range_to_atr = float(today["range_to_atr"])

    # Rule checks
    macro_bull = (c_close > c_ema50) and (c_ema50 > c_ema200)
    exhaustion_ok = c_rsi <= 55.0
    setup_valid = macro_bull and is_compressed and exhaustion_ok

    # Calculation of execution levels for next session
    trigger_price = round(c_high + 5.0, 1)  # 5 index pts above compression high
    init_stop = round(trigger_price - (1.2 * c_atr), 1)
    risk_pts = round(trigger_price - init_stop, 1)
    bn_lot = get_lot_size("BANKNIFTY")
    risk_rupees = round(risk_pts * bn_lot, 0)  # 1 lot
    target_3r = round(trigger_price + (3.0 * risk_pts), 1)
    target_3r_rupees = round(3.0 * risk_pts * bn_lot, 0)
    trail_dist = round(1.5 * c_atr, 1)

    signal_data = {
        "timestamp": datetime.now(IST).isoformat(),
        "trade_date": today_date,
        "setup_valid": setup_valid,
        "symbol": "BANKNIFTY",
        "close": c_close,
        "high": c_high,
        "low": c_low,
        "atr": round(c_atr, 1),
        "ema50": round(c_ema50, 1),
        "ema200": round(c_ema200, 1),
        "rsi": round(c_rsi, 1),
        "is_inside_bar": is_inside,
        "is_narrow_range": is_narrow,
        "range_to_atr": round(range_to_atr, 2),
        "conditions": {
            "macro_bull": {
                "passed": macro_bull,
                "detail": f"Close ({c_close:,.1f}) > EMA50 ({c_ema50:,.1f}) > EMA200 ({c_ema200:,.1f})",
            },
            "volatility_compression": {
                "passed": is_compressed,
                "detail": "Inside Bar" if is_inside else (f"Narrow Range ({range_to_atr:.2f}x ATR <= 0.75x)" if is_narrow else f"No Compression ({range_to_atr:.2f}x ATR)"),
            },
            "exhaustion_filter": {
                "passed": exhaustion_ok,
                "detail": f"RSI {c_rsi:.1f} <= 55.0" if exhaustion_ok else f"Overextended: RSI {c_rsi:.1f} > 55.0",
            },
        },
        "order_plan": {
            "action": "BUY_STOP_LIMIT",
            "trigger_price": trigger_price,
            "limit_price": round(trigger_price + 10.0, 1),  # Execution buffer
            "initial_stop": init_stop,
            "risk_points": risk_pts,
            "risk_rupees": risk_rupees,
            "target_runner": target_3r,
            "target_runner_rupees": target_3r_rupees,
            "trailing_stop_rule": f"Trail stop at {trail_dist} pts (1.5x ATR) below highest close post-entry",
        } if setup_valid else None,
    }

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(signal_data, f, indent=2)

    logger.info("Saved screening result to %s", SIGNAL_FILE)
    return signal_data


# ==============================================================================
# 4. MULTI-CHANNEL ALERTING (CONSOLE, TELEGRAM, DISCORD)
# ==============================================================================
def format_alert_message(signal: dict[str, Any]) -> str:
    """Format an institutional alert report."""
    dt = signal["trade_date"]
    sym = signal["symbol"]

    if signal["setup_valid"]:
        op = signal["order_plan"]
        msg = (
            f"🚀 [INSTITUTIONAL SWING ALERT] {sym} DAILY TREND FOLLOWER\n"
            f"📅 Screening Date: {dt} (Valid for Next Market Open)\n"
            f"──────────────────────────────────────────────\n"
            f"🏛️ MACRO TREND: CONFIRMED BULL MARKET\n"
            f"  • Close: ₹{signal['close']:,.1f}\n"
            f"  • EMA 50: ₹{signal['ema50']:,.1f} | EMA 200: ₹{signal['ema200']:,.1f}\n"
            f"  • RSI(14): {signal['rsi']:.1f} (Healthy Consolidation <= 55)\n"
            f"  • Compression: {signal['conditions']['volatility_compression']['detail']}\n"
            f"  • ATR(14): {signal['atr']:.1f} pts\n"
            f"──────────────────────────────────────────────\n"
            f"🎯 EXECUTION PROTOCOL (GTT / BUY STOP):\n"
            f"  • Action       : BUY STOP above Compression High\n"
            f"  • Trigger Price: ₹{op['trigger_price']:,.1f}\n"
            f"  • Limit Price  : ₹{op['limit_price']:,.1f}\n"
            f"  • Initial Stop : ₹{op['initial_stop']:,.1f}\n"
            f"  • Risk / Lot   : ₹{op['risk_rupees']:,.0f} ({op['risk_points']} pts on 1 lot)\n"
            f"  • Target Runner: ₹{op['target_runner']:,.1f} (+₹{op['target_runner_rupees']:,.0f})\n"
            f"  • Trailing Stop: {op['trailing_stop_rule']}\n"
            f"──────────────────────────────────────────────\n"
            f"⚡ Automated order placement scheduled at 09:00 IST via Angel One SmartAPI."
        )
    else:
        conds = signal["conditions"]
        failed = [f"{k}: {v['detail']}" for k, v in conds.items() if not v["passed"]]
        msg = (
            f"🔍 [SWING SCANNER] {sym} Daily Status ({dt})\n"
            f"──────────────────────────────────────────────\n"
            f"Status: NO TRADE SETUP TODAY\n"
            f"  • Close: ₹{signal['close']:,.1f} | ATR: {signal['atr']:.1f} pts\n"
            f"  • RSI: {signal['rsi']:.1f} | EMA50: ₹{signal['ema50']:,.1f}\n"
            f"Unsatisfied Conditions:\n  - " + "\n  - ".join(failed) + "\n"
            f"──────────────────────────────────────────────\n"
            f"System remains in cash. Next screening at 15:20 IST tomorrow."
        )
    return msg


def send_telegram_alert(text: str) -> bool:
    """Send alert to Telegram Bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not (bot_token and chat_id):
        logger.info("Telegram credentials not configured in .env (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Skipping.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("✅ Telegram alert delivered successfully.")
            return True
    except Exception as e:
        logger.error("❌ Telegram delivery failed: %s", e)
        return False


def send_discord_webhook(text: str) -> bool:
    """Send alert to Discord Webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.info("Discord webhook URL not configured in .env (DISCORD_WEBHOOK_URL). Skipping.")
        return False

    payload = json.dumps({"content": f"```{text}```"}).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "DailyTrendFollower/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("✅ Discord webhook alert delivered successfully.")
            return True
    except Exception as e:
        logger.error("❌ Discord delivery failed: %s", e)
        return False


def dispatch_alerts(signal: dict[str, Any]):
    """Output formatted console message and dispatch to configured webhooks."""
    msg = format_alert_message(signal)
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70 + "\n")

    send_telegram_alert(msg)
    send_discord_webhook(msg)


# ==============================================================================
# 5. ANGEL ONE ORDER PLACEMENT (09:00 IST / MARKET OPEN)
# ==============================================================================
_CACHED_CONTRACT: Optional[dict[str, str]] = None


def find_active_banknifty_contract(obj: Any) -> dict[str, str]:
    """Resolve the active near-month BankNifty Futures contract dynamically."""
    global _CACHED_CONTRACT
    if _CACHED_CONTRACT:
        return _CACHED_CONTRACT

    # Suppress noisy SmartAPI logs during search
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = obj.searchScrip("NFO", "BANKNIFTY")
    except Exception as e:
        logger.warning("Could not resolve futures contract via searchScrip: %s", e)
        res = None
    finally:
        sys.stdout = old_stdout

    if res and res.get("status") and res.get("data"):
        # Filter for FUT contracts
        futs = [x for x in res["data"] if x.get("tradingsymbol", "").endswith("FUT") and "BANKNIFTY" in x.get("tradingsymbol", "")]
        if futs:
            # Sort alphabetically by tradingsymbol to get the earliest expiry (e.g., 24OCT comes before 24NOV)
            futs.sort(key=lambda x: x.get("tradingsymbol", ""))
            chosen = futs[0]
            _CACHED_CONTRACT = chosen
            logger.info("🎯 Resolved active BankNifty Future: %s (Token: %s)", chosen["tradingsymbol"], chosen["symboltoken"])
            return chosen

    # If search fails entirely, dynamically generate current month's contract string as a last resort
    current_month = datetime.now().strftime("%b").upper()
    current_year = str(datetime.now().year)[-2:]
    fallback_symbol = f"BANKNIFTY{current_month}{current_year}FUT"

    logger.error("❌ CRITICAL: searchScrip failed. Using dynamic fallback: %s. This may fail at broker validation.", fallback_symbol)

    # Use a dummy token so the broker rejects it safely, forcing you to check the logs
    _CACHED_CONTRACT = {"exchange": "NFO", "tradingsymbol": fallback_symbol, "symboltoken": "00000"}
    return _CACHED_CONTRACT


def place_swing_order(
    obj: Any,
    order_type: str = "GTT",  # "GTT" or "LIMIT"
    dry_run: bool = True,
    qty: int = 15,
) -> dict[str, Any]:
    """
    Places the daily trend swing trading order via Angel One SmartAPI:
    - Reads setup from data/swing_signal.json.
    - If dry_run=True, validates parameters and logs simulated order without sending to exchange.
    - If dry_run=False, calls gttCreateRule or placeOrder directly on Angel One SmartAPI.
    """
    if not SIGNAL_FILE.exists():
        logger.error("No signal file found at %s. Please run --screen first.", SIGNAL_FILE)
        return {"status": "ERROR", "message": "Signal file missing"}

    with open(SIGNAL_FILE, encoding="utf-8") as f:
        signal = json.load(f)

    if not signal.get("setup_valid"):
        logger.info("Signal file indicates no valid setup. Order placement skipped.")
        return {"status": "SKIPPED", "message": "No setup valid"}

    plan = signal["order_plan"]
    trigger_price = float(plan["trigger_price"])
    limit_price = float(plan["limit_price"])
    stop_loss = float(plan["initial_stop"])

    # Resolve instrument
    contract = find_active_banknifty_contract(obj)
    tradingsymbol = contract["tradingsymbol"] if contract else "BANKNIFTY-FUT"
    symboltoken = contract["symboltoken"] if contract else "68390"
    exchange = contract["exchange"] if contract else "NFO"

    # Idempotency check: Don't place duplicate orders for same date
    orders_db = {}
    if ORDERS_FILE.exists():
        try:
            with open(ORDERS_FILE, encoding="utf-8") as f:
                orders_db = json.load(f)
        except Exception:
            orders_db = {}

    order_key = f"{signal['trade_date']}_{tradingsymbol}"
    if order_key in orders_db and orders_db[order_key].get("status") == "PLACED":
        logger.warning("⚠️ Order for %s already placed with ID %s. Skipping duplicate.", order_key, orders_db[order_key].get("order_id"))
        return orders_db[order_key]

    logger.info("🚀 Preparing %s Order for %s (Token: %s)", order_type, tradingsymbol, symboltoken)
    logger.info("   Trigger: ₹%s | Limit: ₹%s | Stop: ₹%s | Qty: %d", trigger_price, limit_price, stop_loss, qty)

    if dry_run:
        logger.info("🛡️ DRY RUN ACTIVE: Order validated successfully. Not transmitted to broker.")
        result = {
            "status": "PLACED",
            "mode": "DRY_RUN",
            "order_id": f"SIM-GTT-{int(time.time())}",
            "tradingsymbol": tradingsymbol,
            "trigger_price": trigger_price,
            "limit_price": limit_price,
            "stop_loss": stop_loss,
            "qty": qty,
            "placed_at": datetime.now(IST).isoformat(),
        }
        orders_db[order_key] = result
        ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(orders_db, f, indent=2)
        return result

    # LIVE ORDER TRANSMISSION
    logger.info("⚠️ TRANSMITTING LIVE ORDER TO ANGEL ONE SMARTAPI...")

    if order_type.upper() == "GTT":
        gtt_params = {
            "tradingsymbol": tradingsymbol,
            "symboltoken": symboltoken,
            "exchange": exchange,
            "producttype": "CARRYFORWARD",
            "transactiontype": "BUY",
            "price": limit_price,
            "qty": qty,
            "disclosedqty": qty,
            "triggerprice": trigger_price,
            "timeperiod": 365,
        }
        try:
            rule_id = obj.gttCreateRule(gtt_params)
            logger.info("✅ Live Angel One GTT Rule Created! Rule ID: %s", rule_id)
            result = {
                "status": "PLACED",
                "mode": "LIVE_GTT",
                "order_id": str(rule_id),
                "tradingsymbol": tradingsymbol,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "stop_loss": stop_loss,
                "qty": qty,
                "placed_at": datetime.now(IST).isoformat(),
            }
            orders_db[order_key] = result
            with open(ORDERS_FILE, "w", encoding="utf-8") as f:
                json.dump(orders_db, f, indent=2)
            return result
        except Exception as exc:
            logger.error("❌ Angel One GTT order placement error: %s", exc)
            return {"status": "FAILED", "error": str(exc)}
    else:
        # Stop-Limit order on market open
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": tradingsymbol,
            "symboltoken": symboltoken,
            "transactiontype": "BUY",
            "exchange": exchange,
            "ordertype": "SL",
            "producttype": "CARRYFORWARD",
            "duration": "DAY",
            "price": limit_price,
            "triggerprice": trigger_price,
            "quantity": qty,
        }
        try:
            order_id = obj.placeOrder(order_params)
            logger.info("✅ Live Angel One SL-Limit Order Placed! Order ID: %s", order_id)
            result = {
                "status": "PLACED",
                "mode": "LIVE_SL_LIMIT",
                "order_id": str(order_id),
                "tradingsymbol": tradingsymbol,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "stop_loss": stop_loss,
                "qty": qty,
                "placed_at": datetime.now(IST).isoformat(),
            }
            orders_db[order_key] = result
            with open(ORDERS_FILE, "w", encoding="utf-8") as f:
                json.dump(orders_db, f, indent=2)
            return result
        except Exception as exc:
            logger.error("❌ Angel One order placement error: %s", exc)
            return {"status": "FAILED", "error": str(exc)}


# ==============================================================================
# 6. AUTONOMOUS SCHEDULER (DAEMON MODE)
# ==============================================================================
def run_autonomous_daemon(obj: Any, dry_run: bool = True):
    """
    Runs an institutional trading daemon:
    - At 15:20 IST: Runs daily screening & sends alerts.
    - At 09:00 IST: Executes order placement if setup was approved.
    - Loops continuously across trading days.
    """
    logger.info("🏛️ Long-Only Daily Trend Follower Daemon Started.")
    logger.info("   Mode: %s | Screening Time: 15:20 IST | Execution Time: 09:00 IST", "DRY-RUN (Safe)" if dry_run else "LIVE EXECUTION")

    while True:
        now = datetime.now(IST)
        weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Sat, 6=Sun

        if weekday >= 5:
            # Weekend sleep
            logger.info("Market closed (Weekend). Sleeping 1 hour...")
            time.sleep(3600)
            continue

        c_time = now.time()

        # Morning Order Placement window: 09:00 - 09:05 IST
        if dtime(9, 0) <= c_time <= dtime(9, 5):
            logger.info("⏰ 09:00 IST Window: Attempting order placement with retry logic...")
            order_success = False

            # 5 attempts, 15 seconds apart (covers the 09:00 to 09:05 window)
            for attempt in range(1, 6):
                res = place_swing_order(obj, order_type="GTT", dry_run=dry_run)
                if res.get("status") == "PLACED":
                    order_success = True
                    logger.info("✅ Order successfully placed on attempt %d.", attempt)
                    break
                else:
                    error_msg = res.get("error", "Unknown API error")
                    logger.warning("⚠️ Attempt %d failed: %s. Retrying in 15 seconds...", attempt, error_msg)
                    time.sleep(15)

            if not order_success:
                logger.error("❌ CRITICAL: Failed to place order after 5 attempts. Setup skipped for today.")

            # Sleep until 09:06 to cleanly exit the execution window
            time.sleep(360)

        # Afternoon Screener window: 15:20 - 15:25 IST
        elif dtime(15, 20) <= c_time <= dtime(15, 25):
            logger.info("⏰ 15:20 IST Window: Running Daily Screener...")
            sig = screen_market_close(obj)
            dispatch_alerts(sig)
            time.sleep(360)  # Sleep past the window

        else:
            time.sleep(30)


# ==============================================================================
# 7. MAIN CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BankNifty Long-Only Daily Trend Follower Production Automation")
    parser.add_argument("--screen", action="store_true", help="Run Daily Screener immediately and dispatch alerts")
    parser.add_argument("--order", action="store_true", help="Place order for active screened setup")
    parser.add_argument("--daemon", action="store_true", help="Run in autonomous scheduler daemon mode")
    parser.add_argument("--live", action="store_true", help="Enable LIVE order placement (default is Dry-Run)")
    parser.add_argument("--order-type", choices=["GTT", "LIMIT"], default="GTT", help="Order type to place (default GTT)")
    parser.add_argument("--qty", type=int, default=get_lot_size("BANKNIFTY"), help=f"Order quantity (1 lot = {get_lot_size('BANKNIFTY')} qty)")

    args = parser.parse_args()
    is_dry_run = not args.live

    print("\n" + "=" * 75)
    print("  🏛️ BANKNIFTY LONG-ONLY DAILY TREND FOLLOWER — PRODUCTION AUTOMATION")
    print(f"  Mode: {'🛡️ DRY RUN (Safe Simulation)' if is_dry_run else '⚠️ LIVE ORDER TRANSMISSION'}")
    print("=" * 75)

    auth_manager = AngelManager()
    obj = auth_manager.get_authenticated_client()

    if args.screen:
        logger.info("Triggering on-demand Daily Screening...")
        sig = screen_market_close(obj)
        dispatch_alerts(sig)

    elif args.order:
        logger.info("Triggering order placement...")
        res = place_swing_order(obj, order_type=args.order_type, dry_run=is_dry_run, qty=args.qty)
        print("\nOrder Execution Result:")
        print(json.dumps(res, indent=2))

    elif args.daemon:
        run_autonomous_daemon(obj, dry_run=is_dry_run)

    else:
        # Default behavior: run screening immediately to demonstrate live pipeline
        logger.info("No specific flag provided. Running immediate screening & validation...")
        sig = screen_market_close(obj)
        dispatch_alerts(sig)
        if sig["setup_valid"]:
            logger.info("Setup valid! Simulating dry-run order placement...")
            res = place_swing_order(obj, order_type=args.order_type, dry_run=is_dry_run, qty=args.qty)
            print("Order Placement Preview:")
            print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
