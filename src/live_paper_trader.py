"""
Institutional Live Paper Trading Engine — Algo-Trader Platform.

Executes the Optimized 7-Layer Configuration in Real-Time Paper Trading Mode:
1. Loads parameters from `config/optimized_strategy_config.yaml`
2. Maintains real-time candle buffer with live BANKNIFTY spot price & option chain
3. Applies VIX Gating (VIX >= 11.5) & Layer 3 Regime Classifier (CHOP Gating)
4. Evaluates 3 Active Strategies: ORB, Supertrend, Institutional Flow (Excludes VWAP)
5. Enforces Consensus Voting (3/3 = 100%, 2/3 = 50%, <2 = Reject)
6. Manages Stop Loss (1.0x ATR), Target (2.0x ATR), Trailing Stop (1.6R -> 0.8R), and 14:30 Expiry Exit
7. Logs to `data/paper_trading_log.csv` and `logs/paper_trading_errors.log`
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

from src.config.lot_sizes import get_lot_size
from src.data.index_fo_data import fetch_live_option_chain
from src.math.black_scholes import black_scholes, implied_volatility, RISK_FREE_RATE
from src.ml.extended_features import calculate_adx_dmi, calculate_supertrend
from src.ml.scorer import XGBoostScorer
from src.ml.rl_exit_agent import QLearningExitAgent, ExitAction
from src.risk.capital_allocator import CapitalAllocator
from src.strategy.manipulation_detector import JaneStreetDefenseEngine
from src.strategy.regime_detector import MarketRegime, RegimeDetector

IST = ZoneInfo("Asia/Kolkata")
CONFIG_PATH = Path("config/optimized_strategy_config.yaml")
LOGS_DIR = Path("logs")
DATA_DIR = Path("data")
TRADE_LOG_CSV = DATA_DIR / "paper_trading_log.csv"
ERROR_LOG_PATH = LOGS_DIR / "paper_trading_errors.log"
PAPER_OUT_PATH = LOGS_DIR / "paper_trader.out"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("LivePaperTrader")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

file_handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

out_handler = logging.FileHandler(PAPER_OUT_PATH, mode="a", encoding="utf-8")
out_handler.setLevel(logging.INFO)
out_handler.setFormatter(formatter)
logger.addHandler(out_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@dataclass
class PaperPosition:
    """Active open paper position container for Pure Options Buying."""

    trade_id: int
    symbol: str             # e.g. "BANKNIFTY 57200 CE" or "BANKNIFTY 57200 PE"
    underlying: str         # "BANKNIFTY"
    option_type: str        # "CE" or "PE"
    strike: float           # e.g. 57200.0
    direction: str          # "BUY"
    action_label: str       # "CALL OPTION BUY (CE)" or "PUT OPTION BUY (PE)"
    entry_time: datetime
    entry_price: float      # Entry Option Premium (e.g. ₹315.00)
    current_price: float    # Current Option Premium (e.g. ₹340.50)
    spot_at_entry: float    # Underlying spot at entry (e.g. ₹57,236.75)
    quantity: int           # Option contracts qty (e.g. 60)
    stop_loss: float        # Option Premium SL (e.g. ₹236.25 / 25% SL)
    target_price: float     # Option Premium TP (e.g. ₹472.50 / 50% TP)
    spot_stop_loss: float   # Underlying Index Spot SL (e.g. ₹57,100.00)
    spot_target_price: float # Underlying Index Spot TP (e.g. ₹57,550.00)
    delta: float            # Delta sensitivity (e.g. 0.52 for CE, 0.52 for PE)
    atr_at_entry: float
    strategies_agreed: List[str]
    confidence_multiplier: float
    r_target_dist: float
    unrealized_pnl: float = 0.0
    pricing_source: str = "DIRECTIONAL DELTA SIMULATION (MODEL)"
    fair_value: Optional[float] = None
    implied_volatility: Optional[float] = None
    fair_value_ratio: Optional[float] = None


class LivePaperTrader:
    """
    Real-time event-driven paper trading engine with Automated Options Buying.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = self._load_config()

        # Extract optimized parameters
        self.symbol = self.config.get("primary_asset", {}).get("symbol", "BANKNIFTY")
        self.lot_size = int(self.config.get("primary_asset", {}).get("lot_size") or get_lot_size(self.symbol))
        self.active_strategies = self.config["ensemble_configuration"]["active_strategies"]
        self.sl_atr_mult = float(self.config["risk_and_execution_parameters"]["stop_loss_atr_multiplier"])
        self.tp_atr_mult = float(self.config["risk_and_execution_parameters"]["take_profit_atr_multiplier"])
        self.max_trade_risk_pct = float(self.config["risk_and_execution_parameters"]["max_risk_per_trade_pct"])
        self.daily_loss_limit_pct = float(self.config["risk_and_execution_parameters"]["max_daily_loss_pct"])
        self.min_vix_gate = float(self.config["regime_and_volatility_filters"]["min_india_vix_entry"])
        self.max_vix_gate = float(self.config["regime_and_volatility_filters"]["max_india_vix_entry"])
        consensus_cfg = self.config.get("ensemble_configuration", {}).get("consensus_matrix", {})
        self.mult_3_agree = float(consensus_cfg.get("3_of_3_agree", 1.0))
        self.mult_2_agree = float(consensus_cfg.get("2_of_3_agree", 0.5))

        # State tracking
        self.initial_capital: float = 10000.0  # ₹10,000 Dummy-Real Starting Capital
        self.current_capital: float = self.initial_capital
        self.daily_start_capital: float = self.initial_capital
        self.current_day_str: str = datetime.now(IST).strftime("%Y-%m-%d")
        self.daily_circuit_locked: bool = False

        # Market Hours & Over-Trading Risk Guards
        self.market_open_time = dtime(9, 15)   # 09:15 IST market open
        self.market_close_time = dtime(15, 15) # 15:15 IST (auto EOD squareoff)
        self.max_daily_trades: int = 3         # Hard limit: Max 3 high-conviction trades/day
        self.daily_trade_count: int = 0
        self.trade_cooldown_seconds: float = 900.0  # 15-minute cooldown after trade exit
        self.last_trade_exit_time: Optional[datetime] = None

        # Confluence Session Quality Gates (Component 9 Data-Derived from settings.yaml)
        self.earliest_entry_time = dtime(9, 30)   # 09:30 IST opening whip filter
        self.session_block_start = dtime(11, 30)  # 11:30 IST midday chop start
        self.session_block_end = dtime(13, 30)    # 13:30 IST midday chop end
        self.latest_entry_time = dtime(14, 45)    # 14:45 IST pre-squareoff cutoff
        try:
            settings_path = Path("config/settings.yaml")
            if settings_path.exists():
                with open(settings_path, encoding="utf-8") as f:
                    s_cfg = yaml.safe_load(f) or {}
                    conf = s_cfg.get("confluence", {})
                    if conf.get("earliest_entry_time"):
                        h, m = map(int, str(conf["earliest_entry_time"]).split(":"))
                        self.earliest_entry_time = dtime(h, m)
                    if conf.get("session_block_start") and conf.get("session_block_end"):
                        sh, sm = map(int, str(conf["session_block_start"]).split(":"))
                        eh, em = map(int, str(conf["session_block_end"]).split(":"))
                        self.session_block_start = dtime(sh, sm)
                        self.session_block_end = dtime(eh, em)
                    if conf.get("latest_entry_time"):
                        lh, lm = map(int, str(conf["latest_entry_time"]).split(":"))
                        self.latest_entry_time = dtime(lh, lm)
        except Exception as exc:
            logger.debug("Failed to read confluence timings from settings.yaml: %s", exc)

        self.open_position: Optional[PaperPosition] = None
        self.completed_trades: List[Dict[str, Any]] = []
        self.trade_counter: int = 0

        self.regime_detector = RegimeDetector()
        self.allocator = CapitalAllocator(total_capital=self.initial_capital, max_risk_per_trade=self.max_trade_risk_pct)

        # Institutional AI & Quantitative Defense Engines
        self.ml_scorer = XGBoostScorer()
        target_rr = self.tp_atr_mult / max(0.1, self.sl_atr_mult)
        self.rl_exit_agent = QLearningExitAgent(target_rr=target_rr)
        self.jane_street = JaneStreetDefenseEngine()
        self.min_ml_confidence = 0.65  # High-conviction hurdle (default); overridden by settings.yaml ml_filter.min_confidence_threshold if present
        try:
            s_path = Path("config/settings.yaml")
            if s_path.exists():
                _s = yaml.safe_load(s_path.read_text(encoding="utf-8")) or {}
                _ml = _s.get("ml_filter", {})
                if _ml.get("min_confidence_threshold") is not None:
                    self.min_ml_confidence = float(_ml["min_confidence_threshold"])
        except Exception:
            pass
        self.last_ml_prediction: Dict[str, Any] = {
            "p_win": 0.58,
            "confidence_label": "MODERATE",
            "features": {},
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        self._cached_morning_bias: str = ""
        self.notional_cap_pct: float = 0.50
        self.last_rl_action: Optional[ExitAction] = None

        # In-memory candle buffer
        self._candle_buffer = self._init_candle_buffer()
        self._init_csv_log()
        self._last_config_mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
        self._cached_spot: float = 57496.30
        self._last_chain_fetch_ts: float = 0.0
        self._cached_vix: float = 14.5
        self._last_vix_fetch_ts: float = 0.0
        self._cached_pcr: float = 1.0
        self._cached_pcr_signal: str = "NEUTRAL"
        self._cached_pcr_15m_delta: float = 0.0

    @property
    def current_pcr(self) -> float:
        """Current Put-Call Ratio for the primary trading instrument."""
        return self._cached_pcr

    @property
    def current_pcr_signal(self) -> str:
        """Current Put-Call Ratio sentiment signal (BULLISH/BEARISH/NEUTRAL)."""
        return self._cached_pcr_signal

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _check_and_hot_reload_config(self) -> None:
        """Check for configuration file changes on disk and hot-reload parameters."""
        try:
            if not self.config_path.exists():
                return
            mtime = self.config_path.stat().st_mtime
            if mtime > self._last_config_mtime:
                self._last_config_mtime = mtime
                new_cfg = self._load_config()
                self.config = new_cfg

                old_sl = self.sl_atr_mult
                old_tp = self.tp_atr_mult
                self.sl_atr_mult = float(new_cfg["risk_and_execution_parameters"]["stop_loss_atr_multiplier"])
                self.tp_atr_mult = float(new_cfg["risk_and_execution_parameters"]["take_profit_atr_multiplier"])
                self.min_vix_gate = float(new_cfg["regime_and_volatility_filters"]["min_india_vix_entry"])
                self.active_strategies = new_cfg["ensemble_configuration"]["active_strategies"]
                consensus_cfg = new_cfg.get("ensemble_configuration", {}).get("consensus_matrix", {})
                self.mult_3_agree = float(consensus_cfg.get("3_of_3_agree", 1.0))
                self.mult_2_agree = float(consensus_cfg.get("2_of_3_agree", 0.5))
                ver = new_cfg.get("version", "2.0.0")

                logger.info(
                    "[HOT-RELOAD] 🚀 Detected updated configuration version '%s' on disk! Parameters Hot-Reloaded: SL: %.2fx -> %.2fx | TP: %.2fx -> %.2fx | Min VIX: %s | Active: %s | Consensus 3/3: %.1fx, 2/3: %.1fx",
                    ver, old_sl, self.sl_atr_mult, old_tp, self.tp_atr_mult, self.min_vix_gate, self.active_strategies, self.mult_3_agree, self.mult_2_agree
                )
        except Exception as e:
            logger.warning("Config hot-reload check failed: %s", e)

    def _init_candle_buffer(self) -> pd.DataFrame:
        """Initialize in-memory candle buffer from live historical candles or parquet/csv."""
        try:
            from src.data.index_fo_data import fetch_historical_candles
            candle_data = fetch_historical_candles(self.symbol, interval="5m")
            raw_candles = candle_data.get("candles", [])
            if raw_candles and len(raw_candles) >= 30:
                rows = []
                for c in raw_candles:
                    rows.append({
                        "timestamp": pd.to_datetime(c["time"], unit="s", utc=True).tz_convert(IST),
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c.get("volume", 1000)),
                    })
                df = pd.DataFrame(rows)
                logger.info("Initialized candle buffer with %d live 5m candles from exchange feed (latest: %s).", len(df), df["timestamp"].iloc[-1])
                return df.tail(150).copy().reset_index(drop=True)
        except Exception as e:
            logger.warning("Live historical candle init fallback: %s", e)

        hist_csv = Path("data/historical/banknifty_5min.csv")
        if hist_csv.exists():
            df = pd.read_csv(hist_csv)
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.tail(150).copy().reset_index(drop=True)
        else:
            dates = pd.date_range(datetime.now(IST) - timedelta(hours=10), periods=100, freq="5min")
            return pd.DataFrame({
                "timestamp": dates,
                "open": [57450.0] * 100,
                "high": [57500.0] * 100,
                "low": [57400.0] * 100,
                "close": [57490.0] * 100,
                "volume": [15000] * 100,
            })

    def _init_csv_log(self) -> None:
        """Initialize paper trading CSV log header if not present."""
        if not TRADE_LOG_CSV.exists():
            with open(TRADE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "symbol", "direction", "entry_time", "entry_price",
                    "exit_time", "exit_price", "quantity", "gross_pnl", "brokerage_stt",
                    "net_pnl", "return_pct", "exit_reason", "strategies_triggered", "regime"
                ])

    def fetch_live_feed(self) -> Tuple[pd.DataFrame, float, float]:
        """
        Fetch latest spot price, option chain, and update candle buffer in O(1) time.
        """
        now_dt = datetime.now(IST)
        now_ts = time.time()

        # Fetch option chain quotes at most once every 15 seconds
        if now_ts - self._last_chain_fetch_ts >= 15.0:
            self._last_chain_fetch_ts = now_ts
            try:
                chain = fetch_live_option_chain(self.symbol)
                if chain:
                    if chain.get("spot_price"):
                        self._cached_spot = float(chain["spot_price"])
                    pcr_val = chain.get("pcr")
                    if pcr_val is not None and float(pcr_val) > 0:
                        new_pcr = float(pcr_val)
                        if self._cached_pcr > 0:
                            self._cached_pcr_15m_delta = round(new_pcr - self._cached_pcr, 4)
                        self._cached_pcr = new_pcr
                        self._cached_pcr_signal = str(chain.get("pcr_signal", "NEUTRAL"))
            except Exception as e:
                logger.debug("Silent option chain fetch skipped for %s: %s", self.symbol, e)

        spot_p = self._cached_spot

        # Dynamic 5-minute candle roll
        last_idx = len(self._candle_buffer) - 1
        last_ts: Any = self._candle_buffer.loc[last_idx, "timestamp"]
        if not hasattr(last_ts, "minute"):
            last_ts = pd.to_datetime(last_ts)

        current_bucket_min = (now_dt.minute // 5) * 5
        candle_roll = False
        try:
            if getattr(last_ts, "tzinfo", None) is None:
                last_ts_ist = last_ts.tz_localize(IST)
            else:
                last_ts_ist = last_ts.tz_convert(IST)
            candle_roll = (now_dt.date() > last_ts_ist.date()) or (now_dt.hour > last_ts_ist.hour) or (current_bucket_min > (last_ts_ist.minute // 5) * 5)
        except Exception:
            candle_roll = False

        if candle_roll:
            new_time = now_dt.replace(minute=current_bucket_min, second=0, microsecond=0)
            new_row = pd.DataFrame([{
                "timestamp": new_time,
                "open": spot_p,
                "high": spot_p,
                "low": spot_p,
                "close": spot_p,
                "volume": 1000.0,
            }])
            self._candle_buffer = pd.concat([self._candle_buffer, new_row], ignore_index=True).tail(150).reset_index(drop=True)
            last_idx = len(self._candle_buffer) - 1
        else:
            self._candle_buffer.loc[last_idx, "close"] = spot_p
            self._candle_buffer.loc[last_idx, "high"] = max(float(self._candle_buffer.loc[last_idx, "high"]), spot_p)  # type: ignore[arg-type]
            self._candle_buffer.loc[last_idx, "low"] = min(float(self._candle_buffer.loc[last_idx, "low"]), spot_p)  # type: ignore[arg-type]

        # Multi-tiered VIX resolution:
        # Tier 1: Morning Bias Report (data/morning_bias_report.json written daily)
        # Tier 2: Live yfinance Ticker("^INDIAVIX") (60-second TTL cache)
        # Tier 3: In-memory cached VIX, or graceful fallback 14.5
        vix_val = self._cached_vix
        for mb_file in ("data/morning_bias_report.json", "data/morning_bias.json"):
            try:
                mb_path = Path(mb_file)
                if mb_path.exists():
                    with open(mb_path, "r", encoding="utf-8") as f:
                        mb_data = json.load(f)
                        if mb_data.get("india_vix") and float(mb_data["india_vix"]) > 0:
                            vix_val = float(mb_data["india_vix"])
                            self._cached_vix = vix_val
                            break
            except Exception:
                pass

        # Live online check if VIX is uninitialized or 60s elapsed
        if (vix_val <= 0 or vix_val == 14.5) and (now_ts - self._last_vix_fetch_ts >= 60.0):
            self._last_vix_fetch_ts = now_ts
            try:
                import yfinance as yf
                vix_hist = yf.Ticker("^INDIAVIX").history(period="1d")
                if len(vix_hist) >= 1:
                    live_v = float(vix_hist["Close"].iloc[-1])
                    if live_v > 0:
                        vix_val = round(live_v, 2)
                        self._cached_vix = vix_val
            except Exception as e:
                logger.debug("Live yfinance India VIX fetch fallback failed: %s", e)

        return self._candle_buffer, spot_p, vix_val

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate point-in-time features on candle buffer."""
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].astype(float)

        df["ema9"] = c.ewm(span=9, adjust=False).mean()
        df["ema20"] = c.ewm(span=20, adjust=False).mean()
        df["ema21"] = c.ewm(span=21, adjust=False).mean()

        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=1).mean()

        st_dir, st_band = calculate_supertrend(h, l, c, period=10, multiplier=3.0)
        df["supertrend_dir"] = st_dir
        df["supertrend_band"] = st_band

        adx, pdi, mdi = calculate_adx_dmi(h, l, c, period=14)
        df["adx14"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = mdi

        bb_mid = c.rolling(20, min_periods=1).mean()
        bb_std = c.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
        bb_width = (4.0 * bb_std) / (bb_mid + 1e-8)
        df["bb_width"] = bb_width
        bw_arr = bb_width.values
        df["bbw_rank"] = float((bw_arr[-50:] <= bw_arr[-1]).mean()) if len(bw_arr) > 0 else 0.5

        df["vol_ma20"] = v.rolling(20, min_periods=1).mean()

        cum_v = v.cumsum()
        cum_pv = (c * v).cumsum()
        df["vwap"] = np.where(cum_v > 0, cum_pv / cum_v, c)

        # Calculate ORH / ORL dynamically from today's opening session (09:15 - 09:30)
        try:
            today_date = datetime.now(IST).date()
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                ts_col = df["timestamp"]
            else:
                ts_col = pd.to_datetime(df["timestamp"])

            today_mask = ts_col.apply(lambda t: t.date() == today_date if hasattr(t, "date") else True)
            today_bars = df[today_mask]
            if len(today_bars) >= 3:
                df["orh"] = float(today_bars["high"].iloc[:3].max())
                df["orl"] = float(today_bars["low"].iloc[:3].min())
            elif len(today_bars) > 0:
                df["orh"] = float(today_bars["high"].max())
                df["orl"] = float(today_bars["low"].min())
            else:
                df["orh"] = float(h.iloc[-20:].max())
                df["orl"] = float(l.iloc[-20:].min())
        except Exception:
            df["orh"] = float(h.iloc[-20:].max())
            df["orl"] = float(l.iloc[-20:].min())

        return df

    def extract_ml_features(self, df: pd.DataFrame, vix: float = 14.5, pcr: Optional[float] = None) -> Dict[str, float]:
        """Extract standardized feature dictionary matching FEATURE_COLS for ML Scorer."""
        curr = df.iloc[-1]
        c = float(curr["close"])
        vwap_val = float(curr.get("vwap", c))
        vwap_dist_pct = ((c - vwap_val) / max(1.0, vwap_val)) * 100.0

        prev3_vwap = float(df["vwap"].iloc[-4]) if len(df) >= 4 else vwap_val
        vwap_slope = ((vwap_val - prev3_vwap) / max(1.0, prev3_vwap)) * 100.0

        atr_val = float(curr.get("atr14", 50.0))
        atr_pct = (atr_val / max(1.0, c)) * 100.0

        vol_val = float(curr.get("volume", 1000.0))
        vol_ma = float(curr.get("vol_ma20", 1000.0))
        vol_ratio = vol_val / max(1.0, vol_ma)
        vol_surge = 1.0 if vol_ratio >= 1.5 else 0.0

        # Wilder's / EWMA RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

        # Realized Volatility (20-bar return std annualized)
        ret_s = df["close"].pct_change().fillna(0.0)
        realized_vol_20 = float(ret_s.rolling(20, min_periods=1).std().iloc[-1] * np.sqrt(252 * 75)) if len(ret_s) > 0 else 0.15

        orh_val = float(curr.get("orh", c))
        orl_val = float(curr.get("orl", c))
        orh_dist_pct = ((c - orh_val) / max(1.0, orh_val)) * 100.0
        orl_dist_pct = ((c - orl_val) / max(1.0, orl_val)) * 100.0

        now_ist = datetime.now(IST)
        mins = now_ist.hour * 60 + now_ist.minute
        time_angle = 2.0 * np.pi * (mins / 1440.0)
        sin_time = float(np.sin(time_angle))
        cos_time = float(np.cos(time_angle))
        session_period = 0.0 if mins < 630 else (2.0 if mins >= 855 else 1.0)

        pcr_val = float(pcr if pcr is not None else self._cached_pcr)
        pcr_delta = float(self._cached_pcr_15m_delta)

        return {
            "vwap_dist_pct": round(vwap_dist_pct, 4),
            "vwap_slope": round(vwap_slope, 4),
            "atr_pct": round(atr_pct, 4),
            "atr_regime": round(float(curr.get("bbw_rank", 0.5)), 4),
            "realized_vol_20": round(realized_vol_20, 4),
            "vol_ratio": round(vol_ratio, 4),
            "volume_surge": vol_surge,
            "oi_change_pct": 0.0,
            "oi_change_12": 0.0,
            "rsi": round(rsi_val, 2),
            "pdh_dist_pct": 0.0,
            "pdl_dist_pct": 0.0,
            "orh_dist_pct": round(orh_dist_pct, 4),
            "orl_dist_pct": round(orl_dist_pct, 4),
            "sin_time_of_day": round(sin_time, 4),
            "cos_time_of_day": round(cos_time, 4),
            "session_period": session_period,
            "relative_strength_index_pct": 0.0,
            "time_of_day_min": mins,
            "vix": float(vix),
            "pcr_value": round(pcr_val, 2),
            "pcr_15m_delta": round(pcr_delta, 4),
            "pcr_regime_bullish": 1.0 if pcr_val > 1.2 else 0.0,
            "pcr_regime_bearish": 1.0 if pcr_val < 0.8 else 0.0,
        }

    def evaluate_signals(
        self,
        df: pd.DataFrame,
        regime_name: str,
        vix_val: float = 14.5,
        pcr_val: Optional[float] = None,
    ) -> Tuple[str, float, List[str]]:
        """
        Evaluate 3 Active Strategies: ORB, Supertrend, Flow with PCR Guard, ML Gating & Jane Street Defense.
        Returns: (direction, multiplier, triggered_strategies)
        """
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        c_val = float(curr["close"])
        prev_c_val = float(prev["close"])
        vol_val = float(curr["volume"])
        vol_ma_val = float(curr["vol_ma20"])
        orh_val = float(curr["orh"])
        orl_val = float(curr["orl"])
        st_dir_val = float(curr["supertrend_dir"])
        ema9_val = float(curr["ema9"])
        ema21_val = float(curr["ema21"])
        prev_ema9 = float(prev["ema9"])
        prev_ema21 = float(prev["ema21"])

        pcr = float(pcr_val if pcr_val is not None else self._cached_pcr)

        long_strats = []
        short_strats = []

        # Strategy 1: ORB Breakout (09:30-11:30)
        if "ORB" in self.active_strategies:
            if c_val > orh_val and vol_val >= (vol_ma_val * 1.5):
                long_strats.append("ORB")
            elif c_val < orl_val and vol_val >= (vol_ma_val * 1.5):
                short_strats.append("ORB")

        # Strategy 2: Supertrend + EMA Cross
        if "SUPERTREND" in self.active_strategies:
            ema_bull = (ema9_val > ema21_val and st_dir_val == 1.0) and (c_val > prev_c_val or prev_ema9 <= prev_ema21)
            ema_bear = (ema9_val < ema21_val and st_dir_val == -1.0) and (c_val < prev_c_val or prev_ema9 >= prev_ema21)
            if ema_bull:
                long_strats.append("SUPERTREND")
            elif ema_bear:
                short_strats.append("SUPERTREND")

        # Strategy 3: Institutional Flow (enhanced with PCR institutional alignment)
        # Case-insensitive regime check handles both enum names ('TRENDING_UP') and values ('trending_up')
        if "FLOW" in self.active_strategies:
            regime_clean = regime_name.strip().lower()
            regime_up = regime_clean in ("trending_up", "compression")
            regime_dn = regime_clean in ("trending_down", "expansion")
            pcr_long_ok = (pcr >= 0.90) if pcr > 0 else True
            pcr_short_ok = (pcr <= 1.10) if pcr > 0 else True
            if regime_up and vol_val >= (vol_ma_val * 1.05) and pcr_long_ok:
                long_strats.append("FLOW")
            elif regime_dn and vol_val >= (vol_ma_val * 1.05) and pcr_short_ok:
                short_strats.append("FLOW")

        # Strategy 4: VWAP Trend Pulse (Only if enabled in active_strategies)
        if "VWAP" in self.active_strategies:
            vwap_val = float(curr["vwap"])
            if c_val > vwap_val and ema9_val >= ema21_val:
                long_strats.append("VWAP")
            elif c_val < vwap_val and ema9_val <= ema21_val:
                short_strats.append("VWAP")

        # Consensus Rule Evaluation (Derived from ensemble_configuration.consensus_matrix)
        candidate_direction = "NONE"
        base_mult = 0.0
        candidate_strats = []

        if len(long_strats) >= 3:
            candidate_direction, base_mult, candidate_strats = "LONG", self.mult_3_agree, long_strats
        elif len(short_strats) >= 3:
            candidate_direction, base_mult, candidate_strats = "SHORT", self.mult_3_agree, short_strats
        elif len(long_strats) == 2 or len(short_strats) == 2:
            logger.info(
                "[CONSENSUS] ⚠️ Partial %d/3 agreement %s — rejected under max-conviction policy (3/3 required).",
                max(len(long_strats), len(short_strats)),
                "+".join(long_strats or short_strats),
            )

        if candidate_direction == "NONE":
            return "NONE", 0.0, []

        # ── Intelligence Filter 1: Jane Street Anti-Spoofing Shield ───────────
        if self.jane_street is not None:
            if not self.jane_street.validate_anti_spoofing(df, candidate_direction):
                logger.warning(
                    "[Jane Street Defense] 🛡️ Spoofing pattern detected: %s breakout lacked 2nd confirming candle. Trade rejected.",
                    candidate_direction
                )
                return "NONE", 0.0, []

        # ── Intelligence Filter 1B: Put-Call Ratio (PCR) Institutional Guard ──
        # High PCR (>1.25) indicates heavy Put writing support (trap for Short trades).
        # Low PCR (<0.80) indicates heavy Call writing resistance (trap for Long trades).
        if pcr > 0:
            if candidate_direction == "LONG" and pcr < 0.80:
                logger.warning(
                    "[PCR Guard] 🛡️ LONG rejected: PCR=%.2f < 0.80 indicates heavy institutional Call writing resistance overhead.",
                    pcr,
                )
                return "NONE", 0.0, []
            elif candidate_direction == "SHORT" and pcr > 1.25:
                logger.warning(
                    "[PCR Guard] 🛡️ SHORT rejected: PCR=%.2f > 1.25 indicates heavy institutional Put writing support below.",
                    pcr,
                )
                return "NONE", 0.0, []

        # ── Intelligence Filter 2: Calibrated Machine Learning Scorer ─────────
        feat_dict = self.extract_ml_features(df, vix=vix_val, pcr=pcr)
        p_win = self.ml_scorer.predict_confidence(feat_dict)
        conf_label = "HIGH" if p_win >= self.min_ml_confidence else "LOW"

        self.last_ml_prediction = {
            "direction": candidate_direction,
            "p_win": round(p_win, 4),
            "confidence_label": conf_label,
            "features": feat_dict,
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }

        if p_win < self.min_ml_confidence:
            logger.info(
                "[ML Scorer Gating] ❌ %s signal (%s) rejected: Calibrated P(Win)=%.1f%% < %.1f%% minimum hurdle.",
                candidate_direction, "+".join(candidate_strats), p_win * 100.0, self.min_ml_confidence * 100.0
            )
            return "NONE", 0.0, []

        # ── Intelligence Scaling 3: Dynamic Kelly Criterion Multiplier ────────
        # Sizing scales from 0.5x up to 1.5x based on predicted edge:
        # Edge = P(Win) - (1 - P(Win))/2.0
        edge = max(0.02, p_win - ((1.0 - p_win) / 2.0))
        kelly_mult = float(np.clip(edge / 0.25, 0.50, 1.50))

        # PCR Tailwind Multiplier (1.15x when PCR strongly confirms direction)
        pcr_mult = 1.0
        if pcr > 0:
            if (candidate_direction == "LONG" and pcr >= 1.20) or (candidate_direction == "SHORT" and pcr <= 0.80):
                pcr_mult = 1.15

        final_mult = round(float(np.clip(base_mult * kelly_mult * pcr_mult, 0.50, 1.50)), 2)

        logger.info(
            "[AI Gating Approved] 🚀 %s setup PASSED ML Gate: P(Win)=%.1f%% (%s) | PCR=%.2f (Scale: %.2fx) | Kelly Scale: %.2fx → Final Multiplier: %.2fx",
            candidate_direction, p_win * 100.0, conf_label, pcr, pcr_mult, kelly_mult, final_mult
        )

        return candidate_direction, final_mult, candidate_strats

    def manage_position(self, current_spot: float, now: datetime) -> None:
        """Mark-to-market option premium and evaluate SL / Target / Trailing Stops."""
        pos = self.open_position
        if pos is None:
            return

        is_expiry_day = (now.weekday() == 3)
        now_time = now.time()

        # Mark-to-market option premium:
        # Tier 1: Real-time option chain quote if available from Angel One / NSE
        # Tier 2: Delta-sensitivity simulation on underlying spot move
        live_premium = None
        try:
            chain = fetch_live_option_chain(pos.underlying)
            if chain and chain.get("strikes"):
                for s in chain["strikes"]:
                    if abs(s.get("strike", 0.0) - pos.strike) < 25.0:
                        if pos.option_type == "CE" and float(s.get("ce_ltp") or 0.0) > 0:
                            live_premium = float(s["ce_ltp"])
                        elif pos.option_type == "PE" and float(s.get("pe_ltp") or 0.0) > 0:
                            live_premium = float(s["pe_ltp"])
                        break
        except Exception:
            pass

        if live_premium is not None and live_premium > 1.0:
            current_premium = live_premium
        else:
            if pos.option_type == "CE":
                spot_move = current_spot - pos.spot_at_entry
            else:  # PE option gains when spot drops
                spot_move = pos.spot_at_entry - current_spot
            current_premium = max(2.0, round(pos.entry_price + (spot_move * pos.delta), 2))

        pos.current_price = current_premium
        pos.unrealized_pnl = round((current_premium - pos.entry_price) * pos.quantity, 2)

        exit_triggered = False
        exit_reason = ""
        exit_price = current_premium

        # Spot-anchored exit logic (intraday institutional standard)
        is_call = (pos.option_type == "CE")
        spot_loss_hit = (current_spot <= pos.spot_stop_loss) if is_call else (current_spot >= pos.spot_stop_loss)
        spot_target_hit = (current_spot >= pos.spot_target_price) if is_call else (current_spot <= pos.spot_target_price)

        # 1. 14:30 Expiry Cutoff (protect option buyer from 0 theta decay)
        if is_expiry_day and now_time >= dtime(14, 30):
            exit_triggered = True
            exit_reason = "EXPIRY_1430_CUTOFF"

        # 2. 15:15 EOD Squareoff
        elif now_time >= dtime(15, 15):
            exit_triggered = True
            exit_reason = "EOD_1515_SQUAREOFF"

        # 3. Target Hit (Underlying Spot Target reached or +50% Premium gained)
        elif spot_target_hit or current_premium >= pos.target_price:
            exit_triggered = True
            exit_reason = "TARGET_HIT"

        # 4. Stop Loss Hit (Underlying Spot SL reached or -25% Premium lost)
        elif spot_loss_hit or current_premium <= pos.stop_loss:
            exit_triggered = True
            exit_reason = "STOP_LOSS_HIT"

        # 5. Reinforcement Learning Adaptive Exit & Trailing Stop Engine
        if not exit_triggered:
            if self.rl_exit_agent is not None:
                time_elapsed_mins = max(1, int((now - pos.entry_time).total_seconds() / 60.0))
                rl_action = self.rl_exit_agent.evaluate_exit(
                    entry_price=pos.entry_price,
                    current_price=current_premium,
                    initial_stop=pos.entry_price * 0.75,
                    target_price=pos.target_price,
                    direction="LONG",
                    time_elapsed_mins=time_elapsed_mins,
                    atr_14=pos.atr_at_entry,
                )
                self.last_rl_action = rl_action
                if rl_action.action_name == "EXIT_NOW":
                    exit_triggered = True
                    exit_reason = f"RL_AGENT_EXIT ({rl_action.rationale})"
                    logger.info("[RL Exit Agent] 🎯 Recommends immediate profit lock/risk mitigation: %s", rl_action.rationale)
                elif rl_action.action_name in ("TIGHTEN_STOP", "MOVE_BREAKEVEN"):
                    if rl_action.suggested_stop > pos.stop_loss:
                        logger.info(
                            "[RL Exit Agent] 🛡️ Dynamic Stop Tightened: ₹%.2f → ₹%.2f (%s)",
                            pos.stop_loss, rl_action.suggested_stop, rl_action.rationale
                        )
                        pos.stop_loss = round(rl_action.suggested_stop, 2)
            else:
                gain_pct = (current_premium - pos.entry_price) / max(1.0, pos.entry_price)
                if gain_pct >= 0.35:
                    pos.stop_loss = max(pos.stop_loss, round(pos.entry_price * 1.15, 2))
                elif gain_pct >= 0.20:
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)

        if exit_triggered:
            self._close_position(exit_price=exit_price, exit_reason=exit_reason, exit_time=now)

    def _close_position(self, exit_price: float, exit_reason: str, exit_time: datetime) -> None:
        """Execute simulated paper close and record to CSV log and SQLite database."""
        pos = self.open_position
        if pos is None:
            return

        brokerage = 40.0
        stt_fees = (exit_price * pos.quantity) * 0.000625  # STT on option sell
        total_costs = brokerage + stt_fees

        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        net_pnl = gross_pnl - total_costs

        self.current_capital += net_pnl
        ret_pct = ((exit_price - pos.entry_price) / max(1.0, pos.entry_price)) * 100.0

        trade_record = {
            "trade_id": pos.trade_id,
            "symbol": pos.symbol,
            "direction": "BUY",
            "entry_time": pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": pos.entry_price,
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price": round(exit_price, 2),
            "quantity": pos.quantity,
            "gross_pnl": round(gross_pnl, 2),
            "brokerage_stt": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(ret_pct, 2),
            "exit_reason": exit_reason,
            "strategies_triggered": "+".join(pos.strategies_agreed),
            "regime": pos.action_label,
        }
        self.completed_trades.append(trade_record)

        with open(TRADE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(trade_record.values()))

        # Update SQLite DB for UI sync
        self._record_db_trade_exit(pos.trade_id, exit_price, exit_time, net_pnl)

        status_tag = "PROFIT" if net_pnl >= 0 else "LOSS"
        sign = "+" if net_pnl >= 0 else ""
        logger.info(
            ">>> 💰 OPTION EXIT #%s: SELL %d QTY %s @ ₹%s | P&L: %s₹%s (%s %+.1f%%) | Reason: %s",
            pos.trade_id, pos.quantity, pos.symbol, f"{exit_price:,.2f}", sign, f"{net_pnl:,.2f}", status_tag, ret_pct, exit_reason
        )
        self.open_position = None
        self.last_trade_exit_time = exit_time
        self.daily_trade_count += 1

    def _evaluate_option_fair_value_shield(
        self,
        spot_price: float,
        strike: float,
        dte_days: float,
        is_call: bool,
        market_price: float,
        vix_val: float,
    ) -> Tuple[bool, float, float, float, str]:
        """
        Evaluate Black-Scholes Fair Value and IV-Crush risk.
        Returns: (is_safe, fair_value, mkt_iv, overprice_ratio, reason)
        """
        try:
            # Benchmark annual volatility from live India VIX (e.g. 13.0 -> 0.13)
            benchmark_iv = max(0.08, min(0.60, (vix_val or 13.0) / 100.0))
            bs = black_scholes(
                spot=spot_price,
                strike=strike,
                dte_days=dte_days,
                iv=benchmark_iv,
                r=RISK_FREE_RATE,
            )
            fair_value = bs.call_price if is_call else bs.put_price

            # Solve for market implied volatility
            opt_type_str = "CE" if is_call else "PE"
            mkt_iv = implied_volatility(
                market_price=market_price,
                spot=spot_price,
                strike=strike,
                dte_days=dte_days,
                option_type=opt_type_str,
                r=RISK_FREE_RATE,
            )

            # Overpricing ratio (market price vs theoretical benchmark fair price)
            overprice_ratio = round(market_price / max(1.0, fair_value), 2)

            # Rejection criteria:
            # 1. Market premium is > 30% above Black-Scholes Fair Value (overprice_ratio > 1.30)
            # 2. Market IV is pumped > 60% above prevailing India VIX benchmark
            if overprice_ratio > 1.30:
                reason = f"Overpriced Premium ({overprice_ratio:.2f}x Fair Value ₹{fair_value:.1f}, IV: {mkt_iv*100:.1f}%)"
                return False, round(fair_value, 2), round(mkt_iv, 4), overprice_ratio, reason

            if mkt_iv > (benchmark_iv * 1.60):
                reason = f"IV Overheated ({mkt_iv*100:.1f}% vs Benchmark VIX {vix_val:.1f}%)"
                return False, round(fair_value, 2), round(mkt_iv, 4), overprice_ratio, reason

            return True, round(fair_value, 2), round(mkt_iv, 4), overprice_ratio, "FAIR_VALUE_CONFIRMED"
        except Exception as exc:
            logger.debug("Option fair value shield evaluation fallback: %s", exc)
            return True, market_price, 0.15, 1.0, "FALLBACK_ALLOW"

    def execute_paper_entry(
        self,
        signal_direction: str,
        spot_price: float,
        atr_val: float,
        multiplier: float,
        strats: List[str],
        now: datetime,
    ) -> None:
        """Execute simulated Pure Option Buying entry (ATM Call for LONG, ATM Put for SHORT)."""
        if self.open_position is not None:
            return

        # 1. ATM Strike Determination
        step = 100.0 if "BANK" in self.symbol.upper() else (50.0 if "NIFTY" in self.symbol.upper() else 100.0)
        atm_strike = round(spot_price / step) * step

        # 2. Options Buying Selection:
        # - LONG signal -> Buy Call Option (CE)
        # - SHORT signal -> Buy Put Option (PE)
        is_call = (signal_direction in ("LONG", "BUY", "BULLISH"))
        opt_type = "CE" if is_call else "PE"
        opt_symbol = f"BANKNIFTY {int(atm_strike)} {opt_type}"
        action_label = f"CALL OPTION BUY ({opt_type})" if is_call else f"PUT OPTION BUY ({opt_type})"

        # 3. Option Premium & Delta Determination (Real Option Chain vs Directional Delta Model)
        entry_premium = None
        delta = 0.52
        pricing_source = "DIRECTIONAL DELTA SIMULATION (MODEL)"

        try:
            chain = fetch_live_option_chain(self.symbol)
            if chain and chain.get("strikes"):
                for s in chain["strikes"]:
                    if abs(float(s.get("strike") or 0.0) - atm_strike) < (step / 2.0):
                        if is_call and float(s.get("ce_ltp") or 0.0) > 0:
                            entry_premium = float(s["ce_ltp"])
                            raw_delta = float(s.get("ce_delta") or 0.52)
                            delta = round(max(0.15, min(0.95, raw_delta)), 2)
                            pricing_source = "LIVE OPTION CHAIN"
                        elif not is_call and float(s.get("pe_ltp") or 0.0) > 0:
                            entry_premium = float(s["pe_ltp"])
                            raw_delta = abs(float(s.get("pe_delta") or 0.52))
                            delta = round(max(0.15, min(0.95, raw_delta)), 2)
                            pricing_source = "LIVE OPTION CHAIN"
                        break
        except Exception as exc:
            logger.debug("Live option chain pricing fetch fallback: %s", exc)

        if entry_premium is None or entry_premium <= 5.0:
            # Fallback: Directional Delta Simulation model
            base_premium = round(spot_price * 0.0055, 1)
            entry_premium = round(base_premium + 1.5, 2)
            delta = 0.52
            pricing_source = "DIRECTIONAL DELTA SIMULATION (MODEL)"

        # 4. Stop Loss & Target calculation (Dual: Spot Index Level + Option Premium)
        # Spot SL/TP are delta-scaled to match the 25% premium stop so a stop-hit
        # realizes ~25% of premium (not 1.5*ATR, which historically realized ~46%).
        sl_premium = round(entry_premium * 0.75, 2)   # 25% Option SL (hard floor)
        tp_premium = round(entry_premium * 1.50, 2)   # 50% Option TP
        spot_sl_dist = max(100.0, round((entry_premium * 0.25) / delta, 1))
        spot_tp_dist = max(200.0, round((entry_premium * 0.50) / delta, 1))
        spot_sl = round(spot_price - spot_sl_dist, 2) if is_call else round(spot_price + spot_sl_dist, 2)
        spot_tp = round(spot_price + spot_tp_dist, 2) if is_call else round(spot_price - spot_tp_dist, 2)

        chain: Optional[Dict[str, Any]] = None
        # 4b. Jane Street Max-OI Pinning Trap Protection
        if self.jane_street is not None:
            try:
                chain = fetch_live_option_chain(self.symbol)
                if chain and chain.get("strikes"):
                    max_call_s = max(chain["strikes"], key=lambda x: x.get("ce_oi", 0.0)).get("strike", 0.0)
                    max_put_s = max(chain["strikes"], key=lambda x: x.get("pe_oi", 0.0)).get("strike", 0.0)
                    if max_call_s > 0 and max_put_s > 0:
                        spot_tp = self.jane_street.adjust_for_oi_trap(
                            entry_price=spot_price,
                            target_price=spot_tp,
                            direction="LONG" if is_call else "SHORT",
                            max_call_strike=max_call_s,
                            max_put_strike=max_put_s,
                        )
            except Exception as exc:
                logger.debug("OI trap defense lookup skipped: %s", exc)

        # 4c. Black-Scholes Option Fair Value & IV-Crush Shield
        dte_days = 3.0
        if chain and chain.get("selected_expiry"):
            try:
                exp_str = str(chain["selected_expiry"]).strip()
                for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d%b%Y"):
                    try:
                        exp_dt = datetime.strptime(exp_str, fmt)
                        diff = (exp_dt.date() - now.date()).days
                        dte_days = max(0.5, float(diff))
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        else:
            today_weekday = now.weekday()
            days_ahead = (2 - today_weekday) % 7
            dte_days = max(0.5, float(days_ahead if days_ahead > 0 else 7))

        vix_val = getattr(self, "_cached_vix", 13.0) or 13.0
        is_safe, fair_val, mkt_iv, fv_ratio, shield_reason = self._evaluate_option_fair_value_shield(
            spot_price=spot_price,
            strike=atm_strike,
            dte_days=dte_days,
            is_call=is_call,
            market_price=entry_premium,
            vix_val=vix_val,
        )

        if not is_safe:
            logger.warning(
                "🛑 [IV-SHIELD BLOCKED] %s entry rejected: %s | Market: ₹%.2f vs Fair Value: ₹%.2f. Capital protected from IV crush.",
                opt_symbol,
                shield_reason,
                entry_premium,
                fair_val,
            )
            return

        # 5. Position Sizing based on Capital Allocation & Lot Size
        risk_rupees = self.current_capital * self.max_trade_risk_pct
        risk_per_unit = max(20.0, entry_premium - sl_premium)
        target_qty = int(risk_rupees / risk_per_unit)
        allocated_qty = int(target_qty * multiplier)
        allocated_qty = max(self.lot_size, (allocated_qty // self.lot_size) * self.lot_size)

        # 5b. Notional Exposure Cap: premium x qty must stay <= 50% of capital.
        # With a ₹10k book this keeps a single option position at/below ₹5k notional,
        # refusing over-leveraged ATM/monthly contracts on a small account.
        self.notional_cap_pct = 0.50
        notional_exposure = entry_premium * allocated_qty
        if notional_exposure > (self.current_capital * self.notional_cap_pct):
            logger.warning(
                "[NOTIONAL CAP] ❌ %s rejected: Notional ₹%.2f > %.0f%% of capital ₹%.2f. "
                "High-conviction policy: skip over-leveraged option exposure.",
                action_label, notional_exposure, self.notional_cap_pct * 100, self.current_capital,
            )
            return

        self.trade_counter += 1
        r_target_dist = abs(tp_premium - entry_premium) / 2.0

        self.open_position = PaperPosition(
            trade_id=self.trade_counter,
            symbol=opt_symbol,
            underlying=self.symbol,
            option_type=opt_type,
            strike=atm_strike,
            direction="BUY",
            action_label=action_label,
            entry_time=now,
            entry_price=entry_premium,
            current_price=entry_premium,
            spot_at_entry=round(spot_price, 2),
            quantity=allocated_qty,
            stop_loss=sl_premium,
            target_price=tp_premium,
            spot_stop_loss=spot_sl,
            spot_target_price=spot_tp,
            delta=delta,
            atr_at_entry=round(atr_val, 2),
            strategies_agreed=strats,
            confidence_multiplier=multiplier,
            r_target_dist=r_target_dist,
            pricing_source=pricing_source,
            fair_value=fair_val,
            implied_volatility=mkt_iv,
            fair_value_ratio=fv_ratio,
        )

        # Record to SQLite DB for UI sync
        self._record_db_trade_entry(self.open_position)

        logger.info(
            ">>> 🎯 OPTION BUYING ENTRY #%s: BUY %d QTY %s (%s) @ ₹%s [%s] | Delta: %.2f | Spot: ₹%s | SL: ₹%s (-25%%) | TP: ₹%s (+50%%) | Strats: %s",
            self.trade_counter, allocated_qty, opt_symbol, action_label, f"{entry_premium:,.2f}", pricing_source, delta, f"{spot_price:,.2f}", f"{sl_premium:,.2f}", f"{tp_premium:,.2f}", strats
        )

    def _record_db_trade_entry(self, pos: PaperPosition) -> None:
        """Insert active trade into SQLite database for dashboard sync."""
        try:
            import sqlite3
            conn = sqlite3.connect("data/trader.db")
            cur = conn.cursor()
            trade_id_str = f"ORD_{pos.trade_id:06d}"
            notes_str = "+".join(pos.strategies_agreed)
            if pos.fair_value is not None:
                iv_str = f"{pos.implied_volatility*100:.1f}%" if pos.implied_volatility is not None else "N/A"
                ratio_str = f"{pos.fair_value_ratio:.2f}x" if pos.fair_value_ratio is not None else "N/A"
                notes_str += f" | FV:₹{pos.fair_value:.1f}|IV:{iv_str}|Ratio:{ratio_str}"

            cur.execute("""
                INSERT OR REPLACE INTO trades 
                (id, timestamp, symbol, side, entry_price, quantity, stop_price, target_price, ml_confidence, regime, atr_at_entry, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id_str,
                pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                pos.symbol,
                "BUY",
                pos.entry_price,
                pos.quantity,
                pos.stop_loss,
                pos.target_price,
                pos.confidence_multiplier,
                pos.action_label,
                pos.atr_at_entry,
                notes_str,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("DB trade entry error: %s", e)

    def _record_db_trade_exit(self, trade_id: int, exit_price: float, exit_time: datetime, net_pnl: float) -> None:
        """Update closed trade in SQLite database."""
        try:
            import sqlite3
            conn = sqlite3.connect("data/trader.db")
            cur = conn.cursor()
            trade_id_str = f"ORD_{trade_id:06d}"
            cur.execute("""
                UPDATE trades 
                SET exit_timestamp = ?, exit_price = ?, pnl_rupees = ?
                WHERE id = ?
            """, (exit_time.strftime("%Y-%m-%d %H:%M:%S"), exit_price, net_pnl, trade_id_str))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("DB trade exit error: %s", e)

    def _get_morning_bias(self) -> str:
        """Load today's directional bias from data/morning_bias_report.json (BULLISH/BEARISH/NEUTRAL/HIGH_VIX)."""
        for mb_file in ("data/morning_bias_report.json", "data/morning_bias.json"):
            try:
                mb_path = Path(mb_file)
                if mb_path.exists():
                    with open(mb_path, "r", encoding="utf-8") as f:
                        mb_data = json.load(f)
                        if mb_data.get("date") == self.current_day_str and mb_data.get("bias"):
                            self._cached_morning_bias = str(mb_data["bias"]).upper()
                            return self._cached_morning_bias
            except Exception:
                pass
        if not self._cached_morning_bias:
            logger.warning("[BIAS GUARD] Morning bias report missing/expired for %s — defaulting to NEUTRAL (no options entries).", self.current_day_str)
            self._cached_morning_bias = "NEUTRAL"
        return self._cached_morning_bias

    def run_iteration(self, iteration_num: int) -> None:
        """Run single polling cycle."""
        self._check_and_hot_reload_config()
        now = datetime.now(IST)
        date_str = now.strftime("%Y-%m-%d")

        if date_str != self.current_day_str:
            self.current_day_str = date_str
            self.daily_start_capital = self.current_capital
            self.daily_circuit_locked = False
            self.daily_trade_count = 0
            self.last_trade_exit_time = None

        # ── Gate 1: Strict Market Hours (Mon-Fri 09:20 - 15:15 IST) ─────────────
        is_weekday = (now.weekday() < 5)  # 0 = Monday, 4 = Friday
        now_time = now.time()
        is_market_open = is_weekday and (self.market_open_time <= now_time <= self.market_close_time)

        if not is_market_open:
            # Auto-squareoff open position if past 15:15
            if self.open_position is not None:
                self._close_position(
                    exit_price=self.open_position.current_price,
                    exit_reason="EOD_1515_SQUAREOFF",
                    exit_time=now,
                )
            if iteration_num % 12 == 0:  # Print quiet status every 60s
                day_name = now.strftime("%A")
                logger.info("[MARKET CLOSED] (%s %s). Trading Window: Mon-Fri 09:20 - 15:15 IST. Engine idle.", day_name, now.strftime("%H:%M:%S IST"))
            return

        # ── Gate 2: Daily Circuit Breaker ───────────────────────────────────────
        daily_loss_pct = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_loss_pct >= self.daily_loss_limit_pct:
            if not self.daily_circuit_locked:
                logger.warning("[CRITICAL] Daily Circuit Breaker Triggered (Loss >= %.1f%%). System Halted.", self.daily_loss_limit_pct * 100)
                self.daily_circuit_locked = True
            return

        # ── Gate 3: Max Daily Trades Cap (Max 3 Trades/Day) ─────────────────────
        if self.daily_trade_count >= self.max_daily_trades:
            if iteration_num % 12 == 0:
                logger.info("[RISK CAP] Max Daily Trades (%d/%d) reached. Trading locked for today.", self.daily_trade_count, self.max_daily_trades)
            return

        # ── Gate 4: Post-Trade Cooldown (15-min cooldown between trades) ────────
        if self.last_trade_exit_time is not None:
            elapsed_sec = (now - self.last_trade_exit_time).total_seconds()
            if elapsed_sec < self.trade_cooldown_seconds:
                rem_mins = int((self.trade_cooldown_seconds - elapsed_sec) / 60)
                if iteration_num % 6 == 0:
                    logger.info("[COOLDOWN] Post-trade cooldown active (%dm remaining). Waiting for high-conviction setup.", rem_mins)
                return

        # 1. Fetch live data
        df_candles, spot_price, vix_val = self.fetch_live_feed()
        df_feat = self.compute_features(df_candles)
        curr_row = df_feat.iloc[-1]

        # 2. Manage existing position
        self.manage_position(spot_price, now)

        # 3. Check VIX Gate (VIX >= 12.0)
        if vix_val < self.min_vix_gate:
            logger.info("[VIX GATE] India VIX: %.2f < %.2f -> Gated (Low Volatility)", vix_val, self.min_vix_gate)
            return

        # 3b. Confluence Session Timing Quality Gates (Component 9 Data-Derived from settings.yaml)
        # Gate 3b-1: Opening Whip Filter (No entries before 09:30 IST; win rate 4.1% in 09:15-09:30 opening whip)
        if self.open_position is None and now_time < self.earliest_entry_time:
            if iteration_num % 6 == 0:
                logger.info("[SESSION GATE] Pre-%s IST opening whip window — waiting for opening range formation.", self.earliest_entry_time.strftime("%H:%M"))
            return

        # Gate 3b-2: Midday Chop Zone Block (No new entries 11:30–13:30 IST; win rate ~0% due to theta decay + chop)
        if self.open_position is None and (self.session_block_start <= now_time <= self.session_block_end):
            if iteration_num % 6 == 0:
                logger.info("[MIDDAY BLOCK] %s-%s IST dead zone — no new entries (theta decay / chop window).",
                            self.session_block_start.strftime("%H:%M"), self.session_block_end.strftime("%H:%M"))
            return

        # Gate 3b-3: Late Day Entry Cutoff (No new entries after 14:45 IST ahead of 15:15 EOD squareoff)
        if self.open_position is None and now_time > self.latest_entry_time:
            if iteration_num % 6 == 0:
                logger.info("[SESSION GATE] Post-%s IST entry cutoff — no new entries before 15:15 EOD squareoff.", self.latest_entry_time.strftime("%H:%M"))
            return

        # 4. Check Layer 3 Regime Classifier
        regime = self.regime_detector.classify_market_state(df_feat, current_vix=vix_val)
        regime_name = regime.value

        # Directional option buying requires a confirmed trend regime.
        # Flat/chop/compression regimes halt entries (no option-buying edge).
        if regime not in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            logger.info("[REGIME] State: %s (ADX: %.1f) -> no directional option buying outside confirmed trend", regime_name, curr_row["adx14"])
            return

        # 5. Evaluate Active Strategies & Consensus with AI ML Gating
        direction, mult, strats = self.evaluate_signals(df_feat, regime_name, vix_val=vix_val)

        # Print heartbeat
        unrealized = self.open_position.unrealized_pnl if self.open_position else 0.0
        strat_str = f"Signal: {direction} ({'+'.join(strats)})" if direction != "NONE" else "Signal: Neutral"
        pos_str = f"Opt: {self.open_position.symbol} ({self.open_position.quantity} Qty @ ₹{self.open_position.current_price:.2f})" if self.open_position else "Opt: CASH"
        logger.info("Spot: ₹%s | VIX: %.1f | PCR: %.2f (%s) | Regime: %-12s | %-28s | %s | Open P&L: ₹%+.2f", f"{spot_price:,.2f}", vix_val, self._cached_pcr, self._cached_pcr_signal, regime_name, strat_str, pos_str, unrealized)

        # 6. Trigger entry if signal generated (direction must align with trend regime)
        if regime == MarketRegime.TRENDING_UP and direction == "SHORT":
            direction = "NONE"
        if regime == MarketRegime.TRENDING_DOWN and direction == "LONG":
            direction = "NONE"

        # 6b. Morning Bias Direction Guard (macro alignment — high-conviction policy)
        # BULLISH bias → block SHORT/PE buys;  BEARISH → block LONG/CE buys;
        # NEUTRAL / HIGH_VIX → block all directional option entries.
        if direction != "NONE":
            bias = self._get_morning_bias()
            if bias == "BEARISH" and direction == "LONG":
                direction = "NONE"
                logger.warning("[BIAS GUARD] ❌ LONG/CE blocked: Morning bias BEARISH on %s", date_str)
            elif bias == "BULLISH" and direction == "SHORT":
                direction = "NONE"
                logger.warning("[BIAS GUARD] ❌ SHORT/PE blocked: Morning bias BULLISH on %s", date_str)
            elif bias in ("NEUTRAL", "HIGH_VIX", ""):
                direction = "NONE"
                logger.warning("[BIAS GUARD] ❌ Options blocked: Morning bias %s (high-conviction policy).", bias or "UNKNOWN")

        if direction != "NONE" and self.open_position is None:
            atr_val = float(curr_row["atr14"])
            self.execute_paper_entry(direction, spot_price, atr_val, mult, strats, now)

    def print_summary(self) -> None:
        """Print institutional summary of live paper performance."""
        now = datetime.now(IST)
        n_trades = len(self.completed_trades)
        wins = sum(1 for t in self.completed_trades if t["net_pnl"] > 0)
        wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        total_pnl = self.current_capital - self.initial_capital
        unrealized = self.open_position.unrealized_pnl if self.open_position else 0.0
        open_pos_str = f"{self.open_position.direction} {self.open_position.quantity} QTY @ ₹{self.open_position.entry_price}" if self.open_position else "NONE"

        summary_lines = [
            "=" * 80,
            f"  LIVE PAPER TRADING SUMMARY — {now.strftime('%Y-%m-%d %H:%M:%S IST')}",
            "=" * 80,
            f"  Current Capital   : ₹{self.current_capital:,.2f} (P&L: ₹{total_pnl:+,.2f} | {(total_pnl / self.initial_capital) * 100.0:+.2f}%)",
            f"  Open Position     : {open_pos_str}",
            f"  Unrealized P&L    : ₹{unrealized:+,.2f}",
            f"  Completed Trades  : {n_trades} (Wins: {wins}, Losses: {n_trades - wins})",
            f"  Win Rate Today    : {wr:.1f}%",
            f"  Active Strategies : {self.active_strategies}",
            f"  VIX Gate Status   : Active (Min VIX: {self.min_vix_gate})",
            "=" * 80,
        ]
        for line in summary_lines:
            logger.info(line)

    def run_loop(self, max_iterations: Optional[int] = None, poll_interval_sec: float = 5.0) -> None:
        """Run continuous paper trading loop."""
        init_lines = [
            "#" * 80,
            "  🏛️ ALGO-TRADER: INSTITUTIONAL LIVE PAPER TRADING ENGINE ACTIVATED",
            "#" * 80,
            f"  Config Loaded     : {self.config_path}",
            f"  Primary Asset     : {self.symbol} (Lot Size: {self.lot_size})",
            f"  Active Strategies : {self.active_strategies} (VWAP Excluded)",
            f"  Risk Sizing       : Kelly {self.max_trade_risk_pct * 100:.1f}% Hard Risk Cap per trade",
            f"  ATR Stop / Target : SL: {self.sl_atr_mult:.2f}x ATR | TP: {self.tp_atr_mult:.2f}x ATR (1:{self.tp_atr_mult/self.sl_atr_mult:.1f} R:R)",
            f"  VIX Gate          : Min VIX {self.min_vix_gate:.1f} | Max VIX {self.max_vix_gate:.1f}",
            f"  Starting Capital  : ₹{self.initial_capital:,.2f}",
            "#" * 80,
        ]
        for line in init_lines:
            logger.info(line)

        iteration = 0
        last_summary_time = time.time()

        while True:
            try:
                iteration += 1
                self.run_iteration(iteration)

                if time.time() - last_summary_time >= 60.0:
                    self.print_summary()
                    last_summary_time = time.time()

                if max_iterations and iteration >= max_iterations:
                    break

                time.sleep(poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("Paper Trader Stopped by User.")
                break
            except Exception as e:
                logger.error("Error in paper trading iteration: %s", e, exc_info=True)
                time.sleep(5.0)

        self.print_summary()

    def get_ai_telemetry(self) -> Dict[str, Any]:
        """Expose real-time institutional AI intelligence telemetry for Dashboard & APIs."""
        ml_avail = self.ml_scorer.is_available if self.ml_scorer else False
        p_win = self.last_ml_prediction.get("p_win", 0.58)
        conf_label = self.last_ml_prediction.get("confidence_label", "MODERATE")

        return {
            "status": "SUCCESS",
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "ai_layers": {
                "layer_1_regime": {
                    "engine": "ADX + Choppiness + ATR Volatility Classifier",
                    "status": "ACTIVE",
                    "mode": "DIRECTIONAL_TREND_OR_CASH_GATE",
                },
                "layer_2_consensus": {
                    "engine": "Multi-Strategy Consensus Matrix (ORB + Supertrend + Flow)",
                    "voting_threshold": ">= 2 agreeing strategies required",
                },
                "layer_3_jane_street_defense": {
                    "anti_spoofing": "ACTIVE (2-Bar Confirmation Required)",
                    "max_oi_pinning_shield": "ACTIVE (0.5% Pre-Wall Strike Trim)",
                    "expiry_cutoff": "14:30 IST Mandatory Square-Off",
                },
                "layer_4_ml_gating": {
                    "engine": "Calibrated Dual-Model Ensemble (LightGBM + XGBoost)",
                    "is_active": ml_avail,
                    "latest_p_win": round(p_win, 4),
                    "confidence_label": conf_label,
                    "hurdle_rate": self.min_ml_confidence,
                    "verdict": "APPROVED" if p_win >= self.min_ml_confidence else "GATED",
                },
                "layer_5_rl_exit": {
                    "engine": "Q-Learning Dynamic Trailing & Breakeven Lock",
                    "is_active": self.rl_exit_agent is not None,
                    "latest_action": self.last_rl_action.action_name if self.last_rl_action else "IDLE",
                    "rationale": self.last_rl_action.rationale if self.last_rl_action else "Monitoring trade lifecycle",
                },
                "layer_6_pcr_guard": {
                    "engine": "Put-Call Ratio (PCR) Institutional Guard",
                    "current_pcr": round(self._cached_pcr, 2),
                    "pcr_signal": self._cached_pcr_signal,
                    "pcr_15m_delta": round(self._cached_pcr_15m_delta, 4),
                    "long_hurdle": "PCR >= 0.80 (Rejects if Call writing resistance)",
                    "short_hurdle": "PCR <= 1.25 (Rejects if Put writing support)",
                },
            },
        }


if __name__ == "__main__":
    trader = LivePaperTrader()
    trader.run_loop()
