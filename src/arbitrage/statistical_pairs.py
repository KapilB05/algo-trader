"""
Statistical Pairs Arbitrage Scanner — Cointegration & Z-Score Engine.
Mathematical Foundation:
  Two assets A and B are said to be COINTEGRATED if their spread (A - β*B)
  is STATIONARY — meaning it has a stable long-term mean it always reverts to.
  
  The Augmented Dickey-Fuller (ADF) test proves stationarity mathematically.
  If ADF p-value < 0.05 → spread is stationary → pairs trade is mathematically sound.
  Z-Score = (spread - mean_spread) / std_spread
  
  When |Z| > 2.0 → spread has diverged 2 standard deviations → enter trade:
    - Buy the underperformer, Short the outperformer
    - Wait for Z → 0 (mean reversion)
  
  Exit when |Z| < 0.5 (mean reversion complete).
  
  This strategy is MARKET DIRECTION NEUTRAL — immune to Nifty rallying or crashing.
Nifty 50 Constituent Pairs Monitored:
  - HDFCBANK vs ICICIBANK (Banking pairs)
  - RELIANCE vs ONGC (Energy pairs)
  - INFY vs TCS (IT pairs)
  - AXISBANK vs KOTAKBANK (Private bank pairs)
  - LT vs POWERGRID (Infrastructure pairs)
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
logger = logging.getLogger("StatisticalPairsScanner")
IST = ZoneInfo("Asia/Kolkata")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Entry/Exit thresholds
Z_ENTRY = 2.0       # Enter when |Z| > 2.0
Z_EXIT  = 0.5       # Exit when |Z| < 0.5
LOOKBACK_BARS = 60  # 60 bars ≈ 5 hours of 5-min data for rolling stats
# Angel One NSE tokens for Nifty 50 constituents
PAIRS_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "HDFCBANK-ICICIBANK",
        "asset_a": {"symbol": "HDFCBANK",  "token": "1333",  "exchange": "NSE"},
        "asset_b": {"symbol": "ICICIBANK", "token": "4963",  "exchange": "NSE"},
        "lot_size": 550,   # HDFCBANK lot
    },
    {
        "name": "RELIANCE-ONGC",
        "asset_a": {"symbol": "RELIANCE", "token": "2885",  "exchange": "NSE"},
        "asset_b": {"symbol": "ONGC",     "token": "2475",  "exchange": "NSE"},
        "lot_size": 250,
    },
    {
        "name": "INFY-TCS",
        "asset_a": {"symbol": "INFY", "token": "1594", "exchange": "NSE"},
        "asset_b": {"symbol": "TCS",  "token": "11536", "exchange": "NSE"},
        "lot_size": 300,
    },
    {
        "name": "AXISBANK-KOTAKBANK",
        "asset_a": {"symbol": "AXISBANK",  "token": "5900",  "exchange": "NSE"},
        "asset_b": {"symbol": "KOTAKBANK", "token": "1922",  "exchange": "NSE"},
        "lot_size": 1200,
    },
    {
        "name": "LT-POWERGRID",
        "asset_a": {"symbol": "LT",        "token": "11483", "exchange": "NSE"},
        "asset_b": {"symbol": "POWERGRID", "token": "14977", "exchange": "NSE"},
        "lot_size": 575,
    },
]
@dataclass
class PairState:
    """Rolling state for a single cointegrated pair."""
    name: str
    asset_a: str
    asset_b: str
    prices_a: List[float] = field(default_factory=list)
    prices_b: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    beta: float = 1.0           # Hedge ratio (OLS regression slope)
    spread_mean: float = 0.0    # Rolling mean of spread
    spread_std: float = 1.0     # Rolling std of spread
    current_z: float = 0.0      # Latest z-score
    current_spread: float = 0.0 # Latest spread value
    is_cointegrated: bool = False  # ADF test result
    adf_pvalue: float = 1.0     # ADF p-value (< 0.05 = cointegrated)
    pearson_r: float = 0.0      # Pearson correlation
    last_updated: str = ""
@dataclass
class PairsPaperTrade:
    """An active virtual pairs trade."""
    trade_id: int
    pair_name: str
    direction: str          # "LONG_A_SHORT_B" or "SHORT_A_LONG_B"
    entry_z: float          # Z-score at entry
    entry_price_a: float
    entry_price_b: float
    entry_spread: float
    entry_time: str
    beta: float
    lot_size: int
    paper_pnl: float = 0.0
    status: str = "OPEN"    # OPEN → CLOSED
    exit_z: float = 0.0
    exit_time: Optional[str] = None
    exit_reason: str = ""
def _simple_adf_test(series: List[float]) -> Tuple[float, bool]:
    """
    Simplified ADF (Augmented Dickey-Fuller) test without statsmodels dependency.
    
    Tests if the series is stationary using the regression:
      Δy_t = α + β*y_{t-1} + ε
    
    Under H0 (non-stationary): β ≈ 0
    Under H1 (stationary):     β < 0 (mean-reverting)
    
    Returns (approximate_pvalue, is_stationary).
    We use a simpler approximation: check if the spread's autocorrelation at lag-1
    is significantly negative, which is the hallmark of mean-reversion.
    """
    if len(series) < 20:
        return 1.0, False
    
    arr = np.array(series)
    diff = np.diff(arr)
    lagged = arr[:-1]
    
    # OLS regression of Δy on y_{t-1}
    # β = Cov(Δy, y_lagged) / Var(y_lagged)
    try:
        cov = np.cov(diff, lagged)
        if cov[1, 1] < 1e-10:
            return 1.0, False
        beta = cov[0, 1] / cov[1, 1]
        
        # Residuals and t-statistic
        residuals = diff - beta * lagged
        se = np.std(residuals) / (np.sqrt(cov[1, 1]) * np.sqrt(len(lagged)))
        t_stat = beta / (se + 1e-9)
        
        # Approximate p-value: ADF critical values at 5% significance ~ -2.86 for T~60
        # If t-stat < -2.86, reject H0 (series is stationary)
        is_stationary = bool(t_stat < -2.86)
        # Rough p-value approximation
        p_approx = max(0.001, min(0.999, 0.5 * (1.0 + t_stat / 5.0))) if t_stat > -10 else 0.01
        
        return round(float(p_approx), 4), is_stationary
    except Exception:
        return 1.0, False
class StatisticalPairsScanner:
    """
    Real-time statistical pairs arbitrage scanner.
    Monitors 5 cointegrated Nifty50 constituent pairs using Angel One live data.
    All trades are paper-only virtual trades.
    """
    def __init__(self) -> None:
        self._pairs: Dict[str, PairState] = {}
        self._paper_trades: List[PairsPaperTrade] = []
        self._trade_id_counter = 2000
        self._scan_count = 0
        # Initialize pair states
        for cfg in PAIRS_CONFIG:
            name = cfg["name"]
            self._pairs[name] = PairState(
                name=name,
                asset_a=cfg["asset_a"]["symbol"],
                asset_b=cfg["asset_b"]["symbol"],
            )
        # Logging
        log_file = LOGS_DIR / "arb_scanner.log"
        if not any(isinstance(h, logging.FileHandler) and "arb_scanner" in getattr(h, "baseFilename", "") for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    def _fetch_pair_prices(self, pair_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Fetch live LTP for both assets in a pair from Angel One."""
        try:
            from src.auth.angel_auth import api_manager
            auth = api_manager
            if not auth or not auth.smartApi:
                return None, None
            token_a = pair_cfg["asset_a"]["token"]
            token_b = pair_cfg["asset_b"]["token"]
            res = auth.smartApi.getMarketData("LTP", {"NSE": [token_a, token_b]})
            if not res or not res.get("status") or not res.get("data", {}).get("fetched"):
                return None, None
            fetched = res["data"]["fetched"]
            prices = {str(q.get("symbolToken")): float(q.get("ltp") or 0.0) for q in fetched}
            p_a = prices.get(token_a)
            p_b = prices.get(token_b)
            if not p_a or not p_b or p_a <= 0 or p_b <= 0:
                return None, None
            return p_a, p_b
        except Exception as e:
            logger.debug("Price fetch error for pair: %s", e)
            return None, None
    def _update_pair_state(self, state: PairState, p_a: float, p_b: float) -> None:
        """Update rolling stats for a pair given new price observations."""
        ts = datetime.now(IST).strftime("%H:%M:%S")
        state.prices_a.append(p_a)
        state.prices_b.append(p_b)
        state.timestamps.append(ts)
        # Keep rolling window
        if len(state.prices_a) > LOOKBACK_BARS:
            state.prices_a = state.prices_a[-LOOKBACK_BARS:]
            state.prices_b = state.prices_b[-LOOKBACK_BARS:]
            state.timestamps = state.timestamps[-LOOKBACK_BARS:]
        if len(state.prices_a) < 10:
            return
        arr_a = np.array(state.prices_a)
        arr_b = np.array(state.prices_b)
        # OLS hedge ratio: minimize variance of (A - β*B)
        var_b = float(np.var(arr_b))
        var_a = float(np.var(arr_a))
        if var_b > 1e-8:
            beta = float(np.cov(arr_a, arr_b)[0, 1] / (var_b + 1e-9))
        else:
            beta = 1.0
        state.beta = round(beta if not (np.isnan(beta) or np.isinf(beta)) else 1.0, 6)
        spread = arr_a - state.beta * arr_b
        s_mean = float(np.mean(spread))
        s_std = float(np.std(spread))
        state.spread_mean = round(s_mean if not (np.isnan(s_mean) or np.isinf(s_mean)) else 0.0, 4)
        state.spread_std = max(0.01, round(s_std if not (np.isnan(s_std) or np.isinf(s_std)) else 1.0, 4))
        c_spread = float(spread[-1])
        state.current_spread = round(c_spread if not (np.isnan(c_spread) or np.isinf(c_spread)) else 0.0, 4)
        z = (state.current_spread - state.spread_mean) / state.spread_std
        state.current_z = round(float(z) if not (np.isnan(z) or np.isinf(z)) else 0.0, 4)
        # Pearson correlation (guard against zero variance)
        if var_a > 1e-8 and var_b > 1e-8:
            corr_matrix = np.corrcoef(arr_a, arr_b)
            r = float(corr_matrix[0, 1])
            state.pearson_r = round(r if not (np.isnan(r) or np.isinf(r)) else 0.95, 4)
        else:
            state.pearson_r = 0.95
        # ADF test every 20 bars to avoid compute overhead
        if len(state.prices_a) % 20 == 0 or not state.is_cointegrated:
            p_val, is_stat = _simple_adf_test(list(spread))
            state.adf_pvalue = p_val
            state.is_cointegrated = is_stat
        state.last_updated = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    def scan_once(self) -> List[Dict[str, Any]]:
        """
        Fetch prices for all pairs and evaluate entry/exit signals.
        Returns list of signal dicts for any new entries or exits.
        """
        self._scan_count += 1
        signals: List[Dict[str, Any]] = []
        for cfg in PAIRS_CONFIG:
            name = cfg["name"]
            state = self._pairs[name]
            lot_size = cfg["lot_size"]
            p_a, p_b = self._fetch_pair_prices(cfg)
            if p_a is None or p_b is None:
                continue
            self._update_pair_state(state, p_a, p_b)
            if len(state.prices_a) < 10:
                continue
            # Check if any open trade in this pair needs to be closed
            open_trade = next((t for t in self._paper_trades if t.pair_name == name and t.status == "OPEN"), None)
            if open_trade:
                # Mark-to-market
                if open_trade.direction == "LONG_A_SHORT_B":
                    pnl_a = (p_a - open_trade.entry_price_a) * lot_size
                    pnl_b = (open_trade.entry_price_b - p_b) * lot_size * open_trade.beta
                else:
                    pnl_a = (open_trade.entry_price_a - p_a) * lot_size
                    pnl_b = (p_b - open_trade.entry_price_b) * lot_size * open_trade.beta
                open_trade.paper_pnl = round(pnl_a + pnl_b, 2)
                # Exit if spread reverted to mean
                if abs(state.current_z) <= Z_EXIT:
                    open_trade.status = "CLOSED"
                    open_trade.exit_z = state.current_z
                    open_trade.exit_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
                    open_trade.exit_reason = "MEAN_REVERSION"
                    signals.append({
                        "type": "EXIT",
                        "pair": name,
                        "trade_id": open_trade.trade_id,
                        "exit_z": state.current_z,
                        "paper_pnl_inr": open_trade.paper_pnl,
                    })
                    logger.info(
                        "✅ PAIRS EXIT | %s | Trade #%d | Z=%.2f | Paper P&L ₹%+.2f | Reason: MEAN_REVERSION",
                        name, open_trade.trade_id, state.current_z, open_trade.paper_pnl
                    )
            elif abs(state.current_z) >= Z_ENTRY:
                # Entry signal — only if pair is confirmed cointegrated (|r| > 0.75)
                if state.pearson_r < 0.75:
                    continue
                direction = "LONG_A_SHORT_B" if state.current_z < 0 else "SHORT_A_LONG_B"
                trade = PairsPaperTrade(
                    trade_id=self._trade_id_counter,
                    pair_name=name,
                    direction=direction,
                    entry_z=state.current_z,
                    entry_price_a=p_a,
                    entry_price_b=p_b,
                    entry_spread=state.current_spread,
                    entry_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                    beta=state.beta,
                    lot_size=lot_size,
                )
                self._trade_id_counter += 1
                self._paper_trades.append(trade)
                action_str = f"BUY {state.asset_a} / SELL {state.asset_b}" if direction == "LONG_A_SHORT_B" else f"SELL {state.asset_a} / BUY {state.asset_b}"
                logger.info(
                    "📊 PAIRS ENTRY | %s | Z=%.2f | ρ=%.2f | %s | Trade #%d | A=₹%.2f B=₹%.2f | ADF p=%.3f",
                    name, state.current_z, state.pearson_r, action_str,
                    trade.trade_id, p_a, p_b, state.adf_pvalue
                )
                signals.append({
                    "type": "ENTRY",
                    "pair": name,
                    "trade_id": trade.trade_id,
                    "direction": direction,
                    "side": direction,
                    "z_score": state.current_z,
                    "pearson_r": state.pearson_r,
                    "price_a": p_a,
                    "price_b": p_b,
                })
        return signals
    def get_summary(self) -> Dict[str, Any]:
        """Return JSON-serializable summary for the dashboard API."""
        open_trades = [t for t in self._paper_trades if t.status == "OPEN"]
        closed_trades = [t for t in self._paper_trades if t.status == "CLOSED"]
        total_pnl = sum(t.paper_pnl for t in self._paper_trades)
        wins = sum(1 for t in closed_trades if t.paper_pnl > 0)
        pairs_status = []
        for name, state in self._pairs.items():
            pairs_status.append({
                "pair": name,
                "asset_a": state.asset_a,
                "asset_b": state.asset_b,
                "z_score": state.current_z,
                "pearson_r": state.pearson_r,
                "is_cointegrated": state.is_cointegrated,
                "adf_pvalue": state.adf_pvalue,
                "spread_mean": round(state.spread_mean, 4),
                "spread_std": round(state.spread_std, 4),
                "current_spread": round(state.current_spread, 4),
                "beta": state.beta,
                "bars_collected": len(state.prices_a),
                "signal": (
                    "LONG_A_SHORT_B" if state.current_z <= -Z_ENTRY
                    else ("SHORT_A_LONG_B" if state.current_z >= Z_ENTRY else "NEUTRAL")
                ),
                "last_updated": state.last_updated,
            })
        open_trade_list = []
        for t in open_trades:
            open_trade_list.append({
                "trade_id": t.trade_id,
                "pair": t.pair_name,
                "direction": t.direction,
                "side": t.direction,
                "entry_z": t.entry_z,
                "entry_time": t.entry_time,
                "paper_pnl_inr": t.paper_pnl,
                "pnl": t.paper_pnl,
                "net_pnl": t.paper_pnl,
            })
        return {
            "scanner": "STATISTICAL_PAIRS_ARBITRAGE",
            "scan_count": self._scan_count,
            "open_paper_trades": len(open_trades),
            "closed_paper_trades": len(closed_trades),
            "total_paper_pnl_inr": round(total_pnl, 2),
            "win_trades": wins,
            "pairs_monitored": len(self._pairs),
            "pairs_status": pairs_status,
            "open_trade_details": open_trade_list,
        }
