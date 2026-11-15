"""
Pydantic v2 Schemas for Module 1 — Morning Bias & System Health Check.
Master Architecture & Requirements v2.1 (Hardened + Library Locked).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StatusEnum(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class BiasEnum(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VIX = "HIGH_VIX"


class StrategyRegimeEnum(str, Enum):
    DIRECTIONAL_OPTION_BUYING = "DIRECTIONAL_OPTION_BUYING"
    CREDIT_SPREAD_SELLING = "CREDIT_SPREAD_SELLING"
    BREAKOUT_CONFLUENCE = "BREAKOUT_CONFLUENCE"
    STANDBY_CHOP = "STANDBY_CHOP"


class HealthCheckItem(BaseModel):
    name: str
    passed: bool
    details: str
    error: Optional[str] = None


class HealthCheckResult(BaseModel):
    status: StatusEnum
    total_checks: int = 5
    passed_checks: int
    failed_checks: List[str] = Field(default_factory=list)
    items: List[HealthCheckItem] = Field(default_factory=list)


class SystemStatus(BaseModel):
    status: StatusEnum
    timestamp: str
    trace_id: str
    failed_checks: List[str] = Field(default_factory=list)


class PivotLevels(BaseModel):
    pivot: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    previous_high: float
    previous_low: float
    previous_close: float


class ActiveStrategyParams(BaseModel):
    regime: StrategyRegimeEnum
    vix_band: str
    allowed_strategies: List[str]
    max_risk_per_trade_pct: float = 2.0
    min_reward_risk_ratio: float = 2.0
    daily_drawdown_lockout_pct: float = 5.0
    max_open_risk_cap_pct: float = 6.0
    lot_size_nifty: int = 65
    lot_size_banknifty: int = 30
    updated_at: str
    trace_id: str


class MacroBiasReport(BaseModel):
    trace_id: str
    timestamp: str
    date: str
    bias: BiasEnum
    strategy_regime: StrategyRegimeEnum = StrategyRegimeEnum.BREAKOUT_CONFLUENCE
    sp500_change_pct: float
    gift_nifty: float
    india_vix: float
    nifty_prev_close: float
    key_levels: PivotLevels
    banknifty_key_levels: Optional[PivotLevels] = None
    active_params: Optional[ActiveStrategyParams] = None
    notes: str
    health_check: HealthCheckResult


class InstrumentMasterRecord(BaseModel):
    token: str
    symbol: str
    name: str
    expiry: Optional[str] = None
    strike: Optional[float] = None
    lotsize: int
    instrumenttype: str
    exch_seg: str
    tick_size: float


class MorningBiasConfig(BaseModel):
    version: str = "2.1"
    timezone: str = "Asia/Kolkata"
    fail_closed: bool = True
    symbols: Dict[str, str]
    scrip_master: Dict[str, str | int]
    thresholds: Dict[str, float]
    outputs: Dict[str, str]
