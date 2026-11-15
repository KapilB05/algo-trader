"""
Settings Loader — YAML config with Pydantic validation.

Loads config/settings.yaml and validates against a typed Pydantic schema.
Exposes a global Settings singleton with typed access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path("config/settings.yaml")


# ── Pydantic Schema ──────────────────────────────────────────────────

class ExecutionConfig(BaseModel):
    mode: str = "semi"
    min_rr_ratio: float = 2.0

class ConfluenceConfig(BaseModel):
    mode: str = "strict"
    volume_surge_multiplier: float = 1.5
    volume_lookback_periods: int = 20
    require_index_trend: bool = True

class RiskConfig(BaseModel):
    max_risk_per_trade: float = 0.02
    daily_drawdown_limit: float = 0.05
    allocation_mode: str = "auto"
    manual_lot_size: int = 1

class VixConfig(BaseModel):
    high_threshold: float = 14.0
    low_threshold: float = 12.0

class OptionsConfig(BaseModel):
    strategy_selection: str = "auto"

class KillSwitchConfig(BaseModel):
    enabled: bool = True
    sharpe_degradation_pct: float = 0.50
    slippage_multiplier: float = 2.0
    backtest_sharpe: float = 1.5
    backtest_avg_slippage_bps: float = 5.0

class HeartbeatConfig(BaseModel):
    interval_seconds: float = 5.0
    max_failures: int = 3

class TrailingStopConfig(BaseModel):
    mode: str = "atr"
    atr_multiplier: float = 1.5
    fixed_points: float = 10.0

class SquareoffConfig(BaseModel):
    enabled: bool = True
    time: str = "15:15"
    warn_minutes_before: int = 5

class ScannerConfig(BaseModel):
    enabled: bool = True
    refresh_ms: int = 500

class MorningBiasConfig(BaseModel):
    enabled: bool = True
    run_time: str = "09:00"

class TelegramConfig(BaseModel):
    enabled: bool = False
    send_entry_signals: bool = True
    send_exit_fills: bool = True
    send_system_health: bool = True
    send_morning_bias: bool = True

class DataFeedsConfig(BaseModel):
    primary: str = "angelone"
    secondary: str = "dhan"
    tertiary: str = "yfinance"
    auto_failover: bool = True

class WfoConfig(BaseModel):
    enabled: bool = True
    run_day: int = 6
    in_sample_weeks: int = 12
    out_of_sample_weeks: int = 4

class AuditConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "logs/audit"
    retention_days: int = 90

class DatabaseConfig(BaseModel):
    path: str = "data/trading.db"


class Settings(BaseModel):
    """Master settings schema — validated from settings.yaml."""

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    confluence: ConfluenceConfig = Field(default_factory=ConfluenceConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    vix: VixConfig = Field(default_factory=VixConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)
    killswitch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    squareoff: SquareoffConfig = Field(default_factory=SquareoffConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    morning_bias: MorningBiasConfig = Field(default_factory=MorningBiasConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    data_feeds: DataFeedsConfig = Field(default_factory=DataFeedsConfig)
    wfo: WfoConfig = Field(default_factory=WfoConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


# ── Singleton ─────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def load_settings(path: Optional[Path] = None) -> Settings:
    """Load and validate settings from YAML file."""
    global _settings
    config_path = path or SETTINGS_FILE

    if not config_path.exists():
        logger.warning("Settings file not found: %s — using defaults.", config_path)
        _settings = Settings()
        return _settings

    try:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        _settings = Settings(**raw)
        logger.info("Settings loaded from %s", config_path)
        return _settings

    except Exception as exc:
        logger.error("Failed to load settings: %s — using defaults.", exc)
        _settings = Settings()
        return _settings


def get_settings() -> Settings:
    """Get the current settings singleton (loads from disk on first call)."""
    global _settings
    if _settings is None:
        return load_settings()
    return _settings


def reload_settings(path: Optional[Path] = None) -> Settings:
    """Force reload settings from disk."""
    return load_settings(path)
