"""
Main Orchestrator — Async entry point that boots all subsystems.

Boot sequence:
 1. Load config (settings_loader)
 2. Start hot-reload watcher
 3. Authenticate (token_manager)
 4. Download instrument master
 5. Run morning bias engine
 6. Connect WebSocket
 7. Start heartbeat monitor
 8. Start candle aggregator
 9. Start background scanner
10. Start trailing stop manager
11. Schedule auto square-off
12. Enter main event loop

Platform-aware event loop: winloop on Windows, uvloop on Linux.
Graceful shutdown: cancels orders, closes WebSocket, flushes audit log.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


# ── Platform-aware event loop ─────────────────────────────────────────
def _install_fast_event_loop() -> None:
    """Install the fastest available event loop for the platform."""
    if sys.platform == "win32":
        try:
            import winloop
            winloop.install()
            logging.info("Using winloop event loop (Windows).")
            return
        except ImportError:
            pass
    else:
        try:
            import uvloop
            uvloop.install()
            logging.info("Using uvloop event loop (Linux/macOS).")
            return
        except ImportError:
            pass
    logging.info("Using default asyncio event loop.")


# Install before any other asyncio usage
_install_fast_event_loop()

from src.alerts.telegram import AlertType, TelegramAlerts
from src.audit.audit_logger import AuditLogger
from src.auth.token_manager import TokenManager
from src.broker.angel_client import AngelClientFactory
from src.broker.instrument_master import InstrumentMaster
from src.broker.order_api import OrderAPI
from src.broker.portfolio_api import PortfolioAPI
from src.config.hot_reload import HotReloadWatcher
from src.config.settings_loader import Settings, load_settings
from src.data.candle_aggregator import CandleAggregator
from src.data.market_depth import MarketDepthParser
from src.data.protobuf_decoder import DecodedTick
from src.data.tick_validator import TickValidator
from src.data.websocket_manager import WebSocketManager
from src.execution.auto_squareoff import AutoSquareOff
from src.execution.execution_modes import ExecutionMode, ExecutionRouter
from src.execution.trailing_stop import TrailingStopManager
from src.risk.capital_allocator import CapitalAllocator
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.financial_killswitch import FinancialKillSwitch
from src.risk.heartbeat_monitor import HeartbeatMonitor
from src.scanner.background_scanner import BackgroundScanner
from src.state.db import Database
from src.state.state_manager import StateManager
from src.strategy.confluence_engine import ConfluenceEngine, ConfluenceMode, Direction
from src.strategy.regime_detector import RegimeDetector
from src.strategy.signal_confidence import SignalConfidence
from src.strategy.vwap import VWAPEngine

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class LiveValidationError(RuntimeError):
    """Raised when a condition required for real-money trading is not met."""


MIN_LIVE_ALPHA_SIGNIFICANT = True
MIN_LIVE_N_SESSIONS = 20
MIN_LIVE_OOS_SHARPE = 1.5


def validate_live_readiness(mode: str, client_factory: Any, strategy_config: str = "config/optimized_strategy_config.yaml") -> None:
    """
    Raise LiveValidationError if real-money trading must be refused.

    Guards (all must pass when mode == 'auto'):
      1. LIVE_CONFIRM=yes env flag
      2. Real broker SDK client (never the mock fallback)
      3. Deployed strategy config carries statistically-significant OOS evidence
         (alpha_significant, n_sessions >= 20, out_of_sample_sharpe >= 1.5)
    """
    if ExecutionMode(mode) is not ExecutionMode.AUTO:
        return

    live_confirm = os.getenv("LIVE_CONFIRM", "").strip().lower()
    if live_confirm != "yes":
        raise LiveValidationError(
            "❌ LIVE MODE REFUSED: execution.mode == 'auto' requires LIVE_CONFIRM=yes "
            "in the environment. No real orders were placed."
        )
    if getattr(client_factory, "is_mock", False):
        raise LiveValidationError(
            "❌ LIVE MODE REFUSED: broker client is the MOCK fallback (SDK/auth failure). "
            "Refusing to transmit real orders against a simulated client."
        )
    try:
        import yaml as _yaml
        from pathlib import Path as _P

        cfg_path = _P(strategy_config)
        perf = {}
        if cfg_path.exists():
            _cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            perf = _cfg.get("backtested_performance", {}) or {}
        alpha_significant = bool(perf.get("alpha_significant", False))
        n_sessions = int(perf.get("n_sessions", 0) or 0)
        oos_sharpe = float(perf.get("out_of_sample_sharpe", 0.0) or 0.0)
        if alpha_significant != MIN_LIVE_ALPHA_SIGNIFICANT or n_sessions < MIN_LIVE_N_SESSIONS or oos_sharpe < MIN_LIVE_OOS_SHARPE:
            raise LiveValidationError(
                "❌ LIVE MODE REFUSED: strategy not validated for live money.\n"
                f"    alpha_significant={alpha_significant} (need True)\n"
                f"    n_sessions={n_sessions} (need >= {MIN_LIVE_N_SESSIONS})\n"
                f"    out_of_sample_sharpe={oos_sharpe:.2f} (need >= {MIN_LIVE_OOS_SHARPE})\n"
                "    Complete Phase 4 validation (backtest + WFO) and promote the approved config first."
            )
    except LiveValidationError:
        raise
    except Exception as exc:
        raise LiveValidationError(f"❌ LIVE MODE REFUSED: could not validate strategy config: {exc}")


class TradingEngine:
    """
    Main trading engine that orchestrates all subsystems.

    Lifecycle: init → boot → run → shutdown
    """

    def __init__(self) -> None:
        load_dotenv()
        self._settings: Optional[Settings] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # ── Subsystems (initialized during boot) ──────────────────────
        self._audit: Optional[AuditLogger] = None
        self._db: Optional[Database] = None
        self._token_mgr: Optional[TokenManager] = None
        self._client_factory: Optional[Any] = None
        self._order_api: Optional[OrderAPI] = None
        self._portfolio_api: Optional[PortfolioAPI] = None
        self._instruments: Optional[InstrumentMaster] = None
        self._ws_manager: Optional[WebSocketManager] = None
        self._tick_validator: Optional[TickValidator] = None
        self._candle_agg: Optional[CandleAggregator] = None
        self._depth_parser: Optional[MarketDepthParser] = None
        self._vwap: Optional[VWAPEngine] = None
        self._regime: Optional[RegimeDetector] = None
        self._confluence: Optional[ConfluenceEngine] = None
        self._confidence: Optional[SignalConfidence] = None
        self._exec_router: Optional[ExecutionRouter] = None
        self._trailing_mgr: Optional[TrailingStopManager] = None
        self._squareoff: Optional[AutoSquareOff] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._killswitch: Optional[FinancialKillSwitch] = None
        self._heartbeat: Optional[HeartbeatMonitor] = None
        self._allocator: Optional[CapitalAllocator] = None
        self._scanner: Optional[BackgroundScanner] = None
        self._telegram: Optional[TelegramAlerts] = None
        self._hot_reload: Optional[HotReloadWatcher] = None
        self._oi_task: Optional[asyncio.Task] = None

    async def boot(self) -> None:
        """Initialize all subsystems in dependency order."""
        logger.info("═" * 60)
        logger.info("  Algo-Trader — Booting Trading Engine")
        logger.info("═" * 60)

        # ── 1. Load config ────────────────────────────────────────────
        self._settings = load_settings()
        s = self._settings
        logger.info("✅ Configuration loaded.")

        # ── 2. Audit logger ───────────────────────────────────────────
        self._audit = AuditLogger(log_dir=s.audit.log_dir)
        self._audit.log_event("ENGINE_BOOT", source="main", payload={"version": "0.1.0"})
        logger.info("✅ Audit logger initialized.")

        # ── 3. Database ───────────────────────────────────────────────
        self._db = Database()
        await self._db.connect()
        logger.info("✅ Database connected.")

        # ── 4. Authentication ─────────────────────────────────────────
        primary_broker = getattr(s.data_feeds, "primary", "angelone").lower()
        self._token_mgr = TokenManager(broker=primary_broker)
        token = self._token_mgr.load()
        if token:
            logger.info("✅ Authenticated with %s.", primary_broker.upper())
        else:
            logger.error(
                "❌ Authentication failed for %s — check credentials in .env. "
                "Live order routing and market feed will be unavailable.",
                primary_broker.upper(),
            )

        # ── 5. Broker client factory (Angel One Exclusive) ────────────
        self._client_factory = AngelClientFactory(self._token_mgr)
        self._order_api = OrderAPI(self._client_factory.client, self._audit)
        self._portfolio_api = PortfolioAPI(self._client_factory.client)
        logger.info("✅ Broker APIs initialized (ANGELONE).")

        # ── 5b. LIVE SAFETY GATE (Phase 3+4 hardening) ─────────────────
        # Real-money protection: execution.mode == 'auto' demands an explicit
        # LIVE_CONFIRM=yes flag, a genuine broker SDK client (never the mock
        # fallback), and validated OOS strategy evidence (alpha_significant,
        # n_sessions >= 20, out_of_sample_sharpe >= 1.5).
        try:
            validate_live_readiness(s.execution.mode, self._client_factory)
        except LiveValidationError as exc:
            raise RuntimeError(str(exc)) from None
        if ExecutionMode(s.execution.mode) is ExecutionMode.AUTO:
            logger.warning("🚨 LIVE MODE CONFIRMED — real orders WILL be transmitted to %s.", primary_broker.upper())

        # ── 6. Instrument master ──────────────────────────────────────
        self._instruments = InstrumentMaster()
        cached = self._instruments.load_from_cache()
        if cached is None or cached.height == 0:
            now = datetime.now(IST)
            if now.hour >= 9:
                await self._instruments.download()
            else:
                logger.warning("Pre-market: instrument download deferred to 9:00 AM.")
        logger.info("✅ Instrument master ready.")

        # ── 7. Telegram alerts ────────────────────────────────────────
        self._telegram = TelegramAlerts(
            enabled=s.telegram.enabled,
            send_entry_signals=s.telegram.send_entry_signals,
            send_exit_fills=s.telegram.send_exit_fills,
            send_system_health=s.telegram.send_system_health,
            send_morning_bias=s.telegram.send_morning_bias,
        )
        logger.info("✅ Telegram alerts: %s", "enabled" if self._telegram.is_enabled else "disabled")

        # ── 8. Strategy engines ───────────────────────────────────────
        self._vwap = VWAPEngine()
        self._regime = RegimeDetector(
            high_threshold=s.vix.high_threshold,
            low_threshold=s.vix.low_threshold,
        )
        self._candle_agg = CandleAggregator()
        self._depth_parser = MarketDepthParser()
        self._tick_validator = TickValidator()

        self._confluence = ConfluenceEngine(
            vwap_engine=self._vwap,
            candle_aggregator=self._candle_agg,
            mode=ConfluenceMode(s.confluence.mode),
            volume_surge_multiplier=s.confluence.volume_surge_multiplier,
            volume_lookback=s.confluence.volume_lookback_periods,
        )
        self._confidence = SignalConfidence(self._regime)
        logger.info("✅ Strategy engines initialized.")

        # ── 9. Execution engine ───────────────────────────────────────
        self._exec_router = ExecutionRouter(mode=ExecutionMode(s.execution.mode))
        self._trailing_mgr = TrailingStopManager()
        self._squareoff = AutoSquareOff(
            squareoff_time=s.squareoff.time,
            warn_minutes_before=s.squareoff.warn_minutes_before,
            audit=self._audit,
        )
        self._squareoff.set_warning_callback(self._squareoff_warn)
        self._squareoff.set_squareoff_callback(self._execute_squareoff)

        # Get initial capital
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "100000"))
        try:
            funds = await self._portfolio_api.get_funds()
            initial_capital = funds.total_balance or initial_capital
        except Exception as exc:
            logger.debug("Failed to query live funds, using configured capital (₹%.0f): %s", initial_capital, exc)

        self._allocator = CapitalAllocator(
            total_capital=initial_capital,
            max_risk_per_trade=s.risk.max_risk_per_trade,
            allocation_mode=s.risk.allocation_mode,
            manual_lot_size=s.risk.manual_lot_size,
        )
        logger.info("✅ Execution engine initialized (capital: ₹%.0f).", initial_capital)

        # ── 10. Risk management ───────────────────────────────────────
        self._circuit_breaker = CircuitBreaker(
            total_capital=initial_capital,
            daily_drawdown_limit=s.risk.daily_drawdown_limit,
        )
        self._killswitch = FinancialKillSwitch(
            backtest_sharpe=s.killswitch.backtest_sharpe,
            backtest_slippage_bps=s.killswitch.backtest_avg_slippage_bps,
            sharpe_degradation_pct=s.killswitch.sharpe_degradation_pct,
            slippage_multiplier=s.killswitch.slippage_multiplier,
            enabled=s.killswitch.enabled,
        )
        self._heartbeat = HeartbeatMonitor(
            access_token=token or "",
            interval_seconds=s.heartbeat.interval_seconds,
            max_failures=s.heartbeat.max_failures,
        )

        # Wire heartbeat disconnect → cancel all orders
        async def on_disconnect():
            if self._order_api:
                await self._order_api.cancel_all_orders()
            if self._telegram:
                await self._telegram.heartbeat_alert(connected=False)
            if self._audit:
                self._audit.log_event("HEARTBEAT_DISCONNECT", source="heartbeat")

        async def on_reconnect():
            if self._telegram:
                await self._telegram.heartbeat_alert(connected=True)
            if self._audit:
                self._audit.log_event("HEARTBEAT_RECONNECT", source="heartbeat")

        self._heartbeat.set_on_disconnect(on_disconnect)
        self._heartbeat.set_on_reconnect(on_reconnect)
        logger.info("✅ Risk management initialized.")

        # ── 11. Background scanner ────────────────────────────────────
        self._scanner = BackgroundScanner(
            confluence_engine=self._confluence,
            candle_aggregator=self._candle_agg,
            telegram=self._telegram,
        )
        logger.info("✅ Background scanner initialized.")

        # ── 12. State recovery ────────────────────────────────────────
        state_mgr = StateManager(
            self._portfolio_api, self._trailing_mgr, self._db
        )
        recovery = await state_mgr.recover()
        logger.info("✅ State recovered: %s", recovery)

        # ── 13. WebSocket ─────────────────────────────────────────────
        # NOTE: WebSocketManager implements the Upstox V3 feed protocol only.
        # It must only be started with a genuine Upstox access token — passing
        # an Angel One / Fyers JWT to the Upstox authorize endpoint can never
        # succeed and previously masked the failure with a fake "skipped" path.
        if primary_broker == "upstox" and token:
            self._ws_manager = WebSocketManager(access_token=token)

            # Register tick processing pipeline
            async def process_tick(tick: DecodedTick):
                # Validate
                if not self._tick_validator or not self._tick_validator.validate(tick):
                    return
                # Update VWAP
                if self._vwap:
                    self._vwap.on_tick(tick)
                # Aggregate candles
                if self._candle_agg:
                    await self._candle_agg.on_tick(tick)
                # Parse depth
                if self._depth_parser:
                    await self._depth_parser.on_tick(tick)
                # Update trailing stops
                if self._trailing_mgr:
                    await self._trailing_mgr.on_price_update(tick.instrument_token, tick.ltp)

            self._ws_manager.register_handler(process_tick)

            # Subscribe to watchlist instruments
            import yaml
            watchlist_path = "config/watchlist.yaml"
            try:
                with open(watchlist_path, encoding="utf-8") as f:
                    wl = yaml.safe_load(f) or {}
                tokens = []
                for idx in wl.get("indices", []):
                    tokens.append(idx["instrument_token"])
                for eq in wl.get("equities", []):
                    tokens.append(eq["instrument_token"])
                self._ws_manager.subscribe(tokens)
            except Exception as exc:
                logger.warning("Failed to load watchlist: %s", exc)
        else:
            self._ws_manager = None
            logger.warning(
                "Market data feed disabled: WebSocketManager supports Upstox only "
                "(primary broker: %s). No live ticks — strategy engines will idle "
                "until an Angel SmartWebSocket feed is implemented or an Upstox "
                "token is configured.",
                primary_broker,
            )
            if self._telegram and self._telegram.is_enabled:
                try:
                    await self._telegram.send(
                        AlertType.SYSTEM_INFO,
                        f"⚠️ Market data feed disabled: WebSocketManager supports Upstox only "
                        f"(primary broker: {primary_broker}). Live ticks disabled — strategy engines idling.",
                    )
                except Exception as exc:
                    logger.debug("Failed to send telegram alert for disabled WebSocket: %s", exc)

        logger.info("✅ WebSocket configured.")

        # ── 14. Hot-reload watcher ────────────────────────────────────
        def on_config_reload(new_settings: Settings):
            """Propagate config changes to all subsystems."""
            self._settings = new_settings
            if self._regime:
                self._regime.update_thresholds(
                    new_settings.vix.high_threshold,
                    new_settings.vix.low_threshold,
                )
            if self._confluence:
                self._confluence.set_mode(ConfluenceMode(new_settings.confluence.mode))
            if self._exec_router:
                self._exec_router.set_mode(ExecutionMode(new_settings.execution.mode))
            if self._allocator:
                self._allocator.update_config(
                    mode=new_settings.risk.allocation_mode,
                    manual_lot=new_settings.risk.manual_lot_size,
                    max_risk=new_settings.risk.max_risk_per_trade,
                )
            if self._audit:
                self._audit.log_event(
                    "CONFIG_RELOADED", source="hot_reload",
                    payload={"execution_mode": new_settings.execution.mode},
                )

        self._hot_reload = HotReloadWatcher(on_reload=on_config_reload)
        self._hot_reload.start()
        logger.info("✅ Hot-reload watcher started.")

        # Register candle callbacks
        await self._scanner.start()

        # Start OI / index-trend feed refresh (confluence Checks 3 & 4)
        self._oi_task = asyncio.create_task(self._oi_refresh_loop(), name="oi-refresh")

        logger.info("═" * 60)
        logger.info("  ✅ All subsystems booted successfully!")
        logger.info("═" * 60)

    async def _oi_refresh_loop(self, refresh_seconds: int = 120) -> None:
        """
        Periodically refresh PCR / OI-change for index tokens and the
        broader index trend so confluence Checks 3 (OI Alignment) and
        4 (Index Trend) operate on real data instead of never-populated
        defaults.
        """
        from src.data.index_fo_data import fetch_live_option_chain

        chain_symbol_for_token = {
            "NSE_INDEX|Nifty Bank": "BANKNIFTY",
            "NSE_INDEX|Nifty 50": "NIFTY",
        }
        while True:
            try:
                for token, chain_symbol in chain_symbol_for_token.items():
                    try:
                        chain = fetch_live_option_chain(chain_symbol)
                        if not isinstance(chain, dict) or chain.get("status") != "SUCCESS":
                            continue
                        pcr = float(chain.get("pcr", 0.0) or 0.0)
                        atm_oi_chg = 0.0
                        for r in chain.get("strikes", []):
                            if r.get("is_atm"):
                                ce = r.get("ce_oi_chg_pct", 0.0) or 0.0
                                pe = r.get("pe_oi_chg_pct", 0.0) or 0.0
                                atm_oi_chg = float(pe) - float(ce)
                                break
                        if pcr > 0 and self._confluence:
                            self._confluence.update_pcr(token, pcr)
                            self._confluence.update_oi_change(token, atm_oi_chg)
                    except Exception as exc:
                        logger.debug("OI refresh failed for %s: %s", chain_symbol, exc)

                # Index trend from BANKNIFTY PCR signal (primary trading index)
                try:
                    chain = fetch_live_option_chain("BANKNIFTY")
                    sig = str(chain.get("pcr_signal", "")).upper() if isinstance(chain, dict) else ""
                    if self._confluence:
                        if sig == "BULLISH":
                            self._confluence.update_index_trend(Direction.LONG)
                        elif sig == "BEARISH":
                            self._confluence.update_index_trend(Direction.SHORT)
                except Exception as exc:
                    logger.debug("Index trend refresh failed: %s", exc)

                await asyncio.sleep(refresh_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("OI refresh loop error: %s — retrying.", exc)
                await asyncio.sleep(refresh_seconds)

    async def run(self) -> None:
        """Enter the main event loop."""
        self._running = True

        # ── Launch concurrent tasks ───────────────────────────────────
        tasks = []
        if self._heartbeat is not None:
            tasks.append(asyncio.create_task(self._heartbeat.start(), name="heartbeat"))
        if self._ws_manager is not None:
            tasks.append(asyncio.create_task(self._ws_manager.connect(), name="websocket"))
        tasks.append(asyncio.create_task(self._monitoring_loop(), name="monitoring"))

        logger.info("Trading engine is LIVE. Press Ctrl+C to shutdown.")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _squareoff_warn(self) -> None:
        """Warn handler for pre-squareoff notification."""
        if self._squareoff:
            sq_time = self._squareoff.squareoff_time.strftime("%H:%M")
            logger.warning("Square-off warning: positions will be closed at %s IST.", sq_time)

    async def _execute_squareoff(self) -> None:
        """Close all open intraday positions at square-off time."""
        if self._portfolio_api is None or self._order_api is None:
            return
        try:
            positions = await self._portfolio_api.get_positions()
            open_positions = [p for p in positions if p.net_qty != 0]
            if not open_positions:
                logger.info("Auto square-off: no open positions to close.")
                return
            for pos in open_positions:
                try:
                    qty = abs(pos.net_qty)
                    if qty == 0:
                        continue
                    side = "SELL" if pos.net_qty > 0 else "BUY"
                    from src.broker.order_api import (
                        OrderRequest,
                        OrderType,
                        ProductType,
                        TransactionType,
                    )
                    req = OrderRequest(
                        instrument_token=pos.symbol,
                        transaction_type=TransactionType(side),
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        product=ProductType.INTRADAY,
                        symboltoken=pos.instrument_token or (pos.raw.get("symboltoken", "") if isinstance(pos.raw, dict) else ""),
                        exchange=pos.exchange or "NSE_FO",
                    )
                    result = await self._order_api.place_order(req)
                    logger.info(
                        "Square-off order placed for %s: %s",
                        pos.symbol, result,
                    )
                except Exception as exc:
                    logger.error("Failed to square off %s: %s", pos.symbol, exc)
        except Exception as exc:
            logger.error("Auto square-off failed: %s", exc)

    async def _monitoring_loop(self) -> None:
        """
        Periodic monitoring: P&L checks, squareoff, circuit breaker.

        Runs every 1 second.
        """
        while self._running:
            try:
                # Check auto square-off
                if self._squareoff:
                    await self._squareoff.check()

                # Update P&L and circuit breaker (every 30 seconds)
                # This is done less frequently to reduce API calls
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Monitoring loop error: %s", exc)
                await asyncio.sleep(5)

    async def shutdown(self) -> None:
        """Graceful shutdown sequence."""
        logger.info("═" * 60)
        logger.info("  Shutting down trading engine...")
        logger.info("═" * 60)

        self._running = False
        self._shutdown_event.set()

        # 1. Cancel all open orders
        if self._order_api:
            try:
                await self._order_api.cancel_all_orders()
                logger.info("✅ All orders cancelled.")
            except Exception as exc:
                logger.error("Failed to cancel orders: %s", exc)

        # 2. Disconnect WebSocket
        if self._ws_manager:
            await self._ws_manager.disconnect()
            logger.info("✅ WebSocket disconnected.")

        # 3. Stop heartbeat
        if self._heartbeat:
            await self._heartbeat.stop()

        # 4. Stop scanner
        if self._scanner:
            await self._scanner.stop()

        # 4b. Stop OI/index-trend refresh loop
        if self._oi_task:
            self._oi_task.cancel()
            try:
                await self._oi_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ OI refresh loop stopped.")

        # 5. Stop hot-reload
        if self._hot_reload:
            self._hot_reload.stop()

        # 6. Close database
        if self._db:
            await self._db.close()
            logger.info("✅ Database closed.")

        # 7. Final audit entry
        if self._audit:
            self._audit.log_event("ENGINE_SHUTDOWN", source="main")
            logger.info("✅ Audit log flushed.")

        logger.info("═" * 60)
        logger.info("  Trading engine stopped.")
        logger.info("═" * 60)


async def async_main() -> None:
    """Async entry point."""
    engine = TradingEngine()

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal...")
        asyncio.create_task(engine.shutdown())

    # Windows doesn't support add_signal_handler, use try/except
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Windows fallback
        pass

    try:
        await engine.boot()
        await engine.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
    finally:
        await engine.shutdown()


def main() -> None:
    """Synchronous entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
