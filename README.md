# Algo-Trader

Zero-cost, self-healing Python algorithmic trading application and institutional-grade dashboard for the Indian stock market.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev,windows]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your API credentials

# 3. Launch System
./startup.sh all
```

# 4. Run the trading engine
python -m src.main

# 5. Launch the dashboard (separate terminal)
./startup.sh dashboard
#   Dashboard URL: http://localhost:8080

## Architecture

| Layer | Module | Purpose |
|-------|--------|---------|
| Data | `src/data/` | WebSocket ingestion, tick validation, candle aggregation |
| Strategy | `src/strategy/` | VWAP, confluence engine, VIX regime, options strategy |
| Execution | `src/execution/` | Order state machine, position sizing, trailing stops |
| Risk | `src/risk/` | Circuit breakers, kill switch, heartbeat monitor |
| Dashboard | `src/dashboard_v2/` | FastAPI + WebSocket Bloomberg-style terminal |
| Auth | `src/auth/` | Angel One / Fyers broker login & token lifecycle |
| WFO | `src/wfo/` | Walk-forward optimization & regime calibration |

## Configuration

All toggles live in `config/settings.yaml` and are hot-reloadable at runtime.

## License

MIT
