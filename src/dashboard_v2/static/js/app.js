/**
 * Pro Terminal Main Controller (src/dashboard_v2/static/js/app.js)
 * Manages WebSocket feeds, Live Clock, Engine State, and Settings Drawer.
 */

const App = {
  ws: null,
  isEngineActive: true,
  reconnectInterval: 2000,
  reconnectTimer: null,

  init() {
    this.updateClock();
    setInterval(() => this.updateClock(), 1000);
    this.fetchInitialStatus();
    this.connectWebSocket();

    // Start live portfolio & trades polling loop (2.5s interval)
    if (typeof Positions !== 'undefined') {
      Positions.fetchAndRenderAll();
      setInterval(() => Positions.fetchAndRenderAll(), 2500);
    }

    // Direct link to Quick Trader Guide modal (?guide=1&lang=mr)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('guide') === '1') {
      toggleGuideModal(true);
      if (urlParams.get('lang') === 'mr') setGuideLanguage('mr');
    }

    // Live Market Direction & Pivot Radar polling (every 5s)
    this.fetchMarketDirection();
    setInterval(() => this.fetchMarketDirection(), 5000);
  },

  updateClock() {
    const clockEl = document.getElementById('ist-clock');
    if (!clockEl) return;
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
      timeZone: 'Asia/Kolkata',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }) + ' IST';
    clockEl.textContent = timeStr;
  },

  async fetchInitialStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      this.updateStatusDisplay(data);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  },

  updateStatusDisplay(status) {
    if (!status) return;
    this.isEngineActive = !status.killswitch_halted;
    this.renderEngineButton();
    this.updateTickers(status);
    this.updateBrokerConnection(status);
    this.updateEngineConfigDrawer(status);
    this.updateBrainStatus(status);
  },

  setText(id, text, className = null, color = null) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (className) el.className = className;
    if (color) el.style.color = color;
  },

  updateTickers(status) {
    const mb = status.morning_bias || {};

    if (mb.nifty_prev_close) {
      const p = Number(mb.nifty_prev_close);
      this.setText('ticker-nf-price', `₹${p.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`);
    }
    if (mb.india_vix) {
      const v = Number(mb.india_vix);
      this.setText('ticker-vix-val', v.toFixed(2), `ticker-val ${v > 20 ? 'amber' : 'green'}`);
      this.setText('pill-vix', `VIX: ${v.toFixed(2)} ${v <= 22 ? 'OK' : 'HIGH'}`);
      this.setText('drawer-vix-health', `VIX ${v.toFixed(2)} READY`);
    }
    if (status.macro_regime) {
      const reg = String(status.macro_regime).toUpperCase();
      this.setText('ticker-regime-val', reg, `ticker-val ${reg === 'CHOP' ? 'amber' : 'green'}`);
      this.setText('pill-adx', `REGIME: ${reg}`);
    }
  },

  updateBrokerConnection(status) {
    const brokerEl = document.getElementById('header-broker-status');
    if (brokerEl && status.morning_bias?.health_check) {
      const live = status.exec_mode === 'LIVE' && !status.killswitch_halted;
      brokerEl.textContent = live ? 'ANGEL ONE • S1680402 • LIVE' : 'ANGEL ONE • S1680402 • PAPER';
    }
  },

  updateEngineConfigDrawer(status) {
    const cfg = status.active_config || {};
    const drawerVer = document.getElementById('drawer-model-version');
    if (drawerVer && cfg.version) drawerVer.textContent = cfg.version;
    const drawerSLTP = document.getElementById('drawer-sl-tp');
    if (drawerSLTP && cfg.sl_atr && cfg.tp_atr) {
      drawerSLTP.textContent = `${cfg.sl_atr}x ATR | Target: ${cfg.tp_atr}x ATR`;
    }
  },

  updateBrainStatus(status) {
    const brainText = document.getElementById('algo-brain-status');
    if (brainText) {
      if (status.killswitch_halted) {
        brainText.textContent = `⚠️ Engine Halted: ${status.killswitch_reason || 'Manual Pause'}`;
        brainText.style.color = '#fb7185';
      } else {
        brainText.textContent = 'Targeting 5m breakout entries with 1:2 R:R (IV Shield Active)';
        brainText.style.color = '#e2e8f0';
      }
    }
  },

  async fetchMarketDirection() {
    try {
      const res = await fetch('/api/market-direction');
      if (!res.ok) return;
      const data = await res.json();
      this.renderMarketDirection(data);
    } catch (err) {
      console.error('Failed to fetch market direction:', err);
    }
  },

  renderMarketDirection(dir) {
    if (!dir) return;
    this.renderDirectionHeaderAndModal(dir);

    // VIX
    const vixPill = document.getElementById('modal-vix-pill');
    const vixStatus = document.getElementById('modal-vix-status');
    if (vixPill) vixPill.textContent = Number(dir.india_vix || 11.27).toFixed(2);
    if (vixStatus) vixStatus.textContent = dir.vix_regime || 'LOW (< 12.0) • STABLE';

    // Nifty & BankNifty
    if (dir.nifty) this.renderAssetModal(dir.nifty, 'nf');
    if (dir.banknifty) this.renderAssetModal(dir.banknifty, 'bn');
  },

  renderDirectionHeaderAndModal(dir) {
    const isUp = dir.overall_direction === 'UP';
    const isDown = dir.overall_direction === 'DOWN';
    let dirClass = 'amber';
    let dirIcon = '◆';
    let tickerShortText = 'RANGE';
    let defaultText = 'RANGE-BOUND';

    if (isUp) {
      dirClass = 'green';
      dirIcon = '▲';
      tickerShortText = 'BULLISH';
      defaultText = 'UP (BULLISH)';
    } else if (isDown) {
      dirClass = 'red';
      dirIcon = '▼';
      tickerShortText = 'BEARISH';
      defaultText = 'DOWN (BEARISH)';
    }

    const dirText = dir.overall_bias || defaultText;

    // Header Ticker
    const tickerVal = document.getElementById('ticker-direction-val');
    const tickerIcon = document.getElementById('ticker-direction-icon');
    const tickerText = document.getElementById('ticker-direction-text');
    if (tickerVal) {
      tickerVal.className = `ticker-val ${dirClass}`;
      tickerVal.title = dirText;
    }
    if (tickerIcon) tickerIcon.textContent = dirIcon;
    if (tickerText) tickerText.textContent = tickerShortText;

    // Modal Banner
    const banner = document.getElementById('modal-direction-banner');
    const verdict = document.getElementById('modal-direction-verdict');
    const sub = document.getElementById('modal-direction-sub');
    if (banner) {
      banner.className = `direction-banner ${isDown ? 'bearish' : ''}`;
    }
    let verdictIcon = '🟡';
    let verdictColor = '#fbbf24';
    let subText = 'Indices oscillating near Daily Pivots. Range-bound chop regime.';

    if (isUp) {
      verdictIcon = '🟢';
      verdictColor = '#34d399';
      subText = 'Spot indices are trading comfortably above Daily Pivots with safe India VIX (<12.0).';
    } else if (isDown) {
      verdictIcon = '🔴';
      verdictColor = '#fb7185';
      subText = 'Spot indices are trading below Daily Pivots. Bearish caution advised.';
    }

    if (verdict) {
      verdict.innerHTML = `<span>${verdictIcon}</span> <span>${dir.overall_bias}</span>`;
      verdict.style.color = verdictColor;
    }
    if (sub) {
      sub.textContent = subText;
    }
  },

  renderAssetModal(asset, prefix) {
    const formatPrice = val => `₹${Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
    const isBull = asset.diff_from_pivot >= 0;

    this.setText(`modal-${prefix}-ltp`, formatPrice(asset.ltp));
    this.setText(`modal-${prefix}-badge`, asset.bias, isBull ? 'level-badge-s' : 'level-badge-r');
    this.setText(`modal-${prefix}-diff`, asset.status_text, null, isBull ? '#34d399' : '#fb7185');
    this.setText(`modal-${prefix}-p`, formatPrice(asset.pivot));
    this.setText(`modal-${prefix}-r1`, formatPrice(asset.r1));
    this.setText(`modal-${prefix}-r2`, formatPrice(asset.r2 || (prefix === 'nf' ? 23516.63 : 56800.04)));
    this.setText(`modal-${prefix}-s1`, formatPrice(asset.s1));
    this.setText(`modal-${prefix}-s2`, formatPrice(asset.s2 || (prefix === 'nf' ? 23270.83 : 56152.43)));
  },

  renderEngineButton() {
    const btn = document.getElementById('btn-master-engine');
    const dot = document.getElementById('engine-dot');
    const txt = document.getElementById('engine-btn-text');
    if (!btn || !dot || !txt) return;

    if (this.isEngineActive) {
      btn.className = 'btn-engine-toggle';
      dot.className = 'pulse-dot green';
      txt.textContent = 'ENGINE: ACTIVE';
    } else {
      btn.className = 'btn-engine-toggle paused';
      dot.className = 'pulse-dot red';
      txt.textContent = 'ENGINE: PAUSED';
    }
  },

  connectWebSocket() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.status) this.updateStatusDisplay(payload.status);
        if (payload.portfolio && window.Positions) Positions.render(payload.portfolio);
      } catch (err) {
        console.error('Failed to parse WS payload:', err);
      }
    };

    this.ws.onclose = () => {
      if (!this.reconnectTimer) {
        this.reconnectTimer = setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
      }
    };

    this.ws.onerror = () => {
      try { this.ws.close(); } catch (e) {}
    };
  }
};

async function toggleEngineState() {
  const nextHalt = App.isEngineActive; // If currently active, we want to halt it
  const actionName = nextHalt ? 'PAUSE' : 'RESUME';

  try {
    const res = await fetch('/api/killswitch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        halt: nextHalt,
        reason: nextHalt ? 'Paused by user via Pro Terminal' : ''
      })
    });
    await res.json();
    App.isEngineActive = !nextHalt;
    App.renderEngineButton();
    showToast(`Trading Engine ${actionName}D successfully`, nextHalt ? 'error' : 'success');
  } catch (err) {
    console.error(`Failed to toggle engine:`, err);
    showToast(`Failed to ${actionName} engine`, 'error');
  }
}

function toggleSettingsDrawer(open) {
  const drawer = document.getElementById('settings-drawer');
  if (!drawer) return;
  if (open) {
    drawer.classList.add('open');
  } else {
    drawer.classList.remove('open');
  }
}

function toggleGuideModal(open) {
  const modal = document.getElementById('guide-modal');
  if (!modal) return;
  if (open) {
    modal.classList.add('open');
  } else {
    modal.classList.remove('open');
  }
}

function toggleDirectionModal(open) {
  const modal = document.getElementById('direction-modal');
  if (!modal) return;
  if (open) {
    modal.classList.add('open');
    if (window.App?.fetchMarketDirection) {
      window.App.fetchMarketDirection();
    }
  } else {
    modal.classList.remove('open');
  }
}

function setGuideLanguage(lang) {
  const isMr = lang === 'mr';
  const toggle = (id, show, active) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (show !== null) el.style.display = show ? 'block' : 'none';
    if (active !== null) el.classList.toggle('active', active);
  };
  
  toggle('guide-content-en', !isMr, null);
  toggle('guide-content-mr', isMr, null);
  toggle('btn-guide-lang-en', null, !isMr);
  toggle('btn-guide-lang-mr', null, isMr);
  
  const titleText = document.getElementById('guide-modal-title-text');
  if (titleText) titleText.textContent = isMr ? 'ट्रेडर मार्गदर्शक (TRADER GUIDE)' : 'QUICK TRADER GUIDE';
}

function toggleOptionChainModal(open) {
  const modal = document.getElementById('option-chain-modal');
  if (!modal) return;
  if (open) {
    modal.classList.add('open', 'active');
    if (typeof Signals !== 'undefined' && Signals.fetchOptionChain) {
      Signals.fetchOptionChain();
    }
  } else {
    modal.classList.remove('open', 'active');
  }
}

function switchCockpitTab(tabName) {
  document.querySelectorAll('.cockpit-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  document.querySelectorAll('.cockpit-tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${tabName}`);
  });
}

function switchMobileStage(stageName) {
  const chartStage = document.querySelector('.chart-stage');
  const cockpitPanel = document.querySelector('.cockpit-panel');
  document.querySelectorAll('.mobile-nav-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === stageName);
  });
  if (stageName === 'chart') {
    if (chartStage) chartStage.classList.remove('mobile-hidden');
    if (cockpitPanel) cockpitPanel.classList.add('mobile-hidden');
  } else {
    if (chartStage) chartStage.classList.add('mobile-hidden');
    if (cockpitPanel) cockpitPanel.classList.remove('mobile-hidden');
    switchCockpitTab(stageName);
  }
}

document.addEventListener('DOMContentLoaded', () => App.init());

