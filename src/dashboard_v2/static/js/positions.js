/**
 * Positions Component — Pro Terminal Cockpit Manager
 * Renders Hero P&L, Active Position Card, Price Lines, and Today's Trades Log.
 */

const Positions = {
  currentPositions: [],
  futuresPositions: [],
  optionsPositions: [],
  currentFilter: 'ALL',
  knownPositionIds: new Set(),
  notificationsInitialized: false,
  notificationsEnabled: true,
  audioEnabled: true,

  isOption(sym) {
    if (!sym) return false;
    const s = String(sym).toUpperCase().trim();
    return s.endsWith('CE') || s.endsWith('PE') || s.includes(' CE') || s.includes(' PE') || s.includes('CALL') || s.includes('PUT');
  },

  setFilter(filter) {
    this.currentFilter = filter;
    ['all', 'fut', 'opt'].forEach(k => {
      const btn = document.getElementById(`pos-filter-${k}`);
      if (btn) {
        btn.classList.remove('active');
        btn.removeAttribute('style');
      }
    });

    let activeBtnId = 'pos-filter-all';
    if (filter === 'FUTURES') {
      activeBtnId = 'pos-filter-fut';
    } else if (filter === 'OPTIONS') {
      activeBtnId = 'pos-filter-opt';
    }
    const activeBtn = document.getElementById(activeBtnId);
    if (activeBtn) activeBtn.classList.add('active');

    this.renderActivePosition(this.currentPositions);
  },

  async fetchAndRenderAll() {
    try {
      const [portRes, posRes, histRes] = await Promise.all([
        fetch('/api/paper/portfolio'),
        fetch('/api/paper/positions'),
        fetch('/api/paper/history')
      ]);

      let portData = {};
      if (portRes.ok) {
        const portJson = await portRes.json();
        portData = portJson.portfolio || portJson;
      }

      let openPositions = [];
      if (posRes.ok) {
        const posJson = await posRes.json();
        openPositions = posJson.positions || [];
      }

      this.render(portData, openPositions);

      if (histRes.ok) {
        const histJson = await histRes.json();
        const historyData = Array.isArray(histJson) ? histJson : (histJson.history || []);
        this.renderTradesLog(historyData);
      }
    } catch (err) {
      console.error('Failed to fetch positions/portfolio:', err);
    }
  },

  updateFilterCounts(total, futCount, optCount) {
    this.setText('count-all-pos', total);
    this.setText('count-futures-pos', futCount);
    this.setText('count-options-pos', optCount);
  },

  updatePnlSummaryPill(elementId, pnl) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const sign = pnl >= 0 ? '+' : '-';
    el.textContent = `${sign}₹${Math.abs(pnl).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
    el.className = `pill-val ${pnl >= 0 ? 'text-green' : 'text-red'}`;
  },

  updateSegregatedPnls() {
    const getPnl = p => Number(p.pnl_rupees ?? p.pnl ?? p.unrealized_pnl ?? 0);
    const futPnl = this.futuresPositions.reduce((sum, p) => sum + getPnl(p), 0);
    const optPnl = this.optionsPositions.reduce((sum, p) => sum + getPnl(p), 0);
    this.updatePnlSummaryPill('pos-summary-fut-pnl', futPnl);
    this.updatePnlSummaryPill('pos-summary-opt-pnl', optPnl);
  },

  render(data, openPosParam = null) {
    if (!data) return;

    const openPos = openPosParam !== null ? openPosParam : (data.open_positions || this.currentPositions || []);
    this.currentPositions = openPos;

    this.futuresPositions = openPos.filter(p => p.instrument_type === 'FUTURES' || (!p.instrument_type && !this.isOption(p.symbol)));
    this.optionsPositions = openPos.filter(p => p.instrument_type === 'OPTIONS' || (!p.instrument_type && this.isOption(p.symbol)));

    this.updateFilterCounts(openPos.length, this.futuresPositions.length, this.optionsPositions.length);
    this.updateSegregatedPnls();
    this.checkNewPositions(openPos);

    this.renderHeroCards(data, openPos);
    this.renderActivePosition(openPos);

    if (typeof ChartEngine !== 'undefined' && ChartEngine.candlestickSeries) {
      ChartEngine.updateActivePositionPriceLines(openPos);
    }
  },

  setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  },

  setHeroPnl(pnlVal) {
    const pnlEl = document.getElementById('hero-net-pnl');
    const pnlCard = document.getElementById('hero-pnl-card');
    const sign = pnlVal >= 0 ? '+' : '-';
    
    if (pnlEl) {
      pnlEl.textContent = `${sign}₹${Math.abs(pnlVal).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      pnlEl.className = `hero-amount ${pnlVal >= 0 ? '' : 'loss'}`;
    }
    if (pnlCard) {
      pnlCard.className = `hero-pnl-card ${pnlVal >= 0 ? '' : 'loss'}`;
    }
  },

  setHeroPill(pnlVal) {
    const heroPill = document.getElementById('hero-status-pill');
    if (!heroPill) return;

    if (pnlVal >= 0) {
      heroPill.textContent = '🟢 PROFITABLE SESSION';
      heroPill.style.background = 'rgba(16, 185, 129, 0.15)';
      heroPill.style.color = '#34d399';
      heroPill.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else {
      heroPill.textContent = '🛡️ CAPITAL SAFEGUARDED';
      heroPill.style.background = 'rgba(244, 63, 94, 0.15)';
      heroPill.style.color = '#fb7185';
      heroPill.style.borderColor = 'rgba(244, 63, 94, 0.3)';
    }
  },

  updateSubMetrics(data) {
    const wins = data.winning_trades !== undefined ? data.winning_trades : (data.wins || 0);
    const losses = data.losing_trades !== undefined ? data.losing_trades : (data.losses || 0);
    const be = data.breakeven_trades !== undefined ? data.breakeven_trades : (data.breakevens || 0);
    const total = data.total_trades || (wins + losses + be);
    
    let rate = 0.0;
    if (data.win_rate !== undefined) rate = Number(data.win_rate);
    else if (data.win_rate_pct !== undefined) rate = Number(data.win_rate_pct);
    else if (total > 0) rate = Number(((wins / total) * 100).toFixed(2));

    const breakdown = be > 0 ? `${wins}W / ${losses}L / ${be}BE` : `${wins}W / ${losses}L`;
    this.setText('hero-winrate', `${rate.toFixed(1)}% (${breakdown})`);
    const winRateEl = document.getElementById('hero-winrate');
    if (winRateEl) winRateEl.style.color = rate >= 50 ? '#34d399' : '#fb7185';

    const winBar = document.getElementById('hero-win-bar');
    if (winBar) winBar.style.width = `${Math.min(100, Math.max(0, rate))}%`;

    const margin = data.current_capital || data.available_margin || 1000000.0;
    this.setText('hero-margin', `₹${Math.round(margin).toLocaleString('en-IN')}`);

    const ddPct = data.max_drawdown_pct !== undefined ? Number(data.max_drawdown_pct) : 0.0;
    this.setText('hero-risk', `${ddPct.toFixed(1)}% Max DD`);
    const riskEl = document.getElementById('hero-risk');
    if (riskEl) riskEl.style.color = ddPct > 2.0 ? '#fb7185' : '#fff';
  },

  renderHeroCards(data, openPos) {
    const hasSplit = data.total_realized_pnl !== undefined;
    const realized = Number(hasSplit ? data.total_realized_pnl : 0.0);
    const unrealized = openPos.reduce((sum, p) => sum + Number(p.pnl_rupees !== undefined ? p.pnl_rupees : (p.net_pnl || p.pnl || 0.0)), 0);
    const pnlVal = hasSplit ? realized + unrealized : Number(data.total_pnl || 0.0);

    this.setHeroPnl(pnlVal);
    this.setHeroPill(pnlVal);
    this.updateSubMetrics(data);
  },

  parsePositionData(p) {
    const side = String(p.side || p.direction || 'BUY').toUpperCase();
    const isLong = side === 'LONG' || side === 'BUY';
    
    const pnlRaw = p.pnl_rupees ?? p.pnl ?? p.unrealized_pnl ?? p.net_pnl ?? 0.0;
    const pnl = Number(pnlRaw);

    const pnlPct = Number(p.pnl_pct ?? p.return_pct ?? 0.0);
    const sym = p.symbol || 'BANKNIFTY FUT';
    const lotDefault = (sym.includes('NIFTY') && !sym.includes('BANK')) ? 65 : 30;
    const qty = p.quantity ?? p.qty ?? lotDefault;
    const entryP = Number(p.entry_price ?? p.avg_price ?? p.price ?? 0.0);
    const ltpVal = Number(p.ltp ?? p.current_price ?? p.last_price ?? entryP);
    
    const defaultSl = isLong ? entryP * 0.985 : entryP * 1.015;
    const slP = Number(p.stop_loss ?? p.stop_price ?? defaultSl);

    const defaultTp = isLong ? entryP + (entryP - slP) * 2 : entryP - (slP - entryP) * 2;
    const tpP = Number(p.target_price ?? defaultTp);

    const progress = isLong
      ? Math.min(100, Math.max(0, ((ltpVal - entryP) / Math.max(1, (tpP - entryP))) * 100))
      : Math.min(100, Math.max(0, ((entryP - ltpVal) / Math.max(1, (entryP - tpP))) * 100));

    return { isLong, pnl, pnlPct, sym, lotDefault, qty, entryP, ltpVal, slP, tpP, progress };
  },

  getPositionTypeBadge(isOpt, sym, isLong) {
    if (isOpt) {
      const s = String(sym).toUpperCase();
      let optKind = 'CE/PE';
      if (s.includes('CE') || s.includes('CALL')) {
        optKind = 'CALL (CE)';
      } else if (s.includes('PE') || s.includes('PUT')) {
        optKind = 'PUT (PE)';
      }
      return `<span class="pos-type-badge opt">⚡ OPTION • ${optKind}</span>`;
    }
    const side = isLong ? 'LONG' : 'SHORT';
    return `<span class="pos-type-badge fut">📈 FUTURES • ${side}</span>`;
  },

  getRMultipleBadge(rMult) {
    if (rMult === undefined || rMult === null) return '';
    const num = Number(rMult);
    const sign = num >= 0 ? '+' : '';
    return `<span class="r-multiple-pill" title="R-Multiple">${sign}${num.toFixed(2)}R</span>`;
  },

  getTrailingInfo(status, entryP) {
    if (status === 'TRAILED_TO_COST') {
      return {
        badge: ` <span style="font-size:9px; background:rgba(16,185,129,0.25); color:#34d399; padding:1px 5px; border-radius:4px; font-weight:800; border: 1px solid rgba(16,185,129,0.4);">🛡️ ZERO RISK (COST)</span>`,
        banner: `
          <div class="pos-trail-alert cost">
            <span style="font-weight:800;">🛡️ ZERO RISK GUARANTEED</span>
            <span>Stop loss moved to entry price ₹${entryP.toFixed(2)} — no capital loss possible!</span>
          </div>`
      };
    }
    if (status === 'LOCKED_PROFIT_15') {
      return {
        badge: ` <span style="font-size:9px; background:rgba(6,182,212,0.25); color:#38bdf8; padding:1px 5px; border-radius:4px; font-weight:800; border: 1px solid rgba(6,182,212,0.4);">🏆 +15% LOCKED</span>`,
        banner: `
          <div class="pos-trail-alert profit">
            <span style="font-weight:800;">🏆 +15% PROFIT LOCKED IN</span>
            <span>Gains secured even if market pulls back!</span>
          </div>`
      };
    }
    return { badge: '', banner: '' };
  },

  getHealthPill(isProfit) {
    if (isProfit) {
      return `<span style="padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 800; background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.35);">🟢 PROFIT</span>`;
    }
    return `<span style="padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 800; background: rgba(244,63,94,0.18); color: #fb7185; border: 1px solid rgba(244,63,94,0.35);">🟠 RETRACEMENT</span>`;
  },

  renderSinglePositionCard(p) {
    const { isLong, pnl, pnlPct, sym, lotDefault, qty, entryP, ltpVal, slP, tpP, progress } = this.parsePositionData(p);
    const isProfit = pnl >= 0;
    const pnlClass = isProfit ? '' : 'loss';
    const isOpt = p.instrument_type === 'OPTIONS' || this.isOption(sym);

    const { badge: trailBadge, banner: trailBanner } = this.getTrailingInfo(p.trailing_status, entryP);
    const healthPill = this.getHealthPill(isProfit);
    const typeBadge = this.getPositionTypeBadge(isOpt, sym, isLong);
    const rMultBadge = this.getRMultipleBadge(p.r_multiple);
    const pnlSign = pnl >= 0 ? '+' : '-';
    const pctSign = pnlPct >= 0 ? '+' : '';
    const lotsCount = Math.round(qty / lotDefault);
    const lotsLabel = lotsCount > 1 ? `${lotsCount} Lots` : `${lotsCount} Lot`;
    const instLabel = isOpt ? 'Option' : 'Future';

    return `
      <div class="pos-card ${isOpt ? 'options' : 'futures'} ${isProfit ? 'profit' : ''}">
        <div class="pos-header">
          <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            ${typeBadge}
            ${rMultBadge}
            ${healthPill}
          </div>
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="pulse-dot green"></span>
            <span style="font-size: 10px; font-family: var(--font-mono); color: #34d399; font-weight: 700;">LIVE</span>
          </div>
        </div>

        ${trailBanner}

        <div class="pos-symbol">${sym}</div>
        <div class="pos-pnl-row">
          <div>
            <div style="font-size: 9.5px; color: var(--text-dim); text-transform: uppercase; font-weight: 700;">LIVE UNREALIZED P&L</div>
            <div class="pos-live-pnl ${pnlClass}">
              ${pnlSign}₹${Math.abs(pnl).toLocaleString('en-IN', { minimumFractionDigits: 2 })} (${pctSign}${pnlPct.toFixed(2)}%)
            </div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 9.5px; color: var(--text-dim); text-transform: uppercase; font-weight: 700;">QUANTITY</div>
            <div style="font-size: 13px; font-weight: 800; font-family: var(--font-mono); color: #fff;">${qty} QTY (${lotsLabel})</div>
          </div>
        </div>

        <div class="target-progress-wrapper">
          <div class="target-progress-header">
            <span>TARGET PROGRESS: <b>${progress.toFixed(0)}%</b></span>
            <span>GOAL: <b>₹${tpP.toFixed(2)}</b></span>
          </div>
          <div class="target-progress-bar">
            <div class="target-progress-fill" style="width: ${progress}%;"></div>
          </div>
        </div>

        <div class="pos-grid">
          <div><span>Entry Price:</span> <b>₹${entryP.toFixed(2)}</b></div>
          <div><span>Current LTP:</span> <b style="color: #06b6d4;">₹${ltpVal.toFixed(2)}</b></div>
          <div><span>Stop Loss:</span> <b style="color: #fb7185;">₹${slP.toFixed(2)}</b>${trailBadge}</div>
          <div><span>Target:</span> <b style="color: #34d399;">₹${tpP.toFixed(2)}</b></div>
        </div>
        <button onclick="closeCurrentPosition('${sym}')" class="btn-exit-pos">
          ✕ Square Off ${instLabel} (${sym})
        </button>
      </div>
    `;
  },

  renderActivePosition(openPos) {
    const posContainer = document.getElementById('active-position-container');
    if (!posContainer) return;

    const allPositions = openPos || this.currentPositions || [];
    let toShow = [];

    if (this.currentFilter === 'FUTURES') {
      toShow = this.futuresPositions;
    } else if (this.currentFilter === 'OPTIONS') {
      toShow = this.optionsPositions;
    } else {
      toShow = allPositions;
    }

    if (toShow.length === 0) {
      let standbyMsg = 'Capital is safely preserved in cash. Engine scanning 5m candles for high-confluence breakout setups.';
      let standbyTitle = 'STANDBY MODE (NO ACTIVE TRADES)';
      if (this.currentFilter === 'FUTURES') {
        standbyTitle = 'NO ACTIVE FUTURES CONTRACTS';
        standbyMsg = 'Zero open index futures exposure. Futures trend & breakout scanners actively monitoring.';
      } else if (this.currentFilter === 'OPTIONS') {
        standbyTitle = 'NO ACTIVE OPTION CONTRACTS';
        standbyMsg = 'Zero open options contracts. Bangur spreads & 5M confluence radar actively monitoring ATM strikes.';
      }

      posContainer.innerHTML = `
        <div class="standby-card">
          <div class="standby-icon">🛡️</div>
          <div style="font-weight: 800; color: #fff; margin-bottom: 4px; font-size: 12px; letter-spacing: 0.3px;">${standbyTitle}</div>
          <div style="font-size: 11px; color: var(--text-muted); line-height: 1.4;">${standbyMsg}</div>
        </div>
      `;
      return;
    }

    // When filter is ALL and we have both futures and options, render segregated sections
    if (this.currentFilter === 'ALL' && this.futuresPositions.length > 0 && this.optionsPositions.length > 0) {
      let html = '';
      html += `<div class="pos-section-header fut">📈 ACTIVE FUTURES CONTRACTS (${this.futuresPositions.length})</div>`;
      html += this.futuresPositions.map(p => this.renderSinglePositionCard(p)).join('');
      html += `<div class="pos-section-header opt" style="margin-top: 14px;">⚡ ACTIVE OPTION CONTRACTS (${this.optionsPositions.length})</div>`;
      html += this.optionsPositions.map(p => this.renderSinglePositionCard(p)).join('');
      posContainer.innerHTML = html;
      return;
    }

    // Otherwise render each card in toShow
    posContainer.innerHTML = toShow.map(p => this.renderSinglePositionCard(p)).join('');
  },

  renderTradesLog(historyRaw) {
    const listEl = document.getElementById('trades-list-container');
    const badgeEl = document.getElementById('trades-count-badge');
    if (!listEl) return;

    const history = Array.isArray(historyRaw) ? historyRaw : (historyRaw?.history || []);

    if (!history || history.length === 0) {
      listEl.innerHTML = `<div style="text-align: center; color: var(--text-dim); padding: 30px 0;">No trades executed today yet.</div>`;
      if (badgeEl) badgeEl.textContent = '0 CLOSED';
      return;
    }

    if (badgeEl) badgeEl.textContent = `${history.length} CLOSED`;

    const html = history.slice(0, 30).map(t => {
      const pnl = Number(t.pnl_rupees !== undefined ? t.pnl_rupees : (t.net_pnl || 0.0));
      const isProfit = pnl >= 0;
      const pnlClass = isProfit ? 'green' : 'red';
      const side = (t.direction || t.side || 'BUY').toUpperCase();
      const isBuy = side === 'BUY' || side === 'LONG';
      const sym = t.symbol || 'BANKNIFTY';
      const isOpt = t.instrument_type === 'OPTIONS' || this.isOption(sym);
      let timeStr = '--:--';
      if (t.exit_time) {
        timeStr = t.exit_time.includes(' ') ? t.exit_time.split(' ')[1] : t.exit_time;
      } else if (t.timestamp) {
        timeStr = t.timestamp.includes(' ') ? t.timestamp.split(' ')[1] : t.timestamp;
      }

      return `
        <div class="trade-row">
          <div class="trade-sym-side">
            <span class="trade-side-tag ${isBuy ? 'buy' : 'sell'}">${side}</span>
            <span class="trade-inst-pill ${isOpt ? 'opt' : 'fut'}">${isOpt ? 'OPT' : 'FUT'}</span>
            <span style="font-weight: 600; color: #fff;">${sym}</span>
            <span style="color: var(--text-dim); font-size: 10px;">${timeStr}</span>
          </div>
          <div class="trade-pnl ${pnlClass}">
            ${isProfit ? '+' : '-'}₹${Math.abs(pnl).toLocaleString('en-IN', { minimumFractionDigits: 0 })}
          </div>
        </div>
      `;
    }).join('');

    listEl.innerHTML = html;
  },

  playPositionSound(isLong = true) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      const notes = isLong ? [523.25, 659.25, 783.99, 1046.50] : [783.99, 659.25, 523.25, 440.00];
      const startTime = ctx.currentTime;

      notes.forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, startTime + idx * 0.08);

        gain.gain.setValueAtTime(0.001, startTime + idx * 0.08);
        gain.gain.linearRampToValueAtTime(0.2, startTime + idx * 0.08 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, startTime + idx * 0.08 + 0.22);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(startTime + idx * 0.08);
        osc.stop(startTime + idx * 0.08 + 0.23);
      });
    } catch (e) {
      console.warn('Audio playback error:', e);
    }
  },

  async requestNotificationPermission() {
    if (!('Notification' in window)) return false;
    if (Notification.permission === 'granted') return true;
    if (Notification.permission !== 'denied') {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return false;
  },

  showDesktopNotification(title, options = {}) {
    if (!('Notification' in window)) return;
    if (Notification.permission === 'granted' && this.notificationsEnabled) {
      try {
        const notif = new Notification(title, {
          icon: '/static/favicon.ico',
          badge: '/static/favicon.ico',
          silent: true,
          ...options
        });
        notif.onclick = () => {
          window.focus();
          notif.close();
        };
      } catch (err) {
        console.warn('Desktop notification dispatch failed:', err);
      }
    }
  },

  notifyPositionTaken(p) {
    const sym = p.symbol || 'BANKNIFTY';
    const side = (p.side || p.direction || 'BUY').toUpperCase();
    const isLong = side === 'BUY' || side === 'LONG';
    const qty = p.quantity || p.qty || 15;
    const entryP = Number(p.entry_price || p.price || p.avg_price || 0.0);
    const slP = Number(p.stop_loss || p.stop_price || 0.0);
    const tpP = Number(p.target_price || 0.0);
    const inst = p.instrument_type || (this.isOption(sym) ? 'OPTION' : 'FUTURES');
    const strat = p.strategy || 'Algo Strategy';
    const entryStr = entryP > 0 ? ` @ ₹${entryP.toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : '';
    const slStr = slP > 0 ? `SL: ₹${slP.toFixed(2)}` : 'SL: Dynamic';
    const tpStr = tpP > 0 ? `Target: ₹${tpP.toFixed(2)}` : 'Target: 1:2 R:R';

    if (window.Notifications && typeof window.Notifications.add === 'function') {
      Notifications.add({
        type: 'POSITION',
        title: `🟢 POSITION ENTERED: ${side} ${qty}x ${sym}${entryStr}`,
        details: `Type: ${inst} • Qty: ${qty} | Entry: ₹${entryP.toFixed(2)} | ${slStr} | ${tpStr} • Strategy: ${strat}`,
        isLong: isLong,
        meta: { symbol: sym, side, qty, price: entryP, sl: slP, target: tpP }
      });
      return;
    }

    if (this.audioEnabled) {
      this.playPositionSound(isLong);
    }

    showToast(`🟢 POSITION TAKEN: ${side} ${qty}x ${sym}${entryStr}`, 'success');

    this.showDesktopNotification(`🟢 POSITION TAKEN: ${side} ${sym}`, {
      body: `Type: ${inst} • ${side} ${qty} QTY\nPrice: ₹${entryP.toFixed(2)} | ${slStr} | ${tpStr}\nStrategy: ${strat}`,
      tag: `pos-${p.id || p.trade_id || sym}`,
      requireInteraction: false
    });
  },

  checkNewPositions(openPos) {
    if (!Array.isArray(openPos)) return;

    const currentIds = new Set();
    const posList = [];

    openPos.forEach(p => {
      const id = String(p.id || p.trade_id || `${p.symbol}_${p.side}_${p.entry_price || ''}`);
      currentIds.add(id);
      posList.push({ id, pos: p });
    });

    if (!this.notificationsInitialized) {
      this.knownPositionIds = currentIds;
      this.notificationsInitialized = true;
      this.updateAlertsButtonUI();
      return;
    }

    posList.forEach(({ id, pos }) => {
      if (!this.knownPositionIds.has(id)) {
        this.knownPositionIds.add(id);
        this.notifyPositionTaken(pos);
      }
    });

    for (const id of this.knownPositionIds) {
      if (!currentIds.has(id)) {
        this.knownPositionIds.delete(id);
        if (window.Notifications && typeof window.Notifications.add === 'function') {
          Notifications.add({
            type: 'POSITION',
            title: `🔴 POSITION SQUARED OFF: ${id}`,
            details: `Position was squared off and closed successfully from open book.`,
            isLong: false
          });
        }
      }
    }
  },

  async toggleNotifications() {
    if ('Notification' in window && Notification.permission !== 'granted') {
      const granted = await this.requestNotificationPermission();
      if (granted) {
        this.notificationsEnabled = true;
        this.audioEnabled = true;
        this.playPositionSound(true);
        showToast('🔔 Desktop & sound notifications enabled!', 'success');
      } else {
        showToast('⚠️ Desktop notifications blocked. Sound chime is still active.', 'warning');
      }
    } else {
      this.notificationsEnabled = !this.notificationsEnabled;
      this.audioEnabled = this.notificationsEnabled;
      if (this.notificationsEnabled) {
        this.playPositionSound(true);
        showToast('🔔 Position notifications turned ON (Sound & Popups)', 'success');
      } else {
        showToast('🔕 Position notifications turned OFF', 'info');
      }
    }
    this.updateAlertsButtonUI();
  },

  updateAlertsButtonUI() {
    const btn = document.getElementById('btn-alerts-toggle');
    const label = document.getElementById('alerts-toggle-text');
    const icon = document.getElementById('alerts-icon');
    if (!btn || !label) return;

    if (this.notificationsEnabled) {
      label.textContent = 'ALERTS: ON';
      if (icon) icon.textContent = '🔔';
      btn.classList.add('green');
      btn.classList.remove('dim');
      btn.title = 'Position Entry Alerts active (Click to test or toggle)';
    } else {
      label.textContent = 'ALERTS: OFF';
      if (icon) icon.textContent = '🔕';
      btn.classList.remove('green');
      btn.classList.add('dim');
      btn.title = 'Position Entry Alerts disabled (Click to enable)';
    }
  }
};

async function closeCurrentPosition(symbol) {
  const sym = symbol || (Positions.currentPositions.length > 0 ? Positions.currentPositions[0].symbol : 'BANKNIFTY FUT');
  if (!confirm(`Are you sure you want to close position for ${sym}?`)) return;
  try {
    const res = await fetch('/api/position/close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: sym, reason: 'Manual Terminal Close' })
    });
    await res.json();
    showToast(`Position for ${sym} closed successfully`, 'success');
    Positions.fetchAndRenderAll();
    if (typeof ChartEngine !== 'undefined') ChartEngine.loadTradeMarkers();
  } catch (err) {
    console.error('Failed to close position:', err);
    showToast('Failed to close position', 'error');
  }
}

async function emergencySquareOffAll() {
  if (!confirm('⚠️ EMERGENCY: Are you sure you want to SQUARE OFF ALL positions and cancel all pending orders immediately?')) return;
  try {
    const res = await fetch('/api/flatten_all', { method: 'POST' });
    await res.json();
    showToast('ALL POSITIONS SQUARED OFF', 'error');
    Positions.fetchAndRenderAll();
    if (typeof ChartEngine !== 'undefined') ChartEngine.loadTradeMarkers();
  } catch (err) {
    console.error('Emergency square-off failed:', err);
    showToast('Emergency square-off failed', 'error');
  }
}

function showToast(msg, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    setTimeout(() => t.remove(), 300);
  }, 3500);
}

window.Positions = Positions;
document.addEventListener('DOMContentLoaded', () => {
  Positions.fetchAndRenderAll();
});
