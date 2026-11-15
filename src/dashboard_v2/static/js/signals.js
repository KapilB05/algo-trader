/**
 * Option Signals & Execution Controller (src/dashboard_v2/static/js/signals.js)
 * Manages:
 * 1. "When to Buy & When to Sell" Signal Radar
 * 2. 1-Click Quick Option Order Execution
 * 3. Auto-Pilot Autonomous Execution
 * 4. Paper vs. Live Angel One Mode Switching
 * 5. Full Option Chain Modal & Audio Alerts
 */

const Signals = {
  currentSignal: null,
  activeSymbol: 'BANKNIFTY',
  modalSymbol: 'BANKNIFTY',
  quickLots: 1,
  autoPilot: false,
  execMode: 'PAPER',
  atmOptions: null,
  pollTimer: null,
  lastSignalKey: null,
  currentSpreads: null,

  getLotSizeForSymbol(symbol) {
    if (!symbol) return this.activeSymbol === 'NIFTY' ? 65 : 30;
    const s = symbol.toUpperCase();
    if (s.includes('NIFTY') && !s.includes('BANK')) {
      return 65; // Official NSE Nifty 50 lot size
    }
    return 30; // Official NSE Bank Nifty lot size
  },

  getSignalQuantity(sig) {
    if (!sig) return this.quickLots * (this.activeSymbol === 'NIFTY' ? 65 : 30);
    const lotSz = sig.lot_size || this.getLotSizeForSymbol(sig.symbol);
    return this.quickLots * lotSz;
  },

  setSymbol(symbol) {
    if (!symbol) return;
    const s = symbol.toUpperCase();
    const isNifty = s.includes('NIFTY') && !s.includes('BANK');
    this.activeSymbol = isNifty ? 'NIFTY' : 'BANKNIFTY';
    this.modalSymbol = this.activeSymbol;

    // 1. Update Instrument pills in Cockpit
    const btnBn = document.getElementById('btn-cockpit-bn');
    const btnNf = document.getElementById('btn-cockpit-nf');
    if (btnBn && btnNf) {
      if (isNifty) {
        btnBn.classList.remove('active');
        btnNf.classList.add('active');
      } else {
        btnBn.classList.add('active');
        btnNf.classList.remove('active');
      }
    }

    // 2. Update chart symbol buttons
    const chartBtns = document.querySelectorAll('.chart-sym-btn');
    chartBtns.forEach(b => {
      const bSym = (b.dataset.symbol || '').toUpperCase();
      if (isNifty && bSym.includes('NIFTY') && !bSym.includes('BANK')) {
        b.classList.add('active');
      } else if (!isNifty && (bSym.includes('BANK') || bSym.includes('BN'))) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });

    // 3. Update Quick Lots text
    this.updateQuickLotsUI();

    // 4. Trigger scanner fetch and research spreads for selected instrument
    this.fetchSignals();
    this.fetchResearchSpreads();

    // 5. If ChartEngine is present and not currently on this symbol, sync chart
    if (typeof ChartEngine !== 'undefined' && ChartEngine.currentSymbol) {
      const targetChartSym = isNifty ? 'NIFTY 50' : 'BANKNIFTY FUT';
      if (ChartEngine.currentSymbol !== targetChartSym) {
        ChartEngine.currentSymbol = targetChartSym;
        ChartEngine.loadChart(targetChartSym, ChartEngine.currentTimeframe || '5m');
      }
    }
  },

  updateQuickLotsUI() {
    const isNifty = this.activeSymbol === 'NIFTY';
    const baseLot = isNifty ? 65 : 30;

    const b1 = document.getElementById('lot-btn-1');
    const b2 = document.getElementById('lot-btn-2');
    const b4 = document.getElementById('lot-btn-4');
    const title = document.getElementById('quick-ticket-title');

    if (b1) b1.textContent = `1x (${baseLot})`;
    if (b2) b2.textContent = `2x (${baseLot * 2})`;
    if (b4) b4.textContent = `4x (${baseLot * 4})`;
    if (title) title.textContent = `⚡ Quick ATM Strikes (${isNifty ? 'NIFTY 50' : 'BANKNIFTY'})`;

    // Re-render quick ATM strikes with active options
    if (this.atmOptions) {
      this.renderQuickTicket(this.atmOptions);
    }
  },

  init() {
    this.fetchSignals();
    this.fetchResearchSpreads();
    this.pollTimer = setInterval(() => {
      this.fetchSignals();
      this.fetchResearchSpreads();
    }, 3000);
  },

  async fetchSignals() {
    try {
      const res = await fetch(`/api/scanner/signals?symbol=${this.activeSymbol}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.primary_signal) {
        this.renderPrimarySignal(data.primary_signal);
      }
      if (data.atm_options) {
        this.atmOptions = data.atm_options;
        this.renderQuickTicket(data.atm_options);
      }
      if (data.exec_mode) {
        this.execMode = data.exec_mode;
        this.renderModeButton(data.exec_mode);
      }
    } catch (err) {
      console.error('Failed to fetch scanner signals:', err);
    }
  },

  renderPrimarySignal(sig) {
    this.currentSignal = sig;
    const card = document.getElementById('option-signal-card');
    const headline = document.getElementById('signal-headline');
    const badgeText = document.getElementById('signal-badge-text');
    const sigEntry = document.getElementById('sig-entry');
    const sigSl = document.getElementById('sig-sl');
    const sigTarget = document.getElementById('sig-target');
    const sigRr = document.getElementById('sig-rr');
    const reasonsBox = document.getElementById('signal-reasons-box');
    const btnExec = document.getElementById('btn-execute-signal');

    if (!card || !headline) return;

    const action = sig.action || 'STANDBY';
    const isCall = action === 'BUY_CALL';
    const isPut = action === 'BUY_PUT';
    const isStandby = action === 'STANDBY';

    // Play chime on signal change
    const signalKey = `${action}_${sig.symbol}`;
    if (this.lastSignalKey && this.lastSignalKey !== signalKey && !isStandby) {
      this.playSignalChime(isCall ? 'bull' : 'bear');
    }
    this.lastSignalKey = signalKey;

    this._updateSignalCardStyling(card, headline, badgeText, btnExec, isCall, isPut, sig);
    this._updateSignalValues(sigEntry, sigSl, sigTarget, sigRr, sig);
    this._updateReasonsBox(reasonsBox, sig, isStandby);

    // Auto-Pilot Execution if enabled and no current position open
    if (this.autoPilot && !isStandby && window.Positions && (!Positions.currentPositions || Positions.currentPositions.length === 0)) {
      console.info('[Auto-Pilot] Auto-executing primary signal:', sig.symbol);
      this.executePrimarySignal(true);
    }
  },

  _updateSignalCardStyling(card, headline, badgeText, btnExec, isCall, isPut, sig) {
    if (isCall) {
      card.className = 'signal-card buy';
      headline.innerHTML = `🟢 BUY CALL: ${sig.symbol}`;
      headline.className = 'signal-title buy';
      if (badgeText) badgeText.textContent = `BULLISH BREAKOUT (${sig.confidence_pct}% CONF)`;
      if (btnExec) {
        btnExec.className = 'btn-execute-signal buy';
        btnExec.innerHTML = `⚡ 1-Click Buy Call (${this.getSignalQuantity(sig)} Qty @ ₹${sig.entry_price})`;
        btnExec.style.display = 'block';
        btnExec.disabled = false;
        btnExec.style.cursor = 'pointer';
        btnExec.style.background = '';
        btnExec.style.color = '#fff';
      }
    } else if (isPut) {
      card.className = 'signal-card sell';
      headline.innerHTML = `🔴 BUY PUT: ${sig.symbol}`;
      headline.className = 'signal-title sell';
      if (badgeText) badgeText.textContent = `BEARISH BREAKDOWN (${sig.confidence_pct}% CONF)`;
      if (btnExec) {
        btnExec.className = 'btn-execute-signal sell';
        btnExec.innerHTML = `⚡ 1-Click Buy Put (${this.getSignalQuantity(sig)} Qty @ ₹${sig.entry_price})`;
        btnExec.style.display = 'block';
        btnExec.disabled = false;
        btnExec.style.cursor = 'pointer';
        btnExec.style.background = '';
        btnExec.style.color = '#fff';
      }
    } else {
      card.className = 'signal-card standby';
      headline.innerHTML = `⚪ CAPITAL PRESERVATION: CHOP MARKET (DO NOT TRADE)`;
      headline.className = 'signal-title standby';
      if (badgeText) badgeText.textContent = `CHOP GATED (THETA DEFENSE SHIELD)`;
      if (btnExec) {
        btnExec.className = 'btn-execute-signal standby';
        btnExec.innerHTML = `⏸️ Option Buying Gated in Chop (Preserving Capital)`;
        btnExec.style.display = 'block';
        btnExec.disabled = true;
        btnExec.style.background = 'rgba(255, 255, 255, 0.08)';
        btnExec.style.color = '#94a3b8';
        btnExec.style.cursor = 'not-allowed';
      }
    }
  },

  _updateSignalValues(sigEntry, sigSl, sigTarget, sigRr, sig) {
    if (sigEntry) sigEntry.textContent = `₹${sig.entry_price?.toFixed(2) || '--'}`;
    if (sigSl) sigSl.textContent = `₹${sig.stop_loss?.toFixed(2) || '--'} (-25%)`;
    if (sigTarget) sigTarget.textContent = `₹${sig.target?.toFixed(2) || '--'} (+50%)`;
    if (sigRr) sigRr.textContent = `1:${sig.rr_ratio || 2.0} (${sig.confidence_pct || 88}% Prob)`;
  },

  _updateReasonsBox(reasonsBox, sig, isStandby) {
    if (reasonsBox) {
      const reasons = sig.reasons || [sig.standby_reason || 'Waiting for high-confluence entry.'];
      reasonsBox.innerHTML = reasons.map(r => {
        const clean = String(r).replace(/^[•\s*-]+/, '').trim();
        let icon = '⚡';
        const lower = clean.toLowerCase();
        if (lower.includes('chop') || lower.includes('preserve') || lower.includes('shield') || lower.includes('decay')) {
          icon = '🛡️';
        } else if (lower.includes('breakout') || lower.includes('trigger') || lower.includes('target')) {
          icon = '🎯';
        } else if (lower.includes('vix') || lower.includes('volatility')) {
          icon = '🌊';
        } else if (lower.includes('vwap') || lower.includes('ema') || lower.includes('volume')) {
          icon = '📊';
        }
        return `
          <div class="signal-reason-pill">
            <span class="reason-icon">${icon}</span>
            <span class="reason-text">${clean}</span>
          </div>
        `;
      }).join('');
    }
  },

  renderQuickTicket(options) {
    const isNifty = this.activeSymbol === 'NIFTY';
    const ceLabel = document.getElementById('quick-ce-label');
    const peLabel = document.getElementById('quick-pe-label');
    const optCe = isNifty ? options.nf_ce : options.bn_ce;
    const optPe = isNifty ? options.nf_pe : options.bn_pe;

    if (optCe && ceLabel) {
      ceLabel.textContent = `${optCe.strike} CE @ ₹${optCe.ltp}`;
    }
    if (optPe && peLabel) {
      peLabel.textContent = `${optPe.strike} PE @ ₹${optPe.ltp}`;
    }
  },

  async fetchResearchSpreads() {
    try {
      const res = await fetch(`/api/research/spreads?symbol=${this.activeSymbol}&lots=${this.quickLots}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.status === 'SUCCESS') {
        this.currentSpreads = data.spreads;
        this.renderResearchSpreads(data);
      }
    } catch (err) {
      console.warn('[Signals] Failed to fetch research spreads:', err);
    }
  },

  renderResearchSpreads(data) {
    if (!data) return;
    const spreads = data.spreads || [];

    this._renderKellyBadge(data.kelly_sizing);
    this._renderSpreadCard(spreads, 'Bull Call', 'spread-bcs-debit', 'spread-bcs-desc', 'CE');
    this._renderSpreadCard(spreads, 'Bear Put', 'spread-bps-debit', 'spread-bps-desc', 'PE');
  },

  _renderKellyBadge(kelly) {
    const badge = document.getElementById('kelly-sizing-badge');
    if (badge && kelly) {
      badge.textContent = `Kelly: ${kelly.recommended_lots} Lot${kelly.recommended_lots > 1 ? 's' : ''} (2% Cap)`;
      badge.title = `Wu et al. (2022) Kelly: ${kelly.recommended_qty} Qty, Risk Cap ₹${Number(kelly.risk_rupees).toLocaleString('en-IN')}`;
    }
  },

  _renderSpreadCard(spreads, strategyName, debitId, descId, typeSuffix) {
    const spread = spreads.find(s => s.strategy_name.includes(strategyName));
    const debitEl = document.getElementById(debitId);
    const descEl = document.getElementById(descId);
    if (spread && debitEl && descEl) {
      debitEl.textContent = `Net ₹${spread.net_premium.toFixed(1)}`;
      const legBuy = spread.legs.find(l => l.action === 'BUY');
      const legSell = spread.legs.find(l => l.action === 'SELL');
      const buyStr = legBuy ? `${legBuy.strike} ${typeSuffix}` : `ATM ${typeSuffix}`;
      const sellStr = legSell ? `${legSell.strike} ${typeSuffix}` : `OTM ${typeSuffix}`;
      descEl.textContent = `Buy ${buyStr} + Sell ${sellStr} (Loss: ₹${Math.round(spread.max_loss_rupees).toLocaleString('en-IN')} | Profit: ₹${Math.round(spread.max_profit_rupees).toLocaleString('en-IN')})`;
    }
  },

  async executeSpread(strategyType) {
    const isNifty = this.activeSymbol === 'NIFTY';
    const symName = isNifty ? 'NIFTY 50' : 'BANKNIFTY';
    const stratName = strategyType === 'BULL_CALL' ? 'Bull Call Spread (2.75x Edge)' : 'Bear Put Spread';
    const lots = this.quickLots;
    const lotSz = this.getLotSizeForSymbol(this.activeSymbol);
    const totalUnits = lots * lotSz;

    if (!confirm(`Execute ${stratName} for ${symName}?\nAllocating ${lots} Lot (${totalUnits} Qty) with defined max loss.`)) {
      return;
    }

    try {
      const res = await fetch('/api/research/execute-spread', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_type: strategyType,
          symbol: this.activeSymbol,
          lots: lots,
        })
      });
      const data = await res.json();
      if (res.ok && data.status === 'SUCCESS') {
        showToast(`🎓 ${data.strategy_name} Executed! Max Loss: ₹${data.max_loss_rupees.toLocaleString('en-IN')}`, 'success');
        if (window.Positions) Positions.fetchAndRenderAll();
      } else {
        showToast(data.detail || 'Spread execution failed', 'error');
      }
    } catch (err) {
      console.error('Failed to execute spread strategy', err);
      showToast('Failed to execute spread strategy', 'error');
    }
  },

  renderModeButton(mode) {
    const btn = document.getElementById('btn-mode-toggle');
    const dot = document.getElementById('mode-dot');
    const txt = document.getElementById('mode-text');
    if (!btn || !txt) return;

    if (mode === 'LIVE') {
      btn.className = 'btn-mode-toggle live';
      if (dot) dot.className = 'pulse-dot green';
      txt.textContent = 'MODE: LIVE ANGEL';
    } else {
      btn.className = 'btn-mode-toggle paper';
      if (dot) dot.className = 'pulse-dot blue';
      txt.textContent = 'MODE: PAPER SIM';
    }
  },

  async executePrimarySignal(isAuto = false) {
    if (!this.currentSignal || this.currentSignal.action === 'STANDBY') {
      showToast('No active signal to execute', 'error');
      return;
    }

    const sig = this.currentSignal;
    const qty = this.getSignalQuantity(sig);
    const mode = this.execMode;

    if (!isAuto && !confirm(`Place ${mode} Order for ${sig.symbol} (${qty} Qty @ ₹${sig.entry_price})?`)) {
      return;
    }

    try {
      const res = await fetch('/api/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: sig.symbol,
          side: 'BUY',
          quantity: qty,
          price: sig.entry_price,
          stop_loss: sig.stop_loss
        })
      });

      const data = await res.json();
      if (res.ok) {
        showToast(`⚡ ${sig.symbol} Bought (${qty} Qty @ ₹${sig.entry_price})`, 'success');
        if (window.Positions) Positions.fetchAndRenderAll();
        if (typeof ChartEngine !== 'undefined') ChartEngine.loadTradeMarkers();
      } else {
        showToast(data.detail || 'Order execution blocked', 'error');
      }
    } catch (err) {
      console.error('Failed to place order', err);
      showToast('Failed to place order', 'error');
    }
  },

  async quickBuyOption(type) {
    if (!this.atmOptions) return;
    const isNifty = this.activeSymbol === 'NIFTY';
    let opt;
    if (type === 'CE') {
      opt = isNifty ? this.atmOptions.nf_ce : this.atmOptions.bn_ce;
    } else {
      opt = isNifty ? this.atmOptions.nf_pe : this.atmOptions.bn_pe;
    }
    if (!opt) return;

    const qty = this.quickLots * this.getLotSizeForSymbol(opt.symbol);
    const price = opt.ltp;
    const sl = roundPrice(price * 0.75);

    if (!confirm(`Quick Buy ${opt.symbol} (${qty} Qty @ ₹${price})?`)) return;

    try {
      const res = await fetch('/api/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: opt.symbol,
          side: 'BUY',
          quantity: qty,
          price: price,
          stop_loss: sl
        })
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`⚡ Order filled: ${opt.symbol} (${qty} Qty)`, 'success');
        if (window.Positions) Positions.fetchAndRenderAll();
      } else {
        showToast(data.detail || 'Order placement failed', 'error');
      }
    } catch (err) {
      console.error('Failed to place quick order', err);
      showToast('Failed to place quick order', 'error');
    }
  },

  setQuickLots(lots, el) {
    this.quickLots = Number(lots);
    document.querySelectorAll('.lot-btn').forEach(b => b.classList.remove('active'));
    if (el) el.classList.add('active');

    // Update research spreads for new lot size
    this.fetchResearchSpreads();

    // Update signal execute button text
    const btnExec = document.getElementById('btn-execute-signal');
    if (btnExec && this.currentSignal) {
      const side = this.currentSignal.action === 'BUY_CALL' ? 'Call' : 'Put';
      const qty = this.getSignalQuantity(this.currentSignal);
      btnExec.innerHTML = `⚡ 1-Click Buy ${side} (${qty} Qty @ ₹${this.currentSignal.entry_price})`;
    }
  },

  toggleAutoPilot(checked) {
    this.autoPilot = Boolean(checked);
    showToast(`🤖 Auto-Pilot ${this.autoPilot ? 'ENABLED (Will auto-trade signals)' : 'DISABLED (Manual confirm)'}`, this.autoPilot ? 'success' : 'error');
  },

  async toggleExecMode() {
    const nextMode = this.execMode === 'PAPER' ? 'LIVE' : 'PAPER';
    const isLive = nextMode === 'LIVE';

    if (isLive && !confirm('⚠️ SWITCH TO LIVE BROKER MODE?\n\nReal money orders will be routed to your Angel One account (S1680402). Continue?')) {
      return;
    }

    try {
      const res = await fetch('/api/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: nextMode })
      });
      if (res.ok) {
        this.execMode = nextMode;
        this.renderModeButton(nextMode);
        showToast(`Switched to ${nextMode} execution mode`, isLive ? 'error' : 'success');
      }
    } catch (err) {
      console.error('Failed to switch mode', err);
      showToast('Failed to switch mode', 'error');
    }
  },

  playSignalChime(type = 'bull') {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);

      if (type === 'bull') {
        osc.frequency.setValueAtTime(520, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(780, ctx.currentTime + 0.15);
      } else {
        osc.frequency.setValueAtTime(680, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.15);
      }

      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);

      osc.start();
      osc.stop(ctx.currentTime + 0.26);
    } catch (err) {
      // Ignored: Audio context may fail if user hasn't interacted with document yet
    }
  },

  setModalOptionChainSymbol(sym) {
    this.modalSymbol = sym;
    this.loadOptionChainModal(sym);
  },

  async loadOptionChainModal(symbolOverride = null) {
    if (symbolOverride) {
      this.modalSymbol = symbolOverride;
    } else if (!this.modalSymbol) {
      this.modalSymbol = this.activeSymbol || 'BANKNIFTY';
    }

    const sym = this.modalSymbol;
    const isNifty = sym === 'NIFTY';

    this._updateOptionChainModalHeaders(sym, isNifty);

    const tableBody = document.getElementById('oc-modal-table-body');
    if (!tableBody) return;
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 20px; color: var(--text-dim);">Loading live ${sym} Option Chain from Angel One...</td></tr>`;

    try {
      const res = await fetch(`/api/option-chain?symbol=${encodeURIComponent(sym)}`);
      if (!res.ok) return;
      const data = await res.json();

      const modalPcr = document.getElementById('modal-oc-pcr');
      const modalMp = document.getElementById('modal-oc-maxpain');
      const modalSpot = document.getElementById('modal-oc-spot');

      if (modalPcr) modalPcr.textContent = `PCR: ${data.pcr} (${data.pcr_signal || 'BULLISH'})`;
      if (modalMp) modalMp.textContent = `Max Pain: ₹${data.max_pain}`;
      if (modalSpot) modalSpot.textContent = `Spot: ₹${data.spot_price}`;
      // Opportunistic header ticker update (no extra polling — reuse this response).
      if (!isNifty && data.spot_price) {
        const bnTicker = document.getElementById('ticker-bn-price');
        if (bnTicker) bnTicker.textContent = `₹${Number(data.spot_price).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
      }

      const strikes = data.strikes || [];
      if (strikes.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 20px; color: var(--text-dim);">No strike data available</td></tr>`;
        return;
      }

      this._renderOptionChainRows(tableBody, strikes, data.atm_strike);
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 20px; color: var(--color-red);">Failed to load Option Chain</td></tr>`;
      console.error('OC Load Error:', err);
    }
  },

  _updateOptionChainModalHeaders(sym, isNifty) {
    const tabBn = document.getElementById('btn-oc-modal-bn');
    const tabNf = document.getElementById('btn-oc-modal-nf');
    if (tabBn && tabNf) {
      if (isNifty) {
        tabBn.classList.remove('active');
        tabNf.classList.add('active');
      } else {
        tabBn.classList.add('active');
        tabNf.classList.remove('active');
      }
    }
    const modalTitle = document.getElementById('oc-modal-title');
    if (modalTitle) {
      modalTitle.textContent = `📊 ${isNifty ? 'NIFTY 50' : 'BANKNIFTY'} LIVE OPTION CHAIN`;
    }
  },

  _renderOptionChainRows(tableBody, strikes, atm) {
    const maxCeOi = Math.max(...strikes.map(s => Number(s.ce_oi) || 0), 1);
    const maxPeOi = Math.max(...strikes.map(s => Number(s.pe_oi) || 0), 1);

    const html = strikes.map(s => {
      const isAtm = s.strike === atm;
      const atmClass = isAtm ? 'atm-row' : '';
      const ceOiVal = Number(s.ce_oi) || 0;
      const peOiVal = Number(s.pe_oi) || 0;
      const cePct = Math.round((ceOiVal / maxCeOi) * 100);
      const pePct = Math.round((peOiVal / maxPeOi) * 100);

      return `
        <tr class="${atmClass}">
          <td style="color: var(--color-cyan); font-weight: 700; text-align: right; background: linear-gradient(to left, rgba(6, 182, 212, 0.18) ${cePct}%, transparent ${cePct}%);">${ceOiVal ? ceOiVal.toLocaleString('en-IN') : '--'}</td>
          <td style="color: var(--color-green); font-weight: 700;">₹${s.ce_ltp?.toFixed(2) || '--'}</td>
          <td><button onclick="quickBuySpecific('${s.strike} CE', ${s.ce_ltp})" class="mini-buy-btn ce">BUY</button></td>
          <td class="strike-col ${isAtm ? 'atm-strike' : ''}">${s.strike} ${isAtm ? '★ ATM' : ''}</td>
          <td><button onclick="quickBuySpecific('${s.strike} PE', ${s.pe_ltp})" class="mini-buy-btn pe">BUY</button></td>
          <td style="color: var(--color-red); font-weight: 700;">₹${s.pe_ltp?.toFixed(2) || '--'}</td>
          <td style="color: var(--color-amber); font-weight: 700; text-align: left; background: linear-gradient(to right, rgba(245, 158, 11, 0.18) ${pePct}%, transparent ${pePct}%);">${peOiVal ? peOiVal.toLocaleString('en-IN') : '--'}</td>
        </tr>
      `;
    }).join('');

    tableBody.innerHTML = html;

  }
};

function quickBuySpecific(symFull, price) {
  const prefix = Signals.modalSymbol || Signals.activeSymbol || 'BANKNIFTY';
  const sym = `${prefix} ${symFull}`;
  const p = Number(price) || 500.0;
  const qty = Signals.quickLots * Signals.getLotSizeForSymbol(sym);
  const sl = roundPrice(p * 0.75);

  if (!confirm(`Buy ${sym} (${qty} Qty @ ₹${p})?`)) return;

  fetch('/api/order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      symbol: sym,
      side: 'BUY',
      quantity: qty,
      price: p,
      stop_loss: sl
    })
  }).then(r => r.json()).then(d => {
    showToast(`⚡ Order placed for ${sym} (${qty} Qty)`, 'success');
    toggleOptionChainModal(false);
    if (window.Positions) Positions.fetchAndRenderAll();
  }).catch(() => showToast('Order placement failed', 'error'));
}

function executePrimarySignal() {
  Signals.executePrimarySignal(false);
}

function quickBuyOption(type) {
  Signals.quickBuyOption(type);
}

function setQuickLots(lots, el) {
  Signals.setQuickLots(lots, el);
}

function toggleAutoPilot(checked) {
  Signals.toggleAutoPilot(checked);
}

function toggleExecMode() {
  Signals.toggleExecMode();
}

function triggerScannerScan() {
  showToast('⚡ Running Live Confluence Scan...', 'success');
  Signals.fetchSignals();
  if (window.Arb) Arb.scanNow();
}

function toggleOptionChainModal(open) {
  const modal = document.getElementById('option-chain-modal');
  if (!modal) return;
  if (open) {
    modal.classList.add('open');
    Signals.loadOptionChainModal();
  } else {
    modal.classList.remove('open');
  }
}

function executeSpread(type) {
  Signals.executeSpread(type);
}

window.Signals = Signals;
document.addEventListener('DOMContentLoaded', () => Signals.init());
