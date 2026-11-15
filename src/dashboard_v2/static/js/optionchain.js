/**
 * Option Chain JavaScript Module (optionchain.js)
 * Fetches live Option Chain data from /api/option-chain and renders CE/PE matrix.
 * Supports click-to-trade auto-filling in Order Terminal.
 */

window.OptionChainEngine = {
  currentSymbol: 'NIFTY',
  currentExpiry: '',

  init() {
    this.bindEvents();
    this.fetchData(this.currentSymbol);
    // Live continuous 3-second auto-refresh
    if (!this.intervalId) {
      this.intervalId = setInterval(() => {
        const activeTab = document.querySelector('.nav-tab.active');
        if (activeTab?.dataset.tab === 'tab-optionchain') {
          this.fetchData(this.currentSymbol, this.currentExpiry);
        }
      }, 8000);
    }
  },

  bindEvents() {
    const btns = document.querySelectorAll('.oc-sym-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        btns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const sym = e.target.dataset.symbol;
        this.currentSymbol = sym;
        this.currentExpiry = '';

        const badge = document.getElementById('oc-symbol-badge');
        if (badge) {
          if (sym === 'NIFTY') badge.textContent = 'NIFTY 50';
          else if (sym === 'BANKNIFTY') badge.textContent = 'BANKNIFTY';
          else badge.textContent = 'FINNIFTY';
        }

        const tbody = document.getElementById('option-chain-body');
        if (tbody) {
          tbody.innerHTML = `<tr><td colspan="11" style="padding: 24px; color: var(--accent-cyan); font-weight: 500;">⏳ Loading ${sym} Option Chain...</td></tr>`;
        }

        this.fetchData(sym);
      });
    });

    const expSelect = document.getElementById('oc-expiry-select');
    if (expSelect) {
      expSelect.addEventListener('change', (e) => {
        this.currentExpiry = e.target.value;
        const tbody = document.getElementById('option-chain-body');
        if (tbody) {
          tbody.innerHTML = `<tr><td colspan="11" style="padding: 24px; color: var(--accent-cyan); font-weight: 500;">⏳ Loading Expiry ${this.currentExpiry}...</td></tr>`;
        }
        this.fetchData(this.currentSymbol, this.currentExpiry);
      });
    }
  },

  async fetchData(symbol, expiry = '') {
    try {
      let url = `/api/option-chain?symbol=${encodeURIComponent(symbol)}`;
      if (expiry) url += `&expiry=${encodeURIComponent(expiry)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.available_expiries && data.available_expiries.length > 0) {
        this.updateExpiryDropdown(data.available_expiries, data.selected_expiry);
      }
      this.render(data);
    } catch (err) {
      console.error("Failed to fetch option chain:", err);
      const tbody = document.getElementById('option-chain-body');
      if (tbody?.children.length === 0) {
        tbody.innerHTML = `<tr><td colspan="11" style="padding: 24px; color: #ef4444;">⚠️ Failed to load option chain data. Please retry.</td></tr>`;
      }
    }
  },

  updateExpiryDropdown(expiries, selected) {
    const select = document.getElementById('oc-expiry-select');
    if (!select) return;

    if (select.children.length !== expiries.length) {
      select.innerHTML = '';
      expiries.forEach(exp => {
        const opt = document.createElement('option');
        opt.value = exp;
        opt.textContent = exp;
        if (exp === selected) opt.selected = true;
        select.appendChild(opt);
      });
    }
    this.currentExpiry = select.value || selected;
  },

  updateMetrics(data, spot, pcr, maxPain, atmStrike) {
    const setText = (id, txt) => {
      const el = document.getElementById(id);
      if (el) el.textContent = txt;
    };

    setText('oc-updated-time', `Updated: ${data.timestamp_ist || '--'}`);
    setText('oc-spot-price', spot > 0 ? `₹${spot.toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : 'N/A');
    
    let pcrSig = data.pcr_signal;
    if (!pcrSig) pcrSig = pcr >= 1.0 ? 'BULLISH' : 'BEARISH';
    setText('oc-pcr-val', `${pcr.toFixed(2)} (${pcrSig})`);
    
    setText('oc-max-pain', maxPain > 0 ? `₹${maxPain.toLocaleString('en-IN')}` : 'N/A');
    setText('oc-atm-strike', atmStrike > 0 ? `₹${atmStrike.toLocaleString('en-IN')}` : 'N/A');
  },

  updateSourceBadge(source) {
    const badge = document.getElementById('oc-source-badge');
    if (!badge) return;

    if (source.includes('Angel One')) {
      badge.textContent = '🟢 LIVE (Angel One)';
      badge.style.background = 'rgba(34, 197, 94, 0.15)';
      badge.style.color = '#22c55e';
      badge.style.border = '1px solid rgba(34, 197, 94, 0.3)';
    } else if (source.includes('NSE Backup')) {
      badge.textContent = '🔵 LIVE (NSE Backup)';
      badge.style.background = 'rgba(6, 182, 212, 0.15)';
      badge.style.color = '#38bdf8';
      badge.style.border = '1px solid rgba(6, 182, 212, 0.3)';
    } else {
      badge.textContent = '🔴 UNAVAILABLE';
      badge.style.background = 'rgba(239, 68, 68, 0.15)';
      badge.style.color = '#f87171';
      badge.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    }
  },

  render(data) {
    const spot = data.spot_price || 0.0;
    const pcr = data.pcr || 1.0;
    const maxPain = data.max_pain || 0.0;
    const atmStrike = data.atm_strike || 0.0;
    const strikes = data.strikes || [];
    const source = data.source || 'UNAVAILABLE';
    const status = data.status || 'UNAVAILABLE';

    this.updateSourceBadge(source);
    this.updateMetrics(data, spot, pcr, maxPain, atmStrike);

    const tbody = document.getElementById('option-chain-body');
    if (!tbody) return;

    if (status === 'UNAVAILABLE' || strikes.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="11" style="padding: 36px 20px; text-align: center; color: #ef4444; font-weight: 600;">
            ⚠️ REAL OPTION CHAIN DATA CURRENTLY UNAVAILABLE<br>
            <span style="font-size: 11px; color: var(--text-muted); font-weight: normal; margin-top: 6px; display: inline-block;">
              Angel One & NSE live feeds are unreachable. Zero synthetic pricing used.
            </span>
          </td>
        </tr>
      `;
      return;
    }

    let html = '';
    strikes.forEach(row => {
      const isAtm = row.strike === atmStrike;
      const rowStyle = isAtm ? 'background: rgba(234, 179, 8, 0.12); font-weight: bold;' : '';

      const ceDelta = typeof row.ce_delta === 'number' ? row.ce_delta.toFixed(2) : '--';
      const ceIv = typeof row.ce_iv === 'number' ? `${row.ce_iv.toFixed(1)}%` : '--';
      const ceVol = typeof row.ce_volume === 'number' ? row.ce_volume.toLocaleString('en-IN') : '--';
      let ceOiChg = '--';
      if (typeof row.ce_oi_chg_pct === 'number') {
        const sign = row.ce_oi_chg_pct > 0 ? '+' : '';
        ceOiChg = `${sign}${row.ce_oi_chg_pct.toFixed(1)}%`;
      }
      const ceLtp = typeof row.ce_ltp === 'number' ? row.ce_ltp.toFixed(2) : '0.00';

      const peLtp = typeof row.pe_ltp === 'number' ? row.pe_ltp.toFixed(2) : '0.00';
      let peOiChg = '--';
      if (typeof row.pe_oi_chg_pct === 'number') {
        const sign = row.pe_oi_chg_pct > 0 ? '+' : '';
        peOiChg = `${sign}${row.pe_oi_chg_pct.toFixed(1)}%`;
      }
      const peVol = typeof row.pe_volume === 'number' ? row.pe_volume.toLocaleString('en-IN') : '--';
      const peIv = typeof row.pe_iv === 'number' ? `${row.pe_iv.toFixed(1)}%` : '--';
      const peDelta = typeof row.pe_delta === 'number' ? row.pe_delta.toFixed(2) : '--';

      const ceSymbol = row.ce_symbol || `${data.index_symbol || this.currentSymbol} ${row.strike} CE`;
      const peSymbol = row.pe_symbol || `${data.index_symbol || this.currentSymbol} ${row.strike} PE`;

      html += `
        <tr style="${rowStyle} border-bottom: 1px solid rgba(48, 54, 61, 0.4);">
          <td style="color: var(--text-muted);">${ceDelta}</td>
          <td style="color: var(--text-muted);">${ceIv}</td>
          <td>${ceVol}</td>
          <td style="color: #22c55e;">${ceOiChg}</td>
          <td style="border-right: 1px solid var(--border-color); color: #22c55e; cursor: pointer;" onclick="OptionChainEngine.sendToTerminal('${ceSymbol}', 'BUY', ${ceLtp})" title="Click to Trade ${ceSymbol}">
            ₹${ceLtp} 🛒
          </td>
          <td style="font-weight: bold; color: #eab308; background: rgba(234, 179, 8, 0.2);">₹${row.strike}</td>
          <td style="border-left: 1px solid var(--border-color); color: #ef4444; cursor: pointer;" onclick="OptionChainEngine.sendToTerminal('${peSymbol}', 'BUY', ${peLtp})" title="Click to Trade ${peSymbol}">
            ₹${peLtp} 🛒
          </td>
          <td style="color: #ef4444;">${peOiChg}</td>
          <td>${peVol}</td>
          <td style="color: var(--text-muted);">${peIv}</td>
          <td style="color: var(--text-muted);">${peDelta}</td>
        </tr>
      `;
    });

    tbody.innerHTML = html;
  },

  sendToTerminal(symbol, side, price) {
    const symbolInputs = document.querySelectorAll('.order-symbol-input');
    const priceInputs = document.querySelectorAll('.order-price-input');

    symbolInputs.forEach(input => input.value = symbol);
    priceInputs.forEach(input => input.value = price);

    // Switch tab to Order Terminal
    const terminalTab = document.querySelector('.nav-tab[data-tab="tab-terminal"]');
    if (terminalTab) terminalTab.click();
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.OptionChainEngine.init();
});
