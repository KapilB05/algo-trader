/**
 * Stage 6: Segregated Watchlist Component (watchlist.js)
 * Separates universe into Indices, Futures, Stocks, and Options with
 * live market quotes, research quality scores, news sentiment tags, and 1-click trade/chart actions.
 */

window.SegregatedWatchlist = {
  currentCategory: 'all',
  segregatedData: null,
  isInitialized: false,

  init() {
    if (this.isInitialized) return;
    this.isInitialized = true;
    this.bindDelegatedEvents();
    this.fetchData();
    // Refresh every 3 seconds
    setInterval(() => this.fetchData(), 5000);
  },

  bindDelegatedEvents() {
    // 1. Category Switcher (Delegated)
    document.addEventListener('click', (e) => {
      const catBtn = e.target.closest('.wl-category-btn');
      if (catBtn) {
        const cat = catBtn.dataset.category;
        if (cat) this.setCategory(cat);
        return;
      }

      // 2. Trade Button (Delegated)
      const tradeBtn = e.target.closest('.btn-wl-trade');
      if (tradeBtn) {
        const sym = tradeBtn.dataset.symbol;
        const price = Number.parseFloat(tradeBtn.dataset.price || '0');
        this.loadIntoTerminal(sym, price);
        return;
      }

      // 3. Chart Button (Delegated)
      const chartBtn = e.target.closest('.btn-wl-chart');
      if (chartBtn) {
        const sym = chartBtn.dataset.symbol;
        this.loadIntoChart(sym);
      }
    });
  },

  setCategory(cat) {
    this.currentCategory = cat;
    document.querySelectorAll('.wl-category-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.category === cat);
    });
    this.render();
  },

  async fetchData() {
    try {
      const res = await fetch('/api/watchlist/segregated');
      if (!res.ok) return;
      const data = await res.json();
      if (data.status === 'SUCCESS' && data.categories) {
        this.segregatedData = data.categories;
        this.render();
      }
    } catch (err) {
      console.warn("Watchlist fetch error:", err);
    }
  },

  render(legacyItems = null) {
    if (!this.segregatedData) {
      if (Array.isArray(legacyItems)) {
        this.renderFlat(legacyItems);
      }
      return;
    }

    let itemsToRender = [];
    if (this.currentCategory === 'all') {
      itemsToRender = [
        ...(this.segregatedData.indices || []),
        ...(this.segregatedData.futures || []),
        ...(this.segregatedData.stocks || []),
        ...(this.segregatedData.options || []),
      ];
    } else if (this.currentCategory === 'indices') {
      itemsToRender = this.segregatedData.indices || [];
    } else if (this.currentCategory === 'futures') {
      itemsToRender = this.segregatedData.futures || [];
    } else if (this.currentCategory === 'stocks') {
      itemsToRender = this.segregatedData.stocks || [];
    } else if (this.currentCategory === 'options') {
      itemsToRender = this.segregatedData.options || [];
    } else {
      itemsToRender = this.segregatedData[this.currentCategory] || [];
    }

    const tbodyOverview = document.getElementById('watchlist-tbody');
    const tbodyTab = document.getElementById('tab-watchlist-tbody');

    const htmlContent = itemsToRender.length === 0
      ? `<tr><td colspan="7" style="text-align:center; color: var(--text-muted); padding: 16px;">No instruments found in ${this.currentCategory} category.</td></tr>`
      : itemsToRender.map(item => this.createRowHtml(item)).join('');

    if (tbodyOverview) tbodyOverview.innerHTML = htmlContent;
    if (tbodyTab) tbodyTab.innerHTML = htmlContent;
  },

  renderFlat(items) {
    const tbodyOverview = document.getElementById('watchlist-tbody');
    const tbodyTab = document.getElementById('tab-watchlist-tbody');
    const html = items.map(item => {
      const chgClass = item.change_pct >= 0 ? 'text-green' : 'text-red';
      const chgSign = item.change_pct >= 0 ? '+' : '';
      return `
        <tr>
          <td style="font-weight: 700; color: #fff;">${item.symbol}</td>
          <td style="font-weight: 700; font-family: var(--font-mono);">₹${Number(item.ltp || item.price || item.last_price || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 })}</td>
          <td class="${chgClass}" style="font-family: var(--font-mono); font-weight: 700;">${chgSign}${Number(item.change_pct || 0).toFixed(2)}%</td>
          <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${item.volume || '--'}</td>
          <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${item.oi || '--'}</td>
          <td><span class="badge ${item.change_pct >= 0 ? 'badge-ready' : 'badge-not-ready'}">${item.trend || 'NEUTRAL'}</span></td>
          <td>
            <div style="display: flex; gap: 4px;">
              <button class="btn btn-secondary btn-wl-chart" data-symbol="${item.symbol}" title="Chart" style="font-size: 10px; padding: 2px 6px;">📈</button>
              <button class="btn btn-secondary btn-wl-trade" data-symbol="${item.symbol}" data-price="${item.ltp || item.price || 0}" title="Trade" style="font-size: 10px; padding: 2px 6px; color: #22c55e;">⚡ Trade</button>
            </div>
          </td>
        </tr>
      `;
    }).join('');

    if (tbodyOverview) tbodyOverview.innerHTML = html;
    if (tbodyTab) tbodyTab.innerHTML = html;
  },

  createRowHtml(item) {
    const chgClass = item.change_pct >= 0 ? 'text-green' : 'text-red';
    const chgSign = item.change_pct >= 0 ? '+' : '';
    const ltpFormatted = Number(item.ltp || item.price || item.last_price || 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const volFormatted = item.volume ? Number(item.volume).toLocaleString('en-IN') : '--';
    const oiFormatted = item.oi ? Number(item.oi).toLocaleString('en-IN') : '--';

    // Sentiment badge style
    let sentBadge = `<span class="badge" style="background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); font-size: 10px;">⚪ NEUTRAL</span>`;
    if (item.sentiment === 'POSITIVE') {
      sentBadge = `<span class="badge" style="background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); font-size: 10px;">🟢 POSITIVE</span>`;
    } else if (item.sentiment === 'NEGATIVE') {
      sentBadge = `<span class="badge" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); font-size: 10px;">🔴 NEGATIVE</span>`;
    }

    // Category tag
    const catUpper = (item.category || 'STOCK').toUpperCase();
    let catColor = '#22c55e'; // default stock
    if (item.category === 'options') catColor = '#c084fc';
    else if (item.category === 'futures') catColor = '#38bdf8';
    else if (item.category === 'indices') catColor = '#f59e0b';

    return `
      <tr>
        <td>
          <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 9px; font-weight: 700; color: ${catColor}; background: rgba(255,255,255,0.05); padding: 1px 4px; border-radius: 3px; font-family: var(--font-mono);">${catUpper}</span>
            <strong style="color: #fff;">${item.symbol}</strong>
          </div>
        </td>
        <td style="font-family: var(--font-mono); font-weight: 700;">₹${ltpFormatted}</td>
        <td class="${chgClass}" style="font-family: var(--font-mono); font-weight: 700;">${chgSign}${Number(item.change_pct || 0).toFixed(2)}%</td>
        <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${volFormatted}</td>
        <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${oiFormatted}</td>
        <td>${sentBadge}</td>
        <td>
          <div style="display: flex; gap: 4px;">
            <button class="btn btn-secondary btn-wl-chart" data-symbol="${item.symbol}" title="Load Chart" style="font-size: 10px; padding: 2px 6px;">📈</button>
            <button class="btn btn-secondary btn-wl-trade" data-symbol="${item.symbol}" data-price="${item.ltp || item.price || 0}" title="Load into Order Terminal" style="font-size: 10px; padding: 2px 6px; color: #22c55e; border-color: rgba(34, 197, 94, 0.3);">⚡ Trade</button>
          </div>
        </td>
      </tr>
    `;
  },

  loadIntoTerminal(symbol, price) {
    const symbolInputs = document.querySelectorAll('.order-symbol-input');
    symbolInputs.forEach(input => input.value = symbol);

    const priceInputs = document.querySelectorAll('.order-price-input');
    priceInputs.forEach(input => {
      if (price > 0) input.value = price.toFixed(2);
    });

    if (window.LoginGateway) {
      window.LoginGateway.showToast(`Loaded ${symbol} into Order Terminal @ ₹${price.toFixed(2)}`, 'success');
    }
  },

  loadIntoChart(symbol) {
    const chartTabBtn = document.querySelector('.nav-tab[data-tab="tab-chart"]');
    if (chartTabBtn) chartTabBtn.click();

    if (window.LightweightChartEngine) {
      window.LightweightChartEngine.loadSymbol(symbol);
    }
  }
};

// Legacy compatibility wrapper
window.Watchlist = {
  render(items) {
    window.SegregatedWatchlist.render(items);
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => window.SegregatedWatchlist.init());
} else {
  window.SegregatedWatchlist.init();
}
