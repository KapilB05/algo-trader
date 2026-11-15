/**
 * Interactive Order Terminal Component (src/dashboard_v2/static/js/terminal.js)
 * 
 * Features:
 * - One-click execution for BUY (LONG) and SELL (SHORT).
 * - Product Type selection (MIS Intraday vs NRML Carryforward).
 * - Order Type selection (MARKET vs LIMIT vs STOP_LOSS).
 * - Automatic price field disable/enable based on order type.
 * - Loading & disabled button state while processing HTTP requests.
 * - In-panel visual feedback toast (no intrusive browser alert popups).
 */

const TerminalEngine = {
  init() {
    this.bindControls();
  },

  bindControls() {
    const buyBtns = document.querySelectorAll('.btn-order-buy');
    const sellBtns = document.querySelectorAll('.btn-order-sell');
    const typeSelects = document.querySelectorAll('.order-type-select');

    buyBtns.forEach(btn => btn.addEventListener('click', () => this.sendOrder('BUY', btn)));
    sellBtns.forEach(btn => btn.addEventListener('click', () => this.sendOrder('SELL', btn)));

    typeSelects.forEach(sel => {
      sel.addEventListener('change', (e) => {
        const parent = sel.closest('.order-form-container') || document;
        const priceInput = parent.querySelector('.order-price-input');
        if (priceInput) {
          if (e.target.value === 'MARKET') {
            priceInput.disabled = true;
            priceInput.placeholder = 'MKT';
          } else {
            priceInput.disabled = false;
            priceInput.placeholder = 'Limit Price';
          }
        }
      });
    });
  },

  showFeedback(container, message, isSuccess = true) {
    let feedbackEl = container.querySelector('.order-feedback-toast');
    if (!feedbackEl) {
      feedbackEl = document.createElement('div');
      feedbackEl.className = 'order-feedback-toast';
      container.appendChild(feedbackEl);
    }

    feedbackEl.style.cssText = `
      margin-top: 12px; padding: 10px 14px; border-radius: 6px; font-family: var(--font-mono);
      font-size: 12px; font-weight: 500; display: block; border: 1px solid ${isSuccess ? '#22c55e' : '#ef4444'};
      background: ${isSuccess ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)'};
      color: ${isSuccess ? '#22c55e' : '#ef4444'};
    `;
    feedbackEl.textContent = message;

    setTimeout(() => {
      if (feedbackEl) feedbackEl.style.display = 'none';
    }, 6000);
  },

  _validateOrderParams(symbol, quantity, orderType, price, parent) {
    if (!symbol) {
      this.showFeedback(parent, '⚠️ Symbol cannot be empty.', false);
      return false;
    }
    if (Number.isNaN(quantity) || quantity <= 0) {
      this.showFeedback(parent, '⚠️ Quantity must be greater than 0.', false);
      return false;
    }
    if (orderType === 'LIMIT' && (Number.isNaN(price) || price <= 0)) {
      this.showFeedback(parent, '⚠️ Limit price must be greater than zero for LIMIT orders.', false);
      return false;
    }
    return true;
  },

  async sendOrder(side, clickedBtn) {
    const parent = clickedBtn.closest('.order-form-container') || document;

    const symbolInput = parent.querySelector('.order-symbol-input');
    const qtyInput = parent.querySelector('.order-qty-input');
    const typeSelect = parent.querySelector('.order-type-select');
    const productSelect = parent.querySelector('.order-product-select');
    const priceInput = parent.querySelector('.order-price-input');

    const symbol = (symbolInput ? symbolInput.value.trim() : '') || 'BANKNIFTY FUT';
    const quantity = Number.parseInt(qtyInput ? qtyInput.value : '15', 10);
    const orderType = (typeSelect ? typeSelect.value : 'MARKET') || 'MARKET';
    const productType = (productSelect ? productSelect.value : 'MIS') || 'MIS';
    const price = Number.parseFloat(priceInput ? priceInput.value : '0.0') || 0.0;

    if (!this._validateOrderParams(symbol, quantity, orderType, price, parent)) return;

    // Disable button state
    const originalText = clickedBtn.innerHTML;
    clickedBtn.disabled = true;
    clickedBtn.innerHTML = `⏳ Sending ${side}...`;

    try {
      const res = await fetch('/api/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          side: side,
          quantity: quantity,
          order_type: orderType,
          product_type: productType,
          price: price,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        const modeTag = data.mode === 'PAPER' ? '🛡️ [PAPER SIMULATION]' : '⚡ [LIVE BROKER DISPATCH]';
        const fillMsg = data.fill_price ? ` (Fill: ₹${data.fill_price.toFixed(2)})` : '';
        this.showFeedback(parent, `${modeTag} ${side} ${quantity} ${symbol} @ ${orderType}${fillMsg} [ID: ${data.order_id}]`, true);
        if (window.Positions && typeof window.Positions.fetchAndRenderAll === 'function') {
          window.Positions.fetchAndRenderAll();
        }
      } else {
        this.showFeedback(parent, `❌ Order Rejected: ${data.detail || 'Risk/Broker check failed'}`, false);
      }
    } catch (err) {
      console.error('Order dispatch error:', err);
      this.showFeedback(parent, '🚨 Network error while sending order.', false);
    } finally {
      clickedBtn.disabled = false;
      clickedBtn.innerHTML = originalText;
    }
  }
};

document.addEventListener('DOMContentLoaded', () => TerminalEngine.init());
