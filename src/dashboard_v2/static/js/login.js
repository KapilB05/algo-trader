/**
 * Angel One SmartAPI Gateway & Full-Screen Authentication Controller (login.js)
 * Features 1-Click Auto-Login, Credential Setup (.env writer), and Manual 6-Digit TOTP fallback.
 */

window.LoginGateway = {
  activeTab: 'auto',
  authStatus: null,

  init() {
    this.bindEvents();
    this.checkStatus();
  },

  bindEvents() {
    // 1. Tab Switching
    const tabBtns = document.querySelectorAll('.gw-tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tabKey = e.currentTarget.dataset.tab;
        this.switchTab(tabKey);
      });
    });

    // 2. Link to switch to setup account
    const linkSetup = document.getElementById('gw-link-setup');
    if (linkSetup) {
      linkSetup.addEventListener('click', (e) => {
        e.preventDefault();
        this.switchTab('setup');
      });
    }

    // 3. Auto Login Button (Tab 1)
    const btnAuto = document.getElementById('btn-gw-auto-login');
    if (btnAuto) {
      btnAuto.addEventListener('click', () => this.handleAutoLogin());
    }

    // 4. Save Credentials Button (Tab 2)
    const btnSave = document.getElementById('btn-gw-save-creds');
    if (btnSave) {
      btnSave.addEventListener('click', () => this.handleSaveCredentials());
    }

    // 5. Manual TOTP Button (Tab 3)
    const btnManual = document.getElementById('btn-gw-manual-login');
    if (btnManual) {
      btnManual.addEventListener('click', () => this.handleManualLogin());
    }

    // 6. Header Logout Buttons
    const headerLogout = document.getElementById('btn-header-logout');
    if (headerLogout) {
      headerLogout.addEventListener('click', () => this.handleLogout());
    }

    // 7. 6-Digit Manual TOTP Inputs Auto-Advance
    const digits = document.querySelectorAll('.gw-digit-input');
    digits.forEach((input, idx) => {
      input.addEventListener('input', (e) => {
        const val = e.target.value;
        if (val.length >= 1) {
          e.target.value = val.slice(-1);
          if (idx < digits.length - 1) digits[idx + 1].focus();
        }
      });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && !input.value && idx > 0) {
          digits[idx - 1].focus();
        }
      });

      input.addEventListener('paste', (e) => {
        e.preventDefault();
        const text = (e.clipboardData || window.clipboardData).getData('text').trim();
        if (/^\d{6}$/.test(text)) {
          text.split('').forEach((c, i) => {
            if (digits[i]) digits[i].value = c;
          });
          digits[5].focus();
        }
      });
    });
  },

  switchTab(tabKey) {
    this.activeTab = tabKey;
    document.querySelectorAll('.gw-tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabKey);
    });
    document.querySelectorAll('.gw-tab-panel').forEach(p => {
      p.style.display = p.id === `gw-panel-${tabKey}` ? 'block' : 'none';
    });
    if (tabKey === 'manual') {
      const firstDigit = document.querySelector('.gw-digit-input');
      if (firstDigit) firstDigit.focus();
    }
  },

  async checkStatus() {
    try {
      const res = await fetch('/api/auth/status');
      if (!res.ok) return;
      const data = await res.json();
      this.authStatus = data;

      if (data.authenticated && !data.is_expired) {
        this.handleAuthenticatedState(data);
      } else {
        this.handleUnauthenticatedState(data);
      }
    } catch (err) {
      console.warn("Error checking auth status:", err);
    }
  },

  handleAuthenticatedState(data) {
    const overlay = document.getElementById('gateway-overlay');
    const headerUserChip = document.getElementById('header-user-chip');
    const headerUserName = document.getElementById('header-user-name');

    if (overlay) overlay.style.display = 'none';
    if (headerUserChip) headerUserChip.style.display = 'inline-flex';
    if (headerUserName) headerUserName.textContent = data.client_id;
  },

  handleUnauthenticatedState(data) {
    const overlay = document.getElementById('gateway-overlay');
    const headerUserChip = document.getElementById('header-user-chip');
    if (overlay) overlay.style.display = 'none';
    if (headerUserChip) headerUserChip.style.display = 'none';

    const clientDisplay = document.getElementById('gw-auto-client-id');
    const autoStatusChip = document.getElementById('gw-auto-status-chip');
    const btnAuto = document.getElementById('btn-gw-auto-login');

    if (clientDisplay) clientDisplay.textContent = data.client_id || 'Not Configured';

    if (data.is_configured && data.has_secret) {
      this.switchTab('auto');
      if (autoStatusChip) {
        autoStatusChip.innerHTML = `🟢 Credentials Ready in .env`;
        autoStatusChip.style.borderColor = 'rgba(34, 197, 94, 0.4)';
        autoStatusChip.style.color = '#22c55e';
      }
      if (btnAuto) btnAuto.disabled = false;
    } else {
      this.switchTab('setup');
      if (autoStatusChip) {
        autoStatusChip.innerHTML = `⚠️ Credentials Missing in .env`;
        autoStatusChip.style.borderColor = 'rgba(234, 179, 8, 0.4)';
        autoStatusChip.style.color = '#eab308';
      }
    }
  },

  async handleAutoLogin() {
    const btn = document.getElementById('btn-gw-auto-login');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `⏳ Computing TOTP & Logging in...`;
    }

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'auto' }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Auto-login failed.');

      this.showToast(`✅ ${data.message || 'Login Successful!'}`, 'success');
      this.unlockTerminal(data.client_id);
    } catch (err) {
      this.showToast(`❌ ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `🚀 Auto-Generate TOTP & Launch Terminal`;
      }
    }
  },

  async handleSaveCredentials() {
    const apiKey = document.getElementById('gw-setup-apikey')?.value.trim();
    const clientId = document.getElementById('gw-setup-clientid')?.value.trim();
    const pin = document.getElementById('gw-setup-pin')?.value.trim();
    const secret = document.getElementById('gw-setup-secret')?.value.trim();
    const btn = document.getElementById('btn-gw-save-creds');

    if (!apiKey || !clientId || !pin || !secret) {
      this.showToast('Please fill all 4 fields (API Key, Client ID, PIN, TOTP Secret).', 'error');
      return;
    }

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `⏳ Saving & Authenticating...`;
    }

    try {
      const res = await fetch('/api/auth/credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: apiKey,
          client_id: clientId,
          pin: pin,
          totp_secret: secret,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to save credentials.');

      this.showToast(`✅ ${data.message}`, 'success');
      this.unlockTerminal(data.client_id);
    } catch (err) {
      this.showToast(`❌ ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `💾 Save Credentials & Auto-Login`;
      }
    }
  },

  async handleManualLogin() {
    const clientId = document.getElementById('gw-manual-clientid')?.value.trim();
    const pin = document.getElementById('gw-manual-pin')?.value.trim();
    const digits = document.querySelectorAll('.gw-digit-input');
    let totp = '';
    digits.forEach(d => totp += d.value);
    totp = totp.trim();
    const btn = document.getElementById('btn-gw-manual-login');

    if (!clientId || !pin) {
      this.showToast('Please enter your Client ID and 4-Digit PIN.', 'error');
      return;
    }
    if (totp.length !== 6) {
      this.showToast('Please enter the full 6-digit Google Authenticator code.', 'error');
      return;
    }

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `⏳ Verifying Code...`;
    }

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'manual',
          client_id: clientId,
          pin: pin,
          totp: totp,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Manual login failed.');

      this.showToast(`✅ ${data.message}`, 'success');
      this.unlockTerminal(data.client_id);
    } catch (err) {
      this.showToast(`❌ ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `🔑 Verify Code & Login`;
      }
    }
  },

  async handleLogout() {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
      this.showToast('Broker session disconnected.', 'info');
      await this.checkStatus();
    } catch (err) {
      console.error('Logout failed:', err);
      this.showToast('Logout failed.', 'error');
    }
  },

  unlockTerminal(clientId) {
    const overlay = document.getElementById('gateway-overlay');
    if (overlay) {
      overlay.style.opacity = '0';
      overlay.style.transition = 'opacity 0.4s ease';
      setTimeout(() => {
        overlay.style.display = 'none';
        overlay.style.opacity = '1';
      }, 400);
    }

    const headerUserChip = document.getElementById('header-user-chip');
    const headerUserName = document.getElementById('header-user-name');
    if (headerUserChip) headerUserChip.style.display = 'inline-flex';
    if (headerUserName) headerUserName.textContent = clientId || 'Active';

    // Refresh active tab data if available
    if (window.OptionChainEngine) window.OptionChainEngine.fetchData(window.OptionChainEngine.currentSymbol);
    if (window.BrokerAuthEngine) window.BrokerAuthEngine.fetchStatus();
  },

  showToast(message, type = 'info') {
    let container = document.getElementById('gw-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'gw-toast-container';
      container.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999999;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
      `;
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    let bgColor = '#1e293b';
    let borderColor = '#64748b';
    if (type === 'success') {
      bgColor = '#166534';
      borderColor = '#22c55e';
    } else if (type === 'error') {
      bgColor = '#991b1b';
      borderColor = '#ef4444';
    }

    toast.style.cssText = `
      background: ${bgColor};
      border: 1px solid ${borderColor};
      color: #fff;
      padding: 12px 18px;
      border-radius: 6px;
      font-family: var(--font-mono, monospace);
      font-size: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      animation: fadeIn 0.3s ease;
      pointer-events: auto;
      max-width: 380px;
    `;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.4s ease';
      setTimeout(() => toast.remove(), 400);
    }, 4000);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.LoginGateway.init();
});
