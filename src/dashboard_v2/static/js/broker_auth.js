/**
 * Angel One Broker Authentication & Account Sync Module (broker_auth.js)
 * Supports 1-click login, multi-user custom Client ID + PIN + Google Authenticator TOTP,
 * token lifecycle countdown, and real-time live positions/funds sync.
 */

window.BrokerAuthEngine = {
  activeSession: null,
  totpMode: 'auto', // 'auto' or 'manual'

  init() {
    this.bindEvents();
    this.fetchStatus();
    // Periodically refresh broker session status every 10s
    setInterval(() => this.fetchStatus(), 10000);
  },

  bindEvents() {
    // 1. Header Trigger Button
    const triggerBtn = document.getElementById('btn-broker-session');
    if (triggerBtn) {
      triggerBtn.addEventListener('click', () => this.openModal());
    }

    // 2. Modal Close Buttons & Backdrop
    const closeBtn = document.getElementById('broker-modal-close');
    const modalBackdrop = document.getElementById('broker-modal-backdrop');
    if (closeBtn) closeBtn.addEventListener('click', () => this.closeModal());
    if (modalBackdrop) {
      modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) this.closeModal();
      });
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeModal();
    });

    // 3. TOTP Method Tabs
    const tabAuto = document.getElementById('totp-tab-auto');
    const tabManual = document.getElementById('totp-tab-manual');
    const panelAuto = document.getElementById('totp-panel-auto');
    const panelManual = document.getElementById('totp-panel-manual');

    if (tabAuto && tabManual) {
      tabAuto.addEventListener('click', () => {
        this.totpMode = 'auto';
        tabAuto.classList.add('active');
        tabManual.classList.remove('active');
        if (panelAuto) panelAuto.style.display = 'block';
        if (panelManual) panelManual.style.display = 'none';
      });

      tabManual.addEventListener('click', () => {
        this.totpMode = 'manual';
        tabManual.classList.add('active');
        tabAuto.classList.remove('active');
        if (panelAuto) panelAuto.style.display = 'none';
        if (panelManual) panelManual.style.display = 'block';
        const firstDigit = document.querySelector('.totp-digit-input');
        if (firstDigit) firstDigit.focus();
      });
    }

    // 4. 6-Digit Auto-Advancing Input Behavior
    const digitInputs = document.querySelectorAll('.totp-digit-input');
    digitInputs.forEach((input, index) => {
      input.addEventListener('input', (e) => {
        const val = e.target.value;
        if (val.length >= 1) {
          e.target.value = val.slice(-1);
          if (index < digitInputs.length - 1) {
            digitInputs[index + 1].focus();
          }
        }
      });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && !input.value && index > 0) {
          digitInputs[index - 1].focus();
        }
      });

      // Handle full code paste (e.g. "123456")
      input.addEventListener('paste', (e) => {
        e.preventDefault();
        const pasteData = (e.clipboardData || window.clipboardData).getData('text').trim();
        if (/^\d{6}$/.test(pasteData)) {
          pasteData.split('').forEach((char, i) => {
            if (digitInputs[i]) digitInputs[i].value = char;
          });
          digitInputs[5].focus();
        }
      });
    });

    // 5. Submit Form
    const submitBtn = document.getElementById('btn-broker-submit');
    if (submitBtn) {
      submitBtn.addEventListener('click', () => this.handleLogin());
    }

    // 6. Logout Button
    const logoutBtn = document.getElementById('btn-broker-logout');
    if (logoutBtn) {
      logoutBtn.addEventListener('click', () => this.handleLogout());
    }
  },

  openModal() {
    const modal = document.getElementById('broker-login-modal');
    if (modal) {
      modal.style.display = 'flex';
      modal.style.opacity = '1';
    }
  },

  closeModal() {
    const modal = document.getElementById('broker-login-modal');
    if (modal) {
      modal.style.display = 'none';
    }
  },

  getEnteredTOTP() {
    if (this.totpMode === 'auto') {
      return '';
    }
    const digitInputs = document.querySelectorAll('.totp-digit-input');
    let code = '';
    digitInputs.forEach(input => code += input.value);
    return code.trim();
  },

  async fetchStatus() {
    try {
      const res = await fetch('/api/broker/status');
      if (!res.ok) return;
      const data = await res.json();
      this.activeSession = data;
      this.renderStatus(data);
    } catch (err) {
      console.warn("Could not fetch broker status:", err);
    }
  },

  setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  },

  setStyle(id, styles) {
    const el = document.getElementById(id);
    if (el) Object.assign(el.style, styles);
  },

  renderStatus(data) {
    const isAuth = data.authenticated && !data.is_expired;

    this.setText('broker-header-text', isAuth ? `🟢 Angel One: ${data.client_id}` : `🔑 Connect Angel One`);
    this.setText('broker-card-client-id', data.client_id || 'None');
    this.setText('broker-card-margin', `₹${(data.funds || 0.0).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`);
    this.setText('broker-card-status-text', isAuth ? 'Session Active (Live)' : 'Disconnected / Token Expired');

    this.setStyle('btn-broker-session', {
      borderColor: isAuth ? 'rgba(34, 197, 94, 0.4)' : 'rgba(234, 179, 8, 0.4)',
      color: isAuth ? '#22c55e' : '#eab308'
    });

    this.setStyle('broker-card-dot', {
      background: isAuth ? '#22c55e' : '#ef4444',
      boxShadow: isAuth ? '0 0 8px #22c55e' : 'none'
    });

    this.setStyle('broker-card-status-text', {
      color: isAuth ? '#22c55e' : '#f87171'
    });

    this.setStyle('btn-broker-logout', {
      display: isAuth ? 'inline-flex' : 'none'
    });
  },

  async handleLogin() {
    const clientIdInput = document.getElementById('broker-client-id-input');
    const pinInput = document.getElementById('broker-pin-input');
    const apiKeyInput = document.getElementById('broker-api-key-input');
    const submitBtn = document.getElementById('btn-broker-submit');

    const clientId = clientIdInput ? clientIdInput.value.trim() : '';
    const pin = pinInput ? pinInput.value.trim() : '';
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
    const totpCode = this.getEnteredTOTP();
    const isAuto = (this.totpMode === 'auto');

    if (!clientId) {
      this.showToast('Please enter your Angel One Client ID.', 'error');
      return;
    }
    if (!pin) {
      this.showToast('Please enter your 4-digit Trading PIN.', 'error');
      return;
    }
    if (!isAuto && totpCode.length !== 6) {
      this.showToast('Please enter the complete 6-digit TOTP code.', 'error');
      return;
    }

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `⏳ Authenticating...`;
    }

    try {
      const payload = {
        mode: isAuto ? 'auto' : 'manual',
        client_id: clientId,
        pin: pin,
        totp: totpCode,
        api_key: apiKey || null,
        auto_totp: isAuto,
      };

      const res = await fetch('/api/broker/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Login failed. Please check your credentials.');
      }

      this.showToast(`✅ ${data.message || 'Login Successful!'} Available Funds: ₹${(data.funds || 0).toLocaleString('en-IN')}`, 'success');
      await this.fetchStatus();

      // Clear manual digit inputs
      document.querySelectorAll('.totp-digit-input').forEach(i => i.value = '');

      setTimeout(() => {
        this.closeModal();
      }, 1200);

    } catch (err) {
      this.showToast(`❌ ${err.message}`, 'error');
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `🚀 Connect & Sync Account`;
      }
    }
  },

  async handleLogout() {
    try {
      const res = await fetch('/api/broker/logout', { method: 'POST' });
      await res.json();
      this.showToast('Session logged out successfully.', 'info');
      await this.fetchStatus();
    } catch (err) {
      console.error('Logout failed:', err);
      this.showToast('Failed to logout session.', 'error');
    }
  },

  showToast(message, type = 'info') {
    let toastContainer = document.getElementById('broker-toast-container');
    if (!toastContainer) {
      toastContainer = document.createElement('div');
      toastContainer.id = 'broker-toast-container';
      toastContainer.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
      `;
      document.body.appendChild(toastContainer);
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
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.4s ease';
      setTimeout(() => toast.remove(), 400);
    }, 4000);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.BrokerAuthEngine.init();
});
