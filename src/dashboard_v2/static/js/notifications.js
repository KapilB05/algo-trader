/**
 * Notifications Component — Real-Time Trade Alerts & Audit Feed
 * Manages audio chimes, HTML5 desktop notifications, and notification history.
 */

const Notifications = {
  items: [],
  unreadCount: 0,
  currentFilter: 'ALL',
  audioEnabled: true,
  desktopEnabled: true,
  STORAGE_KEY: 'algo_trader_notifications_v1',
  CONFIG_KEY: 'algo_trader_notif_config_v1',

  init() {
    this.loadConfig();
    this.loadHistory();
    this.updateControlsUI();
    this.render();
  },

  loadConfig() {
    try {
      const saved = localStorage.getItem(this.CONFIG_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (typeof parsed.audioEnabled === 'boolean') this.audioEnabled = parsed.audioEnabled;
        if (typeof parsed.desktopEnabled === 'boolean') this.desktopEnabled = parsed.desktopEnabled;
      }
    } catch (err) {
      console.warn('Failed to load notification config:', err);
    }
  },

  saveConfig() {
    try {
      localStorage.setItem(this.CONFIG_KEY, JSON.stringify({
        audioEnabled: this.audioEnabled,
        desktopEnabled: this.desktopEnabled
      }));
    } catch (err) {
      console.warn('Failed to save notification config:', err);
    }
  },

  loadHistory() {
    try {
      const saved = localStorage.getItem(this.STORAGE_KEY);
      if (saved) {
        this.items = JSON.parse(saved);
        this.unreadCount = this.items.filter(item => !item.read).length;
      }
    } catch (err) {
      console.warn('Failed to load notifications history:', err);
      this.items = [];
    }
  },

  saveHistory() {
    try {
      // Retain the last 50 notifications
      const toSave = this.items.slice(0, 50);
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(toSave));
    } catch (err) {
      console.warn('Failed to save notifications history:', err);
    }
  },

  playChime(isLong = true) {
    if (!this.audioEnabled) return;
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
        gain.gain.linearRampToValueAtTime(0.18, startTime + idx * 0.08 + 0.02);
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

  async requestDesktopPermission() {
    if (!('Notification' in window)) return false;
    if (Notification.permission === 'granted') return true;
    if (Notification.permission !== 'denied') {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return false;
  },

  dispatchDesktopNotification(title, body, tag = '') {
    if (!this.desktopEnabled || !('Notification' in window)) return;
    if (Notification.permission !== 'granted') return;

    try {
      const notif = new Notification(title, {
        body,
        icon: '/static/favicon.ico',
        tag: tag || `notif-${Date.now()}`,
        requireInteraction: false,
        silent: true // Audio is handled via our custom synthesizer
      });
      notif.onclick = () => {
        window.focus();
        notif.close();
      };
    } catch (err) {
      console.warn('Failed to dispatch desktop notification:', err);
    }
  },

  add(data) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }) + ' IST';

    const item = {
      id: `ntf_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      type: data.type || 'POSITION', // POSITION, SIGNAL, RISK, SYSTEM
      title: data.title || 'Trading Event',
      details: data.details || '',
      timestamp: now.getTime(),
      timeStr: timeStr,
      read: false,
      isLong: data.isLong !== undefined ? data.isLong : true,
      meta: data.meta || {}
    };

    this.items.unshift(item);
    if (this.items.length > 50) this.items.pop();
    this.unreadCount += 1;
    this.saveHistory();

    // Trigger audio chime
    this.playChime(item.isLong);

    // Trigger desktop notification
    this.dispatchDesktopNotification(item.title, item.details, item.id);

    // Trigger visual toast
    if (typeof showToast === 'function') {
      showToast(item.title, item.type === 'RISK' ? 'warning' : 'success');
    }

    this.render();
  },

  setFilter(filter) {
    this.currentFilter = filter;
    this.render();
  },

  markAllRead() {
    this.items.forEach(item => { item.read = true; });
    this.unreadCount = 0;
    this.saveHistory();
    this.render();
  },

  clearAll() {
    if (this.items.length === 0) return;
    this.items = [];
    this.unreadCount = 0;
    this.saveHistory();
    this.render();
    if (typeof showToast === 'function') {
      showToast('Notifications feed cleared', 'info');
    }
  },

  dismissItem(id) {
    const idx = this.items.findIndex(it => it.id === id);
    if (idx !== -1) {
      if (!this.items[idx].read) {
        this.unreadCount = Math.max(0, this.unreadCount - 1);
      }
      this.items.splice(idx, 1);
      this.saveHistory();
      this.render();
    }
  },

  async toggleSound() {
    this.audioEnabled = !this.audioEnabled;
    this.saveConfig();
    this.updateControlsUI();
    if (this.audioEnabled) {
      this.playChime(true);
      if (typeof showToast === 'function') showToast('🔊 Audio Chimes: Enabled', 'success');
    } else {
      if (typeof showToast === 'function') showToast('🔇 Audio Chimes: Muted', 'info');
    }
  },

  async toggleDesktop() {
    if (!this.desktopEnabled) {
      const granted = await this.requestDesktopPermission();
      if (!granted) {
        if (typeof showToast === 'function') {
          showToast('⚠️ Desktop notifications blocked in browser settings', 'warning');
        }
        return;
      }
      this.desktopEnabled = true;
      if (typeof showToast === 'function') showToast('🔔 Windows Desktop Alerts: Enabled', 'success');
    } else {
      this.desktopEnabled = false;
      if (typeof showToast === 'function') showToast('🔕 Windows Desktop Alerts: Disabled', 'info');
    }
    this.saveConfig();
    this.updateControlsUI();
  },

  testAlert() {
    this.add({
      type: 'POSITION',
      title: '🟢 POSITION ENTERED: BUY 15x BANKNIFTY FUT @ ₹52,476.22',
      details: 'SL: ₹51,977.95 | Target: ₹53,315.84 | Strategy: ML Ensemble (High Confluence)',
      isLong: true,
      meta: { symbol: 'BANKNIFTY FUT', side: 'BUY', qty: 15, price: 52476.22 }
    });
  },

  toggleDrawer(open) {
    const drawer = document.getElementById('notifications-drawer');
    if (!drawer) return;
    if (open) {
      drawer.classList.add('open');
      this.markAllRead();
    } else {
      drawer.classList.remove('open');
    }
  },

  updateControlsUI() {
    const soundText = this.audioEnabled ? '🔊 Sound: ON' : '🔇 Sound: OFF';
    const desktopText = this.desktopEnabled ? '🖥️ Desktop: ON' : '🖥️ Desktop: OFF';

    ['btn-notif-sound-drawer', 'btn-notif-sound-pane'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) {
        btn.textContent = soundText;
        btn.classList.toggle('active', this.audioEnabled);
      }
    });

    ['btn-notif-desk-drawer', 'btn-notif-desk-pane'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) {
        btn.textContent = desktopText;
        btn.classList.toggle('active', this.desktopEnabled);
      }
    });
  },

  getFilteredItems() {
    if (this.currentFilter === 'ALL') return this.items;
    return this.items.filter(item => item.type === this.currentFilter);
  },

  getBadgeMeta(type, isLong) {
    if (type === 'POSITION') {
      const sideText = isLong ? 'LONG ENTRY' : 'SHORT ENTRY';
      const sideClass = isLong ? 'text-green' : 'text-red';
      return { tag: sideText, colorClass: sideClass, icon: isLong ? '📈' : '📉', borderClass: 'pos-entry' };
    }
    if (type === 'SIGNAL') {
      return { tag: 'RADAR SIGNAL', colorClass: 'text-cyan', icon: '⚡', borderClass: 'signal' };
    }
    if (type === 'RISK') {
      return { tag: 'RISK / SL', colorClass: 'text-amber', icon: '🛡️', borderClass: 'risk' };
    }
    return { tag: 'SYSTEM', colorClass: 'text-purple', icon: 'ℹ️', borderClass: 'system' };
  },

  renderNotificationCard(item) {
    const meta = this.getBadgeMeta(item.type, item.isLong);
    return `
      <div class="notif-card ${meta.borderClass} ${item.read ? 'read' : 'unread'}" id="${item.id}">
        <div class="notif-card-header">
          <div class="notif-tag-wrap">
            <span class="notif-icon">${meta.icon}</span>
            <span class="notif-tag ${meta.colorClass}">${meta.tag}</span>
            <span class="notif-time">${item.timeStr}</span>
          </div>
          <button class="btn-dismiss-notif" onclick="Notifications.dismissItem('${item.id}')" title="Dismiss">✕</button>
        </div>
        <div class="notif-title">${item.title}</div>
        ${item.details ? `<div class="notif-details">${item.details}</div>` : ''}
      </div>
    `;
  },

  renderEmptyState() {
    return `
      <div class="notif-empty-card">
        <div class="notif-empty-icon">🛡️</div>
        <div class="notif-empty-title">NO NOTIFICATIONS YET</div>
        <div class="notif-empty-desc">
          Position entries, automated trailing stop adjustments, and execution alerts will appear here in real time.
        </div>
        <button class="hdr-btn cyan" onclick="Notifications.testAlert()" style="margin-top: 14px; padding: 6px 14px;">
          ⚡ Send Test Alert
        </button>
      </div>
    `;
  },

  renderFeedHTML() {
    const filtered = this.getFilteredItems();
    if (filtered.length === 0) {
      return this.renderEmptyState();
    }
    return filtered.map(item => this.renderNotificationCard(item)).join('');
  },

  render() {
    // 1. Update unread count badges
    const unreadPill = document.getElementById('header-notif-badge');
    if (unreadPill) {
      unreadPill.textContent = this.unreadCount;
      unreadPill.style.display = this.unreadCount > 0 ? 'inline-block' : 'none';
    }

    const cockpitCount = document.getElementById('cockpit-notif-count');
    if (cockpitCount) {
      cockpitCount.textContent = this.items.length;
    }

    // 2. Render Feed in Slide-over Drawer
    const drawerFeed = document.getElementById('notif-drawer-feed-list');
    if (drawerFeed) {
      drawerFeed.innerHTML = this.renderFeedHTML();
    }

    // 3. Render Feed in Cockpit Pane
    const paneFeed = document.getElementById('notif-pane-feed-list');
    if (paneFeed) {
      paneFeed.innerHTML = this.renderFeedHTML();
    }

    // 4. Update filter pill counts & active states
    const counts = {
      ALL: this.items.length,
      POSITION: this.items.filter(i => i.type === 'POSITION').length,
      SIGNAL: this.items.filter(i => i.type === 'SIGNAL').length,
      RISK: this.items.filter(i => i.type === 'RISK').length
    };

    ['ALL', 'POSITION', 'SIGNAL', 'RISK'].forEach(f => {
      document.querySelectorAll(`[data-notif-filter="${f}"]`).forEach(btn => {
        btn.classList.toggle('active', this.currentFilter === f);
        const countSpan = btn.querySelector('.filter-count');
        if (countSpan) countSpan.textContent = counts[f];
      });
    });
  }
};

window.Notifications = Notifications;
document.addEventListener('DOMContentLoaded', () => Notifications.init());
