/* Arbitrage Monitor — box-spread + pairs paper telemetry (paper-only, no orders).
 * Polls GET /api/arb/opportunities every 15s; header SCAN triggers
 * POST /api/arb/scan-now via Arb.scanNow(). All figures shown NET of costs.
 * Rendering uses textContent (no innerHTML) — server strings never become HTML.
 */
(function () {
  'use strict';

  var POLL_MS = 15000;
  var STALE_S = 60;

  function esc(v) {
    return v === null || v === undefined ? '' : String(v);
  }

  function el(tag, text, cls) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    n.textContent = esc(text);
    return n;
  }

  function toast(msg, kind) {
    if (typeof showToast === 'function') showToast(msg, kind || 'success');
  }

  function parseTs(s) {
    try {
      var m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(String(s || ''));
      if (!m) return null;
      return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
    } catch (e) { return null; }
  }

  function fmtINR(v) {
    var n = Number(v || 0);
    var sign = n < 0 ? '-' : (n > 0 ? '+' : '');
    return sign + '₹' + Math.abs(n).toLocaleString('en-IN', { maximumFractionDigits: 2 });
  }

  function setFreshness(node, ts) {
    var t = parseTs(ts);
    var age = t === null ? null : Math.max(0, (Date.now() - t) / 1000);
    node.textContent = ts ? ('as of ' + ts) : 'no data yet';
    node.classList.remove('good', 'warn', 'bad');
    if (age === null) { node.classList.add('warn'); return; }
    if (age <= STALE_S) {
      node.classList.add('good');
      node.textContent += ' • LIVE';
    } else if (age < 3600) {
      node.classList.add('warn');
      node.textContent += ' • ' + Math.round(age / 60) + 'm ago';
    } else {
      node.classList.add('warn');
      node.textContent += ' • Market Closed';
    }
  }

  function renderBox(box) {
    var body = document.getElementById('arb-box-body');
    var pills = document.getElementById('arb-box-pills');
    if (!body || !pills) return;
    while (body.firstChild) body.removeChild(body.firstChild);
    while (pills.firstChild) pills.removeChild(pills.firstChild);

    var opps = (box && box.recent_opportunities) || [];
    var head = el('span', 'BOX SPREADS: ' + ((box && box.total_anomalies_detected) || 0) +
      ' detected • ' + ((box && box.open_paper_trades) || 0) + ' open • ' +
      ((box && box.closed_paper_trades) || 0) + ' closed', 'status-pill');
    pills.appendChild(head);
    pills.appendChild(el('span', 'Realized ' + fmtINR(box && box.realized_paper_pnl_inr) +
      ' • Open ' + fmtINR(box && box.open_unrealized_pnl_inr), 'status-pill'));

    if (!opps.length) {
      var empty = el('div', 'No anomalies — markets pricing fairly (net of 4-leg costs).');
      empty.style.cssText = 'text-align:center;color:var(--text-dim);padding:14px 0;font-size:11px;';
      body.appendChild(empty);
      return;
    }
    opps.slice(-6).reverse().forEach(function (o) {
      var row = el('div', '', 'arb-row');
      row.appendChild(el('b', o.strikes + '  ', ''));
      row.appendChild(el('span', 'net ' + fmtINR(o.edge_net_inr) +
        ' (' + esc(o.edge_net_pct) + '%) • costs ₹' + esc(o.all_in_costs) +
        ' • DTE ' + esc(o.dte) + ' • ' + esc(o.status), 'arb-dim'));
      var q = el('div', 'quote ' + esc(o.quote_as_of || 'n/a'), 'arb-dim');
      row.appendChild(q);
      body.appendChild(row);
    });
  }

  function renderPairs(sp) {
    var body = document.getElementById('arb-pairs-body');
    var pills = document.getElementById('arb-pairs-pills');
    if (!body || !pills) return;
    while (body.firstChild) body.removeChild(body.firstChild);
    while (pills.firstChild) pills.removeChild(pills.firstChild);

    pills.appendChild(el('span', 'PAIRS: ' + ((sp && sp.closed_paper_trades) || 0) +
      ' closed • ' + ((sp && sp.win_trades) || 0) + ' wins', 'status-pill'));
    pills.appendChild(el('span', 'P&L ' + fmtINR(sp && sp.total_paper_pnl_inr), 'status-pill'));

    var rows = (sp && sp.pairs_status) || [];
    var tradable = rows.filter(function (r) { return r && r.signal && r.signal !== 'NEUTRAL'; });
    if (!tradable.length) {
      var empty = el('div', 'No pair signals (need |z| > 2 on cointegrated pairs).');
      empty.style.cssText = 'text-align:center;color:var(--text-dim);padding:14px 0;font-size:11px;';
      body.appendChild(empty);
      return;
    }
    tradable.slice(0, 6).forEach(function (r) {
      var row = el('div', '', 'arb-row');
      row.appendChild(el('b', r.pair + '  ', ''));
      row.appendChild(el('span', esc(r.signal) + ' • z ' + esc(r.z_score) +
        ' • adf ' + esc(r.adf_pvalue) + (r.is_cointegrated ? '' : ' (NOT cointegrated — signal gated)'),
        'arb-dim'));
      body.appendChild(row);
    });
  }

  var Arb = {
    refresh: function () {
      return fetch('/api/arb/opportunities').then(function (res) {
        if (!res.ok) throw new Error('arb feed ' + res.status);
        return res.json();
      }).then(function (d) {
        setFreshness(document.getElementById('arb-fresh'),
          (d && (d.timestamp || d.last_updated)) || '');
        renderBox(d && d.box_spread);
        renderPairs(d && d.statistical_pairs);
      }).catch(function (e) {
        var f = document.getElementById('arb-fresh');
        if (f) { f.textContent = 'feed unavailable'; f.classList.add('bad'); }
        if (window.console && console.warn) console.warn('[Arb]', e);
      });
    },
    scanNow: function () {
      toast('📦 Running arbitrage scan (paper)...', 'success');
      return fetch('/api/arb/scan-now', { method: 'POST' }).then(function (res) {
        return res.json();
      }).then(function (d) {
        if (d && d.status === 'SUCCESS' && d.data) {
          setFreshness(document.getElementById('arb-fresh'), d.data.timestamp || '');
          renderBox(d.data.box_spread);
          renderPairs(d.data.statistical_pairs);
          var n = (d.data.metrics && d.data.metrics.box_anomalies_detected) || 0;
          toast(n ? ('📦 ' + n + ' box anomalie(s) net-of-cost') : '📦 Scan done — no anomalies', 'success');
        } else {
          toast('Arb scan: ' + ((d && d.message) || 'error'), 'error');
        }
      }).catch(function (e) {
        toast('Arb scan failed', 'error');
        if (window.console && console.warn) console.warn('[Arb]', e);
      });
    },
    start: function () {
      var self = this;
      this.refresh();
      setInterval(function () { self.refresh(); }, POLL_MS);
    }
  };

  window.Arb = Arb;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { Arb.start(); });
  } else {
    Arb.start();
  }
})();
