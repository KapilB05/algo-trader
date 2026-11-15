/**
 * Modular TradingView Lightweight Charts Engine (src/dashboard_v2/static/js/chart.js)
 * 
 * Features:
 * - Candlesticks, Volume Histogram, EMA 9, EMA 21, and Intraday VWAP overlays.
 * - Dynamic Crosshair Move subscription updating real-time OHLCV legend.
 * - Real-Time candle updating loop (series.update) simulating live market ticks.
 * - Timeframe switching (1m, 5m, 15m, 1h, 1d) & Symbol switching.
 * - Toolbar actions: Fit Content (⟲), Toggle Strategy Levels (🎯), Chart Snapshot (📸).
 * - Responsive auto-resizing via ResizeObserver.
 */

const ChartEngine = {
  chart: null,
  candlestickSeries: null,
  volumeSeries: null,
  ema9Series: null,
  ema21Series: null,
  vwapSeries: null,
  priceLines: [],
  currentSymbol: 'BANKNIFTY FUT',
  currentTimeframe: '5m',
  activeEngine: 'algo', // 'tv' or 'algo'
  resizeObserver: null,
  isLoading: false,
  rawCandles: [],
  liveTickInterval: null,
  showLevels: false,
  showTradeMarkers: false,

  init() {
    this.bindSymbolButtons();
    this.bindTimeframeButtons();
    this.bindActionButtons();
    this.setupResizeObserver();
    this.loadChart(this.currentSymbol, this.currentTimeframe);
    this.startLiveTickingLoop();
  },

  bindSymbolButtons() {
    const symBtns = document.querySelectorAll('.chart-sym-btn');
    symBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        symBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const symbol = btn.dataset.symbol;
        this.currentSymbol = symbol;
        this.loadChart(symbol, this.currentTimeframe);
        if (window.Signals && typeof Signals.setSymbol === 'function') {
          Signals.setSymbol(symbol);
        }
      });
    });
  },

  bindTimeframeButtons() {
    const tfBtns = document.querySelectorAll('.chart-tf-btn');
    tfBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tfBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentTimeframe = btn.dataset.tf;
        this.loadChart(this.currentSymbol, this.currentTimeframe);
      });
    });
  },

  bindActionButtons() {
    const btnFit = document.getElementById('btn-chart-fit');
    if (btnFit) {
      btnFit.addEventListener('click', () => {
        if (this.chart) this.chart.timeScale().fitContent();
      });
    }

    const btnLines = document.getElementById('btn-chart-lines');
    if (btnLines) {
      btnLines.addEventListener('click', () => {
        this.toggleStrategyLevels();
      });
    }

    const btnSnap = document.getElementById('btn-chart-snapshot');
    if (btnSnap) {
      btnSnap.addEventListener('click', () => {
        this.exportChartSnapshot();
      });
    }
  },

  setupResizeObserver() {
    const container = document.getElementById('tv-chart-wrapper');
    if (!container) return;

    this.resizeObserver = new ResizeObserver(entries => {
      if (!entries?.length) return;
      const entry = entries[0];
      const { width, height } = entry.contentRect;
      if (this.chart && width > 0 && height > 0) {
        this.chart.applyOptions({ width, height });
      }
    });
    this.resizeObserver.observe(container);
  },

  showLoadingState(show = true) {
    let loader = document.getElementById('chart-loading-overlay');
    const container = document.getElementById('tv-chart-wrapper');
    if (!container) return;

    if (show) {
      if (!loader) {
        loader = document.createElement('div');
        loader.id = 'chart-loading-overlay';
        loader.style.cssText = `
          position: absolute; top: 0; left: 0; width: 100%; height: 100%;
          background: rgba(13, 17, 23, 0.75); display: flex; justify-content: center;
          align-items: center; z-index: 10; font-family: var(--font-mono);
          font-size: 13px; color: var(--accent-color); gap: 10px;
        `;
        loader.innerHTML = `<span>⏳ Loading ${this.currentSymbol} (${this.currentTimeframe.toUpperCase()})...</span>`;
        container.appendChild(loader);
      } else {
        loader.style.display = 'flex';
      }
    } else if (loader) {
      loader.style.display = 'none';
    }
  },

  async fetchOptionChainSummary(symbol) {
    try {
      let underlying = 'NIFTY';
      if (symbol.includes('BANK')) underlying = 'BANKNIFTY';
      else if (symbol.includes('FIN')) underlying = 'FINNIFTY';

      const res = await fetch(`/api/option-chain?symbol=${encodeURIComponent(underlying)}`);
      if (!res.ok) return;
      const data = await res.json();

      let sig = data.pcr_signal;
      if (!sig) {
        if (data.pcr > 1.2) sig = 'BULLISH';
        else if (data.pcr < 0.8) sig = 'BEARISH';
        else sig = 'NEUTRAL';
      }

      let pcrClass = 'text-amber';
      if (sig === 'BULLISH') pcrClass = 'text-green';
      else if (sig === 'BEARISH') pcrClass = 'text-red';

      this.setText('oc-pcr', `${data.pcr} (${sig})`);
      const pcrEl = document.getElementById('oc-pcr');
      if (pcrEl) pcrEl.className = pcrClass;

      this.setText('oc-maxpain', `₹${data.max_pain.toLocaleString('en-IN')}`);
      this.setText('oc-atm', `${data.atm_strike.toLocaleString('en-IN')} CE/PE`);
    } catch (err) {
      console.error('Failed to fetch option chain summary:', err);
    }
  },

  clearSeriesData() {
    if (!this.candlestickSeries) return;
    try {
      this.candlestickSeries.setData([]);
      if (this.volumeSeries) this.volumeSeries.setData([]);
      if (this.ema9Series) this.ema9Series.setData([]);
      if (this.ema21Series) this.ema21Series.setData([]);
      if (this.vwapSeries) this.vwapSeries.setData([]);
    } catch (e) {
      console.error('Failed to clear cached chart data:', e);
    }
  },

  async loadChart(symbol = 'BANKNIFTY FUT', timeframe = '5m') {
    this.currentSymbol = symbol;
    this.currentTimeframe = timeframe;

    const badge = document.getElementById('chart-symbol-badge');
    if (badge) badge.textContent = `${symbol} (${timeframe.toUpperCase()})`;

    const legendSym = document.getElementById('legend-sym');
    if (legendSym) legendSym.textContent = `${symbol} ${timeframe.toUpperCase()}`;

    const container = document.getElementById('tv-chart');
    if (!container) return;

    // Clear any stale cached data immediately
    this.clearSeriesData();
    this.rawCandles = [];

    this.showLoadingState(true);
    this.fetchOptionChainSummary(symbol);

    try {
      const cacheBuster = Date.now();
      const url = `/api/historical?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(timeframe)}&_t=${cacheBuster}`;
      const res = await fetch(url, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.candles && data.candles.length > 0) {
        this.rawCandles = data.candles;
        this.renderChart(container, data.candles);
        if (data.latest_candle_time_ist) {
          console.info(`[ChartEngine] Loaded ${data.candles.length} candles for ${symbol} | Latest: ${data.latest_candle_time_ist}`);
        }
      }
    } catch (err) {
      console.error('Failed to load chart data:', err);
    } finally {
      this.showLoadingState(false);
    }
  },

  renderChart(container, candles) {
    if (typeof LightweightCharts === 'undefined') return;

    if (this.chart) {
      this.chart.remove();
      this.chart = null;
      this.priceLines = [];
    }

    const wrapper = document.getElementById('tv-chart-wrapper');
    const width = wrapper ? wrapper.clientWidth : (container.clientWidth || 800);
    const height = wrapper ? wrapper.clientHeight : 500;

    this.chart = LightweightCharts.createChart(container, {
      width: width,
      height: height,
      layout: {
        background: { type: 'solid', color: '#0d1117' },
        textColor: '#8b949e',
        fontFamily: "'JetBrains Mono', monospace",
      },
      localization: {
        locale: 'en-IN',
        dateFormat: 'dd MMM yyyy',
        timeFormatter: (timestamp) => {
          const d = new Date(timestamp * 1000);
          return d.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour: 'numeric', minute: '2-digit', hour12: true });
        },
      },
      grid: {
        vertLines: { color: 'rgba(48, 54, 61, 0.4)' },
        horzLines: { color: 'rgba(48, 54, 61, 0.4)' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#30363d',
      },
      timeScale: {
        borderColor: '#30363d',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time, tickMarkType, locale) => {
          const d = new Date(time * 1000);
          const dateStr = d.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short' });
          const timeStr = d.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour: 'numeric', minute: '2-digit', hour12: true });
          
          if (tickMarkType === 2 || tickMarkType === 1 || tickMarkType === 0) {
            return dateStr;
          }
          return timeStr;
        },
      },
    });

    this.candlestickSeries = this.chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    this.volumeSeries = this.chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume_scale',
    });

    this.chart.priceScale('volume_scale').applyOptions({
      scaleMargins: {
        top: 0.82,
        bottom: 0,
      },
    });

    this.ema9Series = this.chart.addLineSeries({
      color: '#eab308',
      lineWidth: 1.5,
      title: 'EMA 9',
    });

    this.ema21Series = this.chart.addLineSeries({
      color: '#f97316',
      lineWidth: 1.5,
      title: 'EMA 21',
    });

    this.vwapSeries = this.chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      title: 'VWAP',
    });

    const formattedCandles = candles.map(c => ({
      time: c.time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    const formattedVolume = candles.map(c => ({
      time: c.time,
      value: c.volume,
      color: c.close >= c.open ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
    }));

    const formattedEma9 = candles.map(c => ({ time: c.time, value: c.ema9 }));
    const formattedEma21 = candles.map(c => ({ time: c.time, value: c.ema21 }));
    const formattedVwap = candles.map(c => ({ time: c.time, value: c.vwap }));

    this.candlestickSeries.setData(formattedCandles);
    this.volumeSeries.setData(formattedVolume);
    this.ema9Series.setData(formattedEma9);
    this.ema21Series.setData(formattedEma21);
    this.vwapSeries.setData(formattedVwap);

    this.chart.timeScale().fitContent();

    // Subscribe to Crosshair Movement to update the real-time Legend
    this.setupCrosshairLegend();

    // Default to latest candle in legend
    if (candles?.length > 0) {
      this.updateLegendWithCandle(candles.at(-1));
    }

    // Load dynamic Buy / Sell execution markers directly on the candlesticks
    this.loadTradeMarkers();
  },

  setupCrosshairLegend() {
    if (!this.chart) return;

    this.chart.subscribeCrosshairMove(param => {
      if (!param?.time || !param?.seriesData) {
        if (this.rawCandles.length > 0) {
          this.updateLegendWithCandle(this.rawCandles.at(-1));
        }
        return;
      }

      const cData = param.seriesData.get(this.candlestickSeries);
      const vData = param.seriesData.get(this.volumeSeries);
      const ema9 = param.seriesData.get(this.ema9Series);
      const ema21 = param.seriesData.get(this.ema21Series);
      const vwap = param.seriesData.get(this.vwapSeries);

      if (cData) {
        this.updateLegendDisplay({
          o: cData.open, h: cData.high, l: cData.low, c: cData.close,
          vol: vData?.value, ema9: ema9?.value, ema21: ema21?.value, vwap: vwap?.value, time: param.time
        });
      }
    });
  },

  updateLegendWithCandle(c) {
    if (!c) return;
    this.updateLegendDisplay({
      o: c.open, h: c.high, l: c.low, c: c.close,
      vol: c.volume, ema9: c.ema9, ema21: c.ema21, vwap: c.vwap, time: c.time
    });
  },

  setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  },

  setLegendColor(id, color) {
    const el = document.getElementById(id);
    if (el) el.style.color = color;
  },

  updateLegendDisplay(data) {
    if (data.time) {
      const d = new Date(data.time * 1000);
      const timeStr = d.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour: 'numeric', minute: '2-digit', hour12: true });
      this.setText('legend-sym', `${this.currentSymbol} ${this.currentTimeframe.toUpperCase()} [${timeStr} IST]`);
    }

    if (data.o != null) this.setText('legend-o', `₹${data.o.toFixed(2)}`);
    if (data.h != null) this.setText('legend-h', `₹${data.h.toFixed(2)}`);
    if (data.l != null) this.setText('legend-l', `₹${data.l.toFixed(2)}`);
    if (data.c != null) {
      this.setText('legend-c', `₹${data.c.toFixed(2)}`);
      this.setLegendColor('legend-c', (data.c >= data.o) ? '#22c55e' : '#ef4444');
    }
    if (data.vol != null) this.setText('legend-vol', data.vol.toLocaleString('en-IN'));
    if (data.ema9 != null) this.setText('legend-ema9', `₹${data.ema9.toFixed(2)}`);
    if (data.ema21 != null) this.setText('legend-ema21', `₹${data.ema21.toFixed(2)}`);
    if (data.vwap != null) this.setText('legend-vwap', `₹${data.vwap.toFixed(2)}`);
  },

  toggleStrategyLevels() {
    if (!this.candlestickSeries || this.rawCandles.length === 0) return;

    if (this.showLevels) {
      // Remove existing lines
      this.priceLines.forEach(line => this.candlestickSeries.removePriceLine(line));
      this.priceLines = [];
      this.showLevels = false;
      const btn = document.getElementById('btn-chart-lines');
      if (btn) btn.classList.remove('active');
    } else {
      const lastCandle = this.rawCandles.at(-1);
      const entryP = lastCandle.close;
      const slP = roundPrice(entryP * 0.985);
      const tgtP = roundPrice(entryP * 1.03);

      const entryLine = this.candlestickSeries.createPriceLine({
        price: entryP,
        color: '#3b82f6',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: 'ENTRY LEVEL',
      });

      const slLine = this.candlestickSeries.createPriceLine({
        price: slP,
        color: '#ef4444',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'STOP LOSS (1.5%)',
      });

      const tgtLine = this.candlestickSeries.createPriceLine({
        price: tgtP,
        color: '#22c55e',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'TARGET (3.0%)',
      });

      this.priceLines = [entryLine, slLine, tgtLine];
      this.showLevels = true;
      const btn = document.getElementById('btn-chart-lines');
      if (btn) btn.classList.add('active');
    }
  },

  exportChartSnapshot() {
    const container = document.getElementById('tv-chart');
    if (!container) return;
    const canvas = container.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `${this.currentSymbol.replace(/\s+/g, '_')}_${this.currentTimeframe}_chart.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  },

  getIntervalSeconds(tf) {
    switch (tf) {
      case '1m': return 60;
      case '5m': return 300;
      case '15m': return 900;
      case '1h': return 3600;
      case '1d': return 86400;
      default: return 300;
    }
  },

  onWatchlistFeed(watchlist) {
    if (!watchlist || !Array.isArray(watchlist) || this.rawCandles.length === 0) return;
    const symUpper = this.currentSymbol.toUpperCase().trim();

    const item = watchlist.find(w => {
      if (!w?.symbol) return false;
      const wSym = w.symbol.toUpperCase().trim();
      
      // Exact match first
      if (wSym === symUpper) return true;

      // If looking at option strike (e.g. 52500 CE), strictly match that strike
      if (symUpper.includes('CE') || symUpper.includes('PE')) {
        return wSym === symUpper;
      }

      // If looking at BANKNIFTY FUT or BANKNIFTY
      if (symUpper.startsWith('BANKNIFTY') && wSym.startsWith('BANKNIFTY') && !wSym.includes('CE') && !wSym.includes('PE')) {
        return true;
      }

      // If looking at FINNIFTY
      if (symUpper.startsWith('FINNIFTY') && wSym.startsWith('FINNIFTY') && !wSym.includes('CE') && !wSym.includes('PE')) {
        return true;
      }

      // If looking at NIFTY 50
      if (symUpper.startsWith('NIFTY') && !symUpper.includes('BANK') && !symUpper.includes('FIN') && wSym.startsWith('NIFTY') && !wSym.includes('BANK') && !wSym.includes('FIN') && !wSym.includes('CE') && !wSym.includes('PE')) {
        return true;
      }

      return false;
    });

    if (item?.ltp != null) {
      const p = Number.parseFloat(item.ltp);
      if (!Number.isNaN(p) && p > 0) {
        this.processLivePrice(p);
      }
    }
  },

  processLivePrice(price, volume = 0, forcedTimestamp = null) {
    if (!this.candlestickSeries || this.rawCandles.length === 0 || !price || Number.isNaN(price)) return;

    let last = this.rawCandles.at(-1);

    // Outlier Sanity Guard: If price differs by > 3.5% from current candle close, discard it
    // to prevent mismatched symbol/feed prices from stretching candle geometry
    if (Math.abs(price - last.close) / last.close > 0.035) {
      return;
    }

    const intervalSec = this.getIntervalSeconds(this.currentTimeframe);
    const nowTs = forcedTimestamp || Math.floor(Date.now() / 1000);
    const bucketTime = Math.floor(nowTs / intervalSec) * intervalSec;

    if (bucketTime > last.time) {
      // ── Bar Rollover: Start a brand NEW forming candle for the current minute/interval! ──
      const newOpen = last.close;
      const newHigh = Math.max(newOpen, price);
      const newLow = Math.min(newOpen, price);
      const newClose = price;
      const newVol = volume || 50;

      const k9 = 2 / (9 + 1);
      const k21 = 2 / (21 + 1);
      const newEma9 = roundPrice((newClose * k9) + (last.ema9 * (1 - k9)));
      const newEma21 = roundPrice((newClose * k21) + (last.ema21 * (1 - k21)));
      const newVwap = roundPrice(((last.vwap * 100) + newClose) / 101);

      const newCandle = {
        time: bucketTime,
        open: newOpen,
        high: newHigh,
        low: newLow,
        close: newClose,
        volume: newVol,
        ema9: newEma9,
        ema21: newEma21,
        vwap: newVwap,
      };

      this.rawCandles.push(newCandle);
      this.updateRealtimeCandle(newCandle);
      this.updateLegendWithCandle(newCandle);
    } else {
      // ── Intra-Bar Update: Update current active forming candle ──
      last.high = Math.max(last.high, price);
      last.low = Math.min(last.low, price);
      last.close = price;
      const randomVolume = (crypto.getRandomValues(new Uint32Array(1))[0] % 15) + 1;
      last.volume += (volume || randomVolume);

      const k9 = 2 / (9 + 1);
      const k21 = 2 / (21 + 1);
      last.ema9 = roundPrice((last.close * k9) + (last.ema9 * (1 - k9)));
      last.ema21 = roundPrice((last.close * k21) + (last.ema21 * (1 - k21)));

      this.updateRealtimeCandle(last);
      this.updateLegendWithCandle(last);
    }
  },

  startLiveTickingLoop() {
    if (this.liveTickInterval) clearInterval(this.liveTickInterval);

    this.liveTickInterval = setInterval(() => {
      const activeTab = document.querySelector('.nav-tab.active');
      if (activeTab?.dataset.tab === 'tab-chart' && this.rawCandles.length > 0) {
        const last = this.rawCandles.at(-1);
        // Micro-oscillation (+/- 0.02%) to simulate real intra-bar heartbeat
        const randomFactor = crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296;
        const delta = (randomFactor - 0.49) * (last.close * 0.0004);
        const livePrice = roundPrice(last.close + delta);
        this.processLivePrice(livePrice);
      }
    }, 1000);
  },

  updateRealtimeCandle(candle) {
    if (!this.candlestickSeries || !candle?.time) return;
    this.candlestickSeries.update({
      time: candle.time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    });
    if (this.volumeSeries && candle.volume) {
      this.volumeSeries.update({
        time: candle.time,
        value: candle.volume,
        color: candle.close >= candle.open ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      });
    }
    if (this.ema9Series && candle.ema9) {
      this.ema9Series.update({ time: candle.time, value: candle.ema9 });
    }
    if (this.ema21Series && candle.ema21) {
      this.ema21Series.update({ time: candle.time, value: candle.ema21 });
    }
  },

  toggleTradeMarkers() {
    this.showTradeMarkers = !this.showTradeMarkers;
    const btn = document.getElementById('btn-toggle-markers');
    if (btn) btn.classList.toggle('active', this.showTradeMarkers);
    this.loadTradeMarkers();
  },

  async loadTradeMarkers() {
    if (!this.candlestickSeries || !this.rawCandles || this.rawCandles.length === 0) return;

    if (!this.showTradeMarkers) {
      this.candlestickSeries.setMarkers([]);
      return;
    }

    try {
      const [histRes, posRes] = await Promise.all([
        fetch('/api/paper/history'),
        fetch('/api/paper/positions')
      ]);
      const histJson = histRes.ok ? await histRes.json() : [];
      const history = Array.isArray(histJson) ? histJson : (histJson.history || []);
      const posData = posRes.ok ? await posRes.json() : { positions: [] };
      const openPos = posData.positions || [];

      const markers = [];
      const minCandleTime = this.rawCandles[0].time;
      const seenTimes = new Set();

      const parseTs = (str) => {
        if (!str) return null;
        let ts = Math.floor(new Date(str).getTime() / 1000);
        if (Number.isNaN(ts)) {
          const today = new Date().toISOString().split('T')[0];
          ts = Math.floor(new Date(`${today} ${str}`).getTime() / 1000);
        }
        if (Number.isNaN(ts)) return null;
        for (let i = this.rawCandles.length - 1; i >= 0; i--) {
          if (this.rawCandles[i].time <= ts) return this.rawCandles[i].time;
        }
        return minCandleTime;
      };

      // Filter only trades for current symbol (e.g. BANKNIFTY or NIFTY)
      const baseSym = this.currentSymbol.split(' ')[0].toUpperCase();
      const relevantHistory = history
        .filter(t => (t.symbol || '').toUpperCase().includes(baseSym))
        .slice(0, 5);

      // Closed trade markers (1 marker per candle maximum to prevent vertical stacking)
      relevantHistory.forEach(trade => {
        const exitTime = parseTs(trade.exit_time || trade.timestamp);
        if (exitTime && exitTime >= minCandleTime && !seenTimes.has(exitTime)) {
          seenTimes.add(exitTime);
          const isProfit = (trade.pnl_rupees || trade.net_pnl || 0) >= 0;
          const pnlText = `${isProfit ? '+' : ''}₹${Math.round(trade.pnl_rupees || trade.net_pnl || 0)}`;
          markers.push({
            time: exitTime,
            position: isProfit ? 'aboveBar' : 'belowBar',
            color: isProfit ? '#10b981' : '#f43f5e',
            shape: isProfit ? 'arrowDown' : 'arrowUp',
            text: pnlText
          });
        }
      });

      // Open trade marker
      openPos.forEach(p => {
        if ((p.symbol || '').toUpperCase().includes(baseSym)) {
          const entryTime = parseTs(p.entry_time || p.timestamp);
          if (entryTime && entryTime >= minCandleTime && !seenTimes.has(entryTime)) {
            seenTimes.add(entryTime);
            markers.push({
              time: entryTime,
              position: 'belowBar',
              color: '#06b6d4',
              shape: 'arrowUp',
              text: `OPEN ₹${Math.round(p.entry_price || p.price)}`
            });
          }
        }
      });

      markers.sort((a, b) => a.time - b.time);
      this.candlestickSeries.setMarkers(markers);
      this.updateActivePositionPriceLines(openPos);
    } catch (err) {
      console.error('Failed to load trade markers:', err);
    }
  },

  updateActivePositionPriceLines(positions) {
    if (!this.candlestickSeries) return;
    if (this.priceLines && this.priceLines.length > 0) {
      this.priceLines.forEach(l => {
        try { this.candlestickSeries.removePriceLine(l); } catch (e) {}
      });
      this.priceLines = [];
    }
    if (!positions || positions.length === 0) return;
    const pos = positions[0];
    const entry = pos.entry_price || pos.price;
    const sl = pos.stop_loss || pos.stop_price;
    const tp = pos.target_price || (pos.side === 'BUY' ? entry * 1.5 : entry * 0.5);

    if (entry && typeof LightweightCharts !== 'undefined') {
      this.priceLines.push(this.candlestickSeries.createPriceLine({
        price: entry,
        color: '#06b6d4',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'ENTRY',
      }));
    }
    if (sl && typeof LightweightCharts !== 'undefined') {
      this.priceLines.push(this.candlestickSeries.createPriceLine({
        price: sl,
        color: '#f43f5e',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'STOP LOSS',
      }));
    }
    if (tp && typeof LightweightCharts !== 'undefined') {
      this.priceLines.push(this.candlestickSeries.createPriceLine({
        price: tp,
        color: '#10b981',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'TARGET',
      }));
    }
  }
};

function roundPrice(val) {
  return Math.round(val * 100) / 100;
}

function toggleIndicator(type) {
  if (type === 'ema') {
    const btn = document.getElementById('btn-toggle-ema');
    const isVisible = ChartEngine.ema9Series && ChartEngine.ema9Series.options().visible !== false;
    const nextState = !isVisible;
    if (ChartEngine.ema9Series) ChartEngine.ema9Series.applyOptions({ visible: nextState });
    if (ChartEngine.ema21Series) ChartEngine.ema21Series.applyOptions({ visible: nextState });
    if (btn) btn.classList.toggle('active', nextState);
  } else if (type === 'vwap') {
    const btn = document.getElementById('btn-toggle-vwap');
    const isVisible = ChartEngine.vwapSeries && ChartEngine.vwapSeries.options().visible !== false;
    const nextState = !isVisible;
    if (ChartEngine.vwapSeries) ChartEngine.vwapSeries.applyOptions({ visible: nextState });
    if (btn) btn.classList.toggle('active', nextState);
  }
}

document.addEventListener('DOMContentLoaded', () => ChartEngine.init());


