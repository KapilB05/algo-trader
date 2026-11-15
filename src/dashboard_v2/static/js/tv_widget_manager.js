/**
 * TradingView External Launcher & Chart Helper (src/dashboard_v2/static/js/tv_widget_manager.js)
 * 
 * Features:
 * - Dynamic JavaScript function to launch official TradingView real-time charts in a clean new tab
 * - Dedicated URL mappings for NSE:NIFTY, NSE:BANKNIFTY, NSE:FINNIFTY, and key equities
 * - Zero external iframe widget overhead, avoiding third-party licensing or symbol errors
 */

window.launchTradingViewWindow = function(symbol = 'NSE:NIFTY') {
  const urlMap = {
    'NSE:NIFTY': 'https://in.tradingview.com/chart/?symbol=NSE%3ANIFTY',
    'NSE:BANKNIFTY': 'https://in.tradingview.com/chart/?symbol=NSE%3ABANKNIFTY',
    'NSE:FINNIFTY': 'https://in.tradingview.com/chart/?symbol=NSE%3AFINNIFTY',
    'NSE:RELIANCE': 'https://in.tradingview.com/chart/?symbol=NSE%3ARELIANCE',
    'NSE:HDFCBANK': 'https://in.tradingview.com/chart/?symbol=NSE%3AHDFCBANK',
    'NSE:ICICIBANK': 'https://in.tradingview.com/chart/?symbol=NSE%3AICICIBANK',
    'NSE:INFY': 'https://in.tradingview.com/chart/?symbol=NSE%3AINFY',
  };

  const targetUrl = urlMap[symbol] || `https://in.tradingview.com/chart/?symbol=${encodeURIComponent(symbol)}`;
  console.info(`[ChartLauncher] Launching official TradingView chart for ${symbol}: ${targetUrl}`);
  window.open(targetUrl, '_blank', 'noopener,noreferrer');
};
