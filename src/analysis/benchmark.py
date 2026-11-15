"""
Alpha / Benchmark Attribution — market-relative edge measurement.

Answers the question every other module avoids: does the strategy beat the
market's own drift (beta), after costs, with statistical significance?

Benchmark definitions (all built on the same intraday DataFrame the strategy
simulates on, so the comparison is apples-to-apples):

  NAIVE_LONG     : each session, buy at open / sell at close. This is exactly
                   the intraday drift a long-biased engine rides — the correct
                   hurdle for intraday systems.
  NAIVE_SHORT    : mirrored (sell open / buy close), for sanity checks.
  CLOSE_TO_CLOSE : buy-and-hold across sessions (overnight-exposed reference).
  REGIME_NAIVE   : NAIVE_LONG restricted to sessions satisfying the same
                   VIX/ADX regime gates the strategy uses.

Significance is established with a bootstrap CI on mean daily excess return
plus a normal-approximation p-value on the same mean — no scipy dependency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("Analysis.Benchmark")

MIN_SESSIONS = 40          # minimum aligned sessions before alpha is trustworthy
BOOTSTRAP_ITERATIONS = 5000
TRADING_DAYS = 252


@dataclass
class AlphaStats:
    """Benchmark-relative performance of one strategy evaluation window."""

    benchmark: str
    n_sessions: int
    bench_return_pct: float
    bench_sharpe: float
    strategy_return_pct: float
    excess_return_pct: float              # strategy total return - benchmark total (window, %)
    alpha_annual_pct: float               # annualized Jensen-style intercept
    beta: float
    information_ratio: float              # annualized mean/excess std
    days_beat_pct: float                  # % of aligned sessions beating benchmark
    mean_daily_excess_pct: float
    p_value: float
    significant: bool


def session_benchmark_returns(
    df: pd.DataFrame,
    benchmark: str = "NAIVE_LONG",
    min_vix: Optional[float] = None,
    max_vix: Optional[float] = None,
    adx_min: Optional[float] = None,
    min_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Build a per-session benchmark daily-return series from intraday bars.

    Returns DataFrame with columns ``date`` and ``ret_pct`` (percent/day).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    has_ind = {"india_vix": "india_vix" in df.columns, "adx14": "adx14" in df.columns}

    sessions: List[Dict[str, Any]] = []
    prev_close: Optional[float] = None
    for date, g in df.groupby("date", sort=True):
        o0 = float(g["open"].iloc[0])
        c0 = float(g["close"].iloc[0])
        c_last = float(g["close"].iloc[-1])

        eligible = True
        if min_vix is not None or max_vix is not None or adx_min is not None:
            n_bar = len(g)
            n_ok = 0
            if min_vix is not None or max_vix is not None:
                if not has_ind["india_vix"]:
                    n_ok = n_bar if (min_vix is None or min_vix <= 0.0) and (max_vix is None or max_vix >= 100.0) else 0
                else:
                    v = g["india_vix"].astype(float)
                    n_ok = ((v >= (min_vix if min_vix is not None else -math.inf)) &
                            (v <= (max_vix if max_vix is not None else math.inf))).sum()
            if adx_min is not None and has_ind["adx14"]:
                n_ok += (g["adx14"].astype(float) >= adx_min).sum()
            eligible = (n_ok / max(1, n_bar)) >= min_fraction

        if benchmark == "CLOSE_TO_CLOSE":
            base = c0 if prev_close is None else prev_close
            ret = (c_last - base) / base * 100.0 if base and base > 0 else 0.0
        else:
            ret = (c_last - o0) / o0 * 100.0 if o0 > 0 else 0.0
            if benchmark == "NAIVE_SHORT":
                ret = -ret

        sessions.append({"date": date, "ret_pct": ret, "_eligible": eligible})
        prev_close = c_last

    out = pd.DataFrame(sessions)
    if benchmark == "REGIME_NAIVE":
        out = out.loc[out["_eligible"]].copy()
        out = out.drop(columns=["_eligible"]).reset_index(drop=True)
        out["ret_pct"] = out["ret_pct"].astype(float)
    else:
        out = out.drop(columns=["_eligible"]).reset_index(drop=True)
    return out[["date", "ret_pct"]]


def compute_alpha(
    strategy_daily_pct: Union[Sequence[float], np.ndarray],
    benchmark_daily_pct: Union[Sequence[float], np.ndarray],
    benchmark: str = "NAIVE_LONG",
) -> AlphaStats:
    """OLS alpha/beta of strategy daily returns on benchmark, + significance."""
    s = np.asarray(strategy_daily_pct, dtype=float)
    b = np.asarray(benchmark_daily_pct, dtype=float)
    n = len(s)
    zero = AlphaStats(benchmark=benchmark, n_sessions=n, bench_return_pct=0.0,
                      bench_sharpe=0.0, strategy_return_pct=0.0, excess_return_pct=0.0,
                      alpha_annual_pct=0.0, beta=0.0, information_ratio=0.0,
                      days_beat_pct=0.0, mean_daily_excess_pct=0.0, p_value=1.0,
                      significant=False)
    if n < 2:
        return zero

    X = np.vstack([np.ones(n), b]).T
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    alpha_daily = float(coef[0])
    beta = float(coef[1])

    excess = s - b
    mean_ex = float(np.mean(excess))
    std_ex = float(np.std(excess, ddof=1))
    se = std_ex / math.sqrt(n) if std_ex > 0 else 0.0
    if se > 0:
        z = mean_ex / se
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    else:
        p_value = 0.5 if abs(mean_ex) < 1e-12 else (0.0 if mean_ex > 0 else 1.0)

    ir = (mean_ex / std_ex) * math.sqrt(TRADING_DAYS) if std_ex > 0 else 0.0
    days_beat = float(np.mean(excess > 0)) * 100.0

    bench_total = (float(np.prod(1.0 + b / 100.0)) - 1.0) * 100.0
    strat_total = (float(np.prod(1.0 + s / 100.0)) - 1.0) * 100.0

    bench_sharpe = (float(np.mean(b)) / float(np.std(b, ddof=1))) * math.sqrt(TRADING_DAYS) \
        if n > 1 and float(np.std(b, ddof=1)) > 0 else 0.0

    # Bootstrap CI on mean daily excess return
    rng = np.random.default_rng(42)
    boots = np.empty(BOOTSTRAP_ITERATIONS)
    for k in range(BOOTSTRAP_ITERATIONS):
        boots[k] = rng.choice(excess, size=n, replace=True).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    significant = bool(n >= MIN_SESSIONS and (lo > 0.0 or hi < 0.0) and p_value < 0.05)

    return AlphaStats(
        benchmark=benchmark,
        n_sessions=n,
        bench_return_pct=round(bench_total, 3),
        bench_sharpe=round(bench_sharpe, 2),
        strategy_return_pct=round(strat_total, 3),
        excess_return_pct=round(strat_total - bench_total, 3),
        alpha_annual_pct=round(alpha_daily * TRADING_DAYS, 3),
        beta=round(beta, 3),
        information_ratio=round(ir, 2),
        days_beat_pct=round(days_beat, 1),
        mean_daily_excess_pct=round(mean_ex, 4),
        p_value=round(p_value, 5),
        significant=significant,
    )


def post_compute_alpha(
    res: Any,
    df: pd.DataFrame,
    benchmark: str = "NAIVE_LONG",
    min_vix: Optional[float] = None,
    max_vix: Optional[float] = None,
    adx_min: Optional[float] = None,
    bench_df: Optional[pd.DataFrame] = None,
) -> Optional[AlphaStats]:
    """Attach benchmark-relative stats to a SimResult carrying daily data."""
    s_dates = list(getattr(res, "daily_dates", None) or [])
    s_ret = list(getattr(res, "daily_return_pct", None) or [])
    if not s_dates:
        return None
    if bench_df is None:
        bench_df = session_benchmark_returns(df, benchmark, min_vix, max_vix, adx_min)

    strat_map = {str(d): r for d, r in zip(s_dates, s_ret)}
    rows = [
        (str(d), r)
        for d, r in zip(bench_df["date"], bench_df["ret_pct"])
        if str(d) in strat_map
    ]
    if len(rows) < 2:
        return None

    s = np.array([strat_map[d] for d, _ in rows], dtype=float)
    b = np.array([r for _, r in rows], dtype=float)
    return compute_alpha(s, b, benchmark=benchmark)


def promotion_eligible(
    sharpe_ratio: float,
    baseline_sharpe: float,
    profit_factor: float,
    alpha: Optional[AlphaStats],
    baseline_alpha: Optional[AlphaStats] = None,
) -> Tuple[bool, str]:
    """
    Promotion gate for the auto-learner.

    A candidate may be promoted only if the legacy Sharpe/PF grid gates pass AND
    it demonstrates statistically significant positive alpha (excess over the
    intraday naive-long benchmark) that also beats the live config's excess.
    """
    ok = True
    reasons: List[str] = []

    if sharpe_ratio < baseline_sharpe * 1.05:
        ok = False
        reasons.append("OOS Sharpe < +5% vs baseline")

    if profit_factor < 1.15:
        ok = False
        reasons.append("OOS PF < 1.15")

    if alpha is None:
        ok = False
        reasons.append("no alpha stats available")
        return ok, "; ".join(reasons)

    if not alpha.significant:
        ok = False
        reasons.append(
            f"alpha NOT significant (p={alpha.p_value:.3f}, n={alpha.n_sessions}, CI includes 0)")
    if alpha.alpha_annual_pct <= 0.0:
        ok = False
        reasons.append(f"annualized alpha not positive ({alpha.alpha_annual_pct:+.1f}%)")
    if alpha.excess_return_pct <= 0.0:
        ok = False
        reasons.append(f"no excess return vs {alpha.benchmark} ({alpha.excess_return_pct:+.2f}%)")
    if baseline_alpha is not None and alpha.excess_return_pct <= baseline_alpha.excess_return_pct:
        ok = False
        reasons.append("excess return not above live config's excess")

    return ok, "; ".join(reasons)