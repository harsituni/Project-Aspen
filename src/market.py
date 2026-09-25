"""Do the two claims pension funds make about timberland actually hold up?

Monetary figures are real CAD.

The renewable resources pitch rests on two empirical claims: that the asset
class is lowly correlated with public markets, and that it hedges inflation.
This module tests both against real market data and reports what comes out,
including where the answer is unhelpful.

The honest caveat, which is repeated on every deliverable that uses these
numbers, is in LIMITATIONS below: listed timber and farmland REITs are the
only daily-priced proxies available, and they are equities first. They carry
market beta that private timberland does not, so measured correlations here
are an upper bound and the diversification of a private portfolio is
understated. Private indices exist but are appraisal-based and smoothed,
which biases correlation the other way.

All downloads are cached under data/raw so a rerun works with no network.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config, style

LIMITATIONS = (
    "Listed timber and farmland REITs are imperfect proxies for private timberland. "
    "They are equities first: they carry market beta, leverage and daily sentiment that "
    "a directly held tract does not, so the correlations measured here are an upper "
    "bound and understate the diversification a private portfolio would deliver. "
    "Appraisal-based private indices have the opposite problem: smoothing suppresses "
    "measured correlation and volatility. The truth sits between the two, and neither "
    "is available for a 20,000 ha Alberta tract specifically."
)

DISPLAY_NAMES = {
    "WY": "Weyerhaeuser (WY)",
    "RYN": "Rayonier (RYN)",
    "ADN.TO": "Acadian Timber (ADN)",
    "PCH": "PotlatchDeltic (PCH)",
    "LAND": "Gladstone Land (LAND)",
    "FPI": "Farmland Partners (FPI)",
    "^GSPTSE": "S&P/TSX Composite",
    "^GSPC": "S&P 500",
    "XBB.TO": "Canadian bonds (XBB)",
    "AGG": "US aggregate bonds (AGG)",
}

CACHE_MAX_AGE_DAYS = 7


def _cache_path(name: str) -> Path:
    safe = name.replace("^", "").replace(".", "_")
    return config.DATA_RAW / f"{safe}.csv"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age.days < CACHE_MAX_AGE_DAYS


def fetch_monthly_close(ticker: str, offline: bool = False) -> Optional[pd.Series]:
    """Monthly adjusted close for one ticker, cached under data/raw.

    Returns None when the ticker has no usable data and no cache, which is how
    the bond proxy falls back from the Canadian ETF to the US one.
    """
    path = _cache_path(ticker)

    def read_cache() -> Optional[pd.Series]:
        if not path.exists():
            return None
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        series = frame.iloc[:, 0].dropna()
        return series.rename(ticker) if not series.empty else None

    if offline or _cache_is_fresh(path):
        cached = read_cache()
        if cached is not None:
            return cached

    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker, period="max", interval="1mo",
                auto_adjust=True, progress=False, threads=False,
            )
        if raw is None or raw.empty:
            return read_cache()
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return read_cache()
        close.index = pd.to_datetime(close.index).tz_localize(None).to_period("M").to_timestamp()
        close.name = ticker
        close.to_frame().to_csv(path)
        return close
    except Exception as exc:  # network down, rate limited, ticker delisted
        print(f"    ! {ticker}: download failed ({type(exc).__name__}), using cache")
        return read_cache()


def fetch_prices(offline: bool = False) -> Tuple[pd.DataFrame, str]:
    """Monthly closes for every proxy and benchmark. Returns (frame, bond ticker used)."""
    tickers: List[str] = (
        list(config.val("market.timber_proxies"))
        + list(config.val("market.farmland_proxies"))
        + list(config.val("market.equity_benchmarks"))
    )
    series: Dict[str, pd.Series] = {}
    for ticker in tickers:
        data = fetch_monthly_close(ticker, offline)
        if data is not None:
            series[ticker] = data
        else:
            print(f"    ! {ticker}: no data available, dropped from the analysis")

    bond = str(config.val("market.bond_proxy"))
    bond_data = fetch_monthly_close(bond, offline)
    if bond_data is None:
        bond = str(config.val("market.bond_proxy_fallback"))
        print(f"    ! primary bond proxy unavailable, falling back to {bond}")
        bond_data = fetch_monthly_close(bond, offline)
    if bond_data is not None:
        series[bond] = bond_data

    frame = pd.DataFrame(series).sort_index()
    return frame, bond


def fetch_cpi(offline: bool = False) -> Tuple[pd.Series, str]:
    """Monthly CPI index. Statistics Canada first, FRED second, cache last.

    Returns (series, a human-readable description of where it came from) so
    every deliverable can state which inflation series was actually used.
    """
    path = _cache_path("cpi")
    meta_path = config.DATA_RAW / "cpi_source.txt"

    def read_cache() -> Optional[Tuple[pd.Series, str]]:
        if not path.exists():
            return None
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        label = meta_path.read_text(encoding="utf-8").strip() if meta_path.exists() else "cached CPI"
        return frame.iloc[:, 0].dropna().rename("CPI"), label

    if offline or _cache_is_fresh(path):
        cached = read_cache()
        if cached is not None:
            return cached

    # Vector v41690973 is the "Canada; All-items" series inside table 18-10-0004-01.
    table = str(config.val("market.cpi_primary_table"))
    try:
        from stats_can import vectors_to_df

        frame = vectors_to_df("v41690973", periods=900)
        series = frame.iloc[:, 0].dropna()
        series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
        series = series.rename("CPI")
        label = (
            f"Statistics Canada table {table}, Consumer Price Index, "
            "all-items, Canada, monthly (vector v41690973)"
        )
        series.to_frame().to_csv(path)
        meta_path.write_text(label, encoding="utf-8")
        return series, label
    except Exception as exc:
        print(f"    ! Statistics Canada CPI failed ({type(exc).__name__}), trying FRED")

    fallback = str(config.val("market.cpi_fallback_series"))
    try:
        from pandas_datareader import data as pdr

        frame = pdr.DataReader(fallback, "fred", "1950-01-01")
        series = frame.iloc[:, 0].dropna()
        series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
        series = series.rename("CPI")
        label = f"US CPI (FRED series {fallback}): the Statistics Canada request failed"
        series.to_frame().to_csv(path)
        meta_path.write_text(label, encoding="utf-8")
        return series, label
    except Exception as exc:
        print(f"    ! FRED CPI failed ({type(exc).__name__}), falling back to cache")

    cached = read_cache()
    if cached is None:
        raise RuntimeError(
            "No CPI data: Statistics Canada and FRED both failed and no cache exists. "
            "Run once with a network connection to populate data/raw."
        )
    return cached


# --- Analysis ---------------------------------------------------------

@dataclass
class MarketResult:
    """Correlation and inflation findings, plus everything needed to caveat them."""

    monthly_returns: pd.DataFrame
    correlation: pd.DataFrame
    rolling_correlation: pd.DataFrame
    annual_returns: pd.DataFrame
    inflation: pd.Series
    regression: pd.DataFrame
    inflation_buckets: pd.DataFrame
    cpi_source: str
    bond_ticker: str
    benchmark: str
    window: int
    coverage: pd.DataFrame
    limitations: str = LIMITATIONS
    figures: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        bench = self.benchmark
        timber = list(config.val("market.timber_proxies"))
        farm = list(config.val("market.farmland_proxies"))
        present = [t for t in timber + farm if t in self.correlation.index]
        timber_present = [t for t in timber if t in self.correlation.index]

        corr_to_bench = self.correlation.loc[present, bench]
        hedges = self.regression.index[
            (self.regression["inflation_beta"] > 0) & (self.regression["p_value"] < 0.10)
        ].tolist()

        return {
            "cpi_source": self.cpi_source,
            "bond_ticker": self.bond_ticker,
            "benchmark": bench,
            "history_start": str(self.monthly_returns.index.min().date()),
            "history_end": str(self.monthly_returns.index.max().date()),
            "n_months": int(len(self.monthly_returns)),
            "n_annual_observations": int(len(self.annual_returns)),
            "mean_timber_corr_to_benchmark": float(
                self.correlation.loc[timber_present, bench].mean()
            ),
            "mean_proxy_corr_to_benchmark": float(corr_to_bench.mean()),
            "min_proxy_corr_to_benchmark": float(corr_to_bench.min()),
            "max_proxy_corr_to_benchmark": float(corr_to_bench.max()),
            "rolling_corr_min": float(self.rolling_correlation.min().min()),
            "rolling_corr_max": float(self.rolling_correlation.max().max()),
            "rolling_corr_latest": {
                c: (None if pd.isna(v) else float(v))
                for c, v in self.rolling_correlation.iloc[-1].items()
            },
            "inflation_betas": {
                idx: float(row["inflation_beta"]) for idx, row in self.regression.iterrows()
            },
            "significant_inflation_hedges": hedges,
            "n_significant_hedges": len(hedges),
            "limitations": self.limitations,
        }


def compute(offline: bool = False) -> MarketResult:
    """Download (or load), then run the correlation and inflation analysis."""
    config.ensure_directories()
    prices, bond = fetch_prices(offline)
    cpi, cpi_source = fetch_cpi(offline)
    benchmark = list(config.val("market.equity_benchmarks"))[0]
    window = int(config.num("market.rolling_window_months"))

    monthly = prices.pct_change().dropna(how="all")
    monthly = monthly[monthly.index >= prices.dropna(how="all").index.min()]

    coverage = pd.DataFrame(
        {
            "first_month": prices.apply(lambda s: s.first_valid_index()),
            "last_month": prices.apply(lambda s: s.last_valid_index()),
            "months": prices.notna().sum(),
        }
    )
    coverage["name"] = [DISPLAY_NAMES.get(t, t) for t in coverage.index]

    correlation = monthly.corr(min_periods=36)

    proxies = [
        t for t in list(config.val("market.timber_proxies"))
        + list(config.val("market.farmland_proxies"))
        if t in monthly.columns
    ]
    rolling = pd.DataFrame(
        {
            t: monthly[t].rolling(window, min_periods=window).corr(monthly[benchmark])
            for t in proxies
        }
    ).dropna(how="all")

    # Annual figures: December-to-December, so returns and inflation line up.
    annual_prices = prices.resample("YE").last()
    annual_returns = annual_prices.pct_change().dropna(how="all")
    annual_cpi = cpi.resample("YE").last()
    inflation = annual_cpi.pct_change().dropna()
    inflation.name = "inflation"

    shared = annual_returns.index.intersection(inflation.index)
    annual_returns = annual_returns.loc[shared]
    inflation = inflation.loc[shared]

    rows = []
    for column in annual_returns.columns:
        pair = pd.concat([annual_returns[column], inflation], axis=1).dropna()
        if len(pair) < 8:
            continue
        fit = stats.linregress(pair["inflation"], pair[column])
        rows.append(
            {
                "asset": DISPLAY_NAMES.get(column, column),
                "ticker": column,
                "inflation_beta": fit.slope,
                "std_error": fit.stderr,
                "t_stat": fit.slope / fit.stderr if fit.stderr else np.nan,
                "p_value": fit.pvalue,
                "r_squared": fit.rvalue ** 2,
                "intercept": fit.intercept,
                "n_years": len(pair),
            }
        )
    regression = pd.DataFrame(rows).set_index("asset")

    median_inflation = float(inflation.median())
    high = inflation >= median_inflation
    buckets = pd.DataFrame(
        {
            "name": [DISPLAY_NAMES.get(c, c) for c in annual_returns.columns],
            "high_inflation_mean": annual_returns[high].mean(),
            "low_inflation_mean": annual_returns[~high].mean(),
        }
    )
    buckets["difference"] = buckets["high_inflation_mean"] - buckets["low_inflation_mean"]
    buckets.attrs["median_inflation"] = median_inflation
    buckets.attrs["n_high"] = int(high.sum())
    buckets.attrs["n_low"] = int((~high).sum())

    return MarketResult(
        monthly_returns=monthly,
        correlation=correlation,
        rolling_correlation=rolling,
        annual_returns=annual_returns,
        inflation=inflation,
        regression=regression,
        inflation_buckets=buckets,
        cpi_source=cpi_source,
        bond_ticker=bond,
        benchmark=benchmark,
        window=window,
        coverage=coverage,
    )


# --- Figures ----------------------------------------------------------

def figure_correlation(res: MarketResult, path):
    """Correlation matrix of monthly returns."""
    import matplotlib.pyplot as plt

    corr = res.correlation
    labels = [DISPLAY_NAMES.get(c, c) for c in corr.columns]
    values = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    ax.grid(False)
    im = ax.imshow(values, cmap=style.CMAP_CORRELATION, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if i == j else "normal",
                    color=style.WHITE if abs(v) > 0.55 else style.INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Correlation of monthly total returns")
    cbar.outline.set_visible(False)

    bench = DISPLAY_NAMES.get(res.benchmark, res.benchmark)
    timber = [t for t in config.val("market.timber_proxies") if t in corr.index]
    mean_corr = corr.loc[timber, res.benchmark].mean()
    style.titled(
        ax,
        f"Timber REIT proxies move with equities: average correlation of {mean_corr:.2f} "
        f"to the {bench}",
        f"Monthly total returns, {res.monthly_returns.index.min():%b %Y} to "
        f"{res.monthly_returns.index.max():%b %Y}. Listed REITs are equities first: see the "
        "proxy caveat.",
    )
    fig.tight_layout()
    return style.save(fig, path, "Real market data from Yahoo Finance")


def figure_rolling_correlation(res: MarketResult, path):
    """Rolling correlation of each proxy against the Canadian equity benchmark."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for i, column in enumerate(res.rolling_correlation.columns):
        ax.plot(res.rolling_correlation.index, res.rolling_correlation[column],
                color=style.SERIES[i % len(style.SERIES)], lw=1.7,
                label=DISPLAY_NAMES.get(column, column))

    mean_line = res.rolling_correlation.mean(axis=1)
    ax.plot(mean_line.index, mean_line, color=style.INK, lw=2.4, ls=":",
            label="Average across proxies")
    ax.axhline(0, color=style.INK_SOFT, lw=1.0)

    ax.set_xlabel("")
    ax.set_ylabel(f"{res.window}-month rolling correlation")
    ax.set_ylim(-0.6, 1.0)
    ax.legend(loc="lower left", ncol=3)

    bench = DISPLAY_NAMES.get(res.benchmark, res.benchmark)
    lo = res.rolling_correlation.min().min()
    hi = res.rolling_correlation.max().max()
    latest = float(mean_line.dropna().iloc[-1])
    average = float(mean_line.mean())
    style.titled(
        ax,
        f"Correlation is not a constant: it swings between {lo:.2f} and {hi:.2f} "
        "over the period",
        f"{res.window}-month rolling correlation of monthly returns against the {bench}. "
        f"The cross-proxy average is {latest:.2f} today against a long-run average of "
        f"{average:.2f}, so any single-number correlation assumption in a portfolio model "
        "is fragile.",
    )
    fig.tight_layout()
    return style.save(fig, path, "Real market data from Yahoo Finance")


def figure_inflation(res: MarketResult, path):
    """Annual returns against annual CPI inflation, with fitted lines."""
    import matplotlib.pyplot as plt

    timber = [t for t in config.val("market.timber_proxies") if t in res.annual_returns.columns]
    farm = [t for t in config.val("market.farmland_proxies") if t in res.annual_returns.columns]
    chosen = timber + farm

    n = len(chosen)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.1 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    x = res.inflation * 100
    grid = np.linspace(float(x.min()) - 0.3, float(x.max()) + 0.3, 50)

    for ax, ticker in zip(axes, chosen):
        pair = pd.concat([res.annual_returns[ticker], res.inflation], axis=1).dropna()
        y = pair[ticker] * 100
        xi = pair["inflation"] * 100
        fit = stats.linregress(xi, y)

        ax.scatter(xi, y, s=44, color=style.FOREST, alpha=0.75, zorder=3,
                   edgecolors=style.WHITE, linewidths=0.6)
        ax.plot(grid, fit.intercept + fit.slope * grid, color=style.ACCENT, lw=2.0, zorder=4)
        ax.axhline(0, color=style.INK_SOFT, lw=0.9)

        beta = fit.slope  # percentage points of return per point of inflation
        ax.set_title(
            f"{DISPLAY_NAMES.get(ticker, ticker)}\n"
            f"β = {beta:+.2f} (s.e. {fit.stderr:.2f}, p = {fit.pvalue:.2f}), "
            f"R² = {fit.rvalue ** 2:.2f}, n = {len(pair)}",
            loc="left", fontsize=10.5, color=style.INK,
        )
        ax.set_ylabel("Annual total return (%)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Annual CPI inflation (%)")
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
            ax.tick_params(labelbottom=True)

    proxy_rows = res.regression[res.regression["ticker"].isin(chosen)]
    significant = proxy_rows[proxy_rows["p_value"] < 0.10]
    positive = int((proxy_rows["inflation_beta"] > 0).sum())
    if significant.empty:
        headline = (
            "No timber or farmland proxy shows a statistically significant inflation beta"
        )
    else:
        headline = (
            f"{len(significant)} of {len(proxy_rows)} proxies show a statistically "
            "significant inflation beta"
        )

    # Reserve the top strip before drawing the header, so the two never overlap.
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.text(0.006, 0.965, headline, ha="left", va="bottom",
             fontsize=13, fontweight="bold", color=style.INK)
    fig.text(
        0.006, 0.915,
        f"Annual total return regressed on annual CPI inflation; {positive} of {len(proxy_rows)} "
        f"point estimates are positive, none significant at the 10% level. With 12-41 annual\n"
        f"observations the standard errors swamp the slopes, so these are indicative only. "
        f"CPI source: {res.cpi_source}.",
        ha="left", va="bottom", fontsize=9.5, color=style.INK_SOFT,
    )
    return style.save(fig, path, "Real market data; annual observations are few, so treat betas as indicative")


def build(figures_dir=None, offline: bool = False) -> Tuple[MarketResult, Dict[str, str]]:
    """Run the market analysis and write the Phase 5 figures."""
    style.apply_style()
    figures_dir = figures_dir or config.FIGURES
    res = compute(offline)
    paths = {
        "correlation_heatmap": str(
            figure_correlation(res, figures_dir / "correlation_heatmap.png")
        ),
        "rolling_correlation": str(
            figure_rolling_correlation(res, figures_dir / "rolling_correlation.png")
        ),
        "inflation_regression": str(
            figure_inflation(res, figures_dir / "inflation_regression.png")
        ),
    }
    res.figures = paths
    return res, paths


if __name__ == "__main__":
    result, figs = build()
    s = result.summary()
    print(f"CPI source          : {s['cpi_source']}")
    print(f"Bond proxy          : {s['bond_ticker']}")
    print(f"History             : {s['history_start']} to {s['history_end']} "
          f"({s['n_months']} months, {s['n_annual_observations']} annual observations)")
    print(f"Mean timber corr    : {s['mean_timber_corr_to_benchmark']:.2f} vs {s['benchmark']}")
    print(f"Rolling corr range  : {s['rolling_corr_min']:.2f} to {s['rolling_corr_max']:.2f}")
    print("\nCorrelation matrix:")
    print(result.correlation.round(2).to_string())
    print("\nInflation regression:")
    print(result.regression.round(3).to_string())
    print("\nHigh vs low inflation years "
          f"(median {result.inflation_buckets.attrs['median_inflation']:.2%}, "
          f"{result.inflation_buckets.attrs['n_high']} high / "
          f"{result.inflation_buckets.attrs['n_low']} low):")
    print((result.inflation_buckets[["name", "high_inflation_mean",
                                     "low_inflation_mean", "difference"]]
           .round(3).to_string(index=False)))
    for name, path in figs.items():
        print(f"  figure {name}: {path}")
