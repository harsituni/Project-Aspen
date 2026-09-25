"""Monte Carlo risk: what can go wrong, and what is the flexibility worth?

Monetary figures are real CAD.

Three things are simulated over the 30-year hold, 10,000 paths at a time:

1. Stumpage price follows a mean-reverting Ornstein-Uhlenbeck process on the
   log price, pulled towards the deterministic real-growth path used in the
   base DCF. Timber prices are cyclical, not a random walk: a sawlog price
   that doubles does not stay doubled, so mean reversion is the honest
   choice, and it is also what makes harvest timing valuable.

2. Each year a wildfire or pest event occurs with a fixed probability and
   destroys a random share of the tract. Burned ground resets to age zero
   and has to be regenerated. No salvage value is assumed, which is
   conservative.

3. Two harvest strategies run on identical price and disturbance paths:

   * Scheduled: cut the even-flow target every year whatever the price.
   * Flexible: cut at the bottom of the permitted +/-15% band when the
     price is below its long-run mean, and at the top when it is above.

   The even-flow band is the only latitude a mill supply agreement leaves,
   so the gap between the two strategies is the value of that latitude: a
   real option on harvest timing. It is positive only because prices mean
   revert; under a random walk today's price is the best forecast of
   tomorrow's and there is nothing to wait for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config, style
from .dcf import initial_areas, regulated_growing_stock, solve_even_flow_schedule
from .growth import GrowthParams, volume
from .rotation import Economics, faustmann_integer_age

SCHEDULED = "Scheduled harvest"
FLEXIBLE = "Flexible harvest"


# --- Vectorised primitives -------------------------------------------

def harvest_all_sims(
    areas: np.ndarray,      # (n_sims, n_ages) hectares
    want: np.ndarray,       # (n_sims,) target volume this year
    floor: np.ndarray,      # (n_sims,) minimum acceptable volume
    per_ha: np.ndarray,     # (n_ages,) m3/ha by age
    rotation_age: int,
    min_age: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oldest-first harvest across every simulation at once.

    The trick is a reverse cumulative sum: for each age, `prior` is the volume
    already taken from older cohorts, so the volume taken at this age is simply
    the remaining appetite clipped to what stands there. That reproduces a
    strictly oldest-first sweep with no Python loop over cohorts.
    """
    remaining = areas.copy()
    cut_volume = np.zeros(areas.shape[0])
    cut_area = np.zeros(areas.shape[0])
    ages = np.arange(areas.shape[1])

    for low, high, appetite in (
        (rotation_age, areas.shape[1] - 1, want),
        (min_age, rotation_age - 1, floor),
    ):
        band = (ages >= low) & (ages <= high)
        available = remaining * per_ha[None, :] * band[None, :]
        # prior[s, a] = volume taken from cohorts strictly older than a
        reverse_cum = np.cumsum(available[:, ::-1], axis=1)[:, ::-1]
        prior = reverse_cum - available
        appetite_left = np.maximum(appetite - cut_volume, 0.0)[:, None]
        take_volume = np.clip(appetite_left - prior, 0.0, available)

        with np.errstate(divide="ignore", invalid="ignore"):
            take_area = np.where(per_ha[None, :] > 0, take_volume / per_ha[None, :], 0.0)
        remaining -= take_area
        cut_volume += take_volume.sum(axis=1)
        cut_area += take_area.sum(axis=1)

    remaining[:, 0] += cut_area
    return remaining, cut_volume, cut_area


def age_forward_all(areas: np.ndarray) -> np.ndarray:
    """Advance every cohort in every simulation by one year."""
    out = np.zeros_like(areas)
    out[:, 1:] = areas[:, :-1]
    out[:, -1] += areas[:, -1]
    return out


def irr_vectorised(cash_flows: np.ndarray, lo: float = -0.95, hi: float = 3.0,
                   iterations: int = 120) -> np.ndarray:
    """IRR for every row of a cash flow matrix, by simultaneous bisection.

    Bisection rather than Newton: 10,000 independent root finds in a Python
    loop would dominate the runtime, and bisection vectorises perfectly.
    """
    t = np.arange(cash_flows.shape[1], dtype=float)

    def npv_at(rates: np.ndarray) -> np.ndarray:
        return np.sum(cash_flows / (1.0 + rates[:, None]) ** t[None, :], axis=1)

    low = np.full(cash_flows.shape[0], lo)
    high = np.full(cash_flows.shape[0], hi)
    f_low = npv_at(low)
    bracketed = f_low * npv_at(high) < 0

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        f_mid = npv_at(mid)
        go_high = (f_mid * f_low) > 0
        low = np.where(go_high, mid, low)
        f_low = np.where(go_high, f_mid, f_low)
        high = np.where(go_high, high, mid)

    out = 0.5 * (low + high)
    return np.where(bracketed, out, np.nan)


# --- Price process ----------------------------------------------------

def simulate_prices(
    n_sims: int,
    years: int,
    rng: np.random.Generator,
    p0: float,
    growth: float,
    kappa: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact discretisation of an OU process on the log price.

    Returns (prices, long-run mean prices), both shaped (n_sims, years).
    The long-run mean drifts at the base-case real growth rate, so the
    simulation is centred on exactly the path the deterministic DCF assumes.
    """
    phi = np.exp(-kappa)
    step_sd = sigma * np.sqrt((1.0 - phi ** 2) / (2.0 * kappa))

    t = np.arange(1, years + 1, dtype=float)
    mu = np.log(p0) + np.log1p(growth) * t          # (years,)

    deviation = np.zeros(n_sims)
    log_prices = np.empty((n_sims, years))
    shocks = rng.standard_normal((n_sims, years))
    for i in range(years):
        deviation = phi * deviation + step_sd * shocks[:, i]
        log_prices[:, i] = mu[i] + deviation

    return np.exp(log_prices), np.exp(np.broadcast_to(mu, (n_sims, years)))


def simulate_disturbances(
    n_sims: int, years: int, rng: np.random.Generator,
    probability: float, mean_severity: float, concentration: float,
) -> np.ndarray:
    """Fraction of the tract destroyed each year, shaped (n_sims, years).

    Severity is Beta-distributed with the configured mean. With a low mean and
    modest concentration the distribution is J-shaped: most events are minor,
    a few are severe. That matches how wildfire loss actually behaves.
    """
    alpha = mean_severity * concentration
    beta = (1.0 - mean_severity) * concentration
    occurs = rng.random((n_sims, years)) < probability
    severity = rng.beta(alpha, beta, size=(n_sims, years))
    return np.where(occurs, severity, 0.0)


# --- Simulation engine -------------------------------------------------

@dataclass
class StrategyOutcome:
    """Per-path results for one harvest strategy."""

    name: str
    irr: np.ndarray
    npv: np.ndarray
    harvest_volumes: np.ndarray   # (n_sims, years)
    total_harvest: np.ndarray
    cash_flows: np.ndarray        # (n_sims, years + 1)

    def percentiles(self, hurdle: float) -> Dict[str, float]:
        finite = self.irr[np.isfinite(self.irr)]
        return {
            "p5": float(np.percentile(finite, 5)),
            "p25": float(np.percentile(finite, 25)),
            "p50": float(np.percentile(finite, 50)),
            "p75": float(np.percentile(finite, 75)),
            "p95": float(np.percentile(finite, 95)),
            "mean": float(np.mean(finite)),
            "prob_below_hurdle": float(np.mean(finite < hurdle)),
            "mean_npv": float(np.mean(self.npv)),
            "prob_negative_npv": float(np.mean(self.npv < 0)),
        }


def _run_paths(
    prices: np.ndarray,
    mean_prices: np.ndarray,
    disturbance: np.ndarray,
    target: float,
    rotation_age: int,
    e: Economics,
    p: GrowthParams,
    flexible: bool,
) -> StrategyOutcome:
    """Push every simulated path through the cohort model and price the result."""
    n_sims, years = prices.shape
    area = config.num("tract.area_ha")
    tolerance = config.num("dcf.even_flow_tolerance") * config.num(
        "monte_carlo.flexible_band_usage"
    )
    min_age = int(config.num("dcf.min_harvest_age"))
    hard_floor = target * (1.0 - config.num("dcf.even_flow_tolerance"))
    per_ha = volume(np.arange(p.max_age + 1, dtype=float), p)

    areas = np.tile(initial_areas(p), (n_sims, 1))
    volumes = np.zeros((n_sims, years))
    cut_areas = np.zeros((n_sims, years))
    burned_areas = np.zeros((n_sims, years))

    for i in range(years):
        areas = age_forward_all(areas)

        # Disturbance strikes the whole tract evenly, so the share of area lost
        # is also the share of standing volume lost. Burned ground regenerates.
        lost = disturbance[:, i]
        if np.any(lost > 0):
            burned = areas * lost[:, None]
            areas = areas - burned
            burned_total = burned.sum(axis=1)
            areas[:, 0] += burned_total
            burned_areas[:, i] = burned_total

        if flexible:
            cheap = prices[:, i] < mean_prices[:, i]
            want = np.where(cheap, target * (1.0 - tolerance), target * (1.0 + tolerance))
        else:
            want = np.full(n_sims, target)

        areas, vol, ha = harvest_all_sims(
            areas, want, np.full(n_sims, hard_floor), per_ha, rotation_age, min_age
        )
        volumes[:, i] = vol
        cut_areas[:, i] = ha

    # Exit valuation at each path's own year-30 price.
    exit_price = prices[:, -1]
    compound = (1.0 + e.discount_rate) ** rotation_age
    levt = (exit_price * per_ha[rotation_age] - e.regen_cost * compound) / (compound - 1.0)

    ages = np.arange(p.max_age + 1)
    wait = np.maximum(rotation_age - ages, 0)
    discount = (1.0 + e.discount_rate) ** (-wait)
    mature = ages >= rotation_age
    per_ha_value = np.where(
        mature[None, :],
        exit_price[:, None] * per_ha[None, :] + levt[:, None],
        (exit_price[:, None] * per_ha[rotation_age] + levt[:, None]) * discount[None, :],
    )
    tv = np.sum(areas * per_ha_value, axis=1) - (e.mgmt_cost / e.discount_rate) * area
    tv *= 1.0 - config.num("dcf.disposition_cost_pct")

    revenue = volumes * prices
    regen = (cut_areas + burned_areas) * e.regen_cost
    mgmt = area * e.mgmt_cost
    if bool(config.val("carbon.enabled")):
        carbon = (
            area
            * config.num("carbon.eligible_area_fraction")
            * config.num("carbon.sequestration_rate")
            * config.num("carbon.price_per_tonne")
        )
    else:
        carbon = 0.0

    annual = revenue + carbon - regen - mgmt
    annual[:, -1] += tv

    acquisition = config.num("dcf.purchase_price") * (
        1.0 + config.num("dcf.acquisition_cost_pct")
    )
    cash_flows = np.concatenate((np.full((n_sims, 1), -acquisition), annual), axis=1)

    t = np.arange(years + 1, dtype=float)
    npvs = np.sum(cash_flows / (1.0 + e.discount_rate) ** t[None, :], axis=1)

    return StrategyOutcome(
        name=FLEXIBLE if flexible else SCHEDULED,
        irr=irr_vectorised(cash_flows),
        npv=npvs,
        harvest_volumes=volumes,
        total_harvest=volumes.sum(axis=1),
        cash_flows=cash_flows,
    )


@dataclass
class MonteCarloResult:
    """Everything Phase 4 hands to the deck, the workbook and the study guide."""

    scheduled: StrategyOutcome
    flexible: StrategyOutcome
    no_disturbance: StrategyOutcome
    prices: np.ndarray
    mean_prices: np.ndarray
    disturbance: np.ndarray
    hurdle: float
    n_sims: int
    years: int
    seed: int

    @property
    def flexibility_value(self) -> float:
        """Mean NPV gain from using the even-flow band, in dollars."""
        return float(np.mean(self.flexible.npv) - np.mean(self.scheduled.npv))

    @property
    def disturbance_cost(self) -> float:
        """Mean NPV lost to fire and pests, measured on identical price paths."""
        return float(np.mean(self.no_disturbance.npv) - np.mean(self.scheduled.npv))

    def summary(self) -> Dict[str, float]:
        sched = self.scheduled.percentiles(self.hurdle)
        flex = self.flexible.percentiles(self.hurdle)
        area = config.num("tract.area_ha")
        events = float(np.mean(np.sum(self.disturbance > 0, axis=1)))
        worst = float(np.max(self.disturbance))
        return {
            "n_simulations": self.n_sims,
            "seed": self.seed,
            "hurdle": self.hurdle,
            "scheduled_p5": sched["p5"],
            "scheduled_p25": sched["p25"],
            "scheduled_p50": sched["p50"],
            "scheduled_p75": sched["p75"],
            "scheduled_p95": sched["p95"],
            "scheduled_mean": sched["mean"],
            "scheduled_prob_below_hurdle": sched["prob_below_hurdle"],
            "scheduled_mean_npv": sched["mean_npv"],
            "scheduled_prob_negative_npv": sched["prob_negative_npv"],
            "flexible_p5": flex["p5"],
            "flexible_p25": flex["p25"],
            "flexible_p50": flex["p50"],
            "flexible_p75": flex["p75"],
            "flexible_p95": flex["p95"],
            "flexible_mean": flex["mean"],
            "flexible_prob_below_hurdle": flex["prob_below_hurdle"],
            "flexible_mean_npv": flex["mean_npv"],
            "flexibility_value": self.flexibility_value,
            "flexibility_value_per_ha": self.flexibility_value / area,
            "flexibility_irr_gain": flex["mean"] - sched["mean"],
            "disturbance_cost": self.disturbance_cost,
            "disturbance_cost_per_ha": self.disturbance_cost / area,
            "mean_disturbance_events": events,
            "worst_single_event_share": worst,
            "prob_any_disturbance": float(np.mean(np.any(self.disturbance > 0, axis=1))),
        }

    def summary_table(self) -> pd.DataFrame:
        """Percentile table in the shape the Excel sheet and the deck want."""
        rows = []
        for outcome in (self.scheduled, self.flexible):
            stats = outcome.percentiles(self.hurdle)
            rows.append(
                {
                    "Strategy": outcome.name,
                    "P5 IRR": stats["p5"],
                    "P25 IRR": stats["p25"],
                    "Median IRR": stats["p50"],
                    "P75 IRR": stats["p75"],
                    "P95 IRR": stats["p95"],
                    "Mean IRR": stats["mean"],
                    "P(IRR < hurdle)": stats["prob_below_hurdle"],
                    "Mean NPV": stats["mean_npv"],
                }
            )
        return pd.DataFrame(rows).set_index("Strategy")


def run(n_sims: Optional[int] = None, seed: Optional[int] = None) -> MonteCarloResult:
    """Run the full Monte Carlo. Deterministic for a given seed."""
    p = GrowthParams.from_config()
    e = Economics.from_config()
    years = int(config.num("dcf.holding_period_years"))
    n_sims = int(config.num("monte_carlo.n_simulations")) if n_sims is None else int(n_sims)
    seed = int(config.num("monte_carlo.random_seed")) if seed is None else int(seed)

    rotation_age = faustmann_integer_age(p, e)
    target = solve_even_flow_schedule(rotation_age, p, years).target

    rng = np.random.default_rng(seed)
    prices, mean_prices = simulate_prices(
        n_sims, years, rng,
        p0=e.price,
        growth=config.num("economics.real_price_growth"),
        kappa=config.num("monte_carlo.mean_reversion_speed"),
        sigma=config.num("monte_carlo.price_volatility"),
    )
    disturbance = simulate_disturbances(
        n_sims, years, rng,
        probability=config.num("monte_carlo.disturbance_probability"),
        mean_severity=config.num("monte_carlo.disturbance_severity_mean"),
        concentration=config.num("monte_carlo.disturbance_severity_concentration"),
    )
    quiet = np.zeros_like(disturbance)

    common = dict(target=target, rotation_age=rotation_age, e=e, p=p)
    scheduled = _run_paths(prices, mean_prices, disturbance, flexible=False, **common)
    flexible = _run_paths(prices, mean_prices, disturbance, flexible=True, **common)
    no_disturbance = _run_paths(prices, mean_prices, quiet, flexible=False, **common)

    return MonteCarloResult(
        scheduled=scheduled,
        flexible=flexible,
        no_disturbance=no_disturbance,
        prices=prices,
        mean_prices=mean_prices,
        disturbance=disturbance,
        hurdle=config.num("dcf.hurdle_rate"),
        n_sims=n_sims,
        years=years,
        seed=seed,
    )


# --- Figures ------------------------------------------------------------

def figure_irr_histogram(res: MonteCarloResult, path):
    """Overlaid IRR distributions for the two strategies, against the hurdle."""
    import matplotlib.pyplot as plt

    sched = res.scheduled.irr[np.isfinite(res.scheduled.irr)] * 100
    flex = res.flexible.irr[np.isfinite(res.flexible.irr)] * 100
    hurdle = res.hurdle * 100
    s_stats = res.scheduled.percentiles(res.hurdle)
    f_stats = res.flexible.percentiles(res.hurdle)

    lo = min(sched.min(), flex.min())
    hi = max(sched.max(), flex.max())
    bins = np.linspace(lo, hi, 70)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    # Solid fill for the base strategy, heavy outline for the alternative: the
    # two stay distinguishable when the page is printed in black and white.
    ax.hist(sched, bins=bins, color=style.FOREST_LIGHT, edgecolor="none", zorder=2,
            label=f"{SCHEDULED}, median {s_stats['p50']:.1%}")
    ax.hist(flex, bins=bins, histtype="step", color=style.ACCENT, lw=1.8, zorder=3,
            label=f"{FLEXIBLE}, median {f_stats['p50']:.1%}")

    ax.axvline(hurdle, color=style.INK, lw=2.0, zorder=4)
    top = ax.get_ylim()[1] * 1.08
    ax.set_ylim(0, top)
    ax.text(hurdle - 0.1, top * 0.985, f"Hurdle {res.hurdle:.1%}", rotation=90,
            ha="right", va="top", fontsize=9.5, color=style.INK, fontweight="bold")

    ax.axvspan(lo, hurdle, color=style.SAND, alpha=0.55, zorder=0)
    ax.text(
        lo + (hurdle - lo) * 0.5, top * 0.55,
        f"Below hurdle\n{s_stats['prob_below_hurdle']:.0%} of paths scheduled\n"
        f"{f_stats['prob_below_hurdle']:.0%} flexible",
        ha="center", va="center", fontsize=9.5, color=style.INK,
    )

    ax.set_xlabel("Project IRR (%)")
    ax.set_ylabel("Simulated paths")
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper right")

    style.titled(
        ax,
        f"{s_stats['prob_below_hurdle']:.0%} of paths miss the {res.hurdle:.0%} hurdle; "
        f"harvest flexibility adds {style.money(res.flexibility_value)} of value",
        f"{res.n_sims:,} paths · mean-reverting prices, random wildfire and pest losses · "
        f"scheduled P5-P95 {s_stats['p5']:.1%} to {s_stats['p95']:.1%} · seed {res.seed}",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative volatility and disturbance assumptions")


def figure_price_fan(res: MonteCarloResult, path):
    """Fan chart of the simulated stumpage price paths."""
    import matplotlib.pyplot as plt

    # Anchor the fan at year 0, where today's price is known with certainty, so
    # the chart visibly spreads from a point rather than starting mid-air.
    p0 = config.num("economics.net_stumpage_price")
    years = np.arange(0, res.years + 1)
    pct = np.percentile(res.prices, [5, 25, 50, 75, 95], axis=0)
    pct = np.column_stack((np.full(5, p0), pct))
    mean_path = np.concatenate(([p0], res.mean_prices[0]))
    paths = np.column_stack((np.full(res.prices.shape[0], p0), res.prices))

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.fill_between(years, pct[0], pct[4], color=style.FOREST_PALE, alpha=0.55,
                    label="5th–95th percentile")
    ax.fill_between(years, pct[1], pct[3], color=style.FOREST_LIGHT, alpha=0.55,
                    label="25th–75th percentile")
    ax.plot(years, pct[2], color=style.FOREST, lw=2.2, label="Median path")
    ax.plot(years, mean_path, color=style.ACCENT, lw=1.8, ls="--",
            label="Long-run mean (base case)")

    for i in range(3):  # a few individual paths, so the cycles are visible
        ax.plot(years, paths[i], color=style.INK_SOFT, lw=0.7, alpha=0.55,
                label="Individual paths" if i == 0 else None)

    ax.set_xlabel("Year")
    ax.set_ylabel("Net stumpage price ($/m³)")
    ax.set_xlim(0, res.years)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.legend(loc="upper left", ncol=2)

    half_life = np.log(2) / config.num("monte_carlo.mean_reversion_speed")
    style.titled(
        ax,
        "Prices swing widely year to year but are pulled back to the long-run mean",
        f"Ornstein-Uhlenbeck on the log price · {config.num('monte_carlo.price_volatility'):.0%} "
        f"annual volatility · shock half-life {half_life:.1f} years · "
        f"the band stops widening, which is what distinguishes this from a random walk",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative volatility assumptions")


def build(figures_dir=None) -> Tuple[MonteCarloResult, Dict[str, str]]:
    """Run the simulation and write the Phase 4 figures."""
    style.apply_style()
    figures_dir = figures_dir or config.FIGURES
    res = run()
    paths = {
        "irr_histogram": str(figure_irr_histogram(res, figures_dir / "irr_histogram.png")),
        "price_fan": str(figure_price_fan(res, figures_dir / "price_fan.png")),
    }
    return res, paths


if __name__ == "__main__":
    config.ensure_directories()
    result, figs = build()
    s = result.summary()
    print(f"Simulations         : {s['n_simulations']:,} (seed {s['seed']})")
    print("                      P5      P25     P50     P75     P95    P(<hurdle)")
    for name, key in ((SCHEDULED, "scheduled"), (FLEXIBLE, "flexible")):
        print(f"{name:20s}" + "".join(
            f"{s[f'{key}_{q}']:7.2%} " for q in ("p5", "p25", "p50", "p75", "p95")
        ) + f"  {s[f'{key}_prob_below_hurdle']:6.1%}")
    print(f"Value of flexibility: ${s['flexibility_value']:,.0f} "
          f"(${s['flexibility_value_per_ha']:,.0f}/ha, "
          f"{s['flexibility_irr_gain']*10000:+.0f} bps of mean IRR)")
    print(f"Disturbance cost    : ${s['disturbance_cost']:,.0f} "
          f"(${s['disturbance_cost_per_ha']:,.0f}/ha)")
    print(f"Disturbance events  : {s['mean_disturbance_events']:.2f} per path on average; "
          f"{s['prob_any_disturbance']:.0%} of paths see at least one; "
          f"worst single event {s['worst_single_event_share']:.1%} of the tract")
    for name, path in figs.items():
        print(f"  figure {name}: {path}")
