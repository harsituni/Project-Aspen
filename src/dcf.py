"""Acquisition DCF: what is the hypothetical tract worth, and at what return?

Monetary figures are real CAD.

The tract is modelled as 151 single-year age cohorts (hectares by age). Each
year the scheduler harvests oldest-first, and harvested hectares return to
age zero. The annual cut is governed by an even-flow rule: we solve for the
largest constant target the tract can sustain for 30 years while every single
year stays inside +/-15% of that target. That is the classic forest-regulation
problem, and it is what a mill supply agreement would force on the owner.

Exit value follows standard forest valuation. The value of land carrying a
stand of age A, with rotation R, is

    A >= R :  P*V(A) + LEVt                      (cut now, keep the land)
    A <  R : (P*V(R) + LEVt) / (1+r)^(R-A)       (wait, then cut)

where LEVt is the land expectation value before perpetual management cost.
Subtracting m/r once over the whole tract gives the reported terminal value,
which is then split into the bare-land and standing-inventory components the
investment committee expects to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config, style
from .growth import GrowthParams, volume
from .rotation import Economics, faustmann_integer_age

# --- Tract state -----------------------------------------------------

def initial_areas(p: GrowthParams) -> np.ndarray:
    """Hectares by single year of age, spread evenly inside each 10-year class."""
    areas = np.zeros(p.max_age + 1, dtype=float)
    for label, hectares in config.val("tract.age_class_distribution_ha").items():
        low, high = (int(x) for x in str(label).split("-"))
        span = high - low + 1
        areas[low:high + 1] += float(hectares) / span
    return areas


def standing_volume(areas: np.ndarray, p: GrowthParams) -> float:
    """Total merchantable volume on the tract, m3."""
    ages = np.arange(areas.size, dtype=float)
    return float(np.sum(areas * volume(ages, p)))


def age_forward(areas: np.ndarray) -> np.ndarray:
    """Advance all cohorts by one year, pinning anything past the top class."""
    out = np.zeros_like(areas)
    out[1:] = areas[:-1]
    out[-1] += areas[-1]  # stands already at max age stay there
    return out


# --- Harvest scheduling ----------------------------------------------

def regulated_growing_stock(rotation_age: int, p: GrowthParams) -> float:
    """Standing volume a fully regulated forest of this rotation would carry, m3.

    A regulated forest holds area/R hectares in every age class from 1 to R.
    This is the natural floor for ending inventory: harvest more than this and
    the tract is being liquidated rather than managed.
    """
    area = config.num("tract.area_ha")
    ages = np.arange(1, rotation_age + 1, dtype=float)
    return float((area / rotation_age) * np.sum(volume(ages, p)))


@dataclass
class HarvestSchedule:
    """Annual harvest volumes and areas over the holding period."""

    years: np.ndarray
    volumes: np.ndarray          # m3 harvested per year
    areas_cut: np.ndarray        # hectares harvested per year
    target: float                # even-flow target, m3/yr
    tolerance: float
    rotation_age: int
    ending_areas: np.ndarray     # age distribution at the end of the hold
    opening_inventory: float     # m3 standing at acquisition
    closing_inventory: float     # m3 standing at exit
    required_ending_inventory: float  # regulated growing stock floor, m3
    binding_years: int           # years that hit the lower even-flow bound
    inventory_constraint_binds: bool  # True when ending inventory set the cut level

    @property
    def mean_volume(self) -> float:
        return float(self.volumes.mean())

    @property
    def max_deviation(self) -> float:
        """Largest absolute deviation from the mean, as a fraction."""
        m = self.mean_volume
        return float(np.max(np.abs(self.volumes - m)) / m) if m else 0.0


def _harvest_one_year(
    areas: np.ndarray,
    target: float,
    rotation_age: int,
    min_age: int,
    floor: float,
    p: GrowthParams,
) -> Tuple[np.ndarray, float, float]:
    """Cut oldest-first up to `target`; dip below rotation age only to reach `floor`.

    Returns (areas after harvest, volume cut, hectares cut).
    """
    ages = np.arange(areas.size)
    per_ha = volume(ages.astype(float), p)
    remaining = areas.copy()
    cut_volume = 0.0
    cut_area = 0.0

    # Pass 1: stands at or past the rotation age, oldest first.
    # Pass 2: immature but merchantable stands, again oldest first, but only
    # as far as the even-flow floor: we never cut young wood for extra profit.
    for pass_no, (low, high, limit) in enumerate(
        ((rotation_age, areas.size - 1, target), (min_age, rotation_age - 1, floor))
    ):
        if limit <= cut_volume:
            continue
        for age in range(high, low - 1, -1):
            if cut_volume >= limit - 1e-9:
                break
            v = per_ha[age]
            if v <= 0 or remaining[age] <= 0:
                continue
            wanted_volume = limit - cut_volume
            ha = min(remaining[age], wanted_volume / v)
            remaining[age] -= ha
            cut_volume += ha * v
            cut_area += ha

    remaining[0] += cut_area  # harvested ground is regenerated immediately
    return remaining, cut_volume, cut_area


def _simulate_schedule(
    target: float,
    years: int,
    rotation_age: int,
    min_age: int,
    tolerance: float,
    p: GrowthParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the scheduler for a given target. Returns (volumes, areas cut, ending areas)."""
    areas = initial_areas(p)
    floor = target * (1.0 - tolerance)
    volumes = np.zeros(years)
    cut_areas = np.zeros(years)
    for t in range(years):
        areas = age_forward(areas)
        areas, vol, ha = _harvest_one_year(areas, target, rotation_age, min_age, floor, p)
        volumes[t] = vol
        cut_areas[t] = ha
    return volumes, cut_areas, areas


def solve_even_flow_schedule(
    rotation_age: int,
    p: Optional[GrowthParams] = None,
    years: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> HarvestSchedule:
    """Find the largest even-flow cut the tract can sustain for the whole hold.

    Bisection on the target. A target is feasible when both forest-regulation
    constraints hold:

    * no single year falls below (1 - tolerance) x target, so the mill supply
      agreement is never broken; and
    * standing volume at exit is still at least the growing stock of a fully
      regulated forest, so the buyer is handed a working forest rather than
      cut-over ground.

    Without the second constraint the optimiser simply liquidates the mature
    surplus and walks away, which flatters the IRR and would not survive due
    diligence.
    """
    p = p or GrowthParams.from_config()
    years = years or int(config.num("dcf.holding_period_years"))
    tolerance = config.num("dcf.even_flow_tolerance") if tolerance is None else tolerance
    min_age = int(config.num("dcf.min_harvest_age"))
    required_ending = regulated_growing_stock(rotation_age, p) * config.num(
        "dcf.min_ending_inventory_multiple"
    )

    def evaluate(target: float):
        volumes, cut_areas, ending = _simulate_schedule(
            target, years, rotation_age, min_age, tolerance, p
        )
        even_flow_ok = bool(np.min(volumes) >= target * (1.0 - tolerance) - 1e-6)
        ending_stock = standing_volume(ending, p)
        return even_flow_ok, ending_stock >= required_ending, volumes, cut_areas, ending

    hi = standing_volume(initial_areas(p), p)  # certainly infeasible: cut everything at once
    lo = 0.0
    if all(evaluate(hi)[:2]):
        lo = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if all(evaluate(mid)[:2]):
                lo = mid
            else:
                hi = mid
    target = lo

    even_flow_ok, _, volumes, cut_areas, ending = evaluate(target)
    closing = standing_volume(ending, p)
    floor = target * (1.0 - tolerance)
    return HarvestSchedule(
        years=np.arange(1, years + 1),
        volumes=volumes,
        areas_cut=cut_areas,
        target=target,
        tolerance=tolerance,
        rotation_age=rotation_age,
        ending_areas=ending,
        opening_inventory=standing_volume(initial_areas(p), p),
        closing_inventory=closing,
        required_ending_inventory=required_ending,
        binding_years=int(np.sum(volumes <= floor * 1.001)),
        # Which constraint set the cut level: if even flow still has slack at the
        # optimum, it was the ending-inventory floor that stopped us.
        inventory_constraint_binds=bool(closing <= required_ending * 1.02),
    )


# --- Valuation helpers ------------------------------------------------

def lev_timber_only(rotation_age: float, price: float, e: Economics, p: GrowthParams) -> float:
    """Land expectation value before the perpetual management charge, $/ha."""
    compound = (1.0 + e.discount_rate) ** rotation_age
    return (price * volume(rotation_age, p) - e.regen_cost * compound) / (compound - 1.0)


def terminal_value(
    areas: np.ndarray,
    rotation_age: int,
    exit_price: float,
    e: Economics,
    p: GrowthParams,
) -> Tuple[float, float, float]:
    """Exit value of the tract. Returns (total, bare-land component, inventory component)."""
    levt = lev_timber_only(rotation_age, exit_price, e, p)
    lev_net = levt - e.mgmt_cost / e.discount_rate
    total_area = float(areas.sum())

    ages = np.arange(areas.size, dtype=float)
    mature = ages >= rotation_age
    per_ha = np.empty_like(ages)
    per_ha[mature] = exit_price * volume(ages[mature], p) + levt
    wait = rotation_age - ages[~mature]
    per_ha[~mature] = (
        exit_price * volume(float(rotation_age), p) + levt
    ) / (1.0 + e.discount_rate) ** wait

    total = float(np.sum(areas * per_ha)) - (e.mgmt_cost / e.discount_rate) * total_area
    land_component = lev_net * total_area
    return total, land_component, total - land_component


def npv(cash_flows: np.ndarray, rate: float) -> float:
    """NPV of a year-indexed cash flow array where element 0 is time zero."""
    t = np.arange(cash_flows.size, dtype=float)
    return float(np.sum(cash_flows / (1.0 + rate) ** t))


def irr(cash_flows: np.ndarray, lo: float = -0.95, hi: float = 3.0) -> float:
    """Internal rate of return by bracketing, or NaN when no sign change exists."""
    cf = np.asarray(cash_flows, dtype=float)
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return float("nan")
    f_lo, f_hi = npv(cf, lo), npv(cf, hi)
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(lambda r: npv(cf, r), lo, hi, xtol=1e-10, maxiter=200))


# --- The model --------------------------------------------------------

@dataclass
class DCFResult:
    """Everything the deck, the workbook and the README need from Phase 3."""

    table: pd.DataFrame
    cash_flows: np.ndarray
    npv: float
    irr: float
    moic: float
    breakeven_price: float
    purchase_price: float
    equity_outflow: float
    terminal_total: float
    terminal_land: float
    terminal_inventory: float
    schedule: HarvestSchedule
    rotation_age: int
    discount_rate: float
    hurdle_rate: float
    pv_operating: float
    carbon_enabled: bool
    grids: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def recommendation(self) -> str:
        return "BUY" if (self.npv > 0 and self.irr >= self.hurdle_rate) else "PASS"

    @property
    def price_per_ha(self) -> float:
        return self.purchase_price / config.num("tract.area_ha")

    @property
    def breakeven_per_ha(self) -> float:
        return self.breakeven_price / config.num("tract.area_ha")

    def summary(self) -> Dict[str, float | str | int]:
        s = self.schedule
        return {
            "recommendation": self.recommendation,
            "purchase_price": self.purchase_price,
            "purchase_price_per_ha": self.price_per_ha,
            "npv": self.npv,
            "irr": self.irr,
            "moic": self.moic,
            "hurdle_rate": self.hurdle_rate,
            "discount_rate": self.discount_rate,
            "breakeven_price": self.breakeven_price,
            "breakeven_price_per_ha": self.breakeven_per_ha,
            "breakeven_premium_vs_ask": self.breakeven_price / self.purchase_price - 1.0,
            "rotation_age": self.rotation_age,
            "even_flow_target_m3": s.target,
            "mean_annual_harvest_m3": s.mean_volume,
            "max_even_flow_deviation": s.max_deviation,
            "even_flow_binding_years": s.binding_years,
            "opening_inventory_m3": s.opening_inventory,
            "closing_inventory_m3": s.closing_inventory,
            "required_ending_inventory_m3": s.required_ending_inventory,
            "inventory_constraint_binds": s.inventory_constraint_binds,
            "total_harvest_m3": float(s.volumes.sum()),
            "terminal_value": self.terminal_total,
            "terminal_land_component": self.terminal_land,
            "terminal_inventory_component": self.terminal_inventory,
            "pv_operating_cash_flows": self.pv_operating,
            "carbon_enabled": self.carbon_enabled,
        }


def run_dcf(
    price: Optional[float] = None,
    discount_rate: Optional[float] = None,
    purchase_price: Optional[float] = None,
    price_growth: Optional[float] = None,
    p: Optional[GrowthParams] = None,
    carbon: Optional[bool] = None,
) -> DCFResult:
    """Build the 30-year acquisition cash flows and the headline return metrics.

    Overrides exist so the sensitivity grids can re-run the whole model, including
    re-solving the Faustmann rotation, rather than flexing a single output.
    """
    p = p or GrowthParams.from_config()
    base = Economics.from_config()
    price = base.price if price is None else price
    discount_rate = base.discount_rate if discount_rate is None else discount_rate
    growth_rate = config.num("economics.real_price_growth") if price_growth is None else price_growth
    purchase_price = config.num("dcf.purchase_price") if purchase_price is None else purchase_price
    carbon_on = bool(config.val("carbon.enabled")) if carbon is None else carbon

    e = Economics(price, base.regen_cost, base.mgmt_cost, discount_rate)
    years = int(config.num("dcf.holding_period_years"))
    area = config.num("tract.area_ha")
    hurdle = config.num("dcf.hurdle_rate")

    rotation_age = faustmann_integer_age(p, e)
    schedule = solve_even_flow_schedule(rotation_age, p, years)

    t = np.arange(1, years + 1)
    prices = price * (1.0 + growth_rate) ** t
    revenue = schedule.volumes * prices
    regen = schedule.areas_cut * e.regen_cost
    mgmt = np.full(years, area * e.mgmt_cost)

    if carbon_on:
        carbon_revenue = np.full(
            years,
            area
            * config.num("carbon.eligible_area_fraction")
            * config.num("carbon.sequestration_rate")
            * config.num("carbon.price_per_tonne"),
        )
    else:
        carbon_revenue = np.zeros(years)

    exit_price = price * (1.0 + growth_rate) ** years
    tv_total, tv_land, tv_inventory = terminal_value(
        schedule.ending_areas, rotation_age, exit_price, e, p
    )
    tv_net = tv_total * (1.0 - config.num("dcf.disposition_cost_pct"))

    operating = revenue + carbon_revenue - regen - mgmt
    annual = operating.copy()
    annual[-1] += tv_net

    acquisition = purchase_price * (1.0 + config.num("dcf.acquisition_cost_pct"))
    cash_flows = np.concatenate(([-acquisition], annual))

    project_npv = npv(cash_flows, discount_rate)
    project_irr = irr(cash_flows)
    inflows = float(np.sum(annual))
    moic = inflows / acquisition
    pv_operating = npv(np.concatenate(([0.0], annual)), discount_rate)
    breakeven = pv_operating / (1.0 + config.num("dcf.acquisition_cost_pct"))

    table = pd.DataFrame(
        {
            "year": np.concatenate(([0], t)),
            "harvest_volume_m3": np.concatenate(([0.0], schedule.volumes)),
            "harvest_area_ha": np.concatenate(([0.0], schedule.areas_cut)),
            "stumpage_price": np.concatenate(([price], prices)),
            "harvest_revenue": np.concatenate(([0.0], revenue)),
            "carbon_revenue": np.concatenate(([0.0], carbon_revenue)),
            "regeneration_cost": np.concatenate(([0.0], -regen)),
            "management_cost": np.concatenate(([0.0], -mgmt)),
            "acquisition": np.concatenate(([-acquisition], np.zeros(years))),
            "terminal_value": np.concatenate((np.zeros(years), [tv_net])),
            "net_cash_flow": cash_flows,
        }
    )
    table["discount_factor"] = 1.0 / (1.0 + discount_rate) ** table["year"]
    table["pv_net_cash_flow"] = table["net_cash_flow"] * table["discount_factor"]
    table["cumulative_pv"] = table["pv_net_cash_flow"].cumsum()

    return DCFResult(
        table=table,
        cash_flows=cash_flows,
        npv=project_npv,
        irr=project_irr,
        moic=moic,
        breakeven_price=breakeven,
        purchase_price=purchase_price,
        equity_outflow=acquisition,
        terminal_total=tv_net,
        terminal_land=tv_land * (1.0 - config.num("dcf.disposition_cost_pct")),
        terminal_inventory=tv_inventory * (1.0 - config.num("dcf.disposition_cost_pct")),
        schedule=schedule,
        rotation_age=rotation_age,
        discount_rate=discount_rate,
        hurdle_rate=hurdle,
        pv_operating=pv_operating,
        carbon_enabled=carbon_on,
    )


# --- Two-way sensitivity grids ---------------------------------------

def sensitivity_price_vs_discount(base: DCFResult) -> pd.DataFrame:
    """IRR across stumpage price (+/-30%) and real discount rate.

    The discount rate moves the IRR through two channels only: it re-solves the
    Faustmann rotation, which reshapes the harvest schedule, and it prices the
    year-30 exit. Operating cash flows themselves are unaffected.
    """
    price0 = config.num("economics.net_stumpage_price")
    prices = price0 * np.array([0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30])
    rates = np.array([0.04, 0.05, 0.06, 0.07, 0.08])
    grid = np.empty((prices.size, rates.size))
    for i, pr in enumerate(prices):
        for j, r in enumerate(rates):
            grid[i, j] = run_dcf(price=pr, discount_rate=r).irr
    return pd.DataFrame(
        grid,
        index=pd.Index([f"${p:,.0f}" for p in prices], name="Net stumpage ($/m³)"),
        columns=pd.Index([f"{r:.0%}" for r in rates], name="Real discount rate"),
    )


def sensitivity_price_paid_vs_growth(base: DCFResult) -> pd.DataFrame:
    """IRR across purchase price and real stumpage price growth."""
    ask = config.num("dcf.purchase_price")
    offers = ask * np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    growths = np.array([-0.005, 0.0, 0.005, 0.010, 0.015])
    grid = np.empty((offers.size, growths.size))
    for i, offer in enumerate(offers):
        for j, g in enumerate(growths):
            grid[i, j] = run_dcf(purchase_price=offer, price_growth=g).irr
    return pd.DataFrame(
        grid,
        index=pd.Index([style.money(o, 0) for o in offers], name="Purchase price"),
        columns=pd.Index([f"{g:+.1%}" for g in growths], name="Real price growth"),
    )


# --- Figures ----------------------------------------------------------

def figure_cash_flows(res: DCFResult, path):
    """Stacked annual cash flows with the acquisition and the exit called out."""
    import matplotlib.pyplot as plt

    tab = res.table
    years = tab["year"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    rev = tab["harvest_revenue"].to_numpy() / 1e6
    carb = tab["carbon_revenue"].to_numpy() / 1e6
    tv = tab["terminal_value"].to_numpy() / 1e6
    costs = (tab["regeneration_cost"] + tab["management_cost"]).to_numpy() / 1e6
    acq = tab["acquisition"].to_numpy() / 1e6

    ax.bar(years, rev, color=style.FOREST, label="Harvest revenue", width=0.74)
    bottom = rev.copy()
    if carb.any():
        ax.bar(years, carb, bottom=bottom, color=style.FOREST_LIGHT,
               label="Carbon revenue", width=0.74)
        bottom = bottom + carb
    # Outlined: the terminal bar is pale and would otherwise vanish on a
    # greyscale printout.
    ax.bar(years, tv, bottom=bottom, color=style.SAND_DEEP, edgecolor=style.INK_SOFT,
           linewidth=0.7, label="Terminal value (net of disposition)", width=0.74)
    ax.bar(years, costs, color=style.ACCENT, label="Regeneration + management", width=0.74)
    ax.bar(years, acq, color=style.INK_SOFT, label="Acquisition (incl. closing costs)", width=0.74)

    ax.axhline(0, color=style.INK, lw=1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Real cash flow ($ millions)")
    ax.set_xlim(-0.8, res.table["year"].max() + 0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}M")
    ax.legend(loc="upper left", ncol=2)

    ax.annotate(
        f"Exit in year {int(years[-1])}\n{style.money(res.terminal_total)}",
        xy=(years[-1], (rev[-1] + tv[-1])),
        xytext=(years[-1] - 7.5, (rev[-1] + tv[-1]) * 1.02),
        fontsize=9, color=style.INK, ha="right",
        arrowprops=dict(arrowstyle="->", color=style.INK_SOFT, lw=1.0),
    )

    avg_cost = -float(costs[1:].mean())
    avg_rev = float(rev[1:].mean())
    style.titled(
        ax,
        f"Steady harvest income plus a {style.money(res.terminal_total)} exit recovers the "
        f"{style.money(res.purchase_price)} outlay",
        f"Real CAD · NPV {style.money(res.npv)} at a {res.discount_rate:.1%} discount rate · "
        f"IRR {res.irr:.1%} · operating costs average {style.money(avg_cost * 1e6)} a year, "
        f"{avg_cost / avg_rev:.0%} of revenue, so they barely register at this scale",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative price and cost assumptions")


def unconstrained_profile(rotation_age: int, p: GrowthParams, years: int) -> np.ndarray:
    """What the tract would yield with no even-flow rule: cut everything mature, at once."""
    areas = initial_areas(p)
    min_age = int(config.num("dcf.min_harvest_age"))
    huge = standing_volume(areas, p)
    volumes = np.zeros(years)
    for t in range(years):
        areas = age_forward(areas)
        areas, vol, _ = _harvest_one_year(areas, huge, rotation_age, min_age, 0.0, p)
        volumes[t] = vol
    return volumes


def figure_harvest_volume(res: DCFResult, path):
    """Annual harvest volume against the even-flow band, and what the rule costs."""
    import matplotlib.pyplot as plt

    s = res.schedule
    p = GrowthParams.from_config()
    free = unconstrained_profile(s.rotation_age, p, len(s.years)) / 1e3

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    lo = s.target * (1 - s.tolerance) / 1e3
    hi = s.target * (1 + s.tolerance) / 1e3
    ax.axhspan(lo, hi, color=style.SAND, zorder=0,
               label=f"Permitted even-flow band (±{s.tolerance:.0%})")
    ax.bar(s.years, s.volumes / 1e3, color=style.FOREST, width=0.74, zorder=3,
           label="Scheduled harvest")
    ax.step(np.concatenate(([0.5], s.years + 0.5)), np.concatenate((free[:1], free)),
            color=style.ACCENT, lw=1.8, ls="--", zorder=4,
            label="Unregulated cut (no even-flow rule)")

    # The unregulated year-1 cut is ~20x the sustainable rate. Plotting it to
    # scale would flatten everything else, so the axis stays on the regulated
    # range and the dashed line is allowed to run off the top.
    top = hi * 1.9
    ax.set_xlabel("Year")
    ax.set_ylabel("Harvest volume (thousand m³)")
    ax.set_xlim(0.3, s.years.max() + 0.7)
    ax.set_ylim(0, top)
    ax.legend(loc="upper right", ncol=1)

    ax.annotate(
        f"Off the chart: an unregulated owner cuts\n{free[0]:,.0f} thousand m³ in year 1 alone, "
        f"{free[0] / (s.target / 1e3):.0f}× the\nsustainable rate, then has nothing left to sell",
        xy=(1.0, top * 0.985), xytext=(2.6, top * 0.62),
        fontsize=9, color=style.ACCENT,
        arrowprops=dict(arrowstyle="->", color=style.ACCENT, lw=1.2),
    )

    style.titled(
        ax,
        f"Even-flow discipline holds the cut at {s.target/1e3:,.0f} thousand m³ a year for the full hold",
        f"Oldest-first from age {s.rotation_age} · exit inventory {s.closing_inventory/1e3:,.0f} thousand m³, "
        f"the regulated growing stock · the binding constraint is ending inventory, not the ±{s.tolerance:.0%} band",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative growth and inventory assumptions")


def figure_heatmap(grid: pd.DataFrame, title: str, subtitle: str, path, hurdle: float):
    """Two-way IRR grid shaded around the hurdle rate."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    values = grid.to_numpy(dtype=float) * 100.0
    h = hurdle * 100.0
    span = max(abs(np.nanmax(values) - h), abs(h - np.nanmin(values)), 0.5)
    norm = TwoSlopeNorm(vmin=h - span, vcenter=h, vmax=h + span)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.grid(False)
    im = ax.imshow(values, cmap=style.CMAP_DIVERGING, norm=norm, aspect="auto")

    ax.set_xticks(range(grid.shape[1]), grid.columns, fontsize=10)
    ax.set_yticks(range(grid.shape[0]), grid.index, fontsize=10)
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9.5,
                        color=style.INK_SOFT)
                continue
            # White text only on the darkest greens, so it stays readable in print.
            light = v > h + span * 0.45
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if v >= h else "normal",
                    color=style.WHITE if light else style.INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(f"Project IRR (%), hurdle {hurdle:.1%}")
    cbar.outline.set_visible(False)
    cbar.ax.axhline(h, color=style.INK, lw=1.4)

    style.titled(ax, title, subtitle)
    fig.tight_layout()
    return style.save(fig, path, "Illustrative assumptions; every cell is a full model re-run")


def breakeven_stumpage(res: DCFResult) -> float:
    """Net stumpage price at which the project IRR equals the hurdle, $/m3."""
    def gap(price: float) -> float:
        return run_dcf(price=price).irr - res.hurdle_rate

    base = config.num("economics.net_stumpage_price")
    return float(brentq(gap, base * 0.4, base * 1.6, xtol=1e-3))


def _grid_swing(grid: pd.DataFrame) -> Tuple[float, float]:
    """Return (row swing, column swing) in percentage points of IRR."""
    values = grid.to_numpy(dtype=float) * 100.0
    mid_col = values[:, values.shape[1] // 2]
    mid_row = values[values.shape[0] // 2, :]
    return float(np.ptp(mid_col)), float(np.ptp(mid_row))


def build(figures_dir=None) -> Tuple[DCFResult, Dict[str, str]]:
    """Run the base DCF, both sensitivity grids and the Phase 3 figures."""
    style.apply_style()
    figures_dir = figures_dir or config.FIGURES

    res = run_dcf()
    res.grids = {
        "price_vs_discount": sensitivity_price_vs_discount(res),
        "price_paid_vs_growth": sensitivity_price_paid_vs_growth(res),
    }

    # Claim-style titles are derived from the grids themselves so the headline
    # can never drift away from the numbers printed underneath it.
    be_price = breakeven_stumpage(res)
    base_price = config.num("economics.net_stumpage_price")
    price_drop = 1.0 - be_price / base_price
    res.grids["price_vs_discount"].attrs["breakeven_stumpage"] = be_price

    grid1 = res.grids["price_vs_discount"]
    _, discount_swing = _grid_swing(grid1)
    price_swing, growth_swing = _grid_swing(res.grids["price_paid_vs_growth"])

    paths = {
        "cash_flows": str(figure_cash_flows(res, figures_dir / "cash_flows.png")),
        "harvest_volume": str(figure_harvest_volume(res, figures_dir / "harvest_volume.png")),
        "sensitivity_price_discount": str(
            figure_heatmap(
                grid1,
                f"Stumpage has to fall {price_drop:.0%}, to about ${be_price:,.0f}/m³, "
                f"before the deal misses the hurdle",
                "Project IRR; each cell re-solves the rotation, the schedule and the exit. The "
                f"discount rate moves the IRR by only {discount_swing:.1f} points across 4-8% "
                "because it touches nothing but the rotation age and the exit price.",
                figures_dir / "sensitivity_price_discount.png",
                res.hurdle_rate,
            )
        ),
        "sensitivity_price_growth": str(
            figure_heatmap(
                res.grids["price_paid_vs_growth"],
                f"Entry price matters {price_swing / growth_swing:.0f}× more than the "
                f"long-run price growth assumption",
                f"Project IRR. Moving the offer ±30% swings the IRR {price_swing:.1f} points; "
                f"moving real price growth across ±1 point swings it {growth_swing:.1f}.",
                figures_dir / "sensitivity_price_growth.png",
                res.hurdle_rate,
            )
        ),
    }
    return res, paths


if __name__ == "__main__":
    config.ensure_directories()
    result, figs = build()
    s = result.summary()
    print(f"Recommendation      : {s['recommendation']}")
    print(f"Purchase price      : ${s['purchase_price']:,.0f} (${s['purchase_price_per_ha']:,.0f}/ha)")
    print(f"NPV @ {s['discount_rate']:.1%}         : ${s['npv']:,.0f}")
    print(f"IRR                 : {s['irr']:.2%}  (hurdle {s['hurdle_rate']:.1%})")
    print(f"MOIC                : {s['moic']:.2f}x")
    print(f"Breakeven price     : ${s['breakeven_price']:,.0f} "
          f"(${s['breakeven_price_per_ha']:,.0f}/ha, {s['breakeven_premium_vs_ask']:+.1%} vs ask)")
    print(f"Rotation age        : {s['rotation_age']} yrs")
    print(f"Even-flow target    : {s['even_flow_target_m3']:,.0f} m3/yr "
          f"(max deviation {s['max_even_flow_deviation']:.1%}, "
          f"{s['even_flow_binding_years']} binding years)")
    print(f"Inventory           : {s['opening_inventory_m3']:,.0f} -> {s['closing_inventory_m3']:,.0f} m3")
    print(f"Terminal value      : ${s['terminal_value']:,.0f} "
          f"(land ${result.terminal_land:,.0f} + timber ${result.terminal_inventory:,.0f})")
    for name, path in figs.items():
        print(f"  figure {name}: {path}")
