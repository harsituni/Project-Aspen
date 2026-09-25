"""Three answers to "when should this stand be cut?", and why they differ.

Monetary figures are real CAD.

1. Maximum mean annual increment: maximise V(T)/T. Ignores money entirely,
   so it is the longest rotation and the one a pure biologist would pick.
2. Single-rotation (Fisher) NPV: maximise the discounted profit of one
   rotation. Money now matters, so the rotation shortens.
3. Faustmann: maximise the land expectation value, the value of the bare
   land used for rotations in perpetuity:

       LEV(T) = [P*V(T) - C*(1+r)^T] / [(1+r)^T - 1] - m/r

   Faustmann is the shortest of the three because waiting costs not just
   interest on the standing timber but also the delayed start of every
   future rotation. That opportunity cost of the land is what the other
   two criteria leave out.

All three are solved with scipy.optimize.minimize_scalar and then checked
against a brute-force grid, because a bounded optimiser that quietly parks
on a boundary is the classic way to get a confidently wrong rotation age.

Rotation analysis holds the stumpage price flat in real terms. Real price
growth enters later, in the acquisition cash flows (see dcf.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from . import config, style
from .growth import GrowthParams, cai, mai, volume

GRID_STEP = 0.01
GRID_TOLERANCE_YEARS = 0.5


@dataclass(frozen=True)
class Economics:
    """Prices and costs used by the rotation criteria."""

    price: float          # net stumpage, $/m3
    regen_cost: float     # $/ha, charged once per rotation
    mgmt_cost: float      # $/ha/yr, charged forever
    discount_rate: float  # real, per year

    @classmethod
    def from_config(cls) -> "Economics":
        return cls(
            price=config.num("economics.net_stumpage_price"),
            regen_cost=config.num("economics.regeneration_cost"),
            mgmt_cost=config.num("economics.management_cost"),
            discount_rate=config.num("dcf.real_discount_rate"),
        )


# --- The three objective functions -----------------------------------

def land_expectation_value(T, p: GrowthParams, e: Economics):
    """Faustmann LEV in $/ha for a perpetual series of rotations of length T."""
    t = np.asarray(T, dtype=float)
    compound = np.power(1.0 + e.discount_rate, t)
    denom = compound - 1.0
    # As T -> 0 the denominator vanishes and LEV is undefined; guard it.
    safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    lev = (e.price * volume(t, p) - e.regen_cost * compound) / safe
    lev = lev - e.mgmt_cost / e.discount_rate
    out = np.where(np.isfinite(lev), lev, -np.inf)
    return float(out) if np.ndim(T) == 0 else out


def single_rotation_npv(T, p: GrowthParams, e: Economics):
    """Fisher criterion: NPV in $/ha of planting now and harvesting once at T."""
    t = np.asarray(T, dtype=float)
    discount = np.power(1.0 + e.discount_rate, -t)
    mgmt_pv = e.mgmt_cost * (1.0 - discount) / e.discount_rate
    npv = e.price * volume(t, p) * discount - e.regen_cost - mgmt_pv
    return float(npv) if np.ndim(T) == 0 else npv


def _maximise(
    objective: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
) -> Tuple[float, float, float]:
    """Maximise on [lower, upper] with scipy, verify on a fine grid.

    Returns (age from scipy, age from the grid, objective value at the grid age).
    The grid value is what downstream code uses: it cannot land on a boundary
    by accident and it is trivially reproducible.
    """
    res = minimize_scalar(
        lambda t: -float(objective(np.asarray(float(t)))),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-4},
    )
    scipy_age = float(res.x)

    grid = np.arange(lower, upper + GRID_STEP, GRID_STEP)
    values = objective(grid)
    best = int(np.nanargmax(values))
    return scipy_age, float(grid[best]), float(values[best])


@dataclass
class RotationResults:
    """The three rotation ages plus the numbers needed to explain them."""

    max_mai_age: float
    max_mai_value: float          # peak MAI, m3/ha/yr
    fisher_age: float
    fisher_npv: float             # $/ha
    faustmann_age: float
    faustmann_lev: float          # $/ha
    faustmann_volume: float       # m3/ha harvested at the Faustmann age
    discount_rate: float
    solver_gap_years: Dict[str, float]  # |scipy - grid| for each criterion

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def solve_rotations(
    p: GrowthParams | None = None,
    e: Economics | None = None,
) -> RotationResults:
    """Solve all three rotation criteria and confirm scipy agrees with the grid."""
    p = p or GrowthParams.from_config()
    e = e or Economics.from_config()
    lower, upper = 1.0, float(p.max_age)

    mai_scipy, mai_age, mai_peak = _maximise(lambda t: mai(t, p), lower, upper)
    fis_scipy, fisher_age, fisher_npv = _maximise(
        lambda t: single_rotation_npv(t, p, e), lower, upper
    )
    # LEV is -inf near T=0; start the search after the singularity.
    lev_scipy, faustmann_age, faustmann_lev = _maximise(
        lambda t: land_expectation_value(t, p, e), 2.0, upper
    )

    gaps = {
        "max_mai": abs(mai_scipy - mai_age),
        "fisher": abs(fis_scipy - fisher_age),
        "faustmann": abs(lev_scipy - faustmann_age),
    }
    for name, gap in gaps.items():
        if gap > GRID_TOLERANCE_YEARS:
            raise AssertionError(
                f"{name} rotation: scipy and grid search disagree by {gap:.3f} years "
                f"(tolerance {GRID_TOLERANCE_YEARS}). The optimiser is not converging."
            )

    return RotationResults(
        max_mai_age=mai_age,
        max_mai_value=mai_peak,
        fisher_age=fisher_age,
        fisher_npv=fisher_npv,
        faustmann_age=faustmann_age,
        faustmann_lev=faustmann_lev,
        faustmann_volume=float(volume(faustmann_age, p)),
        discount_rate=e.discount_rate,
        solver_gap_years=gaps,
    )


def faustmann_integer_age(
    p: GrowthParams | None = None,
    e: Economics | None = None,
) -> int:
    """Best whole-year rotation age.

    Harvests happen in whole years, and the Excel model finds the optimum with
    INDEX/MATCH over a table of integer ages. Defining the scheduling rotation
    this way, rather than rounding the continuous optimum, guarantees the
    workbook and the Python model agree by construction, at any discount rate.
    """
    p = p or GrowthParams.from_config()
    e = e or Economics.from_config()
    ages = np.arange(1, p.max_age + 1, dtype=float)
    lev = land_expectation_value(ages, p, e)
    return int(ages[int(np.nanargmax(lev))])


def faustmann_vs_discount_rate(
    rates: np.ndarray,
    p: GrowthParams | None = None,
    e: Economics | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faustmann rotation age and LEV across a range of real discount rates."""
    p = p or GrowthParams.from_config()
    e = e or Economics.from_config()
    ages, levs = [], []
    for r in rates:
        econ = Economics(e.price, e.regen_cost, e.mgmt_cost, float(r))
        _, age, lev = _maximise(
            lambda t: land_expectation_value(t, p, econ), 2.0, float(p.max_age)
        )
        ages.append(age)
        levs.append(lev)
    return np.array(ages), np.array(levs)


# --- Figures ---------------------------------------------------------

def figure_growth(p: GrowthParams, res: RotationResults, path):
    """Volume, MAI and CAI against stand age, with the MAI culmination marked."""
    import matplotlib.pyplot as plt

    ages = np.arange(1, p.max_age + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    ax.plot(ages, volume(ages, p), color=style.FOREST, label="Standing volume (m³/ha)")
    ax.set_xlabel("Stand age (years)")
    ax.set_ylabel("Standing volume (m³/ha)")
    ax.set_xlim(0, p.max_age)
    ax.set_ylim(0, p.a * 1.14)
    ax.axhline(p.a, color=style.INK_SOFT, lw=1.0, ls=":")
    ax.text(
        p.max_age * 0.985, p.a * 1.012, f"asymptote a = {p.a:,.0f} m³/ha",
        ha="right", va="bottom", fontsize=9, color=style.INK_SOFT,
    )

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(ages, mai(ages, p), color=style.ACCENT, label="MAI (m³/ha/yr)")
    ax2.plot(ages, cai(ages, p), color=style.FOREST_LIGHT, ls="--", label="CAI (m³/ha/yr)")
    ax2.set_ylabel("Annual increment (m³/ha/yr)")
    ax2.set_ylim(0, max(float(np.max(cai(ages, p))), res.max_mai_value) * 1.35)
    ax2.spines["top"].set_visible(False)

    ax2.axvline(res.max_mai_age, color=style.ACCENT, lw=1.2, ls=":")
    ax2.plot([res.max_mai_age], [res.max_mai_value], "o", color=style.ACCENT, ms=7, zorder=5)
    ax2.annotate(
        f"MAI culminates at age {res.max_mai_age:.0f}\n"
        f"peak MAI {res.max_mai_value:.2f} m³/ha/yr\n"
        f"(CAI crosses MAI here)",
        xy=(res.max_mai_age, res.max_mai_value),
        xytext=(res.max_mai_age + 16, res.max_mai_value * 1.02),
        fontsize=9, color=style.INK,
        arrowprops=dict(arrowstyle="-", color=style.ACCENT, lw=1.0),
    )

    handles = ax.get_lines()[:1] + ax2.get_lines()[:2]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left",
              bbox_to_anchor=(0.01, 0.90), ncol=1)

    style.titled(
        ax,
        "Biological growth peaks late: mean annual increment culminates around age "
        f"{res.max_mai_age:.0f}",
        f"Chapman-Richards V(T) = {p.a:,.0f}·(1 − e^(−{p.b}·T))^{p.c:g}",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative growth parameters")


def figure_lev(p: GrowthParams, e: Economics, res: RotationResults, path):
    """Land expectation value against rotation age, with all three criteria marked."""
    import matplotlib.pyplot as plt

    ages = np.arange(2, p.max_age + 1, dtype=float)
    lev = land_expectation_value(ages, p, e)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(ages, lev, color=style.FOREST, label="Land expectation value", zorder=4)
    ax.axhline(0, color=style.INK_SOFT, lw=1.0)

    # LEV dives towards minus infinity as the rotation shortens towards zero.
    # Scale the axis to the economically meaningful range instead, and say so.
    visible = lev[ages >= 15]
    top = float(res.faustmann_lev) * 2.6
    bottom = float(np.nanmin(visible)) * 1.25
    ax.set_ylim(bottom, top)

    ax.plot([res.faustmann_age], [res.faustmann_lev], "o", color=style.ACCENT, ms=9, zorder=6)
    ax.annotate(
        f"Faustmann optimum\nage {res.faustmann_age:.0f} yrs, LEV ${res.faustmann_lev:,.0f}/ha",
        xy=(res.faustmann_age, res.faustmann_lev),
        xytext=(res.faustmann_age + 26, top * 0.62),
        fontsize=10, color=style.INK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=style.ACCENT, lw=1.4),
    )
    ax.text(
        2, bottom * 0.93,
        "Falls off-scale below\nage ~12: too little\nvolume to repay\nregeneration",
        fontsize=8.5, color=style.INK_SOFT, va="bottom", ha="left",
    )

    for age, label, colour in (
        (res.fisher_age, f"Single-rotation (Fisher) optimum: age {res.fisher_age:.0f}", style.FOREST_LIGHT),
        (res.max_mai_age, f"Max MAI (biological): age {res.max_mai_age:.0f}", style.SAND_DEEP),
    ):
        ax.axvline(age, color=colour, lw=1.6, ls="--", label=label, zorder=2)

    ax.set_xlabel("Rotation age (years)")
    ax.set_ylabel("Land expectation value ($/ha)")
    ax.set_xlim(0, p.max_age)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.legend(loc="lower right")

    style.titled(
        ax,
        f"Economics cut the rotation from {res.max_mai_age:.0f} years to "
        f"{res.faustmann_age:.0f} years",
        f"Net stumpage ${e.price:,.0f}/m³ · regeneration ${e.regen_cost:,.0f}/ha · "
        f"management ${e.mgmt_cost:,.0f}/ha/yr · real discount rate {e.discount_rate:.1%}",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative price and cost assumptions")


def figure_discount_sensitivity(p: GrowthParams, e: Economics, res: RotationResults, path):
    """How the Faustmann rotation and land value respond to the discount rate."""
    import matplotlib.pyplot as plt

    rates = np.arange(0.02, 0.0801, 0.0025)
    ages, levs = faustmann_vs_discount_rate(rates, p, e)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(rates * 100, ages, color=style.FOREST, marker="o", ms=4,
            label="Faustmann rotation age")
    ax.set_xlabel("Real discount rate (%)")
    ax.set_ylabel("Faustmann rotation age (years)")
    ax.set_xlim(1.8, 8.2)
    ax.set_ylim(0, max(ages) * 1.2)

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(rates * 100, levs, color=style.ACCENT, ls="--", marker="s", ms=4,
             label="Land expectation value")
    ax2.set_ylabel("Land expectation value ($/ha)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax2.spines["top"].set_visible(False)
    ax2.axhline(0, color=style.INK_SOFT, lw=0.8, ls=":")

    base = e.discount_rate * 100
    ax.axvline(base, color=style.INK_SOFT, lw=1.2, ls=":")
    ax.text(base + 0.08, ax.get_ylim()[1] * 0.96,
            f"base case {base:.1f}%", fontsize=9, color=style.INK_SOFT, va="top")

    handles = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="lower left")

    style.titled(
        ax,
        "A higher discount rate shortens the rotation and destroys land value",
        "Faustmann optimum re-solved at each rate; every other assumption held constant",
    )
    fig.tight_layout()
    return style.save(fig, path, "Illustrative price and cost assumptions")


def build(figures_dir=None) -> Tuple[RotationResults, Dict[str, str]]:
    """Solve the rotations and write the three Phase 2 figures."""
    style.apply_style()
    figures_dir = figures_dir or config.FIGURES
    p = GrowthParams.from_config()
    e = Economics.from_config()
    res = solve_rotations(p, e)

    paths = {
        "growth_mai_cai": str(figure_growth(p, res, figures_dir / "growth_mai_cai.png")),
        "lev_curve": str(figure_lev(p, e, res, figures_dir / "lev_curve.png")),
        "faustmann_vs_discount": str(
            figure_discount_sensitivity(p, e, res, figures_dir / "faustmann_vs_discount.png")
        ),
    }
    return res, paths


if __name__ == "__main__":
    config.ensure_directories()
    result, figs = build()
    print(f"Max-MAI rotation      : {result.max_mai_age:6.2f} yrs  "
          f"(peak MAI {result.max_mai_value:.2f} m3/ha/yr)")
    print(f"Fisher rotation       : {result.fisher_age:6.2f} yrs  "
          f"(NPV ${result.fisher_npv:,.0f}/ha)")
    print(f"Faustmann rotation    : {result.faustmann_age:6.2f} yrs  "
          f"(LEV ${result.faustmann_lev:,.0f}/ha, {result.faustmann_volume:.0f} m3/ha)")
    print(f"Solver vs grid gaps   : {result.solver_gap_years}")
    for name, path in figs.items():
        print(f"  figure {name}: {path}")
