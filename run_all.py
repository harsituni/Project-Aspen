"""Project Aspen: rebuild every deliverable from scratch.

    python run_all.py              full rebuild, refreshing market data
    python run_all.py --offline    use cached market data only
    python run_all.py --quick      1,000 Monte Carlo paths, for a fast smoke test

Everything downstream of the assumptions file is regenerated, so the figures,
the workbook, the deck, the portfolio page and the numbers quoted in the
documentation always describe the same run. Monetary figures are real CAD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src import config, dcf, deck_builder, excel_builder, market, montecarlo
from src import portfolio_page, rotation

BANNER = "=" * 78


def _jsonable(value: Any) -> Any:
    """Make numpy and pandas types survive json.dump."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _step(number: int, total: int, title: str) -> float:
    print(f"\n[{number}/{total}] {title}")
    print("-" * 78)
    return time.perf_counter()


def _done(start: float) -> None:
    print(f"      finished in {time.perf_counter() - start:.1f}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild every Project Aspen deliverable.")
    parser.add_argument("--offline", action="store_true",
                        help="use cached market data instead of downloading")
    parser.add_argument("--quick", action="store_true",
                        help="run 1,000 Monte Carlo paths instead of the configured count")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    print(BANNER)
    print("PROJECT ASPEN: hypothetical Alberta timberland acquisition")
    print("Rebuilding all deliverables from config/assumptions.yaml")
    print(BANNER)

    total = 9
    config.ensure_directories()

    t = _step(1, total, "Validating assumptions")
    config.validate()
    records = config.all_records()
    illustrative = config.illustrative_records()
    print(f"      {len(records)} assumptions, {len(illustrative)} illustrative")
    _done(t)

    t = _step(2, total, "Growth curve and optimal rotation")
    rot, rot_figs = rotation.build()
    print(f"      max-MAI {rot.max_mai_age:.1f} yrs | Fisher {rot.fisher_age:.1f} yrs | "
          f"Faustmann {rot.faustmann_age:.1f} yrs (LEV ${rot.faustmann_lev:,.0f}/ha)")
    _done(t)

    t = _step(3, total, "Acquisition DCF and sensitivity grids")
    res, dcf_figs = dcf.build()
    d = res.summary()
    print(f"      NPV ${d['npv']:,.0f} | IRR {d['irr']:.2%} vs hurdle {d['hurdle_rate']:.1%} | "
          f"MOIC {d['moic']:.2f}x")
    print(f"      breakeven ${d['breakeven_price']:,.0f} "
          f"({d['breakeven_premium_vs_ask']:+.1%} vs ask) -> {d['recommendation']}")
    _done(t)

    t = _step(4, total, "Monte Carlo risk simulation")
    n_sims = 1000 if args.quick else None
    mc_res = montecarlo.run(n_sims=n_sims)
    mc_figs = {
        "irr_histogram": str(montecarlo.figure_irr_histogram(
            mc_res, config.FIGURES / "irr_histogram.png")),
        "price_fan": str(montecarlo.figure_price_fan(
            mc_res, config.FIGURES / "price_fan.png")),
    }
    m = mc_res.summary()
    print(f"      {mc_res.n_sims:,} paths | median IRR {m['scheduled_p50']:.2%} | "
          f"P(IRR < hurdle) {m['scheduled_prob_below_hurdle']:.1%}")
    print(f"      flexibility worth ${m['flexibility_value']:,.0f} | "
          f"disturbances cost ${m['disturbance_cost']:,.0f}")
    _done(t)

    t = _step(5, total, "Market analysis: correlation and inflation")
    mk_res, mk_figs = market.build(offline=args.offline)
    mkt = mk_res.summary()
    print(f"      CPI: {mkt['cpi_source']}")
    print(f"      timber proxy correlation to {mkt['benchmark']}: "
          f"{mkt['mean_timber_corr_to_benchmark']:.2f} | "
          f"{mkt['n_significant_hedges']} significant inflation betas")
    _done(t)

    figures: Dict[str, str] = {**rot_figs, **dcf_figs, **mc_figs, **mk_figs}

    t = _step(6, total, "Writing data/processed/results.json")
    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": config.val("meta.disclaimer"),
        "rotation": rot.to_dict(),
        "dcf": res.summary(),
        "monte_carlo": mc_res.summary(),
        "market": mk_res.summary(),
        "figures": figures,
        "assumptions": [
            {"path": r.path, "value": r.value, "unit": r.unit,
             "source": r.source, "note": r.note}
            for r in records
        ],
    }
    config.RESULTS_PATH.write_text(
        json.dumps(_jsonable(results), indent=2), encoding="utf-8"
    )
    print(f"      {config.RESULTS_PATH}")
    _done(t)

    t = _step(7, total, "Building the Excel model")
    xlsx = excel_builder.build(res, mc_res, mk_res, figures)
    print(f"      {xlsx}")
    _done(t)

    t = _step(8, total, "Building the investment committee deck")
    pptx = deck_builder.build(rot, res, mc_res, mk_res, figures)
    print(f"      {pptx}")
    _done(t)

    t = _step(9, total, "Building the portfolio page")
    pdf = portfolio_page.build(rot, res, mc_res, mk_res, figures)
    print(f"      {pdf}")
    _done(t)

    _print_summary(rot, res, mc_res, mk_res, figures, [xlsx, pptx, pdf], illustrative)
    print(f"\nTotal runtime {time.perf_counter() - started:.1f}s")
    return 0


def _print_summary(rot, res, mc_res, mk_res, figures, deliverables, illustrative) -> None:
    d = res.summary()
    m = mc_res.summary()
    mkt = mk_res.summary()

    print("\n" + BANNER)
    print("HEADLINE RESULTS")
    print(BANNER)
    print(f"  Recommendation            {d['recommendation']} at or below "
          f"${d['breakeven_price']:,.0f} (${d['breakeven_price_per_ha']:,.0f}/ha)")
    print(f"  Asking price              ${d['purchase_price']:,.0f} "
          f"(${d['purchase_price_per_ha']:,.0f}/ha)")
    print(f"  NPV at the {d['hurdle_rate']:.1%} hurdle     ${d['npv']:,.0f}")
    print(f"  IRR                       {d['irr']:.2%}  "
          f"({(d['irr'] - d['hurdle_rate']) * 10000:+.0f} bps vs hurdle)")
    print(f"  MOIC                      {d['moic']:.2f}x over "
          f"{int(config.num('dcf.holding_period_years'))} years")
    print(f"  Faustmann rotation        {d['rotation_age']} yrs "
          f"(max-MAI would be {rot.max_mai_age:.0f} yrs)")
    print(f"  Sustainable annual cut    {d['even_flow_target_m3']:,.0f} m3/yr "
          f"({d['opening_inventory_m3']:,.0f} -> {d['closing_inventory_m3']:,.0f} m3 inventory)")
    print(f"  Risk                      P(IRR < hurdle) {m['scheduled_prob_below_hurdle']:.1%}; "
          f"P5-P95 {m['scheduled_p5']:.1%} to {m['scheduled_p95']:.1%}")
    print(f"  Harvest flexibility       ${m['flexibility_value']:,.0f} "
          f"(${m['flexibility_value_per_ha']:,.0f}/ha)")
    print(f"  Fire and pest cost        ${m['disturbance_cost']:,.0f} "
          f"(${m['disturbance_cost_per_ha']:,.0f}/ha)")
    print(f"  Portfolio fit             timber proxies correlate "
          f"{mkt['mean_timber_corr_to_benchmark']:.2f} with {mkt['benchmark']}; "
          f"{mkt['n_significant_hedges']} of {len(mk_res.regression)} significant "
          f"inflation betas")

    print("\n" + BANNER)
    print("OUTPUT FILES")
    print(BANNER)
    for item in deliverables:
        size = Path(item).stat().st_size / 1024
        print(f"  {str(item):<62} {size:>8,.0f} KB")
    print(f"  {str(config.RESULTS_PATH):<62} "
          f"{config.RESULTS_PATH.stat().st_size / 1024:>8,.0f} KB")
    for name, item in sorted(figures.items()):
        size = Path(item).stat().st_size / 1024
        print(f"  {item:<62} {size:>8,.0f} KB")

    print("\n" + BANNER)
    print("ASSUMPTIONS TO SOURCE BEFORE PRESENTING THIS")
    print(BANNER)
    print(f"  {len(illustrative)} of {len(config.all_records())} assumptions are invented "
          "for this exercise.")
    print("  The ones that move the answer most, in order:\n")
    priority = [
        ("growth.chapman_richards_a", "and b, c: the entire yield curve. Replace with "
                                      "published Alberta growth-and-yield data; the rotation "
                                      "age and every volume depends on these three numbers."),
        ("economics.net_stumpage_price", "drives the IRR more than anything else. A 15% fall "
                                         "takes the deal below the hurdle."),
        ("dcf.purchase_price", "the asking price is assumed, not negotiated. The model's real "
                               "output is the breakeven, not the IRR."),
        ("dcf.real_discount_rate", "sets the rotation age and the exit value. Benchmark it "
                                   "against observed timberland transaction yields."),
        ("tract.age_class_distribution_ha", "a real deal would use a timber cruise, not an "
                                            "assumed distribution."),
        ("monte_carlo.disturbance_probability", "and severity: calibrate to Alberta wildfire "
                                                "and mountain pine beetle records."),
        ("economics.real_price_growth", "0.5% real is a judgement call with 30 years of "
                                        "compounding behind it."),
    ]
    for path, why in priority:
        record = config.rec(path)
        shown = record.value if not isinstance(record.value, dict) else "age-class table"
        print(f"  - {path}")
        print(f"      currently {shown} {record.unit}; {why}")
    print("\n  Market data (prices, CPI) is real and cited. Everything about the tract is not.")
    print(BANNER)


if __name__ == "__main__":
    sys.exit(main())
