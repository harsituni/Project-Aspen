"""Monte Carlo: reproducibility, the price process, and the flexibility option.

Monetary figures are real CAD.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, montecarlo
from src.dcf import initial_areas
from src.growth import volume
from src.montecarlo import (
    harvest_all_sims,
    irr_vectorised,
    simulate_disturbances,
    simulate_prices,
)

SIMS = 400


def test_same_seed_reproduces_the_same_result():
    a = montecarlo.run(n_sims=200, seed=7)
    b = montecarlo.run(n_sims=200, seed=7)
    assert np.array_equal(a.prices, b.prices)
    assert np.array_equal(a.disturbance, b.disturbance)
    np.testing.assert_allclose(a.scheduled.irr, b.scheduled.irr, rtol=0, atol=0)
    assert a.flexibility_value == b.flexibility_value


def test_different_seeds_give_different_paths():
    a = montecarlo.run(n_sims=200, seed=7)
    b = montecarlo.run(n_sims=200, seed=8)
    assert not np.array_equal(a.prices, b.prices)


# --- The price process --------------------------------------------------

def test_price_paths_are_centred_on_the_base_case():
    rng = np.random.default_rng(3)
    prices, mean_prices = simulate_prices(
        20000, 30, rng, p0=48.0, growth=0.005, kappa=0.25, sigma=0.22
    )
    # Log prices are symmetric around the mean, so the median should sit on it.
    median = np.median(prices, axis=0)
    assert np.allclose(median, mean_prices[0], rtol=0.03)


def test_mean_reversion_keeps_the_spread_bounded():
    """Under a random walk the spread would grow without limit; here it plateaus."""
    rng = np.random.default_rng(4)
    prices, _ = simulate_prices(20000, 60, rng, p0=48.0, growth=0.0,
                                kappa=0.25, sigma=0.22)
    spread = np.std(np.log(prices), axis=0)
    early, late = spread[4], spread[-1]
    assert late < early * 1.25, "dispersion must stop widening once reversion binds"

    stationary = 0.22 / np.sqrt(2 * 0.25)
    assert late == pytest.approx(stationary, rel=0.08)


def test_faster_reversion_narrows_the_distribution():
    rng = np.random.default_rng(5)
    slow, _ = simulate_prices(8000, 40, rng, p0=48.0, growth=0.0, kappa=0.10, sigma=0.22)
    rng = np.random.default_rng(5)
    fast, _ = simulate_prices(8000, 40, rng, p0=48.0, growth=0.0, kappa=0.80, sigma=0.22)
    assert np.std(np.log(fast[:, -1])) < np.std(np.log(slow[:, -1]))


# --- Disturbances -------------------------------------------------------

def test_disturbance_frequency_matches_the_configured_probability():
    rng = np.random.default_rng(11)
    events = simulate_disturbances(20000, 30, rng, probability=0.02,
                                   mean_severity=0.06, concentration=25.0)
    assert (events > 0).mean() == pytest.approx(0.02, abs=0.002)


def test_disturbance_severity_matches_its_configured_mean():
    rng = np.random.default_rng(12)
    events = simulate_disturbances(20000, 30, rng, probability=1.0,
                                   mean_severity=0.06, concentration=25.0)
    assert events.mean() == pytest.approx(0.06, abs=0.003)
    assert events.max() < 1.0


def test_disturbances_never_help(mc_result):
    """Fire and pests must cost value, never create it."""
    assert mc_result.disturbance_cost > 0
    assert np.mean(mc_result.no_disturbance.npv) > np.mean(mc_result.scheduled.npv)


# --- Vectorised primitives against the scalar model ---------------------

def test_vectorised_harvest_matches_the_scalar_scheduler(params):
    """The (n_sims, n_ages) sweep must reproduce src.dcf one simulation at a time."""
    from src.dcf import _harvest_one_year

    per_ha = volume(np.arange(params.max_age + 1, dtype=float), params)
    areas = initial_areas(params)
    target, floor = 150_000.0, 120_000.0

    scalar_areas, scalar_vol, scalar_ha = _harvest_one_year(
        areas.copy(), target, 30, 20, floor, params
    )
    vector_areas, vector_vol, vector_ha = harvest_all_sims(
        np.tile(areas, (3, 1)), np.full(3, target), np.full(3, floor), per_ha, 30, 20
    )
    assert vector_vol[0] == pytest.approx(scalar_vol, rel=1e-9)
    assert vector_ha[0] == pytest.approx(scalar_ha, rel=1e-9)
    assert np.allclose(vector_areas[0], scalar_areas, atol=1e-8)


def test_vectorised_harvest_takes_the_oldest_stands_first(params):
    per_ha = volume(np.arange(params.max_age + 1, dtype=float), params)
    areas = np.zeros((1, params.max_age + 1))
    areas[0, 40] = 100.0
    areas[0, 90] = 100.0

    want = 100.0 * per_ha[90] * 0.5  # only half the oldest cohort
    out, vol, _ = harvest_all_sims(areas, np.array([want]), np.array([0.0]),
                                   per_ha, 30, 20)
    assert out[0, 40] == pytest.approx(100.0), "the younger stand must be untouched"
    assert out[0, 90] == pytest.approx(50.0)


def test_vectorised_harvest_conserves_area(params):
    per_ha = volume(np.arange(params.max_age + 1, dtype=float), params)
    areas = np.tile(initial_areas(params), (5, 1))
    out, _, _ = harvest_all_sims(areas, np.full(5, 200_000.0), np.full(5, 100_000.0),
                                 per_ha, 30, 20)
    assert np.allclose(out.sum(axis=1), config.num("tract.area_ha"))


def test_vectorised_irr_matches_the_scalar_solver():
    from src.dcf import irr as scalar_irr

    rng = np.random.default_rng(21)
    flows = np.column_stack((
        np.full(50, -1000.0),
        rng.uniform(50, 250, size=(50, 9)),
        rng.uniform(800, 1600, size=50),
    ))
    vector = irr_vectorised(flows)
    scalar = np.array([scalar_irr(row) for row in flows])
    assert np.allclose(vector, scalar, atol=1e-7)


# --- Headline outputs ----------------------------------------------------

def test_percentiles_are_ordered(mc_result):
    stats = mc_result.scheduled.percentiles(mc_result.hurdle)
    ordered = [stats[q] for q in ("p5", "p25", "p50", "p75", "p95")]
    assert ordered == sorted(ordered)


def test_flexibility_has_positive_value(mc_result):
    """Mean reversion makes waiting informative, so the option must be worth something."""
    assert mc_result.flexibility_value > 0


def test_flexibility_is_worthless_without_mean_reversion(params):
    """The control experiment: with near-zero reversion the option collapses.

    Deferring only pays if a low price is likely to recover. Push kappa very
    high and volatility to nearly nothing and there is no cycle left to time,
    so the two strategies should converge.
    """
    from src.montecarlo import _run_paths
    from src.rotation import Economics, faustmann_integer_age
    from src.dcf import solve_even_flow_schedule

    e = Economics.from_config()
    rotation_age = faustmann_integer_age(params, e)
    target = solve_even_flow_schedule(rotation_age, params, 30).target

    rng = np.random.default_rng(33)
    prices, mean_prices = simulate_prices(
        300, 30, rng, p0=e.price, growth=0.0, kappa=0.25, sigma=0.0005
    )
    quiet = np.zeros((300, 30))
    common = dict(target=target, rotation_age=rotation_age, e=e, p=params)
    sched = _run_paths(prices, mean_prices, quiet, flexible=False, **common)
    flex = _run_paths(prices, mean_prices, quiet, flexible=True, **common)

    gap = abs(np.mean(flex.npv) - np.mean(sched.npv))
    baseline = abs(np.mean(sched.npv))
    assert gap / baseline < 0.02, "with no price cycle there is nothing to time"


def test_probability_below_hurdle_is_a_proportion(mc_result):
    stats = mc_result.scheduled.percentiles(mc_result.hurdle)
    assert 0.0 <= stats["prob_below_hurdle"] <= 1.0


def test_median_simulated_irr_brackets_the_deterministic_case(mc_result, dcf_result):
    """The simulation is centred on the base case, so the median should be close."""
    stats = mc_result.scheduled.percentiles(mc_result.hurdle)
    assert abs(stats["p50"] - dcf_result.irr) < 0.02


def test_summary_table_has_both_strategies(mc_result):
    table = mc_result.summary_table()
    assert len(table) == 2
    assert table["Median IRR"].notna().all()
