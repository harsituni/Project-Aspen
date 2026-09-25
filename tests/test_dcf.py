"""Harvest scheduling, the cash flow build, and the return metrics.

Monetary figures are real CAD.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, dcf
from src.dcf import (
    age_forward,
    initial_areas,
    irr,
    npv,
    regulated_growing_stock,
    solve_even_flow_schedule,
    standing_volume,
)


# --- IRR and NPV against hand-checkable cases -------------------------

def test_npv_of_a_flat_annuity_matches_the_closed_form():
    rate = 0.08
    flows = np.concatenate(([0.0], np.full(10, 100.0)))
    expected = 100.0 * (1 - (1 + rate) ** -10) / rate
    assert npv(flows, rate) == pytest.approx(expected, rel=1e-12)


def test_irr_of_a_doubling_over_one_year_is_one_hundred_percent():
    assert irr(np.array([-100.0, 200.0])) == pytest.approx(1.0, abs=1e-9)


def test_irr_of_a_known_series_is_ten_percent():
    """-1000 now, 100 a year for 10 years, 1000 back at the end: a 10% bond."""
    flows = np.concatenate(([-1000.0], np.full(9, 100.0), [1100.0]))
    assert irr(flows) == pytest.approx(0.10, abs=1e-9)


def test_irr_is_the_rate_that_zeroes_npv():
    flows = np.array([-500.0, 120.0, 130.0, 140.0, 150.0, 260.0])
    assert npv(flows, irr(flows)) == pytest.approx(0.0, abs=1e-6)


def test_irr_returns_nan_without_a_sign_change():
    assert np.isnan(irr(np.array([100.0, 50.0, 25.0])))


# --- Tract bookkeeping -------------------------------------------------

def test_initial_areas_reproduce_the_configured_tract(params):
    areas = initial_areas(params)
    assert areas.sum() == pytest.approx(config.num("tract.area_ha"))
    assert np.all(areas >= 0)
    assert areas[100:].sum() == pytest.approx(0.0), "no stand starts older than 99"


def test_ageing_conserves_area(params):
    areas = initial_areas(params)
    for _ in range(50):
        areas = age_forward(areas)
        assert areas.sum() == pytest.approx(config.num("tract.area_ha"))


def test_ageing_pins_stands_at_the_top_class(params):
    areas = np.zeros(params.max_age + 1)
    areas[params.max_age] = 100.0
    assert age_forward(areas)[params.max_age] == pytest.approx(100.0)


# --- The even-flow schedule --------------------------------------------

def test_schedule_respects_the_even_flow_band(dcf_result):
    s = dcf_result.schedule
    assert s.max_deviation <= s.tolerance + 1e-9, (
        f"harvest deviates {s.max_deviation:.2%} from the mean, "
        f"outside the ±{s.tolerance:.0%} band"
    )


def test_schedule_leaves_a_working_forest(dcf_result):
    """Ending inventory must clear the regulated growing stock floor."""
    s = dcf_result.schedule
    assert s.closing_inventory >= s.required_ending_inventory * 0.999
    assert s.required_ending_inventory > 0


def test_harvested_area_matches_harvested_volume(dcf_result, params):
    """Every cubic metre cut has to come off some hectare."""
    s = dcf_result.schedule
    assert np.all(s.areas_cut > 0)
    implied = s.volumes / s.areas_cut
    assert np.all(implied > 0), "implied volume per hectare must be positive"
    assert np.all(implied < params.a), "and below the asymptotic volume"


def test_area_is_conserved_over_the_whole_hold(dcf_result):
    assert dcf_result.schedule.ending_areas.sum() == pytest.approx(
        config.num("tract.area_ha")
    )


def test_a_tighter_band_cannot_raise_the_sustainable_cut(params):
    """Extra constraint, weakly worse outcome."""
    rotation_age = dcf_result_rotation(params)
    loose = solve_even_flow_schedule(rotation_age, params, 30, tolerance=0.25)
    tight = solve_even_flow_schedule(rotation_age, params, 30, tolerance=0.05)
    assert tight.target <= loose.target + 1e-6


def dcf_result_rotation(params) -> int:
    from src.rotation import Economics, faustmann_integer_age
    return faustmann_integer_age(params, Economics.from_config())


def test_regulated_growing_stock_matches_its_definition(params):
    rotation_age = 30
    area = config.num("tract.area_ha")
    from src.growth import volume
    expected = (area / rotation_age) * sum(
        volume(float(a), params) for a in range(1, rotation_age + 1)
    )
    assert regulated_growing_stock(rotation_age, params) == pytest.approx(expected)


# --- The assembled model ------------------------------------------------

def test_cash_flow_table_ties_to_the_cash_flow_vector(dcf_result):
    table = dcf_result.table
    rebuilt = (
        table["harvest_revenue"] + table["carbon_revenue"] + table["regeneration_cost"]
        + table["management_cost"] + table["acquisition"] + table["terminal_value"]
    )
    assert np.allclose(rebuilt.to_numpy(), dcf_result.cash_flows)


def test_npv_and_irr_are_mutually_consistent(dcf_result):
    assert npv(dcf_result.cash_flows, dcf_result.irr) == pytest.approx(0.0, abs=1.0)
    assert npv(dcf_result.cash_flows, dcf_result.discount_rate) == pytest.approx(
        dcf_result.npv, rel=1e-9
    )


def test_npv_is_positive_exactly_when_irr_beats_the_hurdle(dcf_result):
    assert (dcf_result.npv > 0) == (dcf_result.irr > dcf_result.hurdle_rate)


def test_breakeven_price_drives_npv_to_zero(dcf_result):
    at_breakeven = dcf.run_dcf(purchase_price=dcf_result.breakeven_price)
    assert at_breakeven.npv == pytest.approx(0.0, abs=1.0)
    assert at_breakeven.irr == pytest.approx(dcf_result.discount_rate, abs=1e-6)


def test_terminal_value_splits_into_land_plus_timber(dcf_result):
    assert dcf_result.terminal_land + dcf_result.terminal_inventory == pytest.approx(
        dcf_result.terminal_total, rel=1e-9
    )
    assert dcf_result.terminal_land > 0
    assert dcf_result.terminal_inventory > 0


def test_paying_more_lowers_the_return(dcf_result):
    dearer = dcf.run_dcf(purchase_price=dcf_result.purchase_price * 1.2)
    assert dearer.irr < dcf_result.irr
    assert dearer.npv < dcf_result.npv


def test_a_higher_price_raises_the_return(dcf_result):
    richer = dcf.run_dcf(price=config.num("economics.net_stumpage_price") * 1.2)
    assert richer.irr > dcf_result.irr


def test_carbon_revenue_only_appears_when_enabled(dcf_result):
    with_carbon = dcf.run_dcf(carbon=True)
    assert with_carbon.table["carbon_revenue"].iloc[1:].gt(0).all()
    assert with_carbon.npv > dcf_result.npv
    assert dcf_result.table["carbon_revenue"].eq(0).all()


def test_sensitivity_grids_are_monotonic_in_price(dcf_result):
    grid = dcf.sensitivity_price_vs_discount(dcf_result)
    for column in grid.columns:
        values = grid[column].to_numpy()
        assert np.all(np.diff(values) > 0), "IRR must rise with stumpage price"


def test_sensitivity_grid_is_monotonic_in_price_paid(dcf_result):
    grid = dcf.sensitivity_price_paid_vs_growth(dcf_result)
    for column in grid.columns:
        values = grid[column].to_numpy()
        assert np.all(np.diff(values) < 0), "IRR must fall as the purchase price rises"
