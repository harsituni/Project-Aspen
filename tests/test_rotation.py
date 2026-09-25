"""The three rotation criteria, and the economic ordering between them.

Monetary figures are real CAD.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import rotation
from src.growth import volume
from src.rotation import (
    Economics,
    faustmann_integer_age,
    land_expectation_value,
    single_rotation_npv,
    solve_rotations,
)


def test_solver_agrees_with_brute_force_grid(params, economics):
    """Each optimum is re-found on a 0.01-year grid and must match scipy."""
    res = solve_rotations(params, economics)
    for name, gap in res.solver_gap_years.items():
        assert gap < 0.5, f"{name}: scipy and grid disagree by {gap:.3f} years"


def test_faustmann_is_the_maximum_of_the_lev_curve(params, economics, rotations):
    ages = np.arange(2, params.max_age, 0.05)
    lev = land_expectation_value(ages, params, economics)
    assert rotations.faustmann_lev >= np.nanmax(lev) - 1e-6
    assert rotations.faustmann_age == pytest.approx(
        float(ages[int(np.nanargmax(lev))]), abs=0.1
    )


def test_faustmann_is_shorter_than_the_biological_rotation(rotations):
    """Discounting always pulls the harvest earlier than max-MAI."""
    assert rotations.faustmann_age < rotations.max_mai_age


def test_faustmann_is_no_longer_than_the_single_rotation_optimum(rotations):
    """Faustmann adds the opportunity cost of delaying all future rotations."""
    assert rotations.faustmann_age <= rotations.fisher_age + 1e-6


def test_rotation_shortens_as_the_discount_rate_rises(params, economics):
    rates = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    ages, levs = rotation.faustmann_vs_discount_rate(rates, params, economics)
    assert np.all(np.diff(ages) <= 0), "rotation age must not rise with the discount rate"
    assert ages[0] > ages[-1], "and must actually fall across 2% to 8%"
    assert np.all(np.diff(levs) < 0), "land value must fall as the discount rate rises"


def test_rotation_shortens_as_the_price_rises(params, economics):
    """The counter-intuitive Faustmann comparative static.

    A higher stumpage price makes the land itself more valuable, and the land
    is what waiting ties up. The opportunity cost of delay rises faster than
    the gain from extra volume, so the optimal rotation gets *shorter*. It is
    also why cheap timber is cut late: at a low price the fixed regeneration
    cost takes longer to earn back.
    """
    ages = []
    for multiple in (0.5, 1.0, 2.0):
        econ = Economics(economics.price * multiple, economics.regen_cost,
                         economics.mgmt_cost, economics.discount_rate)
        ages.append(solve_rotations(params, econ).faustmann_age)
    assert ages[0] > ages[1] > ages[2], f"expected a falling rotation, got {ages}"


def test_higher_regeneration_cost_lengthens_the_rotation(params, economics):
    """The mirror image: a larger fixed cost per rotation takes longer to repay."""
    cheap = Economics(economics.price, economics.regen_cost * 0.5,
                      economics.mgmt_cost, economics.discount_rate)
    dear = Economics(economics.price, economics.regen_cost * 2.0,
                     economics.mgmt_cost, economics.discount_rate)
    assert (solve_rotations(params, cheap).faustmann_age
            < solve_rotations(params, dear).faustmann_age)


def test_lev_formula_equals_an_explicit_perpetual_series(params, economics):
    """Check the closed form against 400 rotations summed by hand."""
    T = 30.0
    r = economics.discount_rate
    revenue = economics.price * volume(T, params)

    brute = 0.0
    for k in range(400):
        brute -= economics.regen_cost / (1 + r) ** (k * T)
        brute += revenue / (1 + r) ** ((k + 1) * T)
    brute -= economics.mgmt_cost / r  # perpetual annual charge

    assert land_expectation_value(T, params, economics) == pytest.approx(brute, rel=1e-8)


def test_single_rotation_npv_matches_its_definition(params, economics):
    T = 40.0
    r = economics.discount_rate
    expected = (
        economics.price * volume(T, params) / (1 + r) ** T
        - economics.regen_cost
        - economics.mgmt_cost * (1 - (1 + r) ** -T) / r
    )
    assert single_rotation_npv(T, params, economics) == pytest.approx(expected, rel=1e-12)


def test_integer_rotation_matches_an_index_match_over_whole_years(params, economics):
    """This is the value the Excel INDEX/MATCH formula must reproduce."""
    ages = np.arange(1, params.max_age + 1, dtype=float)
    lev = land_expectation_value(ages, params, economics)
    assert faustmann_integer_age(params, economics) == int(ages[int(np.nanargmax(lev))])


def test_integer_rotation_is_next_to_the_continuous_optimum(params, economics, rotations):
    assert abs(faustmann_integer_age(params, economics) - rotations.faustmann_age) <= 1.0
