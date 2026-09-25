"""Chapman-Richards growth: shape, bounds and the MAI culmination.

Monetary figures are real CAD.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.growth import GrowthParams, cai, mai, max_mai_age, volume


def test_volume_is_zero_at_age_zero(params):
    assert volume(0, params) == pytest.approx(0.0, abs=1e-12)


def test_volume_is_strictly_increasing(params):
    ages = np.arange(0, params.max_age + 1, dtype=float)
    v = volume(ages, params)
    assert np.all(np.diff(v) > 0), "volume must never fall with age"


def test_volume_is_bounded_by_the_asymptote(params):
    modelled = volume(np.arange(0, params.max_age + 1, dtype=float), params)
    assert np.all(modelled < params.a), "volume must stay below the asymptote a"

    # Far beyond the modelled range the curve converges on a, to within float precision.
    far = volume(np.arange(0, 2000, dtype=float), params)
    assert np.all(far <= params.a)
    assert far[-1] == pytest.approx(params.a, rel=1e-9)


def test_curve_is_sigmoid_not_concave(params):
    """c > 1 means slow juvenile growth: CAI rises before it falls."""
    ages = np.arange(1, params.max_age + 1, dtype=float)
    increment = cai(ages, params)
    peak = int(np.argmax(increment))
    assert peak > 0, "CAI should peak after age 1, not at it"
    assert np.all(np.diff(increment[:peak]) > 0)
    assert np.all(np.diff(increment[peak:]) < 0)


def test_cai_matches_a_numerical_derivative(params):
    """The analytic derivative must agree with a finite difference."""
    ages = np.arange(5, 120, 5, dtype=float)
    h = 1e-5
    numeric = (volume(ages + h, params) - volume(ages - h, params)) / (2 * h)
    assert np.allclose(cai(ages, params), numeric, rtol=1e-6)


def test_mai_peaks_where_cai_crosses_it(params):
    """The textbook result: MAI culminates exactly at the CAI/MAI crossover."""
    peak = max_mai_age(params)
    assert cai(peak, params) == pytest.approx(mai(peak, params), rel=1e-3)


def test_mai_peak_is_a_true_maximum(params):
    peak = max_mai_age(params)
    assert mai(peak, params) > mai(peak - 1.0, params)
    assert mai(peak, params) > mai(peak + 1.0, params)


def test_mai_peak_is_in_a_plausible_boreal_range(params):
    """A guard against someone editing the growth parameters into nonsense."""
    peak = max_mai_age(params)
    assert 30 < peak < 120, f"MAI culminating at {peak:.0f} years is not a boreal stand"
    assert 1.0 < mai(peak, params) < 8.0, "peak MAI is outside any plausible boreal range"


def test_faster_growth_shortens_the_mai_peak(params):
    slow = GrowthParams(a=params.a, b=params.b * 0.5, c=params.c, max_age=params.max_age)
    fast = GrowthParams(a=params.a, b=params.b * 2.0, c=params.c, max_age=params.max_age)
    assert max_mai_age(fast) < max_mai_age(params) < max_mai_age(slow)


def test_scalar_and_array_calls_agree(params):
    ages = [3.0, 25.0, 60.0, 140.0]
    for func in (volume, mai, cai):
        scalars = [func(a, params) for a in ages]
        assert np.allclose(scalars, func(np.array(ages), params))
