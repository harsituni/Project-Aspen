"""Shared fixtures. The expensive objects are built once per session.

Monetary figures are real CAD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, dcf, market, montecarlo, rotation  # noqa: E402
from src.growth import GrowthParams  # noqa: E402
from src.rotation import Economics  # noqa: E402

# Small but not trivial: enough paths for the statistics to be stable to a few
# basis points, few enough that the suite stays under a minute.
TEST_SIMS = 400


@pytest.fixture(scope="session")
def params() -> GrowthParams:
    return GrowthParams.from_config()


@pytest.fixture(scope="session")
def economics() -> Economics:
    return Economics.from_config()


@pytest.fixture(scope="session")
def rotations(params, economics) -> rotation.RotationResults:
    return rotation.solve_rotations(params, economics)


@pytest.fixture(scope="session")
def dcf_result() -> dcf.DCFResult:
    return dcf.run_dcf()


@pytest.fixture(scope="session")
def mc_result() -> montecarlo.MonteCarloResult:
    return montecarlo.run(n_sims=TEST_SIMS)


@pytest.fixture(scope="session")
def workbook_path(tmp_path_factory, dcf_result, mc_result) -> Path:
    """Build a real workbook once, into a temp directory."""
    from src import excel_builder

    config.ensure_directories()
    mk, _ = market.build(offline=True)
    dcf_result.grids = {
        "price_vs_discount": dcf.sensitivity_price_vs_discount(dcf_result),
        "price_paid_vs_growth": dcf.sensitivity_price_paid_vs_growth(dcf_result),
    }
    target = tmp_path_factory.mktemp("workbook") / "Project_Aspen_Model.xlsx"
    return excel_builder.build(dcf_result, mc_result, mk, {}, path=target)
