"""Prove the workbook's formulas produce the model's numbers.

Monetary figures are real CAD.

openpyxl writes formulas but never evaluates them, so a workbook can look
perfect and be silently wrong. Three layers of checking here:

1. Structural: the sheets, named ranges and formulas that must exist.
2. Evaluated: the `formulas` package parses and computes the whole workbook,
   and the results are compared with the Python model cell by cell. This is
   the real test: it executes the actual Excel formulas.
3. LibreOffice: if soffice is on PATH, recalculate headlessly as a second
   opinion from a genuine spreadsheet engine. Skipped when it is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

from src import config
from src.excel_builder import NAMED_INPUTS, SHEETS

pytest.importorskip("openpyxl")


# --- 1. Structure -------------------------------------------------------

def test_every_sheet_is_present(workbook_path):
    wb = load_workbook(workbook_path)
    assert wb.sheetnames == [name for name, _ in SHEETS]


def test_named_ranges_are_defined(workbook_path):
    wb = load_workbook(workbook_path)
    defined = set(wb.defined_names)
    for name in NAMED_INPUTS.values():
        assert name in defined, f"named range {name} is missing"
    for name in ("RotationAge", "FaustmannLEV", "ProjectNPV", "ProjectIRR",
                 "BreakevenPrice", "TerminalValue", "ExitPrice"):
        assert name in defined, f"named range {name} is missing"


def test_headline_cells_are_formulas_not_pasted_values(workbook_path):
    """If these ever become hard numbers the workbook has stopped being a model."""
    wb = load_workbook(workbook_path)
    cash = wb["Cash_Flows"]
    growth = wb["Growth_Rotation"]

    assert str(cash["B46"].value).upper().startswith("=$K$13+NPV(")
    assert str(cash["B47"].value).upper().startswith("=IRR(")
    assert str(growth["B9"].value).upper().startswith("=INDEX(")
    assert "MATCH(MAX(" in str(growth["B9"].value).upper()
    assert str(growth["C9"].value).upper().startswith("=MAX(")


def test_growth_table_is_built_from_formulas(workbook_path):
    wb = load_workbook(workbook_path)
    ws = wb["Growth_Rotation"]
    for row in (12, 60, 161):  # first, middle and last age rows
        assert str(ws[f"B{row}"].value).startswith("=CR_a*")
        assert str(ws[f"C{row}"].value).startswith("=$B")
        assert str(ws[f"D{row}"].value).startswith("=CR_a*CR_b*CR_c")


def test_every_cash_flow_year_is_a_formula(workbook_path):
    wb = load_workbook(workbook_path)
    ws = wb["Cash_Flows"]
    hold = int(config.num("dcf.holding_period_years"))
    for year in range(hold + 1):
        row = 13 + year
        for column in ("D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"):
            value = ws[f"{column}{row}"].value
            assert isinstance(value, str) and value.startswith("="), (
                f"Cash_Flows!{column}{row} is a pasted value, not a formula"
            )


def test_inputs_are_marked_as_inputs(workbook_path):
    """Blue on yellow is the promise made on the cover sheet."""
    wb = load_workbook(workbook_path)
    ws = wb["Assumptions"]
    for row in range(6, 60):
        if ws[f"B{row}"].value is not None and ws[f"A{row}"].value is not None:
            if ws[f"C{row}"].value:  # a real assumption row, not a section header
                assert ws[f"B{row}"].fill.fgColor.rgb.endswith("FFF7D6")
                break


def test_conditional_formatting_is_applied(workbook_path):
    wb = load_workbook(workbook_path)
    assert len(wb["Sensitivity"].conditional_formatting._cf_rules) >= 2
    assert len(wb["Market_Analysis"].conditional_formatting._cf_rules) >= 1


def test_print_and_freeze_settings(workbook_path):
    wb = load_workbook(workbook_path)
    for name, _ in SHEETS:
        assert wb[name].print_area, f"{name} has no print area"
    assert wb["Cover"].sheet_view.showGridLines is False
    assert wb["Cash_Flows"].freeze_panes == "B13"


def test_workbook_recalculates_on_open(workbook_path):
    wb = load_workbook(workbook_path)
    assert wb.calculation.fullCalcOnLoad is True


# --- 2. Evaluate the real formulas --------------------------------------

@pytest.fixture(scope="session")
def evaluated(workbook_path):
    """Compute the whole workbook with the `formulas` engine.

    Keys come back as "'[Book.xlsx]SHEET'!A1", where the book keeps its
    original case but the sheet is upper-cased. Rather than depend on that,
    parse every key once into a (SHEET, CELL) lookup.
    """
    engine = pytest.importorskip(
        "formulas", reason="pip install formulas to evaluate the workbook"
    )
    warnings.filterwarnings("ignore")
    model = engine.ExcelModel().loads(str(workbook_path)).finish()
    solution = model.calculate()

    pattern = re.compile(r"^'\[[^\]]+\](?P<sheet>[^']+)'!(?P<cell>[A-Z]+\d+)$")
    lookup = {}
    for key, value in solution.items():
        match = pattern.match(str(key))
        if match:
            lookup[(match.group("sheet").upper(), match.group("cell").upper())] = value

    def get(sheet: str, cell: str):
        try:
            value = lookup[(sheet.upper(), cell.upper())]
        except KeyError:
            raise AssertionError(
                f"{sheet}!{cell} was not evaluated: the formula engine could not "
                f"compute it, which usually means the formula is malformed"
            ) from None
        try:
            return value.value[0, 0]
        except (AttributeError, TypeError, IndexError):
            return value

    return get


def test_excel_reproduces_the_faustmann_rotation(evaluated, dcf_result):
    assert float(evaluated("Growth_Rotation", "B9")) == pytest.approx(
        dcf_result.rotation_age, abs=0
    )


def test_excel_reproduces_the_land_expectation_value(evaluated, rotations):
    """Whole-year optimum, so compare against the LEV at the integer rotation."""
    from src.rotation import Economics, land_expectation_value
    from src.growth import GrowthParams

    integer_lev = land_expectation_value(
        float(evaluated("Growth_Rotation", "B9")),
        GrowthParams.from_config(),
        Economics.from_config(),
    )
    assert float(evaluated("Growth_Rotation", "C9")) == pytest.approx(
        integer_lev, rel=1e-9
    )


def test_excel_reproduces_the_mai_culmination(evaluated, params):
    from src.growth import mai

    age = float(evaluated("Growth_Rotation", "B7"))
    ages = np.arange(1, params.max_age + 1, dtype=float)
    assert age == pytest.approx(float(ages[int(np.argmax(mai(ages, params)))]), abs=0)
    assert float(evaluated("Growth_Rotation", "C7")) == pytest.approx(
        float(np.max(mai(ages, params))), rel=1e-9
    )


def test_excel_reproduces_the_terminal_value(evaluated, dcf_result):
    assert float(evaluated("Cash_Flows", "B10")) == pytest.approx(
        dcf_result.terminal_total, rel=1e-6
    )


def test_excel_reproduces_every_annual_cash_flow(evaluated, dcf_result):
    """Row by row, not just the total: an offsetting pair of errors would hide."""
    hold = int(config.num("dcf.holding_period_years"))
    for year in range(hold + 1):
        excel = float(evaluated("Cash_Flows", f"K{13 + year}"))
        assert excel == pytest.approx(dcf_result.cash_flows[year], rel=1e-6), (
            f"year {year} cash flow differs"
        )


def test_excel_npv_matches_the_model(evaluated, dcf_result):
    assert float(evaluated("Cash_Flows", "B46")) == pytest.approx(
        dcf_result.npv, rel=1e-6
    )


def test_excel_irr_matches_the_model(evaluated, dcf_result):
    assert float(evaluated("Cash_Flows", "B47")) == pytest.approx(
        dcf_result.irr, abs=1e-6
    )


def test_excel_moic_matches_the_model(evaluated, dcf_result):
    assert float(evaluated("Cash_Flows", "B50")) == pytest.approx(
        dcf_result.moic, rel=1e-6
    )


def test_excel_breakeven_matches_the_model(evaluated, dcf_result):
    assert float(evaluated("Cash_Flows", "B52")) == pytest.approx(
        dcf_result.breakeven_price, rel=1e-6
    )


def _find_row(workbook_path, sheet: str, label: str) -> int:
    """Locate a labelled row so these tests survive sheet-layout changes."""
    ws = load_workbook(workbook_path)[sheet]
    for row in range(1, ws.max_row + 1):
        if str(ws.cell(row=row, column=1).value).strip().lower() == label.lower():
            return row
    raise AssertionError(f"no row labelled {label!r} on {sheet}")


def test_excel_opening_inventory_matches_the_model(evaluated, workbook_path, dcf_result):
    """The class-average volume must equal the model's per-age expansion.

    Averaging the yield curve over the ten ages in a class is not the same as
    evaluating it at the midpoint, because the curve is convex. The sheet has
    to do the former or it drifts from the model.
    """
    row = _find_row(workbook_path, "Harvest_Schedule", "Total")
    assert float(evaluated("Harvest_Schedule", f"E{row}")) == pytest.approx(
        dcf_result.schedule.opening_inventory, rel=1e-9
    )


def test_excel_closing_inventory_matches_the_model(evaluated, workbook_path, dcf_result):
    row = _find_row(workbook_path, "Harvest_Schedule", "Total at exit")
    assert float(evaluated("Harvest_Schedule", f"D{row}")) == pytest.approx(
        dcf_result.schedule.closing_inventory, rel=1e-6
    )


def test_excel_even_flow_check_passes(evaluated, workbook_path):
    row = _find_row(workbook_path, "Harvest_Schedule", "Even-flow check")
    assert "PASS" in str(evaluated("Harvest_Schedule", f"B{row}"))


def test_excel_recommendation_agrees_with_the_model(evaluated, dcf_result):
    verdict = str(evaluated("Cash_Flows", "B55"))
    assert verdict.startswith(dcf_result.recommendation.split()[0])


# --- 3. LibreOffice second opinion ---------------------------------------

LIBREOFFICE = shutil.which("soffice") or shutil.which("soffice.exe")


@pytest.mark.skipif(LIBREOFFICE is None, reason="LibreOffice is not installed")
def test_libreoffice_recalculates_to_the_same_numbers(workbook_path, tmp_path, dcf_result):
    """A genuine spreadsheet engine as an independent check on the formulas."""
    subprocess.run(
        [LIBREOFFICE, "--headless", "--convert-to", "xlsx:Calc MS Excel 2007 XML",
         "--outdir", str(tmp_path), str(workbook_path)],
        check=True, capture_output=True, timeout=300,
    )
    recalculated = tmp_path / Path(workbook_path).name
    assert recalculated.exists(), "LibreOffice produced no output"

    wb = load_workbook(recalculated, data_only=True)
    cash = wb["Cash_Flows"]
    assert float(cash["B46"].value) == pytest.approx(dcf_result.npv, rel=1e-4)
    assert float(cash["B47"].value) == pytest.approx(dcf_result.irr, abs=1e-5)
    assert float(wb["Growth_Rotation"]["B9"].value) == pytest.approx(
        dcf_result.rotation_age, abs=0
    )
