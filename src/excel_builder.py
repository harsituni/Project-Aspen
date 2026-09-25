"""Build outputs/Project_Aspen_Model.xlsx with live Excel formulas.

Monetary figures are real CAD.

The point of this workbook is that a reviewer can click any headline number
and trace it back to an input. So wherever a calculation can be expressed in
Excel it is written as a formula, not pasted as a value:

* the whole growth and rotation table, including the Faustmann optimum, which
  is found with INDEX/MATCH/MAX rather than being told the answer;
* every line of the 30-year cash flow, including =NPV and =IRR;
* the entire terminal value, land and standing timber, built off the year-30
  age distribution.

Only three things arrive as values, and each is labelled on the sheet: the
harvest schedule and year-30 age distribution (they need the cohort
simulation), the sensitivity grids (each cell is a full model re-run), and
the Monte Carlo and market statistics.

Colour convention, stated on the cover: blue on yellow is a hard input you
may change, black is a formula, green is a link to another sheet.

openpyxl writes formulas but cannot evaluate them, so no cached results are
stored. Cells show blank until Excel or LibreOffice opens the file and
calculates. tests/test_excel_consistency.py recomputes the key formulas in
Python to prove they agree with the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName

from . import config, style
from .dcf import DCFResult
from .growth import GrowthParams
from .market import MarketResult
from .montecarlo import MonteCarloResult

# --- Shared formatting -------------------------------------------------

FONT = "Calibri"
HEX_FOREST = style.hexcode(style.FOREST)
HEX_SAND = style.hexcode(style.SAND)
HEX_ACCENT = style.hexcode(style.ACCENT)
HEX_INK = style.hexcode(style.INK)

INPUT_FONT = Font(name=FONT, size=11, color="1F4E79", bold=True)       # blue = editable input
INPUT_FILL = PatternFill("solid", fgColor="FFF7D6")                    # light yellow
FORMULA_FONT = Font(name=FONT, size=11, color="000000")                # black = calculated here
LINK_FONT = Font(name=FONT, size=11, color="1B7A3E")                   # green = another sheet
BODY_FONT = Font(name=FONT, size=11, color=HEX_INK)
NOTE_FONT = Font(name=FONT, size=9, color="5C574F", italic=True)
HEADER_FONT = Font(name=FONT, size=11, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor=HEX_FOREST)
SECTION_FONT = Font(name=FONT, size=11, bold=True, color=HEX_FOREST)
SECTION_FILL = PatternFill("solid", fgColor=HEX_SAND)
TITLE_FONT = Font(name=FONT, size=20, bold=True, color=HEX_FOREST)
SUBTITLE_FONT = Font(name=FONT, size=12, color="5C574F")
STAMP_FONT = Font(name=FONT, size=11, bold=True, color=HEX_ACCENT)

THIN = Side(style="thin", color="D9D9D9")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
TOP_RULE = Border(top=Side(style="medium", color=HEX_FOREST))

MONEY = '"$"#,##0;[Red]("$"#,##0)'
MONEY2 = '"$"#,##0.00;[Red]("$"#,##0.00)'
MILLIONS = '"$"#,##0.0,,"M";[Red]("$"#,##0.0,,"M")'
PERCENT1 = "0.0%"
PERCENT2 = "0.00%"
NUMBER0 = "#,##0"
NUMBER2 = "#,##0.00"
VOLUME = '#,##0" m³"'
VOL_PER_HA = '#,##0.0" m³/ha"'
HECTARES = '#,##0" ha"'
YEARS = '0" yrs"'
MULTIPLE = '0.00"x"'

# config path -> Excel defined name. These are the cells everything else reads.
NAMED_INPUTS: Dict[str, str] = {
    "tract.area_ha": "Area_ha",
    "growth.chapman_richards_a": "CR_a",
    "growth.chapman_richards_b": "CR_b",
    "growth.chapman_richards_c": "CR_c",
    "economics.net_stumpage_price": "Stumpage",
    "economics.real_price_growth": "PriceGrowth",
    "economics.regeneration_cost": "RegenCost",
    "economics.management_cost": "MgmtCost",
    "dcf.purchase_price": "PurchasePrice",
    "dcf.acquisition_cost_pct": "AcqCostPct",
    "dcf.real_discount_rate": "DiscRate",
    "dcf.hurdle_rate": "Hurdle",
    "dcf.holding_period_years": "HoldYears",
    "dcf.disposition_cost_pct": "DispCostPct",
    "dcf.even_flow_tolerance": "EvenFlowTol",
    "dcf.min_harvest_age": "MinHarvestAge",
    "carbon.enabled": "CarbonOn",
    "carbon.price_per_tonne": "CarbonPrice",
    "carbon.sequestration_rate": "CarbonRate",
    "carbon.eligible_area_fraction": "CarbonArea",
}

NUMBER_FORMATS: Dict[str, str] = {
    "tract.area_ha": HECTARES,
    "growth.chapman_richards_a": VOL_PER_HA,
    "growth.chapman_richards_b": "0.000",
    "growth.chapman_richards_c": "0.0",
    "growth.max_modelled_age": YEARS,
    "economics.net_stumpage_price": MONEY2,
    "economics.real_price_growth": PERCENT2,
    "economics.regeneration_cost": MONEY,
    "economics.management_cost": MONEY2,
    "dcf.purchase_price": MONEY,
    "dcf.acquisition_cost_pct": PERCENT2,
    "dcf.real_discount_rate": PERCENT2,
    "dcf.hurdle_rate": PERCENT2,
    "dcf.holding_period_years": YEARS,
    "dcf.even_flow_tolerance": PERCENT1,
    "dcf.disposition_cost_pct": PERCENT2,
    "dcf.min_harvest_age": YEARS,
    "dcf.min_ending_inventory_multiple": "0.00",
    "monte_carlo.n_simulations": NUMBER0,
    "monte_carlo.random_seed": "0",
    "monte_carlo.price_volatility": PERCENT1,
    "monte_carlo.mean_reversion_speed": "0.00",
    "monte_carlo.disturbance_probability": PERCENT2,
    "monte_carlo.disturbance_severity_mean": PERCENT1,
    "monte_carlo.disturbance_severity_concentration": "0.0",
    "monte_carlo.flexible_band_usage": "0.00",
    "carbon.price_per_tonne": MONEY2,
    "carbon.sequestration_rate": NUMBER2,
    "carbon.eligible_area_fraction": PERCENT1,
    "market.rolling_window_months": "0",
}

SECTION_TITLES = {
    "meta": "Project identity",
    "tract": "Tract: physical description",
    "growth": "Growth curve (Chapman-Richards)",
    "economics": "Prices and operating costs",
    "dcf": "Transaction, discounting and harvest rules",
    "monte_carlo": "Monte Carlo risk parameters",
    "carbon": "Carbon upside scenario",
    "market": "Market analysis inputs",
}

SHEETS = [
    ("Cover", "Title, contents and headline results"),
    ("Assumptions", "Every input, with its unit and source"),
    ("Growth_Rotation", "Yield curve and the three rotation criteria"),
    ("Harvest_Schedule", "Age classes, annual cut and the year-30 forest"),
    ("Cash_Flows", "30-year DCF, NPV, IRR, MOIC and breakeven"),
    ("Sensitivity", "Two-way IRR grids"),
    ("Monte_Carlo", "10,000-path risk simulation"),
    ("Market_Analysis", "Correlation and inflation evidence"),
]

DISCLAIMER = "HYPOTHETICAL TRACT, FOR PORTFOLIO PURPOSES ONLY. NOT INVESTMENT ADVICE."

# Growth_Rotation lists ages 1..max_age starting here, so age a sits on
# row GROWTH_FIRST_ROW + a - 1. Other sheets sum slices of that column.
GROWTH_FIRST_ROW = 12


# --- Small helpers ------------------------------------------------------

def _set(ws, cell: str, value, font=None, fmt=None, fill=None, align=None, border=None):
    """Write one cell with the usual styling knobs."""
    c = ws[cell]
    c.value = value
    c.font = font or BODY_FONT
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    if align:
        c.alignment = align
    if border:
        c.border = border
    return c


def _header_row(ws, row: int, labels: List[str], start_col: int = 1, widths=None):
    """Dark green header band."""
    for i, label in enumerate(labels):
        c = ws.cell(row=row, column=start_col + i, value=label)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        c.border = BOX
    ws.row_dimensions[row].height = 30
    if widths:
        for i, width in enumerate(widths):
            ws.column_dimensions[get_column_letter(start_col + i)].width = width


def _title_block(ws, title: str, subtitle: str, last_col: str = "H"):
    """Consistent two-line header at the top of every sheet."""
    _set(ws, "A1", title, font=TITLE_FONT)
    ws.merge_cells(f"A1:{last_col}1")
    ws.row_dimensions[1].height = 27
    _set(ws, "A2", subtitle, font=SUBTITLE_FONT)
    ws.merge_cells(f"A2:{last_col}2")
    _set(ws, "A3", DISCLAIMER, font=STAMP_FONT)
    ws.merge_cells(f"A3:{last_col}3")


def _widths(ws, mapping: Dict[str, float]):
    for col, width in mapping.items():
        ws.column_dimensions[col].width = width


def _print_setup(ws, area: str, landscape: bool = True, fit_wide: int = 1):
    ws.print_area = area
    ws.page_setup.orientation = "landscape" if landscape else "portrait"
    ws.page_setup.fitToWidth = fit_wide
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True
    ws.oddFooter.left.text = DISCLAIMER
    ws.oddFooter.left.size = 8
    ws.oddFooter.right.text = "Page &P of &N"
    ws.oddFooter.right.size = 8


# --- Sheet builders -----------------------------------------------------

def _build_assumptions(wb: Workbook) -> Dict[str, str]:
    """Write every assumption and return {config path: 'Assumptions!$B$n'}."""
    ws = wb["Assumptions"]
    _title_block(ws, "Assumptions", "One row per input. Blue on yellow cells are the only "
                                    "hard-coded numbers in this workbook.", "E")

    _header_row(ws, 5, ["Parameter", "Value", "Unit", "Source", "Note"],
                widths=[38, 16, 26, 44, 74])

    refs: Dict[str, str] = {}
    row = 6
    current_section = None

    for record in config.all_records():
        section = record.path.split(".")[0]
        if section != current_section:
            current_section = section
            c = ws.cell(row=row, column=1, value=SECTION_TITLES.get(section, section.title()))
            c.font = SECTION_FONT
            for col in range(1, 6):
                ws.cell(row=row, column=col).fill = SECTION_FILL
                ws.cell(row=row, column=col).border = BOX
            row += 1

        label = record.path.split(".")[-1].replace("_", " ")
        _set(ws, f"A{row}", label[0].upper() + label[1:], border=BOX)

        value = record.value
        if isinstance(value, (list, dict)):
            # Composite inputs (ticker lists, the age-class table) live on their
            # own sheets; here we just say where to look.
            shown = ", ".join(str(v) for v in value) if isinstance(value, list) else \
                "see Harvest_Schedule"
            _set(ws, f"B{row}", shown, font=INPUT_FONT, fill=INPUT_FILL, border=BOX,
                 align=Alignment(horizontal="left"))
        else:
            cell = _set(ws, f"B{row}", value, font=INPUT_FONT, fill=INPUT_FILL,
                        fmt=NUMBER_FORMATS.get(record.path), border=BOX)
            if isinstance(value, str):
                cell.alignment = Alignment(horizontal="left")
            refs[record.path] = f"Assumptions!$B${row}"

        _set(ws, f"C{row}", record.unit, border=BOX)
        source_cell = _set(ws, f"D{row}", record.source, border=BOX)
        if record.is_illustrative:
            source_cell.font = Font(name=FONT, size=11, color=HEX_ACCENT, italic=True)
        _set(ws, f"E{row}", record.note, font=NOTE_FONT, border=BOX,
             align=Alignment(wrap_text=False))
        row += 1

    last = row - 1
    row += 1
    _set(ws, f"A{row}", "Illustrative assumptions shown in orange are invented for this "
                        "exercise and must be replaced with sourced data before any real use.",
         font=Font(name=FONT, size=10, italic=True, color=HEX_ACCENT))

    for path, name in NAMED_INPUTS.items():
        if path in refs:
            wb.defined_names.add(DefinedName(name, attr_text=refs[path]))

    ws.freeze_panes = "A6"
    _print_setup(ws, f"A1:E{last}")
    return refs


def _build_growth(wb: Workbook, p: GrowthParams, res: DCFResult) -> Dict[str, str]:
    """Yield curve, the three rotation criteria, and the Faustmann search."""
    ws = wb["Growth_Rotation"]
    _title_block(
        ws, "Growth and optimal rotation",
        "Chapman-Richards yield curve. Every cell below is a formula reading the "
        "Assumptions sheet: change an input there and the optimum moves.", "G",
    )

    first, last = GROWTH_FIRST_ROW, GROWTH_FIRST_ROW + p.max_age - 1  # ages 1..max_age

    _set(ws, "A5", "Rotation criteria", font=SECTION_FONT)
    _header_row(ws, 6, ["Criterion", "Age (yrs)", "Value", "What it maximises"],
                widths=[30, 12, 16, 62])

    # Each optimum is located by MATCH on the column it maximises: the sheet
    # searches for the answer rather than being handed it.
    _set(ws, "A7", "Maximum mean annual increment", border=BOX)
    _set(ws, "B7", f"=INDEX($A${first}:$A${last},MATCH(MAX($C${first}:$C${last}),"
                   f"$C${first}:$C${last},0))", font=FORMULA_FONT, fmt=YEARS, border=BOX)
    _set(ws, "C7", f"=MAX($C${first}:$C${last})", font=FORMULA_FONT, fmt=VOL_PER_HA, border=BOX)
    _set(ws, "D7", "Volume per year, V(T)/T. Ignores money entirely.", border=BOX)

    _set(ws, "A8", "Single rotation NPV (Fisher)", border=BOX)
    _set(ws, "B8", f"=INDEX($A${first}:$A${last},MATCH(MAX($F${first}:$F${last}),"
                   f"$F${first}:$F${last},0))", font=FORMULA_FONT, fmt=YEARS, border=BOX)
    _set(ws, "C8", f"=MAX($F${first}:$F${last})", font=FORMULA_FONT, fmt=MONEY, border=BOX)
    _set(ws, "D8", "Discounted profit of one rotation. Counts interest, not land.", border=BOX)

    _set(ws, "A9", "Faustmann (land expectation value)", font=Font(name=FONT, bold=True,
                                                                   color=HEX_INK), border=BOX)
    _set(ws, "B9", f"=INDEX($A${first}:$A${last},MATCH(MAX($E${first}:$E${last}),"
                   f"$E${first}:$E${last},0))", font=Font(name=FONT, bold=True), fmt=YEARS,
         border=BOX)
    _set(ws, "C9", f"=MAX($E${first}:$E${last})", font=Font(name=FONT, bold=True), fmt=MONEY2,
         border=BOX)
    _set(ws, "D9", "Value of the bare land in perpetuity. Counts the opportunity "
                   "cost of delaying every future rotation, so it is the shortest.", border=BOX)

    _set(ws, "A10", "Rotation used in the cash flow model", font=SECTION_FONT)
    _set(ws, "B10", "=B9", font=LINK_FONT, fmt=YEARS)
    wb.defined_names.add(DefinedName("RotationAge", attr_text="Growth_Rotation!$B$9"))
    wb.defined_names.add(DefinedName("FaustmannLEV", attr_text="Growth_Rotation!$C$9"))

    _header_row(ws, 11, [
        "Age (yrs)", "Volume V(T) (m³/ha)", "MAI (m³/ha/yr)", "CAI (m³/ha/yr)",
        "LEV (\u0024/ha)", "Single-rotation NPV (\u0024/ha)",
    ], widths=[11, 20, 17, 17, 16, 24])

    for i, age in enumerate(range(1, p.max_age + 1)):
        r = first + i
        _set(ws, f"A{r}", age, font=FORMULA_FONT, fmt="0", border=BOX)
        _set(ws, f"B{r}", f"=CR_a*(1-EXP(-CR_b*$A{r}))^CR_c",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"C{r}", f"=$B{r}/$A{r}", font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"D{r}", f"=CR_a*CR_b*CR_c*(1-EXP(-CR_b*$A{r}))^(CR_c-1)*EXP(-CR_b*$A{r})",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        # LEV is undefined at T=1 (the denominator collapses); blank cells are
        # ignored by MAX, so the search still works.
        _set(ws, f"E{r}",
             f'=IF($A{r}<2,"",(Stumpage*$B{r}-RegenCost*(1+DiscRate)^$A{r})'
             f"/((1+DiscRate)^$A{r}-1)-MgmtCost/DiscRate)",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"F{r}",
             f"=Stumpage*$B{r}/(1+DiscRate)^$A{r}-RegenCost"
             f"-MgmtCost*(1-(1+DiscRate)^-$A{r})/DiscRate",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)

    chart = LineChart()
    chart.title = "Volume, MAI and CAI by stand age"
    chart.height, chart.width = 8.5, 17
    chart.y_axis.title = "m³/ha and m³/ha/yr"
    chart.x_axis.title = "Stand age (years)"
    data = Reference(ws, min_col=2, max_col=4, min_row=11, max_row=last)
    cats = Reference(ws, min_col=1, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, "H6")

    ws.freeze_panes = f"A{first}"
    _print_setup(ws, f"A1:F{last}")
    return {"rotation_age": "Growth_Rotation!$B$9", "lev": "Growth_Rotation!$C$9"}


def _build_harvest(wb: Workbook, p: GrowthParams, res: DCFResult) -> Dict[str, int]:
    """Opening age classes, the annual cut, and the year-30 forest that is sold."""
    ws = wb["Harvest_Schedule"]
    _title_block(
        ws, "Harvest schedule and age-class structure",
        "Hectares by age come from the cohort simulation in src/dcf.py. Volumes and "
        "values beside them are live formulas.", "L",
    )

    s = res.schedule

    _set(ws, "A5", "Opening age-class distribution (acquisition)", font=SECTION_FONT)
    _set(ws, "A6",
         "Hectares are spread evenly across the ten single years inside each class. "
         "Because the yield curve is convex, the average volume across those ten ages is "
         "not the volume at the midpoint age, so column D averages the live growth table "
         "rather than evaluating the curve once.", font=NOTE_FONT)
    _header_row(ws, 7, ["Age class", "Hectares", "Ages included",
                        "Average volume per ha (m³)", "Standing volume (m³)"],
                widths=[14, 13, 15, 24, 21])
    labels, _, hectares = config.age_class_arrays()
    row = 8
    for label, ha in zip(labels, hectares):
        low, high = (int(x) for x in str(label).split("-"))
        # Age a lives on row GROWTH_FIRST_ROW + a - 1; age 0 has zero volume and
        # is not in the table, so the sum starts at age 1 and still divides by 10.
        start = GROWTH_FIRST_ROW + max(low, 1) - 1
        end = GROWTH_FIRST_ROW + high - 1
        _set(ws, f"A{row}", label, border=BOX)
        _set(ws, f"B{row}", ha, font=INPUT_FONT, fill=INPUT_FILL, fmt=NUMBER0, border=BOX)
        _set(ws, f"C{row}", f"{low}-{high}", border=BOX)
        _set(ws, f"D{row}", f"=SUM(Growth_Rotation!$B${start}:$B${end})/{high - low + 1}",
             font=LINK_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"E{row}", f"=$B{row}*$D{row}", font=FORMULA_FONT, fmt=NUMBER0, border=BOX)
        row += 1
    total_row = row
    _set(ws, f"A{total_row}", "Total", font=Font(name=FONT, bold=True), border=TOP_RULE)
    _set(ws, f"B{total_row}", f"=SUM(B8:B{row-1})", font=Font(name=FONT, bold=True),
         fmt=HECTARES, border=TOP_RULE)
    _set(ws, f"E{total_row}", f"=SUM(E8:E{row-1})", font=Font(name=FONT, bold=True),
         fmt=NUMBER0, border=TOP_RULE)
    _set(ws, f"F{total_row}", "← opening inventory, m³", font=NOTE_FONT)

    # Annual harvest -----------------------------------------------------
    sched_top = total_row + 3
    _set(ws, f"A{sched_top - 1}", "Annual harvest (values from the cohort simulation)",
         font=SECTION_FONT)
    _header_row(ws, sched_top, ["Year", "Harvest volume (m³)", "Area cut (ha)",
                                "Lower even-flow bound", "Upper even-flow bound"],
                widths=[8, 20, 15, 21, 21])
    first_sched = sched_top + 1
    for i, (year, vol, ha) in enumerate(zip(s.years, s.volumes, s.areas_cut)):
        r = first_sched + i
        _set(ws, f"A{r}", int(year), font=FORMULA_FONT, fmt="0", border=BOX)
        _set(ws, f"B{r}", float(vol), font=INPUT_FONT, fill=INPUT_FILL, fmt=NUMBER0, border=BOX)
        _set(ws, f"C{r}", float(ha), font=INPUT_FONT, fill=INPUT_FILL, fmt=NUMBER2, border=BOX)
        _set(ws, f"D{r}", f"=AVERAGE($B${first_sched}:$B${first_sched + len(s.years) - 1})"
                          f"*(1-EvenFlowTol)", font=FORMULA_FONT, fmt=NUMBER0, border=BOX)
        _set(ws, f"E{r}", f"=AVERAGE($B${first_sched}:$B${first_sched + len(s.years) - 1})"
                          f"*(1+EvenFlowTol)", font=FORMULA_FONT, fmt=NUMBER0, border=BOX)
    last_sched = first_sched + len(s.years) - 1

    check_row = last_sched + 2
    _set(ws, f"A{check_row}", "Even-flow check", font=SECTION_FONT)
    _set(ws, f"B{check_row}",
         f'=IF(AND(MIN(B{first_sched}:B{last_sched})>=D{first_sched}-1,'
         f'MAX(B{first_sched}:B{last_sched})<=E{first_sched}+1),'
         f'"PASS: every year inside the band","FAIL")', font=FORMULA_FONT)

    # Year-30 forest, which is what the terminal value prices ---------------
    exit_top = check_row + 3
    _set(ws, f"A{exit_top - 1}",
         f"Age distribution at exit (year {int(config.num('dcf.holding_period_years'))}): "
         "this is the forest the next buyer acquires", font=SECTION_FONT)
    _header_row(ws, exit_top, [
        "Age (yrs)", "Hectares", "Volume per ha (m³)", "Standing volume (m³)",
        "Exit value per ha (\u0024)", "Exit value (\u0024)",
    ], widths=[10, 12, 19, 20, 20, 18])

    first_exit = exit_top + 1
    hold = int(config.num("dcf.holding_period_years"))
    for age in range(0, p.max_age + 1):
        r = first_exit + age
        _set(ws, f"A{r}", age, font=FORMULA_FONT, fmt="0", border=BOX)
        _set(ws, f"B{r}", float(s.ending_areas[age]), font=INPUT_FONT, fill=INPUT_FILL,
             fmt=NUMBER2, border=BOX)
        _set(ws, f"C{r}", f"=CR_a*(1-EXP(-CR_b*$A{r}))^CR_c",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"D{r}", f"=$B{r}*$C{r}", font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        # Mature stands are cut at once; immature stands are worth the discounted
        # value of cutting them when they reach the rotation age.
        _set(ws, f"E{r}",
             f"=IF($A{r}>=RotationAge,ExitPrice*$C{r}+LEVT_Exit,"
             f"(ExitPrice*VolAtRotation+LEVT_Exit)/(1+DiscRate)^(RotationAge-$A{r}))",
             font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        _set(ws, f"F{r}", f"=$B{r}*$E{r}", font=FORMULA_FONT, fmt=NUMBER0, border=BOX)
    last_exit = first_exit + p.max_age

    summary_row = last_exit + 1
    _set(ws, f"A{summary_row}", "Total at exit", font=Font(name=FONT, bold=True),
         border=TOP_RULE)
    _set(ws, f"B{summary_row}", f"=SUM(B{first_exit}:B{last_exit})",
         font=Font(name=FONT, bold=True), fmt=HECTARES, border=TOP_RULE)
    _set(ws, f"D{summary_row}", f"=SUM(D{first_exit}:D{last_exit})",
         font=Font(name=FONT, bold=True), fmt=NUMBER0, border=TOP_RULE)
    _set(ws, f"F{summary_row}", f"=SUM(F{first_exit}:F{last_exit})",
         font=Font(name=FONT, bold=True), fmt=MONEY, border=TOP_RULE)
    _set(ws, f"G{summary_row}", "← gross exit value before the perpetual management charge",
         font=NOTE_FONT)

    wb.defined_names.add(
        DefinedName("ExitInventoryValue", attr_text=f"Harvest_Schedule!$F${summary_row}")
    )
    wb.defined_names.add(
        DefinedName("ClosingInventory", attr_text=f"Harvest_Schedule!$D${summary_row}")
    )
    wb.defined_names.add(
        DefinedName("OpeningInventory", attr_text=f"Harvest_Schedule!$E${total_row}")
    )
    wb.defined_names.add(
        DefinedName("HarvestVolumes",
                    attr_text=f"Harvest_Schedule!$B${first_sched}:$B${last_sched}")
    )
    wb.defined_names.add(
        DefinedName("HarvestAreas",
                    attr_text=f"Harvest_Schedule!$C${first_sched}:$C${last_sched}")
    )

    ws.freeze_panes = "A8"
    _print_setup(ws, f"A1:G{summary_row}")
    return {"first_sched": first_sched, "last_sched": last_sched, "hold": hold,
            "opening_total_row": total_row, "even_flow_check_row": check_row}


def _build_cash_flows(wb: Workbook, res: DCFResult, rows: Dict[str, int]) -> Dict[str, str]:
    """The 30-year DCF. Everything on this sheet is a formula."""
    ws = wb["Cash_Flows"]
    _title_block(
        ws, "Acquisition cash flows",
        "Real Canadian dollars. Every cell is a formula: NPV and IRR use Excel's own "
        "functions over the net cash flow column.", "M",
    )

    hold = rows["hold"]
    first_sched = rows["first_sched"]

    # Exit-pricing block, so the terminal value is a formula rather than a paste.
    _set(ws, "A5", "Exit pricing inputs", font=SECTION_FONT)
    _set(ws, "A6", "Stumpage price at exit")
    _set(ws, "B6", "=Stumpage*(1+PriceGrowth)^HoldYears", font=FORMULA_FONT, fmt=MONEY2)
    _set(ws, "C6", "Real price escalated over the hold", font=NOTE_FONT)
    _set(ws, "A7", "Volume per ha at the rotation age")
    _set(ws, "B7", "=CR_a*(1-EXP(-CR_b*RotationAge))^CR_c", font=FORMULA_FONT, fmt=NUMBER2)
    _set(ws, "C7", "Chapman-Richards evaluated at the Faustmann age", font=NOTE_FONT)
    _set(ws, "A8", "Land expectation value at exit prices, before management")
    _set(ws, "B8", "=(ExitPrice*VolAtRotation-RegenCost*(1+DiscRate)^RotationAge)"
                   "/((1+DiscRate)^RotationAge-1)", font=FORMULA_FONT, fmt=MONEY2)
    _set(ws, "C8", "LEV recomputed at the year-30 price", font=NOTE_FONT)
    _set(ws, "A9", "Gross exit value of land and timber")
    _set(ws, "B9", "=ExitInventoryValue-MgmtCost/DiscRate*Area_ha", font=LINK_FONT, fmt=MONEY)
    _set(ws, "C9", "Summed from the year-30 age table on Harvest_Schedule", font=NOTE_FONT)
    _set(ws, "A10", "Net exit proceeds after disposition costs")
    _set(ws, "B10", "=B9*(1-DispCostPct)", font=FORMULA_FONT, fmt=MONEY)

    wb.defined_names.add(DefinedName("ExitPrice", attr_text="Cash_Flows!$B$6"))
    wb.defined_names.add(DefinedName("VolAtRotation", attr_text="Cash_Flows!$B$7"))
    wb.defined_names.add(DefinedName("LEVT_Exit", attr_text="Cash_Flows!$B$8"))
    wb.defined_names.add(DefinedName("TerminalValue", attr_text="Cash_Flows!$B$10"))

    _header_row(ws, 12, [
        "Year", "Harvest volume (m³)", "Area cut (ha)", "Stumpage (\u0024/m³)",
        "Harvest revenue", "Carbon revenue", "Regeneration", "Management",
        "Acquisition", "Terminal value", "Net cash flow", "Discount factor",
        "PV of cash flow", "Cumulative PV",
    ], widths=[7, 18, 13, 14, 16, 14, 14, 14, 16, 16, 16, 13, 16, 16])

    first = 13
    for year in range(0, hold + 1):
        r = first + year
        _set(ws, f"A{r}", year, font=FORMULA_FONT, fmt="0", border=BOX)
        if year == 0:
            _set(ws, f"B{r}", 0, font=FORMULA_FONT, fmt=NUMBER0, border=BOX)
            _set(ws, f"C{r}", 0, font=FORMULA_FONT, fmt=NUMBER2, border=BOX)
        else:
            src = first_sched + year - 1
            _set(ws, f"B{r}", f"=Harvest_Schedule!$B${src}", font=LINK_FONT,
                 fmt=NUMBER0, border=BOX)
            _set(ws, f"C{r}", f"=Harvest_Schedule!$C${src}", font=LINK_FONT,
                 fmt=NUMBER2, border=BOX)
        _set(ws, f"D{r}", f"=Stumpage*(1+PriceGrowth)^$A{r}", font=FORMULA_FONT,
             fmt=MONEY2, border=BOX)
        _set(ws, f"E{r}", f"=$B{r}*$D{r}", font=FORMULA_FONT, fmt=MONEY, border=BOX)
        _set(ws, f"F{r}",
             f"=IF(AND(CarbonOn,$A{r}>0),Area_ha*CarbonArea*CarbonRate*CarbonPrice,0)",
             font=FORMULA_FONT, fmt=MONEY, border=BOX)
        _set(ws, f"G{r}", f"=-$C{r}*RegenCost", font=FORMULA_FONT, fmt=MONEY, border=BOX)
        _set(ws, f"H{r}", f"=IF($A{r}>0,-Area_ha*MgmtCost,0)", font=FORMULA_FONT,
             fmt=MONEY, border=BOX)
        _set(ws, f"I{r}", f"=IF($A{r}=0,-PurchasePrice*(1+AcqCostPct),0)",
             font=FORMULA_FONT, fmt=MONEY, border=BOX)
        _set(ws, f"J{r}", f"=IF($A{r}=HoldYears,TerminalValue,0)", font=FORMULA_FONT,
             fmt=MONEY, border=BOX)
        _set(ws, f"K{r}", f"=SUM($E{r}:$J{r})", font=Font(name=FONT, bold=True),
             fmt=MONEY, border=BOX)
        _set(ws, f"L{r}", f"=1/(1+DiscRate)^$A{r}", font=FORMULA_FONT, fmt="0.0000",
             border=BOX)
        _set(ws, f"M{r}", f"=$K{r}*$L{r}", font=FORMULA_FONT, fmt=MONEY, border=BOX)
        _set(ws, f"N{r}", f"=SUM($M${first}:$M{r})", font=FORMULA_FONT, fmt=MONEY, border=BOX)
    last = first + hold

    out = last + 2
    _set(ws, f"A{out}", "Headline results", font=SECTION_FONT)
    labels = [
        ("Net present value at the discount rate",
         f"=$K${first}+NPV(DiscRate,$K${first + 1}:$K${last})", MONEY,
         "Excel's NPV discounts from period 1, so year 0 is added outside it."),
        ("Internal rate of return", f"=IRR($K${first}:$K${last})", PERCENT2,
         "Real, unlevered, before fees and tax."),
        ("Hurdle rate", "=Hurdle", PERCENT2, "Linked from Assumptions."),
        ("IRR less hurdle", f"=$B${out + 2}-$B${out + 3}", PERCENT2, "Positive means proceed."),
        ("Multiple on invested capital",
         f"=SUM($K${first + 1}:$K${last})/-$K${first}", MULTIPLE,
         "Undiscounted cash returned per dollar invested."),
        ("PV of cash flows excluding the purchase",
         f"=NPV(DiscRate,$K${first + 1}:$K${last})", MONEY,
         "What the tract is worth on these assumptions."),
        ("Breakeven purchase price",
         f"=NPV(DiscRate,$K${first + 1}:$K${last})/(1+AcqCostPct)", MONEY,
         "The price at which NPV is exactly zero."),
        ("Breakeven price per hectare",
         f"=$B${out + 7}/Area_ha", MONEY2, "Compare with the asking price per hectare."),
        ("Headroom versus the asking price",
         f"=$B${out + 7}/PurchasePrice-1", PERCENT1,
         "How far the price could rise before the deal stops clearing the hurdle."),
        ("Recommendation",
         f'=IF(AND($B${out + 1}>0,$B${out + 2}>=Hurdle),"BUY at or below the breakeven price",'
         f'"PASS at this price")', None, "Driven by the two tests above."),
    ]
    for i, (label, formula, fmt, note) in enumerate(labels):
        r = out + 1 + i
        _set(ws, f"A{r}", label, border=BOX)
        _set(ws, f"B{r}", formula, font=Font(name=FONT, bold=True, color="000000"),
             fmt=fmt, border=BOX)
        _set(ws, f"C{r}", note, font=NOTE_FONT)

    wb.defined_names.add(DefinedName("ProjectNPV", attr_text=f"Cash_Flows!$B${out + 1}"))
    wb.defined_names.add(DefinedName("ProjectIRR", attr_text=f"Cash_Flows!$B${out + 2}"))
    wb.defined_names.add(DefinedName("ProjectMOIC", attr_text=f"Cash_Flows!$B${out + 5}"))
    wb.defined_names.add(DefinedName("BreakevenPrice", attr_text=f"Cash_Flows!$B${out + 7}"))
    wb.defined_names.add(DefinedName("Recommendation", attr_text=f"Cash_Flows!$B${out + 10}"))

    ws.freeze_panes = "B13"
    _print_setup(ws, f"A1:N{out + 10}")
    return {"npv": f"Cash_Flows!$B${out + 1}", "irr": f"Cash_Flows!$B${out + 2}"}


def _write_grid(ws, top: int, grid: pd.DataFrame, title: str, note: str, hurdle: float):
    """One two-way sensitivity grid with a three-colour scale."""
    _set(ws, f"A{top}", title, font=SECTION_FONT)
    _set(ws, f"A{top + 1}", note, font=NOTE_FONT)

    header = top + 2
    _set(ws, f"A{header}", f"{grid.index.name} ↓ / {grid.columns.name} →",
         font=HEADER_FONT, fill=HEADER_FILL,
         align=Alignment(horizontal="left", vertical="center", wrap_text=True), border=BOX)
    for j, col in enumerate(grid.columns):
        c = ws.cell(row=header, column=2 + j, value=str(col))
        c.font, c.fill, c.border = HEADER_FONT, HEADER_FILL, BOX
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[header].height = 30

    for i, idx in enumerate(grid.index):
        r = header + 1 + i
        _set(ws, f"A{r}", str(idx), font=Font(name=FONT, bold=True), border=BOX)
        for j, col in enumerate(grid.columns):
            value = float(grid.iloc[i, j])
            c = ws.cell(row=r, column=2 + j, value=None if np.isnan(value) else value)
            c.number_format = PERCENT1
            c.border = BOX
            c.alignment = Alignment(horizontal="center")
            c.font = Font(name=FONT, bold=value >= hurdle)

    last = header + len(grid.index)
    span = f"B{header + 1}:{get_column_letter(1 + len(grid.columns))}{last}"
    ws.conditional_formatting.add(
        span,
        ColorScaleRule(
            start_type="min", start_color=style.hexcode(style.ACCENT),
            mid_type="num", mid_value=hurdle, mid_color="F6F1E7",
            end_type="max", end_color=style.hexcode(style.FOREST),
        ),
    )
    return last


def _build_sensitivity(wb: Workbook, res: DCFResult):
    ws = wb["Sensitivity"]
    _title_block(
        ws, "Two-way sensitivity",
        "Project IRR. Values are computed in Python because each cell is a complete "
        "model re-run, including re-solving the rotation and the harvest schedule.", "H",
    )
    _widths(ws, {"A": 26, **{get_column_letter(i): 13 for i in range(2, 9)}})

    hurdle = res.hurdle_rate
    end = _write_grid(
        ws, 5, res.grids["price_vs_discount"],
        "IRR by net stumpage price and real discount rate",
        "The discount rate barely moves the IRR: it enters only through the rotation age "
        "and the price of the year-30 exit, not through the operating cash flows.",
        hurdle,
    )
    end = _write_grid(
        ws, end + 3, res.grids["price_paid_vs_growth"],
        "IRR by purchase price and real stumpage price growth",
        "Entry price is the dominant variable. Bold cells clear the hurdle.",
        hurdle,
    )
    _set(ws, f"A{end + 2}", f"Hurdle rate for shading and bolding: {hurdle:.1%}",
         font=NOTE_FONT)
    ws.freeze_panes = "B6"
    _print_setup(ws, f"A1:H{end + 2}")


def _build_monte_carlo(wb: Workbook, mc: MonteCarloResult, figures: Dict[str, str]):
    ws = wb["Monte_Carlo"]
    _title_block(
        ws, "Monte Carlo risk analysis",
        f"{mc.n_sims:,} paths, seed {mc.seed}. Mean-reverting stumpage prices and random "
        "wildfire or pest losses, run through the same cohort model.", "J",
    )
    _widths(ws, {"A": 34, "B": 15, "C": 15, "D": 15, "E": 15, "F": 15, "G": 16, "H": 18})

    s = mc.summary()
    _set(ws, "A5", "IRR distribution by strategy", font=SECTION_FONT)
    _header_row(ws, 6, ["Strategy", "P5", "P25", "Median", "P75", "P95", "Mean",
                        "P(IRR < hurdle)"])
    for i, key in enumerate(("scheduled", "flexible")):
        r = 7 + i
        name = "Scheduled harvest" if key == "scheduled" else "Flexible harvest"
        _set(ws, f"A{r}", name, border=BOX)
        for j, q in enumerate(("p5", "p25", "p50", "p75", "p95", "mean")):
            c = ws.cell(row=r, column=2 + j, value=s[f"{key}_{q}"])
            c.number_format, c.border, c.font = PERCENT2, BOX, FORMULA_FONT
        c = ws.cell(row=r, column=8, value=s[f"{key}_prob_below_hurdle"])
        c.number_format, c.border = PERCENT1, BOX
        c.font = Font(name=FONT, bold=True, color=HEX_ACCENT)

    _set(ws, "A10", "Headline risk measures", font=SECTION_FONT)
    rows = [
        ("Hurdle rate", "=Hurdle", PERCENT2, "Linked from Assumptions"),
        ("Probability of missing the hurdle (scheduled)",
         s["scheduled_prob_below_hurdle"], PERCENT1,
         "Share of the 10,000 paths returning less than the hurdle"),
        ("Value of harvest timing flexibility", s["flexibility_value"], MONEY,
         "Mean NPV of the flexible strategy less the scheduled strategy"),
        ("Value of flexibility per hectare", s["flexibility_value_per_ha"], MONEY2,
         "The real option is worth this much per hectare"),
        ("Flexibility gain in mean IRR", s["flexibility_irr_gain"], PERCENT2,
         "Same comparison, expressed as a return"),
        ("Expected cost of fire and pests", s["disturbance_cost"], MONEY,
         "Mean NPV with disturbances switched off, less the base case, on identical price paths"),
        ("Expected cost per hectare", s["disturbance_cost_per_ha"], MONEY2, ""),
        ("Mean disturbance events per path", s["mean_disturbance_events"], NUMBER2,
         "30 years at the configured annual probability"),
        ("Paths seeing at least one event", s["prob_any_disturbance"], PERCENT1, ""),
        ("Worst single event in the simulation", s["worst_single_event_share"], PERCENT1,
         "Share of the tract destroyed in one year"),
    ]
    for i, (label, value, fmt, note) in enumerate(rows):
        r = 11 + i
        _set(ws, f"A{r}", label, border=BOX)
        c = _set(ws, f"B{r}", value, font=Font(name=FONT, bold=True), fmt=fmt, border=BOX)
        if isinstance(value, str) and value.startswith("="):
            c.font = LINK_FONT
        _set(ws, f"C{r}", note, font=NOTE_FONT)

    _set(ws, "A23", "Why flexibility has value", font=SECTION_FONT)
    _set(ws, "A24",
         "Prices mean revert, so a low price today is information: it is likely to rise. "
         "The even-flow agreement still allows the cut to move ±15% around the target, and "
         "shifting volume out of weak years into strong ones is worth the figure above. Under "
         "a random walk the same option would be worth nothing, because today's price would "
         "already be the best forecast of tomorrow's.", font=BODY_FONT)
    ws.merge_cells("A24:J27")
    ws["A24"].alignment = Alignment(wrap_text=True, vertical="top")

    _embed(ws, figures.get("irr_histogram"), "A30", width=760)
    _embed(ws, figures.get("price_fan"), "A70", width=760)
    _print_setup(ws, "A1:H24")


def _build_market(wb: Workbook, mk: MarketResult, figures: Dict[str, str]):
    ws = wb["Market_Analysis"]
    _title_block(
        ws, "Portfolio fit: correlation and inflation",
        "Real market data. This sheet tests the two claims usually made for timberland "
        "and reports what the data says, including where it disagrees.", "J",
    )

    n = len(mk.correlation.columns)
    _widths(ws, {"A": 30, **{get_column_letter(i): 13 for i in range(2, 2 + n)}})

    _set(ws, "A5", "Correlation of monthly total returns", font=SECTION_FONT)
    _set(ws, "A6", f"{mk.monthly_returns.index.min():%b %Y} to "
                   f"{mk.monthly_returns.index.max():%b %Y}. High correlation is shaded "
                   "orange because it is the unhelpful outcome for a diversification case.",
         font=NOTE_FONT)

    header = 7
    from .market import DISPLAY_NAMES
    labels = [DISPLAY_NAMES.get(c, c) for c in mk.correlation.columns]
    _set(ws, f"A{header}", "", fill=HEADER_FILL, border=BOX)
    for j, label in enumerate(labels):
        c = ws.cell(row=header, column=2 + j, value=label)
        c.font, c.fill, c.border = HEADER_FONT, HEADER_FILL, BOX
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[header].height = 34

    for i, label in enumerate(labels):
        r = header + 1 + i
        _set(ws, f"A{r}", label, font=Font(name=FONT, bold=True), border=BOX)
        for j in range(n):
            value = float(mk.correlation.iloc[i, j])
            c = ws.cell(row=r, column=2 + j, value=None if np.isnan(value) else value)
            c.number_format, c.border = "0.00", BOX
            c.alignment = Alignment(horizontal="center")
    corr_last = header + n
    ws.conditional_formatting.add(
        f"B{header + 1}:{get_column_letter(1 + n)}{corr_last}",
        ColorScaleRule(
            start_type="num", start_value=-1, start_color=style.hexcode(style.FOREST),
            mid_type="num", mid_value=0, mid_color="F6F1E7",
            end_type="num", end_value=1, end_color=style.hexcode(style.ACCENT),
        ),
    )

    reg_top = corr_last + 3
    _set(ws, f"A{reg_top}", "Inflation beta: annual total return regressed on annual CPI "
                            "inflation", font=SECTION_FONT)
    _set(ws, f"A{reg_top + 1}", f"CPI source: {mk.cpi_source}", font=NOTE_FONT)
    _header_row(ws, reg_top + 2, ["Asset", "Inflation beta", "Standard error", "t-statistic",
                                  "p-value", "R²", "Years"],
                widths=[30, 14, 15, 13, 12, 10, 9])
    for i, (asset, row) in enumerate(mk.regression.iterrows()):
        r = reg_top + 3 + i
        _set(ws, f"A{r}", str(asset), border=BOX)
        for j, key in enumerate(("inflation_beta", "std_error", "t_stat", "p_value",
                                 "r_squared")):
            c = ws.cell(row=r, column=2 + j, value=float(row[key]))
            c.number_format, c.border, c.font = NUMBER2, BOX, FORMULA_FONT
            if key == "p_value" and float(row[key]) < 0.10:
                c.font = Font(name=FONT, bold=True, color=HEX_FOREST)
        c = ws.cell(row=r, column=7, value=int(row["n_years"]))
        c.number_format, c.border = "0", BOX
    reg_last = reg_top + 2 + len(mk.regression)

    buckets_top = reg_last + 3
    _set(ws, f"A{buckets_top}", "Average annual return in high versus low inflation years",
         font=SECTION_FONT)
    _set(ws, f"A{buckets_top + 1}",
         f"Split at the median annual inflation rate of "
         f"{mk.inflation_buckets.attrs['median_inflation']:.2%} "
         f"({mk.inflation_buckets.attrs['n_high']} high years, "
         f"{mk.inflation_buckets.attrs['n_low']} low years).", font=NOTE_FONT)
    _header_row(ws, buckets_top + 2, ["Asset", "High inflation years", "Low inflation years",
                                      "Difference"], widths=[30, 19, 19, 14])
    for i, (_, row) in enumerate(mk.inflation_buckets.iterrows()):
        r = buckets_top + 3 + i
        _set(ws, f"A{r}", str(row["name"]), border=BOX)
        for j, key in enumerate(("high_inflation_mean", "low_inflation_mean", "difference")):
            c = ws.cell(row=r, column=2 + j, value=float(row[key]))
            c.number_format, c.border, c.font = PERCENT1, BOX, FORMULA_FONT
            if key == "difference":
                c.font = Font(name=FONT, bold=True,
                              color=HEX_FOREST if row[key] > 0 else HEX_ACCENT)
    buckets_last = buckets_top + 2 + len(mk.inflation_buckets)

    lim = buckets_last + 2
    _set(ws, f"A{lim}", "Limitation that qualifies everything on this sheet",
         font=Font(name=FONT, bold=True, color=HEX_ACCENT))
    _set(ws, f"A{lim + 1}", mk.limitations, font=BODY_FONT)
    ws.merge_cells(f"A{lim + 1}:J{lim + 5}")
    ws[f"A{lim + 1}"].alignment = Alignment(wrap_text=True, vertical="top")

    _embed(ws, figures.get("correlation_heatmap"), f"A{lim + 8}", width=700)
    _embed(ws, figures.get("rolling_correlation"), f"A{lim + 46}", width=780)
    ws.freeze_panes = "B8"
    _print_setup(ws, f"A1:{get_column_letter(1 + n)}{lim + 5}")


def _build_cover(wb: Workbook, res: DCFResult, mc: MonteCarloResult, mk: MarketResult):
    ws = wb["Cover"]
    ws.sheet_view.showGridLines = False
    _widths(ws, {"A": 4, "B": 46, "C": 24, "D": 66, "E": 4})

    _set(ws, "B2", "Project Aspen", font=Font(name=FONT, size=30, bold=True, color=HEX_FOREST))
    _set(ws, "B3", "Timberland acquisition analysis: 20,000 ha boreal tract, Alberta",
         font=Font(name=FONT, size=14, color="5C574F"))
    _set(ws, "B4", DISCLAIMER, font=Font(name=FONT, size=12, bold=True, color=HEX_ACCENT))
    _set(ws, "B5", f"Prepared by {config.val('meta.analyst')}  ·  "
                   f"all figures in real {config.val('meta.base_currency')}",
         font=Font(name=FONT, size=11, color="5C574F"))

    _set(ws, "B7", "Headline results", font=Font(name=FONT, size=14, bold=True,
                                                 color=HEX_FOREST))
    _set(ws, "B8", "Every figure below is a live link to the sheet that computes it.",
         font=NOTE_FONT)

    headline = [
        ("Asking price", "=PurchasePrice", MONEY, "Assumptions"),
        ("Net present value at the hurdle", "=ProjectNPV", MONEY, "Cash_Flows"),
        ("Project IRR", "=ProjectIRR", PERCENT2, "Cash_Flows"),
        ("Hurdle rate", "=Hurdle", PERCENT2, "Assumptions"),
        ("Multiple on invested capital", "=ProjectMOIC", MULTIPLE, "Cash_Flows"),
        ("Breakeven purchase price", "=BreakevenPrice", MONEY, "Cash_Flows"),
        ("Faustmann rotation age", "=RotationAge", YEARS, "Growth_Rotation"),
        ("Land expectation value", "=FaustmannLEV", MONEY2, "Growth_Rotation"),
        ("Sustainable annual cut", "=AVERAGE(HarvestVolumes)", VOLUME, "Harvest_Schedule"),
        ("Recommendation", "=Recommendation", None, "Cash_Flows"),
    ]
    _header_row(ws, 9, ["Measure", "Value", "Computed on"], start_col=2,
                widths=[46, 24, 66])
    for i, (label, formula, fmt, sheet) in enumerate(headline):
        r = 10 + i
        _set(ws, f"B{r}", label, border=BOX)
        _set(ws, f"C{r}", formula, font=Font(name=FONT, bold=True, color="1B7A3E"),
             fmt=fmt, border=BOX)
        _set(ws, f"D{r}", sheet, font=NOTE_FONT, border=BOX)

    risk_top = 21
    _set(ws, f"B{risk_top}", "Risk and portfolio fit",
         font=Font(name=FONT, size=14, bold=True, color=HEX_FOREST))
    s = mc.summary()
    m = mk.summary()
    risk_rows = [
        ("Probability of missing the hurdle",
         s["scheduled_prob_below_hurdle"], PERCENT1,
         f"{mc.n_sims:,} simulated paths, seed {mc.seed}"),
        ("IRR range, 5th to 95th percentile",
         f"{s['scheduled_p5']:.1%} to {s['scheduled_p95']:.1%}", None, "Monte_Carlo"),
        ("Value of harvest timing flexibility", s["flexibility_value"], MONEY,
         "A real option created by mean-reverting prices"),
        ("Expected cost of fire and pests", s["disturbance_cost"], MONEY,
         "Measured against identical price paths with no disturbance"),
        ("Timber proxy correlation to the S&P/TSX",
         m["mean_timber_corr_to_benchmark"], "0.00",
         "Listed REITs are equities first, an upper bound on true correlation"),
        ("Proxies with a significant inflation beta",
         f"{m['n_significant_hedges']} of {len(mk.regression)}", None,
         "At the 10% level; see Market_Analysis"),
    ]
    _header_row(ws, risk_top + 1, ["Measure", "Value", "Basis"], start_col=2)
    for i, (label, value, fmt, note) in enumerate(risk_rows):
        r = risk_top + 2 + i
        _set(ws, f"B{r}", label, border=BOX)
        _set(ws, f"C{r}", value, font=Font(name=FONT, bold=True), fmt=fmt, border=BOX)
        _set(ws, f"D{r}", note, font=NOTE_FONT, border=BOX)

    toc = risk_top + 10
    _set(ws, f"B{toc}", "Contents", font=Font(name=FONT, size=14, bold=True, color=HEX_FOREST))
    _header_row(ws, toc + 1, ["Sheet", "", "What it contains"], start_col=2)
    for i, (name, description) in enumerate(SHEETS):
        r = toc + 2 + i
        c = _set(ws, f"B{r}", name, font=Font(name=FONT, bold=True, color="1B7A3E",
                                              underline="single"), border=BOX)
        c.hyperlink = f"#'{name}'!A1"
        _set(ws, f"D{r}", description, border=BOX)

    key = toc + 11
    _set(ws, f"B{key}", "How to read this workbook",
         font=Font(name=FONT, size=14, bold=True, color=HEX_FOREST))
    legend = [
        ("Blue text on yellow", INPUT_FONT, INPUT_FILL,
         "A hard input. These are the only numbers typed in; change one and everything "
         "downstream moves."),
        ("Black text", FORMULA_FONT, None,
         "A formula calculated on that sheet. Click it to see the logic."),
        ("Green text", LINK_FONT, None,
         "A link to another sheet, or a value produced by the Python model where Excel "
         "cannot reproduce the calculation."),
    ]
    for i, (label, font, fill, meaning) in enumerate(legend):
        r = key + 1 + i
        _set(ws, f"B{r}", label, font=font, fill=fill, border=BOX)
        _set(ws, f"D{r}", meaning, border=BOX)

    note = key + 5
    _set(ws, f"B{note}",
         "Note: this file is generated by openpyxl, which writes formulas but does not "
         "calculate them. Formula cells appear blank until Excel or LibreOffice opens the "
         "workbook and recalculates. Press Ctrl+Alt+F9 to force a full rebuild.",
         font=NOTE_FONT)
    ws.merge_cells(f"B{note}:D{note + 2}")
    ws[f"B{note}"].alignment = Alignment(wrap_text=True, vertical="top")

    _print_setup(ws, f"A1:E{note + 2}", landscape=False)


def _embed(ws, image_path: Optional[str], anchor: str, width: int = 720):
    """Drop a PNG onto a sheet, scaled to a sensible width."""
    if not image_path or not Path(image_path).exists():
        return
    img = XLImage(image_path)
    scale = width / img.width
    img.width = int(img.width * scale)
    img.height = int(img.height * scale)
    img.anchor = anchor
    ws.add_image(img)


# --- Entry point --------------------------------------------------------

def build(
    res: DCFResult,
    mc: MonteCarloResult,
    mk: MarketResult,
    figures: Dict[str, str],
    path: Optional[Path] = None,
) -> Path:
    """Assemble the whole workbook and save it."""
    path = Path(path) if path else config.OUTPUTS / "Project_Aspen_Model.xlsx"
    p = GrowthParams.from_config()

    wb = Workbook()
    wb.remove(wb.active)
    for name, _ in SHEETS:
        wb.create_sheet(name)

    _build_assumptions(wb)
    _build_growth(wb, p, res)
    rows = _build_harvest(wb, p, res)
    _build_cash_flows(wb, res, rows)
    _build_sensitivity(wb, res)
    _build_monte_carlo(wb, mc, figures)
    _build_market(wb, mk, figures)
    _build_cover(wb, res, mc, mk)          # last: it links to names defined above

    wb.active = 0
    wb.calculation.fullCalcOnLoad = True   # make Excel evaluate on open
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
