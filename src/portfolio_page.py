"""One-page printable summary of Project Aspen.

Monetary figures are real CAD.

Landscape US Letter, designed to be handed over or pinned up: headline result,
the four charts that carry the argument, how it was built, and a QR code back
to the repository. Checked in greyscale as well as colour, because a printed
copy is the most likely way anyone sees it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from . import config, style
from .dcf import DCFResult
from .market import MarketResult
from .montecarlo import MonteCarloResult
from .rotation import RotationResults

GITHUB_URL = "https://github.com/harsituni/project-aspen"

PAGE_W, PAGE_H = 11.0, 8.5  # landscape US Letter, inches

METHODS = [
    "Chapman-Richards growth curve",
    "Faustmann rotation vs max-MAI and Fisher",
    "30-year DCF on an even-flow harvest schedule",
    "10,000-path Monte Carlo with mean-reverting prices and wildfire",
    "Harvest-timing real option",
    "Correlation and inflation-beta tests on real market data",
]

TOOLS = ["Python", "pandas", "NumPy", "SciPy", "matplotlib",
         "Excel (live formulas)", "PowerPoint", "pytest"]


def _panel(fig, image_path: str, rect):
    """Place a saved figure inside a rectangle given in inches, preserving aspect."""
    left, bottom, width, height = rect
    ax = fig.add_axes([left / PAGE_W, bottom / PAGE_H, width / PAGE_W, height / PAGE_H])
    ax.set_axis_off()
    if not image_path or not Path(image_path).exists():
        ax.text(0.5, 0.5, "figure not available", ha="center", va="center",
                fontsize=9, color=style.INK_SOFT)
        return ax
    image = plt.imread(image_path)
    ax.imshow(image, interpolation="antialiased")
    return ax


def _qr_code(fig, url: str, rect):
    """Render a QR code as an image axes. Pure black so it scans from a print."""
    import qrcode

    left, bottom, width, height = rect
    code = qrcode.QRCode(box_size=10, border=1,
                         error_correction=qrcode.constants.ERROR_CORRECT_M)
    code.add_data(url)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white").convert("L")

    ax = fig.add_axes([left / PAGE_W, bottom / PAGE_H, width / PAGE_W, height / PAGE_H])
    ax.imshow(np.array(image), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_axis_off()
    return ax


def _rule(fig, x0, x1, y, colour=None, lw=1.0):
    fig.add_artist(plt.Line2D([x0 / PAGE_W, x1 / PAGE_W], [y / PAGE_H, y / PAGE_H],
                              color=colour or style.SAND_DEEP, lw=lw,
                              transform=fig.transFigure))


def _band(fig, x, y, w, h, colour):
    fig.add_artist(Rectangle((x / PAGE_W, y / PAGE_H), w / PAGE_W, h / PAGE_H,
                             transform=fig.transFigure, facecolor=colour,
                             edgecolor="none", zorder=0))


def _txt(fig, x, y, text, size=10, weight="normal", colour=None, ha="left", va="baseline",
         style_=None):
    return fig.text(x / PAGE_W, y / PAGE_H, text, fontsize=size, fontweight=weight,
                    color=colour or style.INK, ha=ha, va=va, style=style_ or "normal")


def _wrap(text: str, width_inches: float, size: float, bold: bool = False):
    """Wrap to a width measured in inches, using an average glyph width.

    Good enough for a fixed layout, and it avoids a font-metrics round trip
    just to break two lines of a headline.
    """
    import textwrap

    per_char = size * (0.55 if bold else 0.50) / 72.0
    chars = max(20, int(width_inches / per_char))
    return textwrap.wrap(text, chars) or [text]


def compose(
    rot: RotationResults,
    res: DCFResult,
    mc: MonteCarloResult,
    mk: MarketResult,
    figures: Dict[str, str],
):
    """Build the page as a matplotlib figure."""
    style.apply_style()
    d = res.summary()
    m = mc.summary()
    mkt = mk.summary()

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(style.WHITE)

    left_edge = 0.45
    right_edge = PAGE_W - 0.45
    qr_left = PAGE_W - 1.15          # everything textual must stop short of this
    text_right = qr_left - 0.25

    # --- Header -----------------------------------------------------------
    _band(fig, 0, PAGE_H - 0.14, PAGE_W, 0.14, style.FOREST)

    _txt(fig, left_edge, PAGE_H - 0.60, "Project Aspen", size=24, weight="bold",
         colour=style.FOREST)
    _txt(fig, left_edge, PAGE_H - 0.86,
         "Should a pension fund buy a hypothetical 20,000-hectare boreal timberland tract "
         "in Alberta, and at what price?",
         size=11, colour=style.INK_SOFT)

    answer = (
        f"Answer: buy below {style.money(res.breakeven_price)} "
        f"(${res.breakeven_per_ha:,.0f}/ha). At the {style.money(res.purchase_price)} ask "
        f"the tract returns {res.irr:.1%} real against a {res.hurdle_rate:.0%} hurdle, with "
        f"{m['scheduled_prob_below_hurdle']:.0%} of simulated paths falling short."
    )
    y = PAGE_H - 1.12
    for line in _wrap(answer, text_right - left_edge, 10.5, bold=True):
        _txt(fig, left_edge, y, line, size=10.5, weight="bold", colour=style.INK)
        y -= 0.19

    # QR block, in its own reserved column so nothing can run into it.
    _qr_code(fig, GITHUB_URL, (qr_left, PAGE_H - 1.18, 0.70, 0.70))
    _txt(fig, qr_left + 0.35, PAGE_H - 1.34, "code + write-up", size=7,
         colour=style.INK_SOFT, ha="center")
    _txt(fig, qr_left + 0.35, PAGE_H - 0.36, "HYPOTHETICAL", size=7.5, weight="bold",
         colour=style.ACCENT, ha="center")

    rule_y = PAGE_H - 1.52
    _rule(fig, left_edge, right_edge, rule_y, style.SAND_DEEP, 1.4)

    # --- Headline metrics ---------------------------------------------------
    metrics = [
        ("IRR", f"{res.irr:.1%}", f"vs {res.hurdle_rate:.0%} hurdle"),
        ("NPV", style.money(res.npv), f"at {res.discount_rate:.1%} real"),
        ("Breakeven", style.money(res.breakeven_price),
         f"{d['breakeven_premium_vs_ask']:+.0%} vs ask"),
        ("Rotation", f"{res.rotation_age} yrs",
         f"Faustmann; max-MAI {rot.max_mai_age:.0f}"),
        ("P(miss hurdle)", f"{m['scheduled_prob_below_hurdle']:.0%}",
         f"{mc.n_sims:,} paths"),
        ("Flexibility", style.money(mc.flexibility_value), "real option on timing"),
    ]
    card_top = rule_y - 0.86
    card_h = 0.74
    box_w = (right_edge - left_edge - 5 * 0.09) / 6
    x = left_edge
    for label, value, note in metrics:
        _band(fig, x, card_top, box_w, card_h, style.SAND)
        _txt(fig, x + 0.09, card_top + 0.54, label.upper(), size=7, weight="bold",
             colour=style.INK_SOFT)
        _txt(fig, x + 0.09, card_top + 0.27, value, size=13.5, weight="bold",
             colour=style.FOREST)
        _txt(fig, x + 0.09, card_top + 0.09, note, size=7, colour=style.INK_SOFT)
        x += box_w + 0.09

    # --- Bottom strips, laid out first so the grid can claim what is left ----
    _band(fig, 0, 0, PAGE_W, 0.08, style.FOREST)
    _txt(fig, left_edge, 0.18,
         f"{config.val('meta.analyst')}  ·  the tract is fictional; every assumption about "
         f"it is illustrative and carries a source field in config/assumptions.yaml  ·  "
         f"market data from Yahoo Finance and Statistics Canada is real and cited",
         size=7, colour=style.INK_SOFT)
    _txt(fig, left_edge, 0.33,
         "Listed timber and farmland REITs are equities first, so the measured correlations "
         "are an upper bound and understate the diversification private timberland would "
         "deliver.",
         size=7, colour=style.INK_SOFT, style_="italic")

    _band(fig, left_edge, 0.46, right_edge - left_edge, 0.28, style.SAND)
    _txt(fig, left_edge + 0.12, 0.55, "TOOLS", size=8, weight="bold", colour=style.ACCENT)
    _txt(fig, left_edge + 0.80, 0.55, "   ·   ".join(TOOLS), size=8, colour=style.INK)

    # The methods list is long; wrap it rather than let it run off the sheet.
    methods_x = left_edge + 0.73
    _txt(fig, left_edge, 1.04, "METHODS", size=8, weight="bold", colour=style.ACCENT)
    y = 1.04
    for line in _wrap("   ·   ".join(METHODS), right_edge - methods_x, 8):
        _txt(fig, methods_x, y, line, size=8, colour=style.INK)
        y -= 0.16

    grid_bottom = 1.26
    _rule(fig, left_edge, right_edge, 1.20, style.SAND_DEEP, 1.4)

    # --- Two by two figure grid ----------------------------------------------
    grid_top = card_top - 0.14
    cell_h = (grid_top - grid_bottom - 0.08) / 2
    cell_w = (right_edge - left_edge - 0.16) / 2

    panels = [
        ("lev_curve", left_edge, grid_top - cell_h),
        ("sensitivity_price_growth", left_edge + cell_w + 0.16, grid_top - cell_h),
        ("irr_histogram", left_edge, grid_bottom),
        ("correlation_heatmap", left_edge + cell_w + 0.16, grid_bottom),
    ]
    for key, left, bottom in panels:
        _panel(fig, figures.get(key, ""), (left, bottom, cell_w, cell_h))

    return fig


def _overflow_inches(fig) -> float:
    """How far the widest artist runs past the edges of the sheet, in inches.

    A fixed-size page silently clips anything too wide, so this is checked on
    every build rather than left to be noticed on a printout.
    """
    renderer = fig.canvas.get_renderer()
    worst = 0.0
    for artist in fig.get_children():
        if not artist.get_visible():
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        worst = max(worst, box.x1 / fig.dpi - PAGE_W, -box.x0 / fig.dpi,
                    box.y1 / fig.dpi - PAGE_H, -box.y0 / fig.dpi)
    return worst


def greyscale_preview(pdf_path: Path, figures: Dict[str, str]) -> Optional[Path]:
    """Save a desaturated PNG of the page so the print version can be eyeballed."""
    try:
        import fitz  # PyMuPDF, optional
    except ImportError:
        return None

    out = Path(pdf_path).with_name("portfolio_page_greyscale.png")
    with fitz.open(pdf_path) as doc:
        pixmap = doc[0].get_pixmap(dpi=110, colorspace=fitz.csGRAY)
        pixmap.save(out)
    return out


def build(
    rot: RotationResults,
    res: DCFResult,
    mc: MonteCarloResult,
    mk: MarketResult,
    figures: Dict[str, str],
    path: Optional[Path] = None,
) -> Path:
    """Render the page to PDF and report any greyscale legibility problems."""
    path = Path(path) if path else config.OUTPUTS / "Project_Aspen_Portfolio_Page.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = compose(rot, res, mc, mk, figures)
    # The global style saves figures with bbox='tight', which would resize this
    # sheet to fit its contents. A printable page must stay exactly Letter, so
    # the setting is suspended here and overflow is caught by the check below.
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        with PdfPages(path) as pdf:
            pdf.savefig(fig, facecolor=style.WHITE)

    overflow = _overflow_inches(fig)
    if overflow > 0.02:
        print(f"    ! layout warning: page content overruns the sheet by "
              f"{overflow:.2f} in and will be clipped when printed")
    plt.close(fig)

    clashes = style.check_print_contrast(style.SERIES_PRINT_CRITICAL)
    if clashes:
        pairs = ", ".join(f"{a}/{b} (gap {gap:.2f})" for a, b, gap in clashes)
        print(f"    ! greyscale warning: these palette colours are too close: {pairs}")
    else:
        print("      greyscale check: all series colours separable in black and white")

    preview = greyscale_preview(path, figures)
    if preview:
        print(f"      greyscale preview: {preview}")
    return path
