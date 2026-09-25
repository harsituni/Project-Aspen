"""One visual identity for every chart, the deck and the portfolio page.

Monetary figures are real CAD.

The palette is defined once here and imported everywhere else. Colours were
picked so that their greyscale luminances stay far apart, which keeps the
portfolio page readable when it is printed on a black-and-white printer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

matplotlib.use("Agg")

# --- Core palette ----------------------------------------------------
# Luminances are spaced roughly 0.15 apart so the series stay separable in
# greyscale; see check_print_contrast, which the portfolio page runs on build.
FOREST = "#1B4332"        # primary: dark forest green   (luminance 0.21)
FOREST_MID = "#2D6A4F"
FOREST_LIGHT = "#85B7A2"  #                              (0.65)
FOREST_PALE = "#C6DBCF"

SAND = "#E8E0D2"          # secondary: warm neutral      (0.88)
SAND_DEEP = "#D6CBB4"     #                              (0.80)
INK = "#22201D"           # body text
INK_SOFT = "#5C574F"      # secondary text, axis labels
GRIDLINE = "#DAD4C8"

ACCENT = "#C1663B"        # single accent: burnt sienna, used sparingly (0.49)
ACCENT_PALE = "#EBC3AC"

WHITE = "#FFFFFF"

# Ordered series colours. Sorted by greyscale luminance these run
# 0.21, 0.34, 0.49, 0.65, 0.80: every neighbouring pair at least 0.13 apart,
# so a line chart with five series still reads on a black-and-white printer.
SERIES = [FOREST, ACCENT, FOREST_LIGHT, SAND_DEEP, INK_SOFT, FOREST_MID, ACCENT_PALE]

#: The five colours that actually encode data, in luminance order.
SERIES_PRINT_CRITICAL = [FOREST, INK_SOFT, ACCENT, FOREST_LIGHT, SAND_DEEP]

# Divergent map for sensitivity grids: accent (bad) -> cream -> forest (good).
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "aspen_diverging", [ACCENT, "#E9C9B4", "#F6F1E7", FOREST_LIGHT, FOREST]
)
# Sequential map for correlation magnitude.
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "aspen_sequential", ["#F6F1E7", FOREST_LIGHT, FOREST]
)
# Correlation map. Deliberately the reverse of the IRR map: for a diversification
# thesis a HIGH correlation is the bad outcome, so high reads accent, not forest.
CMAP_CORRELATION = CMAP_DIVERGING.reversed()

DISCLAIMER = "Hypothetical tract, for portfolio purposes only"

# Hex values without the leading hash, for python-pptx and openpyxl.
def hexcode(colour: str) -> str:
    """'#1B4332' -> '1B4332', the form pptx and openpyxl expect."""
    return colour.lstrip("#").upper()


def greyscale(colour: str) -> float:
    """Relative luminance, used by the print-legibility check."""
    r, g, b = to_rgb(colour)
    return 0.299 * r + 0.587 * g + 0.114 * b


def apply_style() -> None:
    """Set matplotlib defaults. Call once before building any figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans", "Arial"],
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK_SOFT,
            "axes.labelpad": 7,
            "axes.edgecolor": GRIDLINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": INK,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            # Dollar signs are everywhere in these labels; without this a pair of
            # them is silently parsed as mathtext and the text renders as italics.
            "text.parse_math": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
        }
    )


def titled(ax, title: str, subtitle: str | None = None, wrap_at: int = 92) -> None:
    """Bold title with an optional lighter subtitle underneath it.

    Both are drawn as text above the axes rather than with set_title, so the
    two never land on the same baseline no matter how long the strings are.
    """
    import textwrap

    title_lines = textwrap.wrap(title, wrap_at) or [title]
    sub_lines = textwrap.wrap(subtitle, wrap_at + 14) if subtitle else []

    # Work downwards from the top so extra lines push the title up, not into the axes.
    line_height = 0.052
    sub_height = 0.044
    y = 1.012
    for line in reversed(sub_lines):
        ax.text(0.0, y, line, transform=ax.transAxes, fontsize=9.5,
                color=INK_SOFT, va="bottom", ha="left")
        y += sub_height
    if sub_lines:
        y += 0.012
    for line in reversed(title_lines):
        ax.text(0.0, y, line, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=INK, va="bottom", ha="left")
        y += line_height


def stamp(fig, note: str | None = None) -> None:
    """Footer stamp so a figure lifted out of context is still labelled hypothetical."""
    text = DISCLAIMER if note is None else f"{DISCLAIMER}  |  {note}"
    fig.text(0.005, 0.005, text, fontsize=7.5, color=INK_SOFT, ha="left", va="bottom")


def save(fig, path: Path | str, note: str | None = None) -> Path:
    """Stamp, save and close a figure. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp(fig, note)
    fig.savefig(out)
    plt.close(fig)
    return out


def money(value: float, decimals: int = 1) -> str:
    """Compact currency label: 70000000 -> '$70.0M'."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}${v / 1e9:.{decimals}f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:.{decimals}f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:.0f}k"
    return f"{sign}${v:,.0f}"


def percent(value: float, decimals: int = 1) -> str:
    """0.0642 -> '6.4%'."""
    return f"{value * 100:.{decimals}f}%"


def check_print_contrast(colours: Iterable[str], min_gap: float = 0.12) -> list[tuple[str, str, float]]:
    """Return colour pairs whose greyscale luminances are too close to tell apart."""
    items = list(colours)
    clashes = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            gap = abs(greyscale(a) - greyscale(b))
            if gap < min_gap:
                clashes.append((a, b, gap))
    return clashes
