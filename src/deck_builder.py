"""Six-slide investment committee deck.

Monetary figures are real CAD.

Every slide title is a claim, not a label: a reader who sees only the titles
should get the whole argument. Every number in those claims is pulled from
the model at build time, so the deck cannot drift away from the analysis.

Speaker notes on each slide explain the numbers in plain language, including
what would change the recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from . import config, style
from .dcf import DCFResult
from .market import MarketResult
from .montecarlo import MonteCarloResult
from .rotation import RotationResults

WIDTH = Inches(13.333)
HEIGHT = Inches(7.5)
MARGIN = Inches(0.62)
CONTENT_W = WIDTH - 2 * MARGIN

FONT = "Calibri"


def _rgb(colour: str) -> RGBColor:
    return RGBColor.from_string(style.hexcode(colour))


FOREST = _rgb(style.FOREST)
FOREST_MID = _rgb(style.FOREST_MID)
SAND = _rgb(style.SAND)
SAND_DEEP = _rgb(style.SAND_DEEP)
ACCENT = _rgb(style.ACCENT)
INK = _rgb(style.INK)
INK_SOFT = _rgb(style.INK_SOFT)
WHITE = _rgb(style.WHITE)


# --- Layout primitives --------------------------------------------------

def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _box(slide, left, top, width, height, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def _text(slide, left, top, width, height, runs: Sequence[dict],
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing: float = 1.0):
    """Add a textbox. `runs` is a list of paragraph specs."""
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    for i, spec in enumerate(runs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = spec.get("align", align)
        para.line_spacing = spec.get("spacing", spacing)
        if spec.get("space_before"):
            para.space_before = Pt(spec["space_before"])
        if spec.get("space_after"):
            para.space_after = Pt(spec["space_after"])
        run = para.add_run()
        run.text = spec["text"]
        font = run.font
        font.name = FONT
        font.size = Pt(spec.get("size", 14))
        font.bold = spec.get("bold", False)
        font.italic = spec.get("italic", False)
        font.color.rgb = spec.get("color", INK)
    return box


def _slide_header(slide, takeaway: str, eyebrow: str, number: int):
    """Eyebrow label, claim-style takeaway title, and a rule beneath."""
    _text(slide, MARGIN, Inches(0.36), CONTENT_W, Inches(0.28), [
        {"text": eyebrow.upper(), "size": 11, "bold": True, "color": ACCENT},
    ]).name = "eyebrow"
    _text(slide, MARGIN, Inches(0.66), CONTENT_W, Inches(1.0), [
        {"text": takeaway, "size": 26, "bold": True, "color": FOREST, "spacing": 0.92},
    ]).name = "takeaway"
    rule = _box(slide, MARGIN, Inches(1.62), CONTENT_W, Pt(2.2), fill=SAND_DEEP)
    rule.line.fill.background()
    _footer(slide, number)


def _footer(slide, number: int):
    _text(slide, MARGIN, HEIGHT - Inches(0.44), CONTENT_W - Inches(0.6), Inches(0.26), [
        {"text": "Project Aspen  ·  hypothetical tract, prepared for portfolio purposes "
                 "only  ·  not investment advice", "size": 9, "color": INK_SOFT},
    ])
    _text(slide, WIDTH - MARGIN - Inches(0.6), HEIGHT - Inches(0.44), Inches(0.6),
          Inches(0.26), [
        {"text": str(number), "size": 9, "color": INK_SOFT, "align": PP_ALIGN.RIGHT},
    ])


def _notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text.strip()


def _breakeven_stumpage(res: DCFResult) -> float:
    """Stumpage price at which the IRR equals the hurdle, cached by dcf.build."""
    from .dcf import breakeven_stumpage

    cached = res.grids.get("price_vs_discount")
    if cached is not None and "breakeven_stumpage" in cached.attrs:
        return float(cached.attrs["breakeven_stumpage"])
    return breakeven_stumpage(res)


def _picture(slide, path: Optional[str], left, top, max_w, max_h):
    """Insert a figure scaled to fit the box, centred within it."""
    if not path or not Path(path).exists():
        return None
    from PIL import Image

    with Image.open(path) as img:
        aspect = img.width / img.height
    width = max_w
    height = Emu(int(width / aspect))
    if height > max_h:
        height = max_h
        width = Emu(int(height * aspect))
    left = Emu(int(left + (max_w - width) / 2))
    top = Emu(int(top + (max_h - height) / 2))
    return slide.shapes.add_picture(path, left, top, width, height)


def _stat_card(slide, left, top, width, height, label: str, value: str, note: str,
               highlight: bool = False):
    """One headline metric in a filled card."""
    _box(slide, left, top, width, height,
         fill=FOREST if highlight else SAND,
         line=None)
    colour_label = SAND_DEEP if highlight else INK_SOFT
    colour_value = WHITE if highlight else FOREST
    colour_note = SAND if highlight else INK_SOFT
    pad = Inches(0.18)
    _text(slide, left + pad, top + Inches(0.14), width - 2 * pad, Inches(0.26), [
        {"text": label.upper(), "size": 10, "bold": True, "color": colour_label},
    ])
    _text(slide, left + pad, top + Inches(0.42), width - 2 * pad, Inches(0.52), [
        {"text": value, "size": 27, "bold": True, "color": colour_value},
    ])
    _text(slide, left + pad, top + height - Inches(0.52), width - 2 * pad, Inches(0.44), [
        {"text": note, "size": 9.5, "color": colour_note, "spacing": 0.9},
    ])


def _bullets(slide, left, top, width, height, items: List[Tuple[str, str]],
             size: float = 13):
    """Bold lead-in followed by explanatory text, one paragraph per item."""
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    for i, (lead, rest) in enumerate(items):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.space_after = Pt(9)
        para.line_spacing = 1.08
        marker = para.add_run()
        marker.text = "▪  "
        marker.font.name, marker.font.size = FONT, Pt(size)
        marker.font.color.rgb = ACCENT
        marker.font.bold = True

        head = para.add_run()
        head.text = lead
        head.font.name, head.font.size = FONT, Pt(size)
        head.font.bold = True
        head.font.color.rgb = FOREST

        tail = para.add_run()
        tail.text = rest
        tail.font.name, tail.font.size = FONT, Pt(size)
        tail.font.color.rgb = INK
    return box


# --- Slides --------------------------------------------------------------

def _slide_title(prs, res: DCFResult):
    slide = _blank(prs)
    _box(slide, 0, 0, WIDTH, HEIGHT, fill=FOREST)
    _box(slide, 0, HEIGHT - Inches(0.16), WIDTH, Inches(0.16), fill=ACCENT)

    _text(slide, MARGIN, Inches(1.5), CONTENT_W, Inches(0.36), [
        {"text": "INVESTMENT COMMITTEE  ·  PRELIMINARY SCREENING", "size": 13,
         "bold": True, "color": SAND_DEEP},
    ])
    _text(slide, MARGIN, Inches(2.0), CONTENT_W, Inches(1.2), [
        {"text": "Project Aspen", "size": 58, "bold": True, "color": WHITE},
    ])
    _text(slide, MARGIN, Inches(3.15), Inches(9.6), Inches(1.3), [
        {"text": "Should we acquire a 20,000-hectare boreal timberland tract in "
                 "north-central Alberta, and at what price?",
         "size": 21, "color": SAND, "spacing": 1.06},
    ])
    _box(slide, MARGIN, Inches(4.55), Inches(1.4), Pt(2.4), fill=ACCENT)

    _text(slide, MARGIN, Inches(4.85), Inches(9.6), Inches(0.8), [
        {"text": "HYPOTHETICAL TRANSACTION: the tract does not exist. Built to "
                 "demonstrate method, not to recommend an investment.",
         "size": 12.5, "bold": True, "color": SAND_DEEP, "spacing": 1.05},
    ])
    _text(slide, MARGIN, Inches(5.95), CONTENT_W, Inches(0.8), [
        {"text": config.val("meta.analyst"), "size": 16, "bold": True, "color": WHITE},
        {"text": "Renewable Resources, timberland and farmland  |  all figures in real "
                 f"{config.val('meta.base_currency')}",
         "size": 12, "color": SAND_DEEP, "space_before": 4},
    ])

    _notes(slide, f"""
Project Aspen is a hypothetical timberland acquisition I built end to end to show how I
would underwrite a real one: a 20,000-hectare boreal mixedwood tract in north-central
Alberta, offered at {style.money(res.purchase_price)}, or
${res.price_per_ha:,.0f} a hectare.

Say clearly up front that the tract is invented. Everything about the property is an
illustrative assumption documented in a single YAML file with a source field. The market
data in the last section is real and cited.

The question the deck answers is not just "is this a good asset" but "what is the most we
should pay, and what could go wrong". Three deliverables sit behind it: a live-formula
Excel model, a Monte Carlo risk engine, and a market study testing the diversification
and inflation claims the asset class is usually sold on.
""")


def _slide_recommendation(prs, res: DCFResult, mc: MonteCarloResult):
    slide = _blank(prs)
    s = res.summary()
    m = mc.summary()
    gap_bps = (res.irr - res.hurdle_rate) * 10000

    _slide_header(
        slide,
        f"Buy, but only below {style.money(res.breakeven_price)}: at the "
        f"{style.money(res.purchase_price)} ask the tract returns {res.irr:.1%} against a "
        f"{res.hurdle_rate:.0%} hurdle",
        "Recommendation", 2,
    )

    top = Inches(1.95)
    card_h = Inches(1.62)
    gap = Inches(0.19)
    card_w = Emu(int((CONTENT_W - 3 * gap) / 4))
    cards = [
        ("Project IRR", f"{res.irr:.1%}", f"{gap_bps:+.0f} bps above the "
                                          f"{res.hurdle_rate:.0%} hurdle", True),
        ("Net present value", style.money(res.npv),
         f"discounted at {res.discount_rate:.1%} real", False),
        ("Breakeven price", style.money(res.breakeven_price),
         f"{s['breakeven_premium_vs_ask']:+.0%} vs the ask "
         f"(${res.breakeven_per_ha:,.0f}/ha)", False),
        ("Money multiple", f"{res.moic:.2f}x",
         f"undiscounted, over {int(config.num('dcf.holding_period_years'))} years", False),
    ]
    for i, (label, value, note, highlight) in enumerate(cards):
        _stat_card(slide, MARGIN + i * (card_w + gap), top, card_w, card_h,
                   label, value, note, highlight)

    body_top = Inches(3.86)
    _text(slide, MARGIN, body_top, Inches(7.5), Inches(0.3), [
        {"text": "WHY", "size": 11, "bold": True, "color": ACCENT},
    ])
    _bullets(slide, MARGIN, body_top + Inches(0.32), Inches(7.5), Inches(2.6), [
        ("We are buying standing inventory, not growth.  ",
         f"The tract carries {s['opening_inventory_m3']/1e6:.2f} million m³ today, "
         f"{s['opening_inventory_m3']/config.num('tract.area_ha'):.0f} m³ a hectare, against "
         f"{s['required_ending_inventory_m3']/1e6:.2f} million for a fully regulated forest. "
         f"That mature surplus is the return; the biology adds little over 30 years."),
        ("The price, not the forecast, decides the outcome.  ",
         f"Moving the offer ±30% swings the IRR by roughly six points; moving long-run real "
         f"price growth across a full point moves it by two. Discipline on entry is the "
         f"whole game, which is why the recommendation is a price, not a yes."),
        ("The margin is real but thin.  ",
         f"Stumpage can fall about 15% before the deal breaks, and "
         f"{m['scheduled_prob_below_hurdle']:.0%} of simulated paths miss the hurdle. "
         f"This clears, but it is not a bargain."),
    ])

    panel_l = MARGIN + Inches(7.9)
    panel_w = CONTENT_W - Inches(7.9)
    _box(slide, panel_l, body_top, panel_w, Inches(2.75), fill=SAND)
    _text(slide, panel_l + Inches(0.22), body_top + Inches(0.18),
          panel_w - Inches(0.44), Inches(0.3), [
        {"text": "CONDITIONS", "size": 11, "bold": True, "color": ACCENT},
    ])
    _text(slide, panel_l + Inches(0.22), body_top + Inches(0.55),
          panel_w - Inches(0.44), Inches(2.1), [
        {"text": f"1.  Cap the bid at {style.money(res.breakeven_price)}. Above that the "
                 f"deal destroys value at our cost of capital.", "size": 12,
         "color": INK, "spacing": 1.04, "space_after": 7},
        {"text": "2.  Verify the inventory with an independent timber cruise before "
                 "confirming price. The whole case rests on the volume that is standing "
                 "there today.", "size": 12, "color": INK, "spacing": 1.04,
         "space_after": 7},
        {"text": "3.  Negotiate harvest-timing latitude into any mill supply agreement. "
                 f"That flexibility alone is worth {style.money(mc.flexibility_value)}.",
         "size": 12, "color": INK, "spacing": 1.04},
    ])

    _notes(slide, f"""
The recommendation is conditional and the condition is the price.

At the {style.money(res.purchase_price)} ask the model returns {res.irr:.2%} real and
unlevered against a {res.hurdle_rate:.1%} hurdle, an NPV of {style.money(res.npv)}. That
clears, so the answer is buy. But the number that matters is the breakeven,
{style.money(res.breakeven_price)} or ${res.breakeven_per_ha:,.0f} a hectare. That is
{s['breakeven_premium_vs_ask']:.0%} above the ask, and it is the ceiling I would take
into a negotiation.

On the first bullet: this is fundamentally a liquidation-of-surplus story. The tract holds
{s['opening_inventory_m3']/config.num('tract.area_ha'):.0f} m³ a hectare against a
regulated level of about
{s['required_ending_inventory_m3']/config.num('tract.area_ha'):.0f}. We harvest the excess
down over thirty years and sell a normalised forest. If asked "where does the return come
from", the answer is the mature timber, not the growth.

Be honest about the weaknesses: the growth curve is illustrative, and
{m['scheduled_prob_below_hurdle']:.0%} of paths miss the hurdle. I would not present this
as a high-conviction deal. It is a priced-about-right deal where execution and entry
discipline decide the result.
""")


def _slide_rotation(prs, rot: RotationResults, res: DCFResult, figures: Dict[str, str]):
    slide = _blank(prs)
    _slide_header(
        slide,
        f"Discounting cuts the optimal rotation from {rot.max_mai_age:.0f} years to "
        f"{res.rotation_age}, and that decision sets every cash flow that follows",
        "Biological growth and optimal rotation", 3,
    )

    _picture(slide, figures.get("lev_curve"), MARGIN, Inches(1.9),
             Inches(8.1), Inches(4.7))

    panel_l = MARGIN + Inches(8.35)
    panel_w = CONTENT_W - Inches(8.35)
    _text(slide, panel_l, Inches(1.95), panel_w, Inches(0.3), [
        {"text": "THREE ANSWERS, ONE QUESTION", "size": 11, "bold": True, "color": ACCENT},
    ])

    # One decimal, because Fisher and Faustmann round to the same whole year here
    # and the whole point of the slide is that the criteria differ.
    rows = [
        ("Maximum MAI", f"{rot.max_mai_age:.1f} yrs",
         f"Peak {rot.max_mai_value:.2f} m³/ha/yr. Maximises wood, ignores money."),
        ("Single rotation (Fisher)", f"{rot.fisher_age:.1f} yrs",
         "Counts interest on the standing timber, but not on the land under it."),
        ("Faustmann", f"{rot.faustmann_age:.1f} yrs",
         f"LEV ${rot.faustmann_lev:,.0f}/ha. Also counts the delay to every future "
         f"rotation, so it is the shortest. Rounded to {res.rotation_age} years for "
         f"scheduling."),
    ]
    y = Inches(2.3)
    for i, (name, age, note) in enumerate(rows):
        highlight = i == 2
        h = Inches(1.28)
        _box(slide, panel_l, y, panel_w, h, fill=FOREST if highlight else SAND)
        _text(slide, panel_l + Inches(0.18), y + Inches(0.12), panel_w - Inches(0.36),
              Inches(0.3), [
            {"text": name, "size": 12, "bold": True,
             "color": WHITE if highlight else FOREST},
        ])
        _text(slide, panel_l + Inches(0.18), y + Inches(0.40), panel_w - Inches(0.36),
              Inches(0.34), [
            {"text": age, "size": 19, "bold": True,
             "color": WHITE if highlight else INK},
        ])
        _text(slide, panel_l + Inches(0.18), y + Inches(0.76), panel_w - Inches(0.36),
              Inches(0.48), [
            {"text": note, "size": 9.5, "color": SAND if highlight else INK_SOFT,
             "spacing": 0.92},
        ])
        y = Emu(int(y + h + Inches(0.13)))

    _text(slide, MARGIN, Inches(6.65), CONTENT_W, Inches(0.4), [
        {"text": f"Caveat: a {res.rotation_age}-year rotation is far shorter than the 70-100 "
                 f"years Alberta boreal stands are actually managed on. That gap is driven by "
                 f"the illustrative growth curve and the {res.discount_rate:.0%} real discount "
                 f"rate, and it is the first assumption I would replace with published "
                 f"growth-and-yield data.",
         "size": 10, "italic": True, "color": ACCENT, "spacing": 0.95},
    ])

    _notes(slide, f"""
This slide is the forestry content, and it is where the model earns its keep.

Three rotation ages, all defensible, all different. Maximum mean annual increment says cut
at {rot.max_mai_age:.0f} years: that maximises wood per year and ignores money entirely.
Fisher's single-rotation NPV says {rot.fisher_age:.0f} years: it discounts, so it cuts
earlier. Faustmann says {res.rotation_age} years, and it is the shortest because it counts
one thing the others miss: while you wait, the land itself is tied up, and so is every
rotation that would have followed. That opportunity cost is the land expectation value,
${rot.faustmann_lev:,.0f} a hectare here.

A good follow-up question is why the curve falls so steeply after the peak. Past the
optimum you are earning a biological growth rate below the discount rate on an
increasingly valuable asset, so you are destroying value every year you wait.

Another: a higher timber price actually *shortens* the Faustmann rotation, which surprises
people. Richer land raises the opportunity cost of waiting faster than the extra volume
repays it.

Do not oversell this. A 30-year rotation is not what anyone operates in the Alberta boreal.
That is a function of my illustrative parameters, and I say so on the slide rather than
hoping nobody notices.
""")


def _slide_valuation(prs, res: DCFResult, figures: Dict[str, str]):
    slide = _blank(prs)
    s = res.summary()
    from .dcf import breakeven_stumpage

    be_price = res.grids["price_vs_discount"].attrs.get("breakeven_stumpage")
    if be_price is None:
        be_price = breakeven_stumpage(res)
    drop = 1 - be_price / config.num("economics.net_stumpage_price")

    _slide_header(
        slide,
        f"The return absorbs a {drop:.0%} fall in stumpage, but not a "
        f"{s['breakeven_premium_vs_ask']:.0%} rise in the price we pay",
        "Valuation and sensitivity", 4,
    )

    _picture(slide, figures.get("cash_flows"), MARGIN, Inches(1.85),
             Inches(6.35), Inches(3.55))
    _picture(slide, figures.get("sensitivity_price_growth"), MARGIN + Inches(6.55),
             Inches(1.85), Inches(6.05), Inches(3.55))

    strip_top = Inches(5.5)
    items = [
        ("Sustainable cut", f"{s['even_flow_target_m3']/1e3:,.0f}k m³/yr",
         f"held within ±{res.schedule.tolerance:.0%} every year"),
        ("Exit value", style.money(res.terminal_total),
         f"{style.money(res.terminal_land)} land + "
         f"{style.money(res.terminal_inventory)} standing timber"),
        ("Breakeven stumpage", f"${be_price:,.0f}/m³",
         f"vs ${config.num('economics.net_stumpage_price'):,.0f} assumed"),
        ("Breakeven price", f"${res.breakeven_per_ha:,.0f}/ha",
         f"vs ${res.price_per_ha:,.0f}/ha asked"),
    ]
    gap = Inches(0.19)
    card_w = Emu(int((CONTENT_W - 3 * gap) / 4))
    for i, (label, value, note) in enumerate(items):
        left = MARGIN + i * (card_w + gap)
        _box(slide, left, strip_top, card_w, Inches(1.18), fill=SAND)
        _text(slide, left + Inches(0.16), strip_top + Inches(0.12),
              card_w - Inches(0.32), Inches(0.24), [
            {"text": label.upper(), "size": 9.5, "bold": True, "color": INK_SOFT},
        ])
        _text(slide, left + Inches(0.16), strip_top + Inches(0.38),
              card_w - Inches(0.32), Inches(0.36), [
            {"text": value, "size": 18, "bold": True, "color": FOREST},
        ])
        _text(slide, left + Inches(0.16), strip_top + Inches(0.76),
              card_w - Inches(0.32), Inches(0.36), [
            {"text": note, "size": 9, "color": INK_SOFT, "spacing": 0.9},
        ])

    _notes(slide, f"""
Left chart: the cash flow shape. One large outflow at close, thirty years of steady harvest
income of roughly {style.money(float(res.table['harvest_revenue'][1:].mean()))} a year, and
an exit worth {style.money(res.terminal_total)}. Operating costs are about 5% of revenue,
which is why they barely show.

The exit is not a hand-waved multiple. It is the bare land value in perpetuity,
{style.money(res.terminal_land)}, plus the standing timber we leave behind,
{style.money(res.terminal_inventory)}, priced stand by stand off the year-30 age
distribution. In the workbook that whole calculation is live formulas.

Right chart: the two-way grid. Read down for the entry price, across for the long-run real
price growth assumption. The vertical spread dwarfs the horizontal one, which is the point
- we control the price we pay and we do not control the commodity.

The single most useful number here is the breakeven stumpage of ${be_price:,.0f} a cubic
metre against ${config.num('economics.net_stumpage_price'):,.0f} assumed. Prices would have
to sit {drop:.0%} below assumption, on average, for thirty years, before this deal fails.
That is a real cushion, though not a large one by timberland standards.

If challenged on the discount rate: it barely moves the IRR, because it touches only the
rotation age and the exit price, not the operating cash flows. The grid in the workbook
shows that explicitly.
""")


def _slide_risk(prs, res: DCFResult, mc: MonteCarloResult, figures: Dict[str, str]):
    slide = _blank(prs)
    m = mc.summary()
    price_cushion = 1 - _breakeven_stumpage(res) / config.num("economics.net_stumpage_price")

    _slide_header(
        slide,
        f"{m['scheduled_prob_below_hurdle']:.0%} of {mc.n_sims:,} simulated paths miss the "
        f"hurdle; harvest timing flexibility is worth "
        f"{style.money(mc.flexibility_value)} against that",
        "Risk", 5,
    )

    _picture(slide, figures.get("irr_histogram"), MARGIN, Inches(1.82),
             Inches(6.5), Inches(3.5))

    _text(slide, MARGIN, Inches(5.42), Inches(6.5), Inches(0.28), [
        {"text": "WHERE THE RISK SITS", "size": 11, "bold": True, "color": ACCENT},
    ])
    _text(slide, MARGIN, Inches(5.72), Inches(6.5), Inches(1.2), [
        {"text": f"IRR spans {m['scheduled_p5']:.1%} at the 5th percentile to "
                 f"{m['scheduled_p95']:.1%} at the 95th, median {m['scheduled_p50']:.1%}. "
                 f"Fire and pests cost {style.money(mc.disturbance_cost)} of expected value "
                 f"- real, but second order next to price.",
         "size": 11.5, "color": INK, "spacing": 1.06},
        {"text": f"Flexibility has value only because prices mean revert: a weak price today "
                 f"signals a stronger one later, so shifting volume inside the ±"
                 f"{res.schedule.tolerance:.0%} band pays. Under a random walk it would be "
                 f"worth nothing.",
         "size": 11.5, "color": INK_SOFT, "spacing": 1.06, "space_before": 6},
    ])

    panel_l = MARGIN + Inches(6.8)
    panel_w = CONTENT_W - Inches(6.8)
    _text(slide, panel_l, Inches(1.86), panel_w, Inches(0.28), [
        {"text": "KEY RISKS AND MITIGANTS", "size": 11, "bold": True, "color": ACCENT},
    ])

    risks = [
        ("Wildfire",
         f"Costs {style.money(mc.disturbance_cost)} of expected NPV. Spread harvest across "
         f"the tract, insure where priced sensibly, budget salvage capacity."),
        ("Mountain pine beetle",
         "Modelled inside the same disturbance term. Mixedwood composition limits pure-pine "
         "exposure; monitor and accelerate harvest in infested stands."),
        ("Lumber and stumpage prices",
         f"The dominant driver. Prices can sit {price_cushion:.0%} below assumption for the "
         f"whole hold before the deal breaks; keep the harvest band wide and stay unlevered."),
        ("Regulatory and Indigenous consultation",
         "Not quantified here, and that is a genuine gap. Requires early engagement, "
         "consent-based planning and schedule contingency before any bid."),
        ("Liquidity",
         "Thirty-year hold with no interim market. Only appropriate for a long-dated "
         "liability pool; size the position accordingly."),
    ]
    y = Inches(2.2)
    for name, mitigant in risks:
        h = Inches(0.86)
        _box(slide, panel_l, y, Pt(3), h, fill=ACCENT)
        _text(slide, panel_l + Inches(0.14), y, panel_w - Inches(0.14), Inches(0.24), [
            {"text": name, "size": 11.5, "bold": True, "color": FOREST},
        ])
        _text(slide, panel_l + Inches(0.14), y + Inches(0.24), panel_w - Inches(0.14),
              Inches(0.62), [
            {"text": mitigant, "size": 9.5, "color": INK, "spacing": 0.93},
        ])
        y = Emu(int(y + h + Inches(0.06)))

    _notes(slide, f"""
Ten thousand paths. Stumpage follows a mean-reverting Ornstein-Uhlenbeck process on the log
price: cyclical, not a random walk, because a sawlog price that doubles does not stay
doubled. Each year carries a {config.num('monte_carlo.disturbance_probability'):.0%} chance
of a fire or pest event destroying a random share of the tract.

The honest headline is {m['scheduled_prob_below_hurdle']:.0%} of paths below the hurdle.
Median {m['scheduled_p50']:.1%}, 5th percentile {m['scheduled_p5']:.1%}. This is not a
safe deal; it is an adequately priced one.

The flexibility number is the part I would most want to talk about. Two strategies run on
identical price and fire paths. One cuts the target every year regardless of price. The
other cuts at the bottom of the permitted band when prices are below their long-run mean
and at the top when above. The difference, {style.money(mc.flexibility_value)} or
${m['flexibility_value_per_ha']:,.0f} a hectare, is a real option on harvest timing, and it
exists only because prices revert. I verified that with a control run: set volatility near
zero and the option value collapses.

One caveat worth volunteering: the flexible strategy raises mean value but also widens the
distribution, so its probability of missing the hurdle is slightly higher, at
{m['flexible_prob_below_hurdle']:.0%}. More value, more variance.

On regulatory and Indigenous consultation, I have not quantified it and I would not
pretend otherwise. On a real Alberta transaction that is front-and-centre due diligence,
not a footnote, and it would need legal and community work before a bid.
""")


def _slide_portfolio(prs, mk: MarketResult, figures: Dict[str, str]):
    slide = _blank(prs)
    m = mk.summary()
    bench = "S&P/TSX Composite"

    _slide_header(
        slide,
        f"The diversification case is weaker than the pitch: listed proxies correlate "
        f"{m['mean_timber_corr_to_benchmark']:.2f} with the {bench}",
        "Portfolio fit", 6,
    )

    _picture(slide, figures.get("rolling_correlation"), MARGIN, Inches(1.82),
             Inches(7.4), Inches(3.9))

    panel_l = MARGIN + Inches(7.7)
    panel_w = CONTENT_W - Inches(7.7)
    _text(slide, panel_l, Inches(1.86), panel_w, Inches(0.28), [
        {"text": "WHAT THE DATA SAYS", "size": 11, "bold": True, "color": ACCENT},
    ])

    n_reg = len(mk.regression)
    findings = [
        ("Correlation is moderate, not low.  ",
         f"Timber proxies average {m['mean_timber_corr_to_benchmark']:.2f} against the "
         f"{bench} over {m['n_months']} months, and the 36-month rolling figure has ranged "
         f"from {m['rolling_corr_min']:.2f} to {m['rolling_corr_max']:.2f}."),
        ("Inflation hedging is unproven here.  ",
         f"Regressing annual returns on annual CPI gives no statistically significant "
         f"positive beta for any of the {n_reg} series tested. With 12 to 41 annual "
         f"observations the standard errors swamp the slopes."),
        ("Farmland looks better than timber.  ",
         "Both farmland proxies earned materially more in above-median inflation years, "
         "but on barely a decade of data that is suggestive, not evidence."),
    ]
    _bullets(slide, panel_l, Inches(2.2), panel_w, Inches(3.2), findings, size=11)

    caveat_top = Inches(5.95)
    _box(slide, MARGIN, caveat_top, CONTENT_W, Inches(0.92), fill=SAND)
    _text(slide, MARGIN + Inches(0.2), caveat_top + Inches(0.12),
          CONTENT_W - Inches(0.4), Inches(0.24), [
        {"text": "THE CAVEAT THAT QUALIFIES ALL OF THIS", "size": 10, "bold": True,
         "color": ACCENT},
    ])
    _text(slide, MARGIN + Inches(0.2), caveat_top + Inches(0.38),
          CONTENT_W - Inches(0.4), Inches(0.5), [
        {"text": "Listed timber and farmland REITs are equities first. They carry market "
                 "beta, leverage and daily sentiment that a directly held tract does not, so "
                 "these correlations are an upper bound and understate the diversification "
                 "private timberland would deliver. Appraisal-based private indices have the "
                 "opposite bias: smoothing suppresses measured correlation. The truth sits "
                 "between them, and neither is available for one Alberta tract.",
         "size": 10, "color": INK, "spacing": 0.95},
    ])

    _notes(slide, f"""
This is the slide where I report a result that does not help the pitch.

The standard case for timberland is low correlation and inflation protection. Tested
against real data, using Yahoo Finance monthly returns and Statistics Canada CPI, neither
claim comes through cleanly. Timber proxies correlate
{m['mean_timber_corr_to_benchmark']:.2f} with the TSX, which is moderate, not low. And no
proxy shows a statistically significant inflation beta.

I want to be precise about why that is not the same as "the thesis is wrong". The proxies
are listed REITs, and they are equities before they are trees. They trade with the market
because they are traded. A directly held tract has no daily price and no equity beta, so
its true correlation is almost certainly lower than what I measured. The opposite bias
exists too: private appraisal-based indices update slowly and understate correlation.
Honest answer is that the truth is between them and I cannot observe it.

What I would do with more time and data: get NCREIF Timberland Index history and run the
same analysis on a de-smoothed series, which is the standard correction.

The farmland result is interesting: both proxies did much better in high-inflation years,
but it rests on twelve and thirteen annual observations, so I would present it as a
hypothesis worth testing, not a finding.

CPI source is {mk.cpi_source}.
""")


# --- Entry point ---------------------------------------------------------

def build(
    rot: RotationResults,
    res: DCFResult,
    mc: MonteCarloResult,
    mk: MarketResult,
    figures: Dict[str, str],
    path: Optional[Path] = None,
) -> Path:
    """Assemble the six-slide deck and save it."""
    path = Path(path) if path else config.OUTPUTS / "Project_Aspen_IC_Deck.pptx"

    prs = Presentation()
    prs.slide_width = WIDTH
    prs.slide_height = HEIGHT

    _slide_title(prs, res)
    _slide_recommendation(prs, res, mc)
    _slide_rotation(prs, rot, res, figures)
    _slide_valuation(prs, res, figures)
    _slide_risk(prs, res, mc, figures)
    _slide_portfolio(prs, mk, figures)

    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)
    return path
