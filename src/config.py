"""Load config/assumptions.yaml and expose it as values plus documented records.

Every number the model uses comes from here. A "record" is any mapping that
contains a ``value`` key; it also carries ``unit``, ``source`` and ``note`` so
that the Excel assumptions sheet and the README table can render provenance
without a second copy of the data. Monetary figures are real CAD.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
RESULTS_PATH = DATA_PROCESSED / "results.json"

ILLUSTRATIVE = "Illustrative assumption"


class Record(NamedTuple):
    """One assumption with its provenance."""

    path: str
    value: Any
    unit: str
    source: str
    note: str

    @property
    def is_illustrative(self) -> bool:
        return self.source.strip().lower() == ILLUSTRATIVE.lower()

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'dcf.real_discount_rate' -> 'Real discount rate'."""
        leaf = self.path.split(".")[-1]
        return leaf.replace("_", " ").capitalize()


def ensure_directories() -> None:
    """Create every directory the pipeline writes into."""
    for path in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def load(path: Path | None = None) -> Dict[str, Any]:
    """Return the parsed YAML tree. Cached, so repeated calls are free."""
    target = Path(path) if path is not None else CONFIG_PATH
    with open(target, "r", encoding="utf-8") as fh:
        tree = yaml.safe_load(fh)
    if not isinstance(tree, dict):
        raise ValueError(f"{target} did not parse into a mapping")
    return tree


def _is_record(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


def rec(dotted: str) -> Record:
    """Return the full record at a dotted path, e.g. ``rec('dcf.purchase_price')``."""
    node: Any = load()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"assumptions.yaml has no entry at '{dotted}'")
        node = node[part]
    if not _is_record(node):
        raise KeyError(f"'{dotted}' is a section, not an assumption record")
    return Record(
        path=dotted,
        value=node["value"],
        unit=str(node.get("unit", "")),
        source=str(node.get("source", "")),
        note=str(node.get("note", "")),
    )


def val(dotted: str) -> Any:
    """Return just the value at a dotted path."""
    return rec(dotted).value


def num(dotted: str) -> float:
    """Return a value coerced to float, failing loudly if it is not numeric."""
    raw = val(dotted)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"'{dotted}' is {raw!r}, which is not a number")
    return float(raw)


def _walk(node: Any, prefix: str = "") -> Iterator[Record]:
    if not isinstance(node, dict):
        return
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _is_record(child):
            yield Record(
                path=path,
                value=child["value"],
                unit=str(child.get("unit", "")),
                source=str(child.get("source", "")),
                note=str(child.get("note", "")),
            )
        else:
            yield from _walk(child, path)


def all_records() -> List[Record]:
    """Every assumption in file order, for the Excel sheet and README table."""
    return list(_walk(load()))


def records_in(section: str) -> List[Record]:
    """Every assumption under one top-level section."""
    return [r for r in all_records() if r.path.startswith(f"{section}.")]


def illustrative_records() -> List[Record]:
    """Assumptions that still need a real source before this is shown to anyone."""
    return [r for r in all_records() if r.is_illustrative]


def age_class_arrays() -> Tuple[List[str], List[float], List[float]]:
    """Return (labels, midpoint ages, hectares) for the tract age-class table."""
    dist = val("tract.age_class_distribution_ha")
    labels: List[str] = []
    midpoints: List[float] = []
    hectares: List[float] = []
    for label, area in dist.items():
        low, high = (int(x) for x in str(label).split("-"))
        labels.append(str(label))
        midpoints.append((low + high) / 2.0)
        hectares.append(float(area))
    return labels, midpoints, hectares


def validate() -> None:
    """Fail fast on the mistakes that would silently corrupt every downstream number."""
    problems: List[str] = []

    _, _, hectares = age_class_arrays()
    total = sum(hectares)
    area = num("tract.area_ha")
    if abs(total - area) > 1e-6:
        problems.append(
            f"age classes sum to {total:,.0f} ha but tract.area_ha is {area:,.0f} ha"
        )

    for path in (
        "dcf.real_discount_rate",
        "dcf.hurdle_rate",
        "economics.real_price_growth",
        "dcf.even_flow_tolerance",
        "monte_carlo.disturbance_probability",
    ):
        value = num(path)
        if not 0.0 <= value < 1.0:
            problems.append(f"'{path}' is {value}, which is not a decimal fraction below 1")

    if num("economics.real_price_growth") >= num("dcf.real_discount_rate"):
        problems.append("real price growth must stay below the discount rate or values diverge")

    for path in (
        "growth.chapman_richards_a",
        "growth.chapman_richards_b",
        "growth.chapman_richards_c",
        "economics.net_stumpage_price",
        "dcf.purchase_price",
        "monte_carlo.n_simulations",
    ):
        if num(path) <= 0:
            problems.append(f"'{path}' must be positive")

    for record in all_records():
        if not record.source.strip():
            problems.append(f"'{record.path}' has no source field")
        if not record.note.strip():
            problems.append(f"'{record.path}' has no note field")

    if problems:
        raise ValueError("assumptions.yaml failed validation:\n  - " + "\n  - ".join(problems))
