"""Deterministic scientific checks of claims, no model involved. Three rules, each conservative (it fires only when a
claim value matches no plausible reading of its source), because a false alarm makes arm C delete a correct claim:
- number fidelity: every number with a unit in a claim must appear in its cited source(s), after unit conversion within
one dimension (GPa ↔ MPa ↔ N/mm², µN ↔ mN, °C ↔ K, min ↔ h …) and German number formats ("1,5", "1.000", "Gew.-%").
Ambiguous numbers are read every plausible way ("1.000" = 1.0 or 1000); rounding to the claim's own precision passes.
- unit consistency: a property named next to a value implies a dimension (fracture toughness → MPa·m^0.5, hardness →
pressure or HV, density → g/cm³ or %, temperature → °C/K, grain size → length …); a mismatch is flagged.
- phase rule: a Greek-prefixed phase (α-/alpha-Al2O3) contradicts the source if the source names the same compound only
with a different prefix (γ-Al2O3); a compound the source never prefixes is not flagged.
A library chooses its rules in its domain profile (`claim_rules`); RQ3 ran all three.

Figure claims: values read from a plot are usually not in the caption. `figure_mode`: "skip" (default) leaves their
numbers to the vision check; "conflict" flags a claim number only if the caption has values of the same dimension and
none matches (wrongly flagged about 7 % of figure claims on real answers); "require" treats captions like text."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from memento.graph.registry_base import normalize

# ── units: synonym → (canonical unit, dimension, factor to the dimension's base) ──────────────────────────────────
_U = {}


def _unit(canon: str, dim: str, factor: float, *syns: str):
    for s in (canon,) + syns:
        _U[s] = (canon, dim, factor)


_unit("GPa", "pressure", 1e9)
_unit("MPa", "pressure", 1e6, "N/mm²", "N/mm2", "N/mm^2", "N mm-2", "N mm⁻²")
_unit("kPa", "pressure", 1e3)
_unit("Pa", "pressure", 1.0)
_unit("HV", "hardness_hv", 1.0, "HV0.1", "HV1", "HV5", "HV10", "HV30")
_unit("MPa·m^0.5", "toughness", 1.0, "MPa√m", "MPa·√m", "MPa m^0.5", "MPa·m^1/2", "MPa m^1/2", "MPa m1/2",
      "MPa·m1/2", "MPa m½", "MPa·m½", "MPa·m0.5", "MPa m0.5")
_unit("°C", "temperature", 1.0, "ºC", "° C", "℃", "degC")
_unit("K", "temperature", 1.0)                        # converted by offset, see _to_base
_unit("wt%", "percent", 1.0, "Gew.-%", "Gew%", "Gew.%", "wt.%", "wt. %", "mass%", "Ma.-%", "Masse-%")
_unit("vol%", "percent", 1.0, "Vol.-%", "Vol%", "vol.%", "vol. %")
_unit("mol%", "percent", 1.0, "mol.-%", "mol-%", "mol.%", "Mol-%")
_unit("at%", "percent", 1.0, "at.%", "at.-%", "At.-%")
_unit("%", "percent", 1.0)
_unit("nm", "length", 1e-9)
_unit("µm", "length", 1e-6, "μm", "um")
_unit("mm", "length", 1e-3)
_unit("cm", "length", 1e-2)
_unit("g/cm³", "density", 1.0, "g/cm3", "g cm-3", "g cm⁻³", "g/cm^3", "g·cm⁻³")
_unit("kg/m³", "density", 1e-3, "kg/m3", "kg m-3", "kg m⁻³", "kg/m^3")
_unit("µN", "force", 1e-6, "μN", "uN")
_unit("mN", "force", 1e-3)
_unit("kN", "force", 1e3)
_unit("N", "force", 1.0)
_unit("ms", "time", 1e-3)
_unit("s", "time", 1.0, "sec")
_unit("min", "time", 60.0)
_unit("h", "time", 3600.0, "hr", "Std.")
_unit("J/g", "specific_energy", 1.0, "J g-1", "J g⁻¹")
_unit("kJ/m²", "area_energy", 1.0, "kJ/m2", "kJ m-2", "kJ m⁻²")
_unit("W/(m·K)", "conductivity", 1.0, "W/mK", "W/(mK)", "W m-1 K-1", "W m⁻¹ K⁻¹", "W/m·K", "W·m⁻¹·K⁻¹")
_unit("cm⁻¹", "wavenumber", 1.0, "cm-1", "cm^-1", "1/cm")
_unit("eV", "energy_ev", 1.0)
_unit("Hz", "frequency", 1.0)
_unit("kHz", "frequency", 1e3)
_unit("MHz", "frequency", 1e6)
_unit("rpm", "rotation", 1.0, "U/min")
_unit("°", "angle", 1.0)

# longest synonyms first so "MPa·m^0.5" wins over "MPa", "wt%" over "%", "°C" over "°"
_UNIT_ALT = "|".join(re.escape(s) for s in sorted(_U, key=len, reverse=True))
_NUM = r"[-−–]?\d+(?:[.,]\d+)*"
_TIMES10 = r"(?:\s*[×x·*]\s*10\s*(?:\^|\*\*)?\s*(?P<exp>[-−⁻]?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
_RANGE_SEP = r"\s*(?:–|—|-|bis|to|und|and|…|\.\.\.?)\s*"
_QTY = re.compile(
    rf"(?<![\w.,])(?P<a>{_NUM})(?:\s*(?:±|\+/-|\+-)\s*{_NUM})?(?:{_RANGE_SEP}(?P<b>{_NUM}))?"
    rf"{_TIMES10}\s*(?P<unit>{_UNIT_ALT})(?![\wµμ/^⁻¹²³·])")      # "s^-1", "s⁻¹", "mm²": not the bare unit
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


_LATEX = [
    (r"\^\{?\\circ\}?", "°"), (r"\\circ", "°"), (r"\\mu\b", "µ"), (r"\\micro\b", "µ"), (r"\\%", "%"),
    (r"\\times", "×"), (r"\\cdot", "·"), (r"\\pm", "±"), (r"\\sim", "~"), (r"\\approx", "≈"), (r"\\sqrt\{([^{}]*)\}", r"√\1"),
    (r"\\(?:mathrm|text|mathit|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1"), (r"\^\{([^{}]*)\}", r"^\1"),
    (r"_\{([^{}]*)\}", r"\1"), (r"\\[,;:! ]", " "), (r"\\~", " "), (r"[{}$]", ""),
]
_LATEX = [(re.compile(p), r) for p, r in _LATEX]


def plain(text: str) -> str:
    """MinerU writes values as LaTeX ("4.06 $\\mu$ m", "$141.7^{\\circ}\\mathrm{C}$") — turn it into plain text."""
    t = text or ""
    for _ in range(2):                                 # nested \mathrm{\mu} etc.
        for rx, rp in _LATEX:
            t = rx.sub(rp, t)
    t = re.sub(r"µ\s+(?=[mgNlLs])", "µ", t)             # "µ m" → "µm"
    t = re.sub(r"(\d)\s+°\s*C", r"\1 °C", t)
    return re.sub(r"[ \t]{2,}", " ", t)


@dataclass(frozen=True)
class Quantity:
    values: tuple[float, ...]   # every plausible reading, in the canonical unit ("1.000" → 1.0 and 1000.0)
    unit: str                   # canonical unit
    dim: str
    raw: str
    decimals: int               # digits after the decimal separator in the written number (precision)


def _readings(num: str) -> tuple[tuple[float, ...], int]:
    """All plausible numeric readings of a written number and its decimal precision.
    "1,5" → 1.5 · "1.5" → 1.5 · "1.000" → 1.0 or 1000 · "1,000" → 1.0 or 1000 · "12.345,6" → 12345.6."""
    s = num.replace("−", "-").replace("–", "-")
    neg = s.startswith("-")
    s = s.lstrip("-")
    out = set()
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        th = "." if dec == "," else ","
        body = s.replace(th, "").replace(dec, ".")
        out.add(float(body)); decimals = len(s.split(dec)[-1])
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) == 2:
            out.add(float(parts[0] + "." + parts[1])); decimals = len(parts[1])
            if len(parts[1]) == 3:                     # could be a thousands separator
                out.add(float(parts[0] + parts[1]))
        else:                                          # 1.000.000 / 1,000,000
            out.add(float("".join(parts))); decimals = 0
    else:
        out.add(float(s)); decimals = 0
    return tuple(sorted(-v if neg else v for v in out)), decimals


def extract_quantities(text: str) -> list[Quantity]:
    """Every number (or range endpoint) followed by a known unit. Plus-minus keeps the central value; "×10^n" scales."""
    out = []
    for m in _QTY.finditer(plain(text)):
        canon, dim, _ = _U[m.group("unit")]
        exp = m.group("exp")
        scale = 10 ** int(exp.translate(_SUP)) if exp else 1
        for key in ("a", "b"):
            if m.group(key):
                vals, dec = _readings(m.group(key))
                out.append(Quantity(tuple(v * scale for v in vals), canon, dim, m.group(0), dec))
    return out


def _to_base(v: float, unit: str) -> float:
    canon, dim, factor = _U[unit]
    if dim == "temperature":
        return v if canon == "°C" else v - 273.15
    return v * factor


_PCT_SPECIFIC = {"wt%", "vol%", "mol%", "at%"}


def _units_comparable(u1: str, u2: str) -> bool:
    d1, d2 = _U[u1][1], _U[u2][1]
    if d1 != d2:
        return False
    if d1 == "percent":                                # generic % matches any; wt% vs vol% do not
        return u1 == "%" or u2 == "%" or u1 == u2
    return True


def _matches(c: Quantity, s: Quantity, rel_tol: float) -> bool:
    """Same value after conversion, within 1 % of the source value or half a unit of the claim's last written digit
    (the claim may round: "99 %" matches 98.7 %, "1.5 GPa" matches 1.52 GPa)."""
    if not _units_comparable(c.unit, s.unit):
        return False
    half_digit = 0.5 * 10 ** (-c.decimals)
    step = half_digit if c.dim == "temperature" else half_digit * _U[c.unit][2]   # as a difference in base units
    for cv in c.values:
        for sv in s.values:
            a, b = _to_base(cv, c.unit), _to_base(sv, s.unit)
            if abs(a - b) <= max(rel_tol * abs(b), step) + 1e-12:
                return True
    return False


_BARE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)(?![\w])")


def _bare_numbers(texts: list[str]) -> list[tuple[float, ...]]:
    """Numbers in the sources written WITHOUT a unit next to them (table cells carry the unit in the column header).
    A number that has a unit is only matched through its unit (10 vol% must not pass as "10 wt%")."""
    out = []
    for t in texts:
        bare_text = _QTY.sub(" ", plain(t))
        out += [_readings(m.group(1))[0] for m in _BARE.finditer(bare_text)]
    return out


def _matches_bare(c: Quantity, bare: list[tuple[float, ...]], rel_tol: float) -> bool:
    half_digit = 0.5 * 10 ** (-c.decimals)
    return any(abs(cv - bv) <= max(rel_tol * abs(bv), half_digit) + 1e-12 for vals in bare for bv in vals for cv in c.values)


def _found(c: Quantity, src_q: list[Quantity], bare: list[tuple[float, ...]], rel_tol: float = 0.01) -> bool:
    return any(_matches(c, s, rel_tol) for s in src_q) or _matches_bare(c, bare, rel_tol)


@dataclass
class CheckIssue:
    claim_id: str
    kind: str       # "number_not_in_source" | "number_conflicts_caption" | "unit_mismatch" | "phase_contradicts_source"
    detail: str


def number_fidelity(claim_id: str, claim: str, source_texts: list[str], *, rel_tol: float = 0.01,
                    mode: str = "require") -> list[CheckIssue]:
    """mode "require": every claim quantity needs a matching source quantity. mode "conflict" (figure captions): only
    quantities whose dimension occurs in the source and match none of those values are flagged."""
    src = [q for t in source_texts for q in extract_quantities(t)]
    bare = _bare_numbers(source_texts)
    issues = []
    for q in extract_quantities(claim):
        if _found(q, src, bare, rel_tol):
            continue
        if mode == "conflict":
            if any(_units_comparable(q.unit, s.unit) for s in src):
                issues.append(CheckIssue(claim_id, "number_conflicts_caption", f"{q.raw.strip()} not among the caption's values"))
        else:
            issues.append(CheckIssue(claim_id, "number_not_in_source", f"{q.raw.strip()} not found in the cited source"))
    return issues


# ── unit consistency ──────────────────────────────────────────────────────────
# property words (EN + DE, lower-case stems) → dimensions a value of that property may have
PROPERTY_DIMENSION = [
    (r"fracture toughness|bruchzähigkeit|risszähigkeit|k_?ic\b|kic\b", {"toughness"}),
    (r"hardness|härte", {"pressure", "hardness_hv"}),
    (r"(?:relative )?density|dichte", {"density", "percent"}),
    (r"grain size|korngröße|korngroesse|particle size|partikelgröße|pore size|porengröße|thickness|dicke", {"length"}),
    (r"sinter(?:ing)?[- ]temperatur|(?<!room )(?<!raum)temperature|(?<!raum)temperatur|\btg\b|\btm\b|glass transition|"
     r"melting point|schmelzpunkt|glasübergang", {"temperature"}),
    (r"(?:tensile|flexural|compressive|bending|yield) strength|festigkeit|young'?s modulus|elastic modulus|"
     r"e-modul|elastizitätsmodul|modulus|modul\b|stress|spannung", {"pressure"}),
    (r"holding time|dwell time|haltezeit|sinter(?:ing)? time|time|zeit|dauer", {"time"}),
    (r"thermal conductivity|wärmeleitfähigkeit", {"conductivity"}),
]
_PROP = [(re.compile(p, re.I), dims) for p, dims in PROPERTY_DIMENSION]


# property word + a short connector + the value: "fracture toughness of 4.5 GPa", "Die Härte betrug 20 GPa",
# "hardness: 20 GPa", "density was about 3.9 g/cm³". Anything looser (a condition in between) is not judged.
_CONNECT = (r"(?:\s*\([^)]{0,20}\))?\s*(?:[:=]|of|is|was|were|are|reached|reaches|von|betrug|beträgt|betragen|"
            r"lag bei|liegt bei|erreichte|erreicht|mit)?\s*(?:about|approximately|approx\.|around|ca\.|etwa|rund|"
            r"~|≈|nearly|only|up to|bis zu)?\s*$")
_CONNECT_RX = re.compile(_CONNECT, re.I)


def unit_consistency(claim_id: str, claim: str) -> list[CheckIssue]:
    """Flags a value whose unit cannot belong to the property named DIRECTLY before it (see _CONNECT)."""
    text, issues = plain(claim), []
    for m in _QTY.finditer(text):
        before = text[max(0, m.start() - 60):m.start()]
        for rx, dims in _PROP:
            hit = None
            for pm in rx.finditer(before):
                hit = pm
            if hit and _CONNECT_RX.match(before[hit.end():]) and _U[m.group("unit")][1] not in dims:
                issues.append(CheckIssue(claim_id, "unit_mismatch",
                                         f"{m.group(0).strip()} is not a unit of {hit.group(0).lower()}"))
                break
    return issues


# ── phase rule ────────────────────────────────────────────────────────────────
_PHASE = re.compile(r"(?<![A-Za-z])(α|β|γ|δ|θ|κ|η|χ|alpha|beta|gamma|delta|theta|kappa|eta|chi)\s?[-‐]\s?"
                    r"([A-Za-z][A-Za-z0-9₀-₉]*)", re.I)


def _phases(text: str) -> set[tuple[str, str]]:
    return {(normalize(m.group(1)), normalize(m.group(2))) for m in _PHASE.finditer(text or "")}


def phase_rule(claim_id: str, claim: str, source_texts: list[str]) -> list[CheckIssue]:
    src = set().union(*(_phases(t) for t in source_texts)) if source_texts else set()
    issues = []
    for letter, compound in _phases(claim):
        if (letter, compound) in src:
            continue
        others = sorted(l for l, c in src if c == compound and l != letter)
        if others:
            issues.append(CheckIssue(claim_id, "phase_contradicts_source",
                                     f"{letter}-{compound} but the source has {', '.join(o + '-' + compound for o in others)}"))
    return issues


# a library's rules come from its domain profile (`claim_rules`); all three = what RQ3 evaluated
CLAIM_RULES = ("numbers", "units", "phases")


def claim_rules(names) -> tuple[str, ...]:
    """A domain profile's `claim_rules`, checked: an unknown rule name is an error, not a silently skipped check."""
    names = tuple(names)
    unknown = [n for n in names if n not in CLAIM_RULES]
    if unknown:
        raise ValueError(f"unknown claim rule(s) {', '.join(unknown)}; known: {', '.join(CLAIM_RULES)}")
    return names


def _number_issues(claim_id: str, claim: str, texts: list[str], captions: list[str], figure_mode: str) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    if texts and not captions:
        issues += number_fidelity(claim_id, claim, texts)
    elif captions and figure_mode != "skip":
        if figure_mode == "require":
            issues += number_fidelity(claim_id, claim, texts + captions)
        else:
            # a value is fine if the text sources have it; otherwise only a conflict with a caption counts
            tq, tb = [s for t in texts for s in extract_quantities(t)], _bare_numbers(texts)
            missing = [q for q in extract_quantities(claim) if not _found(q, tq, tb)]
            if missing:
                cap_q, cap_b = [s for t in captions for s in extract_quantities(t)], _bare_numbers(captions)
                for q in missing:
                    if _found(q, cap_q, cap_b):
                        continue
                    if any(_units_comparable(q.unit, s.unit) for s in cap_q):
                        issues.append(CheckIssue(claim_id, "number_conflicts_caption",
                                                 f"{q.raw.strip()} not among the caption's values"))
    return issues


def run_checks(claim_id: str, claim: str, cited_hits: list[dict], *, figure_mode: str = "skip",
               rules: tuple[str, ...] = CLAIM_RULES) -> list[CheckIssue]:
    """The chosen `rules` for one claim against the hits it cites. Text/table hits are required sources (with the
    authors' text under a table or an equation, as the answer model read them); figure hits count per `figure_mode`
    ("skip" | "conflict" | "require"). Only claims citing text/tables alone are held to number fidelity in "skip": a
    claim that also cites a figure may read its values from the plot (the vision check judges those). Unit consistency
    and the phase rule, when chosen, apply to every claim."""
    from memento.retrieval.answer_pipeline import source_text
    texts = [source_text(h) for h in cited_hits if h.get("type") not in ("image", "chart")]
    captions = [h.get("text", "") for h in cited_hits if h.get("type") in ("image", "chart")]
    issues: list[CheckIssue] = []
    if "numbers" in rules:
        issues += _number_issues(claim_id, claim, texts, captions, figure_mode)
    if "units" in rules:
        issues += unit_consistency(claim_id, claim)
    if "phases" in rules:
        issues += phase_rule(claim_id, claim, texts + captions)
    return issues
