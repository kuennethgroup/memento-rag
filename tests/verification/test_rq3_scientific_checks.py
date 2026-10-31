"""RQ3 task 4: deterministic scientific checks (memento/agentic_validation/scientific_checks.py)."""
import pytest

from memento.agentic_validation.scientific_checks import (extract_quantities, number_fidelity, phase_rule, run_checks,
                                                  unit_consistency)

V = lambda text: {(q.values, q.unit) for q in extract_quantities(text)}


def test_extract_ranges_plus_minus_german_and_scientific():
    assert V("sintered at 1400–1600 °C, grain size 2.3 ± 0.4 µm") == {((1400.0,), "°C"), ((1600.0,), "°C"), ((2.3,), "µm")}
    assert V("Härte 1,5 GPa bei 10 Gew.-% SiO2; Zugfestigkeit 85 N/mm²") == {((1.5,), "GPa"), ((10.0,), "wt%"), ((85.0,), "MPa")}
    assert V("bei 1.000 °C") == {((1.0, 1000.0), "°C")}                  # ambiguous → both readings
    assert V("1.5 × 10⁻³ Pa") == {((0.0015,), "Pa")}
    assert V("zwischen 140 °C bis etwa 160 °C") >= {((140.0,), "°C"), ((160.0,), "°C")}
    assert V("Figure 3 and Table 2 in 2019") == set()                    # numbers without a unit are ignored


def test_number_fidelity_german_formats_and_synonyms():
    src = ["Die Härte betrug 1,5 GPa bei 10 Gew.-% SiO2; Zugfestigkeit 85 N/mm²."]
    assert number_fidelity("c1", "Hardness was 1.5 GPa with 10 wt% SiO2", src) == []
    assert number_fidelity("c2", "tensile strength of 85 MPa", src) == []
    assert [i.kind for i in number_fidelity("c3", "Hardness was 2.5 GPa", src)] == ["number_not_in_source"]


@pytest.mark.parametrize("claim,source,ok", [
    ("peak load of 10 mN", "maximum load 10000 µN", True),              # unit conversion
    ("sintered at 1873 K", "sintered at 1600 °C", True),                  # K ↔ °C
    ("held for 2 h", "held for 120 min", True),
    ("relative density of 99 %", "relative density of 98.7 %", True),   # claim rounds to its own precision
    ("hardness of 1.5 GPa", "hardness of 1.52 GPa", True),
    ("hardness of 1.50 GPa", "hardness of 1.58 GPa", False),             # precision stated → must match
    ("10 wt% SiO2", "10 vol% SiO2", False),                              # wt% is not vol%
    ("10 % SiO2", "10 Gew.-% SiO2", True),                              # generic % matches a specific one
    ("sintered at 1000 °C", "Sinterung bei 1.000 °C", True),            # German thousands separator
    ("a density of 3.95 g/cm³", "Dichte 3950 kg/m3", True),
    ("sintered at 1500 °C", "sintered between 1400 and 1600 °C", False), # a value inside a range is not stated
])
def test_number_fidelity_cases(claim, source, ok):
    assert (number_fidelity("c", claim, [source]) == []) is ok


def test_unit_consistency():
    assert [i.kind for i in unit_consistency("c1", "fracture toughness of 4.5 GPa")] == ["unit_mismatch"]
    assert unit_consistency("c2", "fracture toughness of 4.5 MPa·m^0.5") == []
    assert unit_consistency("c3", "a relative density of 99 %") == []
    assert unit_consistency("c4", "sintered at 1600 °C for 2 h, reaching a hardness of 20 GPa") == []
    assert [i.kind for i in unit_consistency("c5", "Die Bruchzähigkeit betrug 4,5 GPa")] == ["unit_mismatch"]
    assert [i.kind for i in unit_consistency("c6", "a grain size of 3 GPa")] == ["unit_mismatch"]
    assert unit_consistency("c7", "The hardness reached 18 GPa at 1350 °C after 2 h") == []   # only the first value


def test_phase_rule():
    assert phase_rule("c1", "transforms to α-Al2O3 at 1200 °C", ["γ-Al2O3 was observed"])[0].kind == "phase_contradicts_source"
    assert phase_rule("c2", "transforms to alpha-Al2O3", ["α-Al2O3 forms"]) == []
    assert phase_rule("c3", "α-Al2O3 forms", ["corundum forms"]) == []            # not prefixed in the source → no flag
    assert phase_rule("c4", "β-TCP and α-TCP coexist", ["α-TCP and β-TCP were found"]) == []


def test_run_checks_figure_modes():
    fig = {"type": "chart", "text": "Fig. 3 Hardness of samples sintered at 1150–1350 °C"}
    txt = {"type": "text", "text": "The samples were sintered at 1150 °C and 1350 °C."}
    plot_value = "The hardness reached 18 GPa at 1350 °C"                 # 18 GPa read from the plot
    # default: figure values are left to the vision check
    assert run_checks("c1", plot_value, [fig]) == [] and run_checks("c1b", "18 GPa at 1725 °C", [fig]) == []
    # option B ("conflict"): a plot value passes, a value clashing with the caption is flagged
    assert run_checks("c2", plot_value, [fig], figure_mode="conflict") == []
    assert [i.kind for i in run_checks("c3", "The hardness reached 18 GPa at 1725 °C", [fig],
                                       figure_mode="conflict")] == ["number_conflicts_caption"]
    assert [i.kind for i in run_checks("c4", plot_value, [fig], figure_mode="require")] == ["number_not_in_source"]
    assert run_checks("c5", "sintered at 1150 °C", [txt]) == []
    assert [i.kind for i in run_checks("c6", "sintered at 1250 °C", [txt])] == ["number_not_in_source"]
    assert run_checks("c7", "sintered at 1350 °C, reaching 18 GPa", [txt, fig], figure_mode="conflict") == []
    assert run_checks("c8", "sintered at 1250 °C", [txt, fig]) == []       # default: cites a figure → vision check


def test_latex_sources_and_tables():
    # MinerU writes LaTeX; tables carry the unit in the header and bare numbers in the cells
    assert number_fidelity("c1", "grain size of 4.06 µm", ["decreased from 4.06 $\\mu$ m in the monolithic"]) == []
    assert number_fidelity("c2", "Tg of 141.7 °C", ["drops to $141.7^{\\circ}\\mathrm{C}$ for BMI"]) == []
    table = "<table><tr><td>Material</td><td>E-Modul [GPa]</td></tr><tr><td>PA rCF30</td><td>26,7</td></tr></table>"
    assert number_fidelity("c3", "The modulus of PA rCF30 is 26.7 GPa", [table]) == []
    assert [i.kind for i in number_fidelity("c4", "The modulus of PA rCF30 is 28.7 GPa", [table])] == ["number_not_in_source"]


def test_rate_units_and_room_temperature_are_not_misread():
    assert V("strain rate of 2.2 × 10^-2 s^-1") == set() and V("rate 5 s⁻¹") == set()
    assert unit_consistency("c1", "elongation at break at room temperature is about 0.2%") == []
