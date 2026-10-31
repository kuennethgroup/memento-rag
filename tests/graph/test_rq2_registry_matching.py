# tests/graph/test_rq2_registry_matching.py
from memento.graph.registry_base import EntityDef, build_registry

REG = build_registry((
    EntityDef("polystyrene", "Polystyrene", "Material", ("PS", "Polystyrol")),
    EntityDef("hot_pressing", "Hot pressing", "Process", ("HP",)),
    EntityDef("alumina", "Alumina", "Material", ("Al2O3", "aluminium oxide")),
))


def ids(text):
    return [e.canonical_id for e in REG.find_entities_in_text(text)]


def test_short_alias_is_case_sensitive():
    assert ids("PS foams were extruded") == ["polystyrene"]
    assert ids("ps foams, ups and downs") == []
    assert ids("samples were hot pressed (HP) at 1600 C") == ["hot_pressing"]
    assert ids("an engine with 300 hp") == []


def test_short_alias_needs_boundaries():
    assert ids("PSU membranes") == []          # PS inside PSU
    assert ids("(PS)") == ["polystyrene"]


def test_long_aliases_stay_case_insensitive():
    assert ids("AL2O3 and ALUMINIUM OXIDE") == ["alumina"]
    assert ids("polystyrol") == ["polystyrene"]


def test_canonicalize_short_alias_exact_case():
    assert REG.canonicalize("PS").canonical_id == "polystyrene"
    assert REG.canonicalize("ps") is None
    assert REG.canonicalize("polystyrene").canonical_id == "polystyrene"


LATEX_REG = build_registry((
    EntityDef("alumina", "Alumina", "Material", ("Al2O3",)),
    EntityDef("delta_alumina", "delta-Al2O3", "Material", ("δ-Al2O3", "δ-alumina")),
    EntityDef("zirconium_diboride", "Zirconium diboride", "Material", ("ZrB2",)),
))


def latex_ids(text):
    return sorted(e.canonical_id for e in LATEX_REG.find_entities_in_text(text))


def test_latex_formulas_are_folded_before_matching():
    assert latex_ids(r"only $Al_{2}O_{3}$ peaks were observed") == ["alumina"]
    assert latex_ids(r"$\mathrm{HfB} _ {2} + \mathrm{ZrB} _ {2}$") == ["zirconium_diboride"]
    assert latex_ids(r"transforms to $\delta$-$Al_2O_3$ at 900 C") == ["delta_alumina"]


def test_latex_fold_leaves_non_digit_subscripts_alone():
    assert latex_ids(r"$SrLaAl_{1 - x}O_4$ ceramics") == []


CAPS_REG = build_registry((
    EntityDef("sps", "Spark plasma sintering", "Process", ("SPS", "FAST")),
    EntityDef("tem", "Transmission electron microscopy", "Characterization", ("TEM", "STEM")),
    EntityDef("ytzp", "Y-TZP", "Material", ("3Y-TZP",)),
))


def caps_ids(text):
    return sorted(e.canonical_id for e in CAPS_REG.find_entities_in_text(text))


def test_all_caps_acronyms_are_case_sensitive():
    assert caps_ids("densified by FAST at 1500 C") == ["sps"]
    assert caps_ids("the phase fraction increases fast") == []
    assert caps_ids("STEM-EDX maps") == ["tem"]
    assert caps_ids("mesenchymal stem cells") == []
    assert caps_ids("3Y-TZP crowns") == ["ytzp"]
    assert CAPS_REG.canonicalize("fast") is None and CAPS_REG.canonicalize("FAST").canonical_id == "sps"


def test_mixed_case_names_stay_case_insensitive():
    assert caps_ids("SPARK PLASMA SINTERING and transmission Electron Microscopy") == ["sps", "tem"]


def test_latex_fold_handles_split_math_spans_and_greek_hyphen_spacing():
    assert latex_ids(r"PDF 00-046-1215 (δ-Al $_{2}$ O $_{3}$ ), and") == ["delta_alumina"]
    assert latex_ids(r"the $\gamma$ - and $\delta$ -alumina phases") == ["delta_alumina"]
    assert latex_ids(r"Al $_{2}$ O $_{3}$ and") == ["alumina"]


OVERLAP_REG = build_registry((
    EntityDef("boron_nitride", "Boron nitride", "Material", ("BN", "h-BN")),
    EntityDef("cubic_boron_nitride", "Cubic boron nitride", "Material", ("c-BN", "cubic boron nitride")),
    EntityDef("alumina", "Alumina", "Material", ("Al2O3",)),
))


def test_exact_case_match_inside_a_longer_match_is_dropped():
    found = [e.canonical_id for e in OVERLAP_REG.find_entities_in_text("c-BN inserts")]
    assert found == ["cubic_boron_nitride"]


def test_matches_are_returned_in_first_occurrence_order():
    found = [e.canonical_id for e in OVERLAP_REG.find_entities_in_text("BN coated Al2O3 and c-BN")]
    assert found == ["boron_nitride", "alumina", "cubic_boron_nitride"]


def test_position_map_reproduces_normalize():
    from memento.graph.registry_base import _normalize_with_map, normalize
    for t in ["  α-Al₂O₃  (Corundum) ", "Δ Temperatur\n\n und   ZrO2", "µCT of SiC"]:
        norm, pos = _normalize_with_map(t)
        assert norm.strip(" \t-_.,;:()[]") == normalize(t)
        assert len(pos) == len(norm)


def test_canonicalize_all_caps_input_of_mixed_case_names():
    assert CAPS_REG.canonicalize("SPARK PLASMA SINTERING").canonical_id == "sps"
    assert REG.canonicalize("POLYSTYRENE").canonical_id == "polystyrene"
    assert CAPS_REG.canonicalize("fast") is None and REG.canonicalize("ps") is None


def test_exact_case_aliases_match_unicode_subscript_digits():
    reg = build_registry((EntityDef("fracture_toughness", "Fracture toughness", "Property", ("K1c",)),
                          EntityDef("ysz", "Yttria-stabilized zirconia", "Material", ("3YSZ",))))
    assert [e.canonical_id for e in reg.find_entities_in_text("K₁c of 4 MPa for ³YSZ")] == ["fracture_toughness", "ysz"]
    assert [e.canonical_id for e in reg.find_entities_in_text("toughness K₁c")] == ["fracture_toughness"]
