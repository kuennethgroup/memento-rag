#!/usr/bin/env python3
"""
memento/graph/registries/entity_registry_ceramics.py — Ceramics entity registry (RQ2 dual-graph).

This is one *domain* registry — the generic normalization/matching/fusion
machinery lives in memento/graph/registry_base.py, shared by every domain. To deploy
Memento's graph layer for a different scientific domain, copy this file (see
the "Add a new domain" instructions in registry_base.py) and point
`config.DOMAIN["entity_registry"]` at it; memento/graph/__init__.py loads whichever
domain registry is configured, so no other code references "ceramics" by name.

Provides the canonical vocabulary + alias table for the ceramics six-type
schema:

    Material | Process | Property | Condition | Characterization | Relation

`Material`/`Process`/`Property`/`Condition`/`Characterization` become
`Entity` nodes (memento/graph/store.py); `Relation` types become the `rel_type`
property on `RELATION` edges between entities.

This registry makes cross-document, cross-lingual entity fusion work: a
material's name, formula, German name and abbreviation all resolve to one
`canonical_id`, so chunks in English and German papers `MENTIONS` one shared
Entity node, the prerequisite for traversal to surface cross-document evidence.

A curated seed set: the schema's main entities plus the common cross-lingual
and abbreviation variants. Entities it does not know come from the LLM
extraction (`memento/graph/build/extract_entities.py`), so a registry gap means no
canonical fusion, not a dropped entity.

No I/O — pure Python, fully unit-testable.
"""
from __future__ import annotations

# `normalize` is re-exported (unused directly here) so memento/graph/__init__.py's
# domain loader can expose a uniform interface across domain registries.
from memento.graph.registry_base import DEFAULT_ENTITY_TYPES, EntityDef, build_registry, normalize  # noqa: F401

ENTITY_TYPES = DEFAULT_ENTITY_TYPES

# Semantic relation types extracted between Entity nodes. All share the
# single Neo4j "RELATION" relationship type (memento/graph/store.py); the name here
# becomes the `rel_type` edge property. Descriptions double as the schema
# memento/graph/build/extract_entities.py gives the LLM.
RELATION_TYPES = {
    "transforms_to":    "a material/phase transforms into another material/phase "
                         "(e.g. boehmite transforms_to gamma-Al2O3)",
    "sintered_at":      "a material/process is sintered/processed at a given "
                         "condition (temperature, dwell time, atmosphere, ...)",
    "produces":         "a process produces a material/phase",
    "measured_by":      "a property is measured using a characterization technique",
    "compared_to":      "an entity's value/behaviour is compared against another "
                         "entity (e.g. across papers — useful for contradiction "
                         "detection)",
    "consists_of":      "a material consists of / is composed of another "
                         "material or phase",
    "characterized_by": "a material/process is characterized by a property",
}


# ── Registry ──────────────────────────────────────────────────────────────
# Curated seed set (v1). canonical_id is snake_case and stable — it becomes
# the Entity node's `id` in memento/graph/store.py, so do not rename existing ids
# without a migration.
REGISTRY: tuple[EntityDef, ...] = (
    # ── Material ────────────────────────────────────────────────────────
    EntityDef("alumina", "Alumina", "Material",
              ("Al2O3", "Al₂O₃", "aluminium oxide", "aluminum oxide", "Aluminiumoxid")),
    EntityDef("alpha_alumina", "alpha-Al2O3", "Material",
              ("α-Al2O3", "alpha-alumina", "α-alumina", "corundum")),
    EntityDef("gamma_alumina", "gamma-Al2O3", "Material",
              ("γ-Al2O3", "gamma-alumina", "γ-alumina")),
    EntityDef("theta_alumina", "theta-Al2O3", "Material",
              ("θ-Al2O3", "theta-alumina", "θ-alumina")),
    EntityDef("delta_alumina", "delta-Al2O3", "Material",
              ("δ-Al2O3", "delta-alumina", "δ-alumina")),
    EntityDef("boehmite", "Boehmite", "Material",
              ("AlO(OH)", "AlOOH", "gamma-AlOOH", "γ-AlOOH", "Böhmit", "boehmit")),
    EntityDef("spinel", "Spinel", "Material",
              ("MgAl2O4", "magnesium aluminate spinel")),
    EntityDef("yag", "YAG", "Material",
              ("Y3Al5O12", "yttrium aluminium garnet", "yttrium aluminum garnet")),
    EntityDef("zirconia", "Zirconia", "Material",
              ("ZrO2", "zirconium dioxide", "Zirkonoxid", "Zirkondioxid",
               "PSZ", "partially stabilized zirconia", "partially stabilised zirconia")),
    EntityDef("ysz", "Yttria-stabilized zirconia", "Material",
              ("YSZ", "yttria stabilized zirconia", "Y2O3-ZrO2",
               "3YSZ", "8YSZ", "3Y-YSZ", "8Y-YSZ")),
    EntityDef("ytzp", "Y-TZP", "Material",
              ("TZP", "Y-TZP", "3Y-TZP", "5Y-TZP", "2Y-TZP",
               "tetragonal zirconia polycrystal",
               "yttria-stabilized tetragonal zirconia polycrystal")),
    EntityDef("zta", "Zirconia-toughened alumina", "Material",
              ("ZTA", "zirconia toughened alumina",
               "alumina-zirconia composite")),
    EntityDef("hydroxyapatite", "Hydroxyapatite", "Material",
              ("HA", "HAp", "hydroxylapatite", "Ca10(PO4)6(OH)2",
               "calcium hydroxyapatite", "Hydroxylapatit")),
    EntityDef("boron_nitride", "Boron nitride", "Material",
              ("BN", "h-BN", "hexagonal boron nitride", "Bornitrid")),
    EntityDef("cubic_boron_nitride", "Cubic boron nitride", "Material",
              ("c-BN", "cBN", "cubic boron nitride")),
    EntityDef("tungsten_carbide", "Tungsten carbide", "Material",
              ("WC", "Wolframcarbid")),
    EntityDef("carbon_nanotube", "Carbon nanotube", "Material",
              ("CNT", "CNTs", "carbon nanotubes", "MWCNT", "SWCNT",
               "multi-walled carbon nanotube", "single-walled carbon nanotube")),
    # Bare "MAX" is excluded deliberately — it collides with the English word.
    EntityDef("max_phase", "MAX phase", "Material",
              ("MAX phases", "MAX-phase", "MAX-phases")),
    EntityDef("silicon_carbide", "Silicon carbide", "Material",
              ("SiC", "Siliziumcarbid", "Siliziumkarbid")),
    EntityDef("silicon_nitride", "Silicon nitride", "Material",
              ("Si3N4", "Siliziumnitrid")),
    EntityDef("magnesia", "Magnesia", "Material", ("MgO", "magnesium oxide")),
    EntityDef("calcia", "Calcia", "Material", ("CaO", "calcium oxide")),
    EntityDef("zinc_oxide", "Zinc oxide", "Material", ("ZnO",)),
    EntityDef("titania", "Titania", "Material", ("TiO2", "titanium dioxide")),
    EntityDef("silica", "Silica", "Material", ("SiO2", "silicon dioxide")),
    EntityDef("titanium_carbide", "Titanium carbide", "Material", ("TiC",)),
    EntityDef("zirconium_carbide", "Zirconium carbide", "Material", ("ZrC",)),
    EntityDef("hafnium_carbide", "Hafnium carbide", "Material", ("HfC",)),
    EntityDef("tantalum_carbide", "Tantalum carbide", "Material", ("TaC",)),
    EntityDef("zirconium_diboride", "Zirconium diboride", "Material", ("ZrB2",)),
    EntityDef("aluminium_nitride", "Aluminium nitride", "Material",
              ("AlN", "aluminum nitride")),
    EntityDef("titanium_nitride", "Titanium nitride", "Material", ("TiN",)),
    EntityDef("barium_titanate", "Barium titanate", "Material", ("BaTiO3",)),
    EntityDef("pzt", "Lead zirconate titanate", "Material",
              ("PZT", "lead zirconium titanate")),
    EntityDef("uhtc", "Ultra-high temperature ceramic", "Material",
              ("UHTC", "UHTCs", "ultra-high temperature ceramics",
               "ultra high temperature ceramic", "ultra high temperature ceramics")),
    EntityDef("glass_ceramic", "Glass-ceramic", "Material",
              ("glass-ceramics", "glass ceramic", "glass ceramics")),
    EntityDef("thermal_barrier_coating", "Thermal barrier coating", "Material",
              ("TBC", "TBCs", "thermal barrier coatings")),

    # ── Process ─────────────────────────────────────────────────────────
    EntityDef("uhs", "Ultrafast high-temperature sintering", "Process",
              ("UHS", "ultra-fast high temperature sintering",
               "ultrafast high-temperature sintering")),
    EntityDef("sps", "Spark plasma sintering", "Process",
              ("SPS", "field-assisted sintering technique", "FAST")),
    EntityDef("flash_sintering", "Flash sintering", "Process", ()),
    EntityDef("pressureless_sintering", "Pressureless sintering", "Process",
              ("conventional sintering",)),
    EntityDef("hot_pressing", "Hot pressing", "Process",
              ("HP", "hot-pressed", "hot-pressing", "hot pressed")),
    EntityDef("calcination", "Calcination", "Process",
              ("calcining", "Kalzinierung", "Kalzinieren")),
    # "HIP" matches with exact case only (registry_base.SHORT_ALIAS_MAX), so the anatomical "hip" in dental and
    # orthopaedic papers does not collide
    EntityDef("hot_isostatic_pressing", "Hot isostatic pressing", "Process",
              ("HIP", "hot isostatic press", "hot-isostatic pressing",
               "HIPing", "HIPed", "heiß-isostatisches Pressen")),
    EntityDef("cvd", "Chemical vapour deposition", "Process",
              ("CVD", "chemical vapor deposition", "chemical vapour deposition")),
    EntityDef("cvi", "Chemical vapour infiltration", "Process",
              ("CVI", "chemical vapor infiltration", "chemical vapour infiltration")),
    # "AM" matches with exact case only, so the English "am" does not collide
    EntityDef("additive_manufacturing", "Additive manufacturing", "Process",
              ("AM", "3D printing", "3-D printing", "additive fabrication",
               "additive Fertigung")),
    EntityDef("ball_milling", "Ball milling", "Process",
              ("ball-milled", "ball-milling", "ball milled")),
    EntityDef("sol_gel", "Sol-gel", "Process", ("sol–gel", "sol gel", "sol-gel process")),
    EntityDef("liquid_phase_sintering", "Liquid-phase sintering", "Process",
              ("liquid phase sintering", "LPS")),
    EntityDef("pvd", "Physical vapour deposition", "Process",
              ("PVD", "physical vapor deposition")),

    # ── Property ────────────────────────────────────────────────────────
    EntityDef("density", "Density", "Property",
              ("relative density", "bulk density", "Dichte")),
    EntityDef("hardness", "Hardness", "Property",
              ("Vickers hardness", "HV", "Härte")),
    EntityDef("grain_size", "Grain size", "Property", ("Korngröße",)),
    EntityDef("fracture_toughness", "Fracture toughness", "Property",
              ("KIC", "K1c", "KIc", "Bruchzähigkeit")),
    EntityDef("flexural_strength", "Flexural strength", "Property",
              ("bending strength", "Biegefestigkeit")),
    EntityDef("porosity", "Porosity", "Property", ("Porosität",)),
    EntityDef("thermal_conductivity", "Thermal conductivity", "Property",
              ("Wärmeleitfähigkeit",)),
    EntityDef("cte", "Coefficient of thermal expansion", "Property",
              ("CTE", "thermal expansion coefficient", "thermal expansion",
               "Wärmeausdehnungskoeffizient")),

    # ── Condition ───────────────────────────────────────────────────────
    EntityDef("temperature", "Temperature", "Condition",
              ("sintering temperature", "Temperatur", "Sintertemperatur")),
    EntityDef("dwell_time", "Dwell time", "Condition",
              ("holding time", "soaking time", "Haltezeit")),
    EntityDef("heating_rate", "Heating rate", "Condition",
              ("ramp rate", "Heizrate", "Aufheizrate")),
    EntityDef("atmosphere", "Atmosphere", "Condition",
              ("sintering atmosphere", "Atmosphäre")),
    EntityDef("pressure", "Pressure", "Condition",
              ("applied pressure", "Druck", "Sinterdruck")),

    # ── Characterization ────────────────────────────────────────────────
    EntityDef("xrd", "X-ray diffraction", "Characterization",
              ("XRD", "Röntgendiffraktometrie", "Röntgenbeugung")),
    EntityDef("sem", "Scanning electron microscopy", "Characterization",
              ("SEM", "Rasterelektronenmikroskopie", "FESEM", "FE-SEM",
               "field emission scanning electron microscopy", "BSE",
               "backscattered electron")),
    EntityDef("tem", "Transmission electron microscopy", "Characterization",
              ("TEM", "Transmissionselektronenmikroskopie", "HRTEM", "STEM", "HAADF",
               "high-resolution transmission electron microscopy",
               "high resolution transmission electron microscopy",
               "scanning transmission electron microscopy")),
    EntityDef("micro_ct", "Micro-computed tomography", "Characterization",
              ("micro-CT", "µCT", "μCT", "X-ray computed tomography")),
    EntityDef("edx", "Energy-dispersive X-ray spectroscopy", "Characterization",
              ("EDX", "EDS", "energy dispersive X-ray spectroscopy")),
    EntityDef("raman", "Raman spectroscopy", "Characterization", ("Raman",)),
    EntityDef("xps", "X-ray photoelectron spectroscopy", "Characterization", ("XPS",)),
    EntityDef("ftir", "Fourier-transform infrared spectroscopy", "Characterization",
              ("FTIR", "FT-IR", "Fourier transform infrared spectroscopy")),
    EntityDef("afm", "Atomic force microscopy", "Characterization",
              ("AFM", "atomic force microscope")),
    EntityDef("dsc", "Differential scanning calorimetry", "Characterization", ("DSC",)),
    EntityDef("tga", "Thermogravimetric analysis", "Characterization",
              ("TGA", "thermogravimetry")),
)


# ── Public API (delegates to memento/graph/registry_base.py) ──────────────────────
_registry = build_registry(REGISTRY, ENTITY_TYPES)
get_by_id = _registry.get_by_id
canonicalize = _registry.canonicalize
find_entities_in_text = _registry.find_entities_in_text
