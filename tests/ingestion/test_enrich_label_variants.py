# Figure and table labels match the authors' citations in any spelling: a caption "Fig. 5" is linked to text that writes
# "Figure 5 shows …", and German labels ("Abb. 3", "Abbildung 3", "Tabelle 2") are part of the domain vocabulary.
import json
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.ingestion import enrich_parsed_content_list as en  # noqa: E402

FILLER = "The samples were prepared as described in the methods section of this work."


def text(t, page=0):
    return {"type": "text", "text": t, "page_idx": page, "bbox": [80, 100, 480, 200]}


def image(caption, page=1):
    return {"type": "image", "img_path": "", "image_caption": [caption], "image_footnote": [], "page_idx": page,
            "bbox": [100, 300, 800, 600]}


def table(caption, page=1):
    return {"type": "table", "table_caption": [caption], "table_footnote": [], "table_body": "<table></table>",
            "page_idx": page, "bbox": [100, 300, 800, 600]}


def enrich(items, tmp_path):
    p = tmp_path / "doc_content_list.json"
    p.write_text(json.dumps(items))
    out = en.enrich_content_list(p)
    return json.loads(Path(out).read_text())


def element(items, typ):
    return next(i for i in items if i["type"] == typ)


# ── abbreviation and full word ──────────────────────────────────────────────────────────────────────────────────────

def test_caption_abbreviation_finds_full_word_citation(tmp_path):
    items = [text("Figure 5 shows the flexural strength of the specimens after quenching."), text(FILLER, page=3),
             image("Fig. 5 Thermal shock resistance of porous alumina ceramics.")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["enrichment_method"] == "label_match"
    assert "Figure 5 shows the flexural strength" in fig["enriched_context"]


def test_caption_full_word_finds_abbreviated_citation(tmp_path):
    items = [text("The reproducibility of the process is high (Fig. 12), as the coatings show."),
             image("Figure 12: Coating thickness over twenty runs.")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["enrichment_method"] == "label_match"
    assert "(Fig. 12)" in fig["enriched_context"]


def test_plural_abbreviation_counts_as_the_same_word(tmp_path):
    items = [text("Figs. 3 and 4 compare the two sintering routes at equal density."),
             image("Figure 3: Microstructure after spark plasma sintering.")]
    assert element(enrich(items, tmp_path), "image")["enrichment_method"] == "label_match"


def test_number_guard_still_holds_across_spellings():
    items = [text("Figure 10 shows the grain size distribution for all specimens.")]
    assert en.find_all_label_reference_indices(items, "Fig. 1") == []


def test_other_label_words_stay_separate(tmp_path):
    items = [text("Scheme 1 illustrates the reaction pathway of the precursor."), text(FILLER, page=3),
             image("Fig. 1 SEM image of the fracture surface.")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["enrichment_method"] != "label_match"


def test_table_words_do_not_match_figure_citations(tmp_path):
    assert en.find_all_label_reference_indices([text("Fig. 2 shows the density of all sintered samples.")],
                                               "Table 2") == []


def test_subpanel_label_falls_back_to_the_figure_number(tmp_path):
    items = [text("As Figure 3 shows, the porosity decreases with temperature."),
             image("Fig. 3a Porosity at 1200 °C.")]
    assert element(enrich(items, tmp_path), "image")["enrichment_method"] == "label_match"


# ── supplementary figures ───────────────────────────────────────────────────────────────────────────────────────────

def test_supplementary_figure_is_not_the_main_figure():
    items = [text("From the conversion spectra (Supplementary Fig. 2) we conclude that the phase is stable.")]
    assert en.find_all_label_reference_indices(items, "Figure 2") == []


def test_supplementary_and_main_citation_in_one_passage_matches():
    items = [text("Supplementary Fig. 2 gives the raw data; Fig. 2 shows the fitted curves for all samples.")]
    assert en.find_all_label_reference_indices(items, "Figure 2") == [0]


# ── German labels ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_german_figure_caption_gets_a_label():
    assert en._extract_figure_label("Abb. 3: Schematische Darstellung des Versuchsaufbaus") == "Abb. 3"
    assert en._extract_figure_label("Abbildung 4.2: DSC-Kurven der Proben") == "Abbildung 4.2"
    assert en._extract_figure_label("Bild 7: Bruchfläche nach dem Zugversuch") == "Bild 7"


def test_german_figure_caption_finds_full_word_citation(tmp_path):
    items = [text("Abbildung 3 zeigt den schematischen Aufbau der Spritzgießanlage mit Werkzeug."),
             text(FILLER, page=3), image("Abb. 3: Schematische Darstellung des Versuchsaufbaus")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["enrichment_method"] == "label_match"
    assert "Abbildung 3 zeigt" in fig["enriched_context"]


def test_german_table_caption_finds_abbreviated_citation(tmp_path):
    items = [text("Die Kennwerte der Werkstoffe sind in Tab. 2 zusammengefasst und werden verglichen."),
             text(FILLER, page=3), table("Tabelle 2: Mechanische Kennwerte der untersuchten Werkstoffe")]
    tab = element(enrich(items, tmp_path), "table")
    assert tab["table_label"] == "Tabelle 2"
    assert tab["enrichment_method"] == "label_match"


def test_chapter_numbered_figures_match_exactly():
    items = [text("Abbildung 4.1 zeigt die Proben vor der Prüfung im Klimaschrank."),
             text("Wie Abb. 4.2 zeigt, sinkt die Festigkeit mit steigender Temperatur deutlich.")]
    assert en.find_all_label_reference_indices(items, "Abbildung 4.2") == [1]
    assert en.find_all_label_reference_indices(items, "Abbildung 4") == []
    assert en._extract_table_label("Table 3.1: Processing parameters") == "Table 3.1"


def test_three_level_numbers_match_exactly():
    assert en._extract_figure_label("Figure 4.5.5: K_Ic of the cured bulk epoxy resin systems") == "Figure 4.5.5"
    items = [text("Figure 4.5.7 shows that the hybrid system is a tough and stiff matrix resin."),
             text("The fracture toughness in Fig. 4.5.5 rises with the rubber content of all systems.")]
    assert en.find_all_label_reference_indices(items, "Figure 4.5.5") == [1]


def test_label_followed_by_a_full_stop_is_not_a_chapter_number():
    assert en._extract_figure_label("Fig. 5. Thermal shock resistance of porous alumina.") == "Fig. 5"
    items = [text("The strength drops sharply above 600 °C, see Fig. 5. All specimens behave alike.")]
    assert en.find_all_label_reference_indices(items, "Figure 5") == [0]


def test_bild_and_abbildung_stay_separate():
    items = [text("Bild 3 zeigt die Probengeometrie nach DIN EN ISO 527 für alle Versuche.")]
    assert en.find_all_label_reference_indices(items, "Abb. 3") == []


def test_german_equation_reference_matches():
    items = [text("Mit Gl. (3) ergibt sich die Viskosität der Schmelze bei dieser Temperatur.")]
    assert en.find_all_equation_reference_indices(items, "3") == [0]


# ── lists and ranges: "Figs. 3 and 4", "Figures 3.31 to 3.35", "Abb. 2–4" ──
# Used only when the figure has no direct citation.

def test_second_item_of_a_list_is_linked(tmp_path):
    items = [text("Figs. 3 and 4 compare the two sintering routes at equal density."), text(FILLER, page=3),
             image("Fig. 4 Microstructure after hot pressing.")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["enrichment_method"] == "label_match"
    assert "Figs. 3 and 4 compare" in fig["enriched_context"]


def test_comma_list_with_conjunction_is_linked():
    items = [text("Figures 3, 4 and 5 show the fracture surfaces of the three composites.")]
    assert en.find_all_label_reference_indices(items, "Fig. 4") == [0]


def test_chapter_numbered_range_is_linked():
    items = [text("Figures 3.31 to 3.35 show the stereo micrographs of all damaged specimens.")]
    assert en.find_all_label_reference_indices(items, "Figure 3.33") == [0]
    assert en.find_all_label_reference_indices(items, "Figure 3.36") == []


def test_german_range_is_linked():
    items = [text("Abb. 2–4 zeigen die Bruchflächen der Proben nach dem Schlagbiegeversuch.")]
    assert en.find_all_label_reference_indices(items, "Abbildung 3") == [0]


def test_direct_citation_wins_over_list():
    items = [text("Figs. 3 and 4 compare the two sintering routes at equal density."),
             text("Figure 4 shows the grain boundaries after hot pressing at 1600 °C.")]
    assert en.find_all_label_reference_indices(items, "Fig. 4") == [1]


def test_list_number_guard_holds():
    items = [text("Figs. 3 and 14 compare the two sintering routes at equal density.")]
    assert en.find_all_label_reference_indices(items, "Fig. 1") == []


def test_count_after_a_comma_is_not_a_list():
    items = [text("In Fig. 3, 25 samples were tested at room temperature and at 800 °C.")]
    assert en.find_all_label_reference_indices(items, "Figure 25") == []


def test_range_does_not_cross_chapters():
    items = [text("Figures 3.9 to 4.2 summarise the results of the creep tests on all alloys.")]
    assert en.find_all_label_reference_indices(items, "Figure 4.1") == []
    assert en.find_all_label_reference_indices(items, "Figure 4.2") == [0]


def test_supplementary_list_is_not_the_main_figures():
    items = [text("The raw spectra are in Supplementary Figs. 3 and 4 for all compositions studied.")]
    assert en.find_all_label_reference_indices(items, "Figure 4") == []


def test_hyphen_is_no_range_in_a_hyphen_numbered_book():
    items = [text("Figure 3-8 shows the sandwich element with two thin face sheets of robust material."),
             image("Figure 3-8. Sandwich element", page=1), image("Figure 8-1. Mould for the face sheets", page=2)]
    assert en.find_all_label_reference_indices(items, "Figure 8") == []
    assert en.find_all_label_reference_indices([text("Figure 41-43 show the 3D models of the strand borders.")],
                                               "Figure 42") == [0]


# ── supplementary labels and recovered captions ──
# An "Extended Data Figure 2" caption is not "Figure 2" of the main paper; running text "Abbildung 4.2 zeigt …" beside
# an uncaptioned image is not its caption.

def test_supplementary_captions_get_no_main_label():
    assert en._extract_figure_label("Extended Data Figure 2 | More channels reduce the pressure drop.") is None
    assert en._extract_figure_label("Supplementary Fig. 3: Raw spectra of all samples") is None
    assert en._extract_table_label("Supplementary Table 1. Composition of the glasses") is None
    assert en._extract_figure_label("Figure 2: Pressure drop over the channel count") == "Figure 2"


def test_main_and_extended_data_figure_2_stay_apart(tmp_path):
    items = [text("Fig. 2 shows the pressure drop of the heat exchanger for all channel counts."),
             image("Figure 2: Pressure drop over the channel count", page=1), text(FILLER, page=5),
             image("Extended Data Figure 2 | More channels reduce the pressure drop.", page=10)]
    out = [i for i in enrich(items, tmp_path) if i["type"] == "image"]
    assert out[0]["enrichment_method"] == "label_match"
    assert "figure_label" not in out[1] and out[1]["enrichment_method"] != "label_match"


def test_si_citation_is_not_the_main_figure():
    assert en.find_all_label_reference_indices(
        [text("The raw data are given in SI Fig. 2 for all compositions studied here.")], "Figure 2") == []
    assert en.find_all_label_reference_indices(
        [text("See Supplementary Information Fig. 2 for the fitted curves of all samples.")], "Figure 2") == []


def test_chapter_numbered_prose_is_not_a_caption_line():
    assert en._CAPTION_LINE_RE.match("Abbildung 4.2 zeigt die Ergebnisse des Versuchs.") is None
    assert en._CAPTION_LINE_RE.match("Figure 4.2 shows the results of the test series.") is None
    for caption in ("Abbildung 4.2: DSC-Kurven der Proben", "Figure 4.2. Results of the test series",
                    "Fig. 5. Thermal shock resistance of porous alumina"):
        assert en._CAPTION_LINE_RE.match(caption), caption


def test_uncaptioned_image_gets_no_label_from_chapter_prose(tmp_path):
    items = [image("", page=1), text("Abbildung 4.2 zeigt die Ergebnisse des Versuchs an allen Proben.", page=1)]
    items[0]["image_caption"] = []
    fig = element(enrich(items, tmp_path), "image")
    assert "figure_label" not in fig
    assert fig["enrichment_method"] != "label_match"


def test_hyphen_numbered_citation_is_not_the_first_number():
    # a book numbering "Table 2-2" is not "Table 2"
    items = [text("The benefits of a high ballistic coefficient are listed in Table 2-2 for all vehicles.")]
    assert en.find_all_label_reference_indices(items, "Table 2") == []
    assert en.find_all_label_reference_indices([text("Figs. 3-5 show the fracture surfaces of all samples tested.")],
                                               "Fig. 3") == [0]


def test_label_families_of_every_domain_are_the_intended_ones():
    # a word added to a domain YAML must not silently join a family by prefix
    import yaml
    expected_fig = [{"fig", "figs", "figure", "figures"}, {"abb", "abbildung", "abbildungen"}]
    for path in sorted((ROOT / "config/domains").glob("*.yaml")):
        labels = yaml.safe_load(path.read_text())["structural_labels"]
        fig = [f for f in _families(labels["figure"]) if len(f) > 1]
        tab = [f for f in _families(labels["table"]) if len(f) > 1]
        assert sorted(map(sorted, fig)) == sorted(map(sorted, expected_fig)), path.name
        assert tab == [{"tab", "table", "tables", "tabelle", "tabellen"}], path.name


def _families(words):
    saved = dict(en._STRUCTURAL_LABELS)
    try:
        en._STRUCTURAL_LABELS.clear()
        en._STRUCTURAL_LABELS["x"] = words
        return en._label_families("x")
    finally:
        en._STRUCTURAL_LABELS.clear()
        en._STRUCTURAL_LABELS.update(saved)


# ── hyphen chapter numbering: "Abbildung 4-1", "Table 5-1" ──
# A caption cut to "Abbildung 4" would match the text about 4-1 for every figure of chapter 4.

def test_hyphen_numbered_captions_keep_the_whole_number():
    assert en._extract_figure_label("Abbildung 4-1: Chemische Struktur des Monomers") == "Abbildung 4-1"
    assert en._extract_table_label("Tabelle 5-1 Gemessene Kennwerte der Simulation") == "Tabelle 5-1"
    assert en._extract_figure_label("Figure 10-25 Fracture origin in a glass rod") == "Figure 10-25"


def test_hyphen_numbered_label_matches_exactly(tmp_path):
    items = [text("Abbildung 4-12 zeigt die DMTA-Kurven der ausgewählten Proben nach der Lagerung."),
             text("In Abbildung 4-1 ist die chemische Struktur des verwendeten Monomers dargestellt."),
             text(FILLER, page=3), image("Abbildung 4-1: Chemische Struktur des Monomers")]
    fig = element(enrich(items, tmp_path), "image")
    assert fig["figure_label"] == "Abbildung 4-1"
    assert fig["enrichment_method"] == "label_match"
    assert "In Abbildung 4-1 ist" in fig["enriched_context"] and "4-12" not in fig["enriched_context"]


def test_hyphen_ranges_still_work_in_plain_numbered_papers():
    assert en.find_all_label_reference_indices([text("Figs. 3-5 show the fracture surfaces of all samples tested.")],
                                               "Fig. 4") == [0]


# ── a figure's caption mentioning "Table 1" is not Table 1 ──

def test_supplementary_figure_mentioning_a_table_gets_no_table_label(tmp_path):
    items = [text("Table 1 lists the processing parameters of all heat-exchanger plates in this study."),
             text(FILLER, page=5),
             image("Extended Data Figure 2 | Additional measurements; compare with Table 1 for processing parameters.",
                   page=10)]
    fig = element(enrich(items, tmp_path), "image")
    assert "figure_label" not in fig and "table_label" not in fig
    assert fig["enrichment_method"] != "label_match"


def test_chart_whose_caption_is_a_table_keeps_its_table_label(tmp_path):
    items = [text("Table 3 summarises the wear rates of all textured samples after the sliding tests."),
             text(FILLER, page=5), image("Table 3: Wear rates of the textured samples", page=6)]
    fig = element(enrich(items, tmp_path), "image")
    assert fig.get("table_label") == "Table 3" and fig["enrichment_method"] == "label_match"


def test_parser_space_inside_a_chapter_number_is_closed():
    # MinerU writes "Figure 1. 2 Global carbon …" for Figure 1.2
    assert en._extract_figure_label("Figure 1. 2 Global carbon emissions from fossil fuels") == "Figure 1.2"
    assert en._extract_table_label("Table 2. 5 Cycling profile for four groups") == "Table 2.5"
    assert en._extract_figure_label("Fig. 5. 3D map of the porosity") == "Fig. 5"
    assert en._extract_figure_label("Fig. 5. Thermal shock resistance") == "Fig. 5"
