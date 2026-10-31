# Naming a document without a graph: several names per document and guarded short names ("the X paper").
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from memento.query.scope.paper_resolve import PaperResolver, resolve_short_name  # noqa: E402

NAMES = {
    "polyBERT - a chemical language model to enable fully machine-driven ultrafast polymer informatics": [
        "polyBERT - a chemical language model to enable fully machine-driven ultrafast polymer informatics",
        "polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics"],
    "The origin of ferroelectricity in Hf1-xZrxO2": ["The origin of ferroelectricity in Hf1-xZrxO2"],
    "Domain pinning - Comparison of hafnia and PZT based ferroelectrics": [
        "Domain pinning - Comparison of hafnia and PZT based ferroelectrics"],
    "Stabilization of ferroelectric properties in hafnia with doping": [
        "Stabilization of ferroelectric properties in hafnia with doping"],
    "Dissertation_Hofbauer_final": ["Dissertation_Hofbauer_final", "Einfluss von DOPO auf Epoxidharze"],
    "151105_PhD-Kolloquium_Hofbauer_final": ["151105_PhD-Kolloquium_Hofbauer_final"],
    "160514_Diss Brandauer final": ["160514_Diss Brandauer final"],
    "Vorabversion_Diss_Brandauer": ["Vorabversion_Diss_Brandauer"],
    "160128_Promotionskolloquium_Brandauer": ["160128_Promotionskolloquium_Brandauer"],
}


def test_a_second_name_of_the_same_document_is_not_a_runner_up():
    q = "What does a chemical language model to enable fully machine-driven ultrafast polymer informatics propose?"
    both = PaperResolver((s, a) for s, al in NAMES.items() for a in al).resolve(q)
    assert both is not None and both.source.startswith("polyBERT")


def test_short_names_in_english_and_german():
    assert resolve_short_name("Summarize the polyBERT paper", NAMES).startswith("polyBERT")
    assert resolve_short_name("Fasse das polyBERT Paper zusammen", NAMES).startswith("polyBERT")
    assert resolve_short_name("Fasse die Dissertation von Hofbauer zusammen", NAMES) == "Dissertation_Hofbauer_final"
    assert resolve_short_name("Summarize Hofbauer's colloquium talk", NAMES) == "151105_PhD-Kolloquium_Hofbauer_final"
    assert resolve_short_name("Zeig mir den Vortrag von Brandauer", NAMES) == "160128_Promotionskolloquium_Brandauer"


def test_short_names_abstain_when_unsure():
    assert resolve_short_name("Summarize the hafnia paper", NAMES) is None                 # several hafnia documents
    assert resolve_short_name("What is the Brandauer dissertation about?", NAMES) is None      # final and draft
    assert resolve_short_name("What does the paper say about ferroelectricity?", NAMES) is None   # a topic
    assert resolve_short_name("Which papers discuss polyBERT?", NAMES) is None             # plural: a set
    assert resolve_short_name("What is polyBERT?", NAMES) is None                          # no document named
    assert resolve_short_name("Tell me about Hofbauer's work on flame retardants", NAMES) is None
