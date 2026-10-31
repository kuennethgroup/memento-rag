# Answer wording: plain wording when nothing answers it, "library" not "corpus", the document's name when asked where
# something comes from; an honest answer to "which model are you?"; a table shown when one is asked for; figures under
# equation / table questions and "which figures show …".
import openwebui.functions.query_pipe as qp
from memento.query import prompts
from memento.query.flows import general


def test_answer_style_rules_are_in_every_domain_config():
    # in the domain configs, not in the code
    import yaml
    from pathlib import Path
    for f in sorted((Path(qp.__file__).resolve().parents[2] / "config/domains").glob("*.yaml")):
        p = yaml.safe_load(f.read_text())["system_prompt"]
        assert "own words" in p and "The sources found do not cover" in p and "never claim" in p, f.name
        assert '"the library"' in p and "corpus." not in p.split("\n")[0], f.name
        assert "where something comes from" in p and "title" in p, f.name


def test_which_model_is_answered_honestly():
    p = general.GENERAL_SYSTEM_PROMPT
    assert "which model" in p and "Answered by" in p


def test_a_table_question_shows_the_table():
    assert "Markdown table" in prompts.element_addendum("table")
    assert prompts.element_addendum("figure") == "" and prompts.element_addendum("none") == ""


def test_figures_under_equation_and_table_questions_and_figure_lists():
    assert prompts.figures_for(element="equation", image_intent=False) == (False, False)   # no figures under an equation
    assert prompts.figures_for(element="table", image_intent=False) == (False, False)
    assert prompts.figures_for(element="figure", image_intent=False) == (True, True)     # "which figures show …": shown
    assert prompts.figures_for(element="figure", image_intent=True) == (True, True)
    assert prompts.figures_for(element="none", image_intent=False) == (True, False)      # a normal question: as before
