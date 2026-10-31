# At most MAX_FIGURES_SHOWN figures under an answer, the ones the answer discusses first.
import openwebui.functions.query_pipe as qp
from memento.query.flows import figures


def _fig(page=3, w=500, h=300):
    return {"type": "image", "img_path": "/f.png", "bbox": [100, 100, 100 + w, 100 + h], "source": "doc", "page": page,
            "text": "t"}


HITS = [_fig() for _ in range(10)]


def test_figures_named_in_the_text():
    answer = ("Common methods are DFT [2] and GW [3].\n\nNice figures:\n- Fig. 1.1 [1] – a Venn diagram.\n"
              "- Figure 5.4 [6][7] – the spectrum.\nAbbildung 3 zeigt das Gitter [9].\nNothing else [10].")
    assert figures._figures_named(answer) == {0, 5, 6, 8}


def test_discussed_figures_first_then_seen_then_citation_order():
    answer = "Text [1][2][3].\n- Fig. 2 [5].\n- Fig. 7 [8]."
    shown, more = figures.figures_to_show(HITS, cited=set(range(10)), answer=answer, seen={0, 1}, cap=4)
    assert shown == [0, 1, 4, 7] and more == 6            # named 4, 7; seen 0, 1; shown in citation order


def test_no_cap_shows_every_cited_figure():
    shown, more = figures.figures_to_show(HITS, cited={2, 5}, answer="", seen=set(), cap=None)
    assert shown == [2, 5] and more == 0


def test_figures_already_shown_after_the_first_paragraph_are_skipped():
    shown, more = figures.figures_to_show(HITS, cited={0, 1, 2, 3}, answer="", seen=set(), cap=2, skip={0})
    assert shown == [1, 2] and more == 1


def test_only_figures_the_display_would_show_count():
    hits = [_fig(), _fig(w=60, h=40), _fig(page=0), {"type": "text", "text": "x"}, _fig()]
    shown, more = figures.figures_to_show(hits, cited={0, 1, 2, 3, 4}, answer="", seen=set(), cap=1)
    assert shown == [0] and more == 1                     # the icon, the page-1 image and the text are not figures
    shown, more = figures.figures_to_show(hits, cited={0, 1, 2, 3, 4}, answer="", seen=set(), cap=5, figure_asked=True)
    assert shown == [0, 2, 4] and more == 0               # a figure request shows page-1 figures, as before


def test_the_note_on_further_figures():
    assert figures.more_figures_note(1, de=False) == "\n\n_1 more figure is in the cited sources._"
    assert figures.more_figures_note(9, de=False) == "\n\n_9 more figures are in the cited sources._"
    assert figures.more_figures_note(2, de=True) == "\n\n_2 weitere Abbildungen sind in den zitierten Quellen._"
    assert figures.more_figures_note(1, de=True) == "\n\n_1 weitere Abbildung ist in den zitierten Quellen._"
    assert figures.more_figures_note(0, de=False) == ""


def test_the_cap_is_a_valve_of_six():
    assert qp.Pipe().valves.MAX_FIGURES_SHOWN == 6
