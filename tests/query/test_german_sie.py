# German texts address the reader with "Sie": Memento's fixed German texts, and the model's German through one rule
# added where answers are generated (generation.open_answer, checking.thorough_check, the rework prompt). Older chats
# keep their "du" texts, which are still recognised.
import asyncio
import re
from types import SimpleNamespace

import pytest

import openwebui.functions.ingestion_pipe as ip
import openwebui.functions.query_pipe as qp
from memento.query.models import load_variants
from memento.query import cache, checking, generation, request, texts
from memento.query.scope import conversation, documents, resolve
from memento.query.flows import rework
from memento.ingestion.upload import messages

INFORMAL = re.compile(r"\b(?:du|Du|dich|Dich|dir|Dir|dein\w*|Dein\w*|kannst|Kannst|möchtest|willst|Entschuldige)\b")
# du imperatives; status lines ("Prüfe die Übersicht...", "Füge zuerst ... hinzu...") are Memento's first person
IMPERATIVE = re.compile(r"\b(?:nutze|frag|Frag|versuch|Versuch|wähle|Wähle|hänge|Hänge|füge|Füge|prüfe|Prüfe|sag|"
                        r"nenne|öffne|sende|sieh|Sieh|schalte|antworte)\b")


def _status(en) -> bool:
    return isinstance(en, str) and en.endswith("...")


def _german_texts():
    out = []
    for en, de in list(texts.DE_FIXED.items()) + list(messages._DE_UPLOAD_FIXED.items()):
        out.append((de, _status(en)))
    out += [(de, False) for de in list(texts._DE_REASONS.values()) + list(texts._DE_ERRORS.values())]
    for rx, rep in list(texts._DE_PATTERNS) + list(messages._DE_UPLOAD_PATTERNS):
        if isinstance(rep, str):
            out.append((rep, rep.endswith("...")))
    token = request.LANG.set("de")
    try:
        out.append((texts.tr("⚠️ **Sorry, something went wrong.** Please try again in a minute. If it keeps happening, "
                           "tell the study organiser and mention `E-x`."), False))
    finally:
        request.LANG.reset(token)
    out += [(texts.NAMED_NOT_FOUND_DE, False), (texts.NO_UPLOADS_DE, False), (conversation.unnamed_document_msg(True), False),
            (resolve.processing_msg(["A"], all_busy=True, de=True), False),
            (documents.existence_reply("X", "de", title="X"), False)]
    return out


def test_no_german_text_says_du():
    bad = [de for de, status in _german_texts() if INFORMAL.search(de) or (not status and IMPERATIVE.search(de))]
    assert not bad, "\n".join(bad)


def test_formal_texts_keep_the_house_style():
    for de, _ in _german_texts():
        assert "—" not in de
    assert "Können Sie" in conversation.unnamed_document_msg(True)
    assert texts.DE_FIXED[texts.DOC_CHAT_READY_MSG] == "Ihre Datei ist bereit. Was möchten Sie darüber wissen?"


# ── older chats: the "du" texts are still recognised ──
def test_every_german_note_maps_back_to_english_without_an_umlaut():
    # every German note maps back, also one without an umlaut (else the stopped note could be stored as an answer)
    for en, de in texts.DE_FIXED.items():
        if len(de) > 12:
            assert en in texts.untr(f"Text. {de}"), de
    for en, old in texts._DE_FIXED_DU.items():
        assert en in texts.untr(f"Text. {old}"), old


def test_a_stopped_german_answer_is_not_stored():
    text = "Eine halbe Antwort" + "\n\n" + texts.DE_FIXED[texts.STOPPED_NOTE]
    assert any(n in texts.untr(text) for n in cache.UNSTORED_NOTES)
    old = "Eine halbe Antwort" + "\n\n" + texts._DE_FIXED_DU[texts.STOPPED_NOTE]
    assert any(n in texts.untr(old) for n in cache.UNSTORED_NOTES)


@pytest.mark.parametrize("reply", ["new", "old"])
def test_ja_after_the_not_found_reply_searches_the_library(reply):
    text = texts.NAMED_NOT_FOUND_DE if reply == "new" else texts._DE_FIXED_DU[texts.NAMED_NOT_FOUND_MSG]
    history = [{"role": "user", "content": "Was sagt das Zebra-Paper?"}, {"role": "assistant", "content": text}]
    assert resolve.search_all_request("ja", history) == "Was sagt das Zebra-Paper?"


# ── the model's German: the rule in the prompt that reaches the model ─────────────────────────────────────────────
V = {v.id: v for v in load_variants()}


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _Stream:
    def __init__(self):
        self.response = SimpleNamespace(headers={})

    def __aiter__(self):
        self._done = False
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return _chunk("Antwort")

    async def close(self):
        pass


def _pipe(monkeypatch, tmp_path):
    p = qp.Pipe()
    p.valves.LLM_BASE_URL, p.valves.LLM_MODEL = "http://local/v1", "qwen3.8-27b"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    calls = []

    def fake_client(base_url, api_key, cloud):
        async def create(**kw):
            calls.append(kw)
            return _Stream()
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(generation, "client", fake_client)
    return p, calls


def _open(p, messages, lang):
    async def go():
        token = request.LANG.set(lang)
        try:
            chunks, _, _ = await generation.open_answer(p.valves, V["local"], messages=messages, think=False)
            return "".join([c.choices[0].delta.content async for c in chunks])
        finally:
            request.LANG.reset(token)
    return asyncio.run(go())


def _system(call) -> str:
    return next(m["content"] for m in call["messages"] if m["role"] == "system")


def test_a_german_question_gets_the_rule_once(monkeypatch, tmp_path):
    p, calls = _pipe(monkeypatch, tmp_path)
    msgs = [{"role": "system", "content": "Answer from the sources." + texts.DE_ADDRESS_RULE},
            {"role": "user", "content": "Wie lade ich ein Dokument hoch?"}]
    _open(p, msgs, "de")
    assert _system(calls[0]).count(texts.DE_ADDRESS_RULE.strip()) == 1
    plain = [{"role": "system", "content": "Answer."}, {"role": "user", "content": "Kannst du mir helfen?"}]
    _open(p, plain, "de")
    assert _system(calls[1]) == "Answer." + texts.DE_ADDRESS_RULE
    assert plain[0]["content"] == "Answer."                       # the caller's messages are not changed
    no_system = [{"role": "user", "content": "Hallo"}]
    _open(p, no_system, "de")
    assert calls[2]["messages"][0] == {"role": "system", "content": texts.DE_ADDRESS_RULE.strip()}
    assert no_system == [{"role": "user", "content": "Hallo"}]


def test_an_english_question_is_unchanged(monkeypatch, tmp_path):
    p, calls = _pipe(monkeypatch, tmp_path)
    msgs = [{"role": "system", "content": "Answer."}, {"role": "user", "content": "How do I upload?"}]
    _open(p, msgs, "en")
    assert calls[0]["messages"] == msgs


def test_the_rule_says_sie_and_keeps_quotes_and_names():
    rule = texts.DE_ADDRESS_RULE
    assert "Sie, Ihnen and Ihr" in rule and "informally" in rule and "quotations" in rule and "names" in rule


def test_the_rework_prompt_always_carries_the_rule():
    # "translate it into German" asked in English: an exception to "English prompts unchanged"
    msgs = rework.rework_messages("System.", "An answer [1].", [{"source": {"name": "Doc (p. 1)"}, "document": ["x"]}],
                               "translate it into German")
    assert texts.DE_ADDRESS_RULE.strip() in msgs[0]["content"]


def test_an_english_rework_request_states_english():
    # an English "summarize it in short" after a summary came back in German (the address rule was the only language
    # line)
    msgs = rework.rework_messages("System.", "An answer [1].", [{"source": {"name": "Doc (p. 1)"}, "document": ["x"]}],
                                  "Please summarize the paper in short")
    assert msgs[0]["content"].endswith(rework.REWORK_EN_RULE)
    assert "write the reworked answer in English" in rework.REWORK_EN_RULE and "another language" in rework.REWORK_EN_RULE


def test_a_german_rework_request_gets_no_english_line():
    token = request.LANG.set("de")
    try:
        msgs = rework.rework_messages("System.", "Eine Antwort [1].", [], "Bitte kürzer")
    finally:
        request.LANG.reset(token)
    assert rework.REWORK_EN_RULE not in msgs[0]["content"] and texts.DE_ADDRESS_RULE.strip() in msgs[0]["content"]


def test_thorough_check_revisions_get_the_rule(monkeypatch, tmp_path):
    import memento.agentic_validation.strategies as st
    seen = {}

    class Ctx:
        def __init__(self, gen, model, prompt, *a, **k):
            seen["prompt"] = prompt

    def run(*a, **k):
        raise RuntimeError("stop here")
    monkeypatch.setattr(st, "ArmContext", Ctx)
    monkeypatch.setattr(st, "run_strategy", run)
    p = qp.Pipe()

    async def status(*a, **k):
        pass

    async def go(lang):
        token = request.LANG.set(lang)
        try:
            await checking.thorough_check(p.valves, answer="A [1].", hits=[], question="q", check_prompt="Check.", think=False,
                                    client=None, dom={}, retrieve_filter=None, api_key="k", status=status)
        finally:
            request.LANG.reset(token)
    asyncio.run(go("de"))
    assert seen["prompt"] == "Check." + texts.DE_ADDRESS_RULE
    asyncio.run(go("en"))
    assert seen["prompt"] == "Check."


def test_german_extensions_are_stored_under_the_rule_english_unchanged():
    from memento.library import answers as ac
    from memento.research_gaps.passages import SUGGEST_SYSTEM_PROMPT
    assert cache.suggest_prompt_version("authors", german=False) == ac.prompt_version(SUGGEST_SYSTEM_PROMPT, "authors")
    de = cache.suggest_prompt_version("authors", german=True)
    assert de != cache.suggest_prompt_version("authors", german=False)
    assert de == ac.prompt_version(SUGGEST_SYSTEM_PROMPT, "authors", texts.DE_ADDRESS_RULE)
