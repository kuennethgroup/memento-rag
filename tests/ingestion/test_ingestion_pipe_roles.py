# Roles: admins and CONTRIBUTOR_GROUP members may ingest.
import asyncio
import sys
import types
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.ingestion.upload import intake, messages
from memento.ingestion.upload import document

ip = load_function(PROJECT_ROOT / "openwebui/functions/ingestion_pipe.py", "memento_ingestion")


def _groups(monkeypatch, names):
    class G:
        @staticmethod
        async def get_groups_by_member_id(uid):
            return [types.SimpleNamespace(name=n) for n in names]
    mod = types.ModuleType("open_webui.models.groups")
    mod.Groups = G
    monkeypatch.setitem(sys.modules, "open_webui.models.groups", mod)


def _may(pipe, user):
    return asyncio.run(intake.may_ingest(pipe.valves, user))


def test_admin_always_contributor_by_group_reader_denied(monkeypatch):
    p = document.Ingester(ip.Pipe().valves)
    _groups(monkeypatch, ["contributors"])
    assert _may(p, {"id": "a", "role": "admin"})
    assert _may(p, {"id": "u", "role": "user"})
    _groups(monkeypatch, ["readers"])
    assert not _may(p, {"id": "u", "role": "user"})
    p.valves.CONTRIBUTOR_GROUP = ""
    _groups(monkeypatch, ["contributors"])
    assert not _may(p, {"id": "u", "role": "user"})


def test_group_lookup_failure_denies(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_webui.models.groups", types.ModuleType("broken"))
    assert not _may(ip.Pipe(), {"id": "u", "role": "user"})


def test_denied_user_gets_a_friendly_message(monkeypatch):
    _groups(monkeypatch, [])
    async def go():
        return [c async for c in ip.Pipe().pipe({"messages": []}, {"id": "u", "role": "user"})]
    out = asyncio.run(go())
    assert len(out) == 1 and "contributors only" in out[0]


def test_several_groups_comma_separated_including_the_trial_group(monkeypatch):
    # several upload groups
    p = document.Ingester(ip.Pipe().valves)
    p.valves.CONTRIBUTOR_GROUP = "contributors, research group"
    _groups(monkeypatch, ["research group"])
    assert _may(p, {"id": "u", "role": "user"})
    _groups(monkeypatch, ["research Group "])                     # case/whitespace typed differently in the admin UI
    assert _may(p, {"id": "u", "role": "user"})
    _groups(monkeypatch, ["Users"])
    assert not _may(p, {"id": "u", "role": "user"})


# ── members may only upload: no server paths, a hint for plain text ──
def _body(text):
    return {"messages": [{"role": "user", "content": text}]}


def _run_body(monkeypatch, tmp_path, user, text, files=(), seen=()):
    p = document.Ingester(ip.Pipe().valves)
    # the default library: its processed attachments are the manifest's "seen_file_ids" (per library; the tests' process
    # domain is polymer, so the collection is named here)
    p.valves.COLLECTION = "memento-comp-mat-rag"
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text(__import__("json").dumps({"seen_file_ids": list(seen), "documents": {}}))
    touched = []
    monkeypatch.setattr(workflow, "process_docs", lambda ing, *a_, **k_: (lambda docs: touched.append(docs) or iter(["processed"]))(*a_, **k_))
    monkeypatch.setattr(workflow, "run_batch", lambda ing, *a_, **k_: (lambda docs: touched.append(docs) or iter(["processed"]))(*a_, **k_))
    return "".join(str(x) for x in workflow.pipe_body(p, _body(text), user, list(files))), touched


def test_member_text_gets_the_upload_hint_and_server_paths_are_never_read(monkeypatch, tmp_path):
    folder = tmp_path / "someone_elses_papers"
    folder.mkdir()
    (folder / "private.pdf").write_bytes(b"%PDF-1.4")
    out, touched = _run_body(monkeypatch, tmp_path, {"id": "u", "role": "user"}, str(folder))
    assert out.startswith(messages.ONLY_UPLOADS_MSG) and not touched
    out, _ = _run_body(monkeypatch, tmp_path, {"id": "u", "role": "user"}, "test")
    assert "Not found" not in out and "Upload Files" in out and "Library Chat" in out


def test_member_follow_up_text_does_not_relist_earlier_uploads(monkeypatch, tmp_path):
    earlier = [{"id": "f1", "name": "paper.pdf", "file": {"path": "/nonexistent"}}]
    out, touched = _run_body(monkeypatch, tmp_path, {"id": "u", "role": "user"}, "thanks", earlier, seen=["f1"])
    assert out.startswith("Library Upload doesn't answer messages") and out.endswith(messages.EARLIER_UPLOADS_NOTE)
    assert "skipped" not in out
    assert not touched


def test_admin_keeps_the_typed_path_mode(monkeypatch, tmp_path):
    out, _ = _run_body(monkeypatch, tmp_path, {"id": "a", "role": "admin"}, "no_such_file_xyz")
    assert "Not found: `no_such_file_xyz`" in out


def test_text_sent_with_an_upload_is_flagged_as_not_used(monkeypatch, tmp_path):
    # files + a typed request: the files are added, and the reply says the text is not carried out
    n = iter(range(100))

    def run(content):
        p = document.Ingester(ip.Pipe().valves)
        monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
        monkeypatch.setattr(workflow, "handle_uploads", lambda ing, *a_, **k_: (lambda files, user, *a: iter(["stored\n"]))(*a_, **k_))
        files = [{"id": f"new{next(n)}", "name": "paper.pdf", "file": {"path": "/x"}}]   # a new upload each call
        return "".join(workflow.pipe_body(p, {"messages": [{"role": "user", "content": content}]}, {"id": "u", "role": "user"},
                                    files))
    # the request is handed over to Library Chat (a link that asks it there)
    assert "[Ask it in Library Chat](" in run("please add these and summarise them")
    assert "[Ask it in Library Chat](" in run([{"type": "text", "text": "summarise"}])
    assert "Ask it in Library Chat" not in run("") and "Ask it in Library Chat" not in run("   ")


def test_openwebui_attached_files_block_is_not_user_text():
    # OpenWebUI 0.11 prepends this block to every message with files (utils/middleware.py add_file_context)
    block = '<attached_files>\n<file type="file" id="f1" name="paper.pdf"/>\n</attached_files>\n\n'
    assert messages.last_text({"messages": [{"role": "user", "content": block}]}) == ""
    assert messages.last_text({"messages": [{"role": "user", "content": block + "summarise"}]}) == "summarise"
    assert messages.last_text({"messages": [{"role": "user", "content": [{"type": "text", "text": block},
                                                                   {"type": "text", "text": ""}]}]}) == ""


# ── chat titles: the pipe answers Open WebUI's title task itself (file names, else date and time) ──

def test_upload_title_names_the_files():
    assert messages.upload_title(["polyBERT.pdf"]) == "Upload: polyBERT"
    assert messages.upload_title(["a.pdf", "b.docx", "a.pdf", "c.png"]) == "Upload: a (+2 more)"
    long = messages.upload_title(["x" * 80 + ".pdf", "y.pdf"])
    assert long.startswith("Upload: " + "x" * 59 + "…") and long.endswith("(+1 more)")


def test_upload_title_falls_back_to_date_and_time():
    from datetime import datetime
    assert messages.upload_title([], now=datetime(2026, 9, 29, 14, 5)) == "Library upload, 29 Sep 2026 14:05"
    assert messages.upload_title(["", None], now=datetime(2026, 9, 29, 9, 0)) == "Library upload, 29 Sep 2026 09:00"


def test_title_task_is_answered_without_ingesting(monkeypatch):
    import json as _json
    async def names(chat_id):
        assert chat_id == "c1"
        return ["Macromolecular Rapid Communications - Becker.pdf", "s41467-023-39868-6.pdf"]
    monkeypatch.setattr(intake, "chat_file_names", names)
    async def never(*a, **k):
        raise AssertionError("a title task must not reach the ingestion path")
    p = ip.Pipe()
    monkeypatch.setattr(intake, "may_ingest", lambda valves, *a_, **k_: never(*a_, **k_))
    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "### Task: title"}]},
                                        {"id": "u", "role": "user"}, __task__="title_generation", __chat_id__="c1")]
    out = asyncio.run(run())
    assert _json.loads("".join(out)) == {"title": "Upload: Macromolecular Rapid Communications - Becker (+1 more)"}


def test_chat_file_names_unreadable_chat_is_empty():
    assert asyncio.run(intake.chat_file_names(None)) == []
    assert asyncio.run(intake.chat_file_names("local:abc")) == []


# ── only the files attached to the message being sent ──
# Open WebUI sends every file attached anywhere in the chat with each request. A file attached for Library Chat (or a
# private one for Document Chat) earlier in the same chat must not be added when the chat is switched to Library Upload.

def test_current_message_files_are_split_from_earlier_ones():
    f1 = {"id": "f1", "name": "from_library_chat.pdf"}
    f2 = {"id": "f2", "name": "new.pdf"}
    meta = {"user_message": {"id": "m2", "role": "user", "files": [{"type": "file", "id": "f2", "name": "new.pdf"}]}}
    assert intake.current_message_files(meta, [f1, f2]) == ([f2], [f1])
    no_attachment = {"user_message": {"id": "m3", "role": "user", "content": "add it"}}   # OpenWebUI omits "files"
    assert intake.current_message_files(no_attachment, [f1, f2]) == ([], [f1, f2])
    assert intake.current_message_files(None, [f1, f2]) == ([f1, f2], [])                    # API call: as before
    assert intake.current_message_files({}, [f1]) == ([f1], [])


def test_files_attached_to_earlier_messages_are_not_added(monkeypatch, tmp_path):
    earlier = [{"id": "f1", "name": "polymers-15-03907.pdf", "file": {"path": "/x"}}]
    p = document.Ingester(ip.Pipe().valves)
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text('{"seen_file_ids": [], "documents": {}}')
    called = []
    monkeypatch.setattr(workflow, "handle_uploads", lambda ing, *a_, **k_: (lambda files, user, *a: called.append(files) or iter(["stored\n"]))(*a_, **k_))
    out = "".join(workflow.pipe_body(p, _body(""), {"id": "u", "role": "user"}, [], earlier=earlier))
    assert not called
    assert out.startswith("Library Upload doesn't answer messages") and messages.EARLIER_ATTACHMENTS_NOTE in out
    new = [{"id": "f2", "name": "new.pdf", "file": {"path": "/y"}}]
    out = "".join(workflow.pipe_body(p, _body(""), {"id": "u", "role": "user"}, new, earlier=earlier))
    assert [f["id"] for f in called[0]] == ["f2"] and messages.EARLIER_ATTACHMENTS_NOTE in out


def test_pipe_passes_only_the_current_message_files(monkeypatch):
    p = ip.Pipe()
    async def yes(user):
        return True
    monkeypatch.setattr(intake, "may_ingest", lambda valves, *a_, **k_: yes(*a_, **k_))
    seen = {}
    def fake_sync(body, user, files, earlier=()):
        seen.update(files=[f["id"] for f in files], earlier=[f["id"] for f in earlier])
        return iter(["ok"])
    monkeypatch.setattr(workflow, "pipe_sync", lambda ing, *a_, **k_: (fake_sync)(*a_, **k_))

    async def all_owned(files, user, lookup=None):          # ownership is tested on its own
        return list(files), []
    monkeypatch.setattr(intake, "owned_uploads", all_owned)
    meta = {"user_message": {"id": "m2", "role": "user", "files": [{"type": "file", "id": "f2"}]}}
    files = [{"id": "f1", "name": "a.pdf"}, {"id": "f2", "name": "b.pdf"}]
    async def run():
        return [c async for c in p.pipe(_body(""), {"id": "u", "role": "user"}, files, __metadata__=meta)]
    assert asyncio.run(run()) == ["ok"] and seen == {"files": ["f2"], "earlier": ["f1"]}


def test_earlier_uploads_note_also_for_files_from_earlier_messages(monkeypatch, tmp_path):
    # after the split, files of earlier Library Upload messages arrive as "earlier", not as the message's own files
    p = document.Ingester(ip.Pipe().valves)
    p.valves.COLLECTION = "memento-comp-mat-rag"          # the default library: the manifest's "seen_file_ids"
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text('{"seen_file_ids": ["f1"], "documents": {}}')
    earlier = [{"id": "f1", "name": "paper.pdf", "file": {"path": "/x"}}]
    out = "".join(workflow.pipe_body(p, _body("thanks"), {"id": "u", "role": "user"}, [], earlier=earlier))
    assert out.endswith(messages.EARLIER_UPLOADS_NOTE) and messages.EARLIER_ATTACHMENTS_NOTE not in out


# ── the upload reply points on to Library Chat ──
def test_the_upload_reply_links_each_new_document_to_its_summary_and_limitations():
    from urllib.parse import unquote
    out = messages.ask_links(["Inverse Modeling through Machine Learning"])
    assert "[Summarize it](/?model=memento-query&q=" in out and "[Limitations & future work](" in out
    links = [unquote(u) for u in __import__("re").findall(r"\]\((/\?model=memento-query&q=[^)]+)\)", out)]
    assert links == ['/?model=memento-query&q=Summarize the document "Inverse Modeling through Machine Learning"',
                     '/?model=memento-query&q=What research gaps does the document '
                     '"Inverse Modeling through Machine Learning" name?']   # the picker's own wordings: stored answers
    assert " " not in __import__("re").findall(r"\]\(([^)]+)\)", out)[0]   # the link itself is encoded
    many = messages.ask_links([f"Doc {i}" for i in range(5)])
    assert "[open Library Chat](/?model=memento-query)" in many and "Summarize it" not in many
    assert messages.ask_links([]) == ""
    tricky = messages.ask_links(["A *bold* [x] title"])
    assert "**A \\*bold\\* \\[x\\] title**" in tricky


def test_the_upload_only_hint_links_to_library_chat():
    # the Library Chat link comes with the hint (OPEN_CHAT_LINE without typed text, else "Ask it in Library Chat")
    assert "[Library Chat](/?model=memento-query)" in messages.ask_line("")
    assert "in the model menu" not in messages.ONLY_UPLOADS_MSG + messages.ask_line("")


# ── Library Upload hands a question over to Library Chat ──
from urllib.parse import parse_qs, urlparse  # noqa: E402
from memento.ingestion.upload import workflow


def _ask_url(out):
    m = __import__("re").search(r"\[Ask it in Library Chat\]\(([^)]+)\)", out)
    return m and m.group(1)


def test_a_question_without_a_file_gets_a_link_that_asks_it_in_library_chat(monkeypatch, tmp_path):
    out, touched = _run_body(monkeypatch, tmp_path, {"id": "u", "role": "user"}, "Summarise the paper on bioplastics")
    url = _ask_url(out)
    assert url and url.startswith("/?model=memento-query&q=")
    assert parse_qs(urlparse(url).query)["q"] == ["Summarise the paper on bioplastics"]
    assert "[Document Chat](/?model=memento-document-chat)" in out and not touched


def test_no_text_no_question_link(monkeypatch, tmp_path):
    earlier = [{"id": "f1", "name": "paper.pdf", "file": {"path": "/nonexistent"}}]
    out, _ = _run_body(monkeypatch, tmp_path, {"id": "u", "role": "user"}, "", earlier, seen=["f1"])
    assert _ask_url(out) is None and "Library Chat" in out


def _upload_run(monkeypatch, tmp_path, content, stems, titles):
    p = document.Ingester(ip.Pipe().valves)
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")

    def handle(files, user, stems_out=None):
        if stems_out is not None:
            stems_out.extend(stems)
        yield "handled\n"
    monkeypatch.setattr(workflow, "handle_uploads", lambda ing, *a_, **k_: (handle)(*a_, **k_))
    monkeypatch.setattr(p, "stored_titles", lambda s: [titles.get(x, x) for x in s])
    files = [{"id": f"n{k}", "name": f"{s}.pdf", "file": {"path": "/x"}} for k, s in enumerate(stems or ["x"])]
    return "".join(workflow.pipe_body(p, {"messages": [{"role": "user", "content": content}]}, {"id": "u", "role": "user"},
                                files))


def test_a_question_sent_with_one_upload_is_asked_about_that_document(monkeypatch, tmp_path):
    out = _upload_run(monkeypatch, tmp_path, "Summarise this paper to me", ["s43246"],
                      {"s43246": "Bioplastic design using multitask deep neural networks"})
    q = parse_qs(urlparse(_ask_url(out)).query)["q"][0]
    assert q == 'Summarise this paper to me (the document "Bioplastic design using multitask deep neural networks")'
    assert "isn't used here" not in out


def test_a_question_sent_with_several_uploads_names_them(monkeypatch, tmp_path):
    # up to ASK_NAMED_MAX uploaded documents are named in the handed-over question
    out = _upload_run(monkeypatch, tmp_path, "Compare these", ["a", "b"], {})
    assert parse_qs(urlparse(_ask_url(out)).query)["q"] == ['Compare these (the documents "a", "b")']


def test_a_file_already_in_the_library_gets_the_ask_links(monkeypatch, tmp_path):
    p = document.Ingester(ip.Pipe().valves)
    monkeypatch.setattr(p, "input_dir", lambda: tmp_path / "in")
    src = tmp_path / "s43246.pdf"
    src.write_bytes(b"%PDF-1.4 x")
    dup = __import__("types").SimpleNamespace(action="duplicate", name="s43246.pdf", stem="s43246", filename="s43246.pdf",
                                              path=src)
    monkeypatch.setattr(p, "decide", lambda items, uploader, tag: [dup])
    monkeypatch.setattr(p, "stored_titles", lambda s: ["Bioplastic design" for _ in s])
    stems = []
    out = "".join(workflow.handle_uploads(p, [{"name": "s43246.pdf", "file": {"path": str(src)}}], {"id": "u"}, stems))
    assert "already in the library as **s43246**" in out and "Nothing to process" not in out
    assert "Ask about **Bioplastic design**: [Summarize it]" in out and stems == ["s43246"]


# ── uploads come only from the uploader's own Open WebUI files ──
# The pipe's file list comes from the browser: a forged entry could name any server-readable path. Only files whose Open
# WebUI record belongs to the user are taken, from the path Open WebUI stored.
def _rec(uid, path, filename="paper.pdf"):
    return __import__("types").SimpleNamespace(user_id=uid, path=path, filename=filename, meta={})


def _owned(files, user, records):
    async def lookup(fid):
        return records.get(fid)
    return __import__("asyncio").run(intake.owned_uploads(files, user, lookup=lookup))


def test_a_forged_path_is_never_used():
    files = [{"id": "f1", "name": "paper.pdf", "file": {"path": "/home/x/memento/config/.env"}}]
    kept, refused = _owned(files, {"id": "u1"}, {"f1": _rec("u1", "/srv/uploads/f1_paper.pdf")})
    assert kept[0]["file"]["path"] == "/srv/uploads/f1_paper.pdf" and refused == []


def test_someone_elses_or_unknown_file_is_refused():
    files = [{"id": "f2", "name": "theirs.pdf", "file": {"path": "/srv/uploads/f2.pdf"}},
             {"id": "f3", "name": "ghost.pdf", "file": {"path": "/etc/hosts"}},
             {"name": "no_id.pdf", "file": {"path": "/etc/hosts"}}]
    kept, refused = _owned(files, {"id": "u1"}, {"f2": _rec("u2", "/srv/uploads/f2.pdf")})
    assert kept == [] and refused == ["theirs.pdf", "ghost.pdf", "no_id.pdf"]


def test_the_pipe_checks_ownership_before_anything_is_read(monkeypatch, tmp_path):
    p = ip.Pipe()
    seen = []

    async def may(user):
        return True

    async def owned(files, user, lookup=None):
        return [], [f.get("name") for f in files]
    monkeypatch.setattr(intake, "may_ingest", lambda valves, *a_, **k_: may(*a_, **k_))
    monkeypatch.setattr(intake, "owned_uploads", owned)
    monkeypatch.setattr(workflow, "pipe_sync", lambda ing, *a_, **k_: (lambda body, user, files, earlier=(): seen.append(files) or iter([]))(*a_, **k_))

    async def go():
        files = [{"id": "f9", "name": "x.pdf", "file": {"path": "/etc/hosts"}}]
        return "".join([c async for c in p.pipe({"messages": [{"role": "user", "content": ""}]},
                                                  {"id": "u1", "role": "user"}, files, None, None, "c1",
                                                  {"user_message": {"files": files}})])
    out = __import__("asyncio").run(go())
    assert "x.pdf" in out and "unable to access" in out and seen and seen[0] == []
