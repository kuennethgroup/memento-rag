# Deployment mode of scripts/setup_openwebui.py (offline, requests mocked).
import importlib.util

import pytest
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("so", PROJECT_ROOT / "scripts/setup_openwebui.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)


def test_trial_valves_for_generic_domain():
    q = so.trial_valves("memento_query", "generic", {})
    i = so.trial_valves("memento_ingestion", "generic", {"MINERU_BACKEND": "vlm-engine"})
    assert q == {"DOMAIN_NAME": "generic", "COLLECTION": "memento-generic-rag", "GRAPH_RETRIEVAL": False,
                 "THINKING_AUTO": True, "VALIDATION_ENABLED": True, "VALIDATION_STRATEGY": "A",
                 "VALIDATION_MAX_ROUNDS": 2,                           # RQ3's best configuration
                 # the answer model of config/.env, and the load its warm-up reads
                 "LLM_BASE_URL": "http://localhost:8023/v1", "LLM_MODEL": "qwen-flash-next",
                 "LLM_API_KEY_ENV": "REASONING_API_KEY", "METRICS_URL": "http://localhost:8023/metrics",
                 "METRICS_API_KEY_ENV": "REASONING_API_KEY"}
    assert so.trial_valves("memento_thorough", "generic", {}) == {}
    assert i["COLLECTION"] == "memento-generic-rag" and i["SKIP_PARSE"] is False and i["FORCE_REEMBED"] is False
    assert i["BACKEND"] == "vlm-engine" and i["CONTRIBUTOR_GROUP"] == "contributors, members"
    assert (i["LLM_BASE_URL"], i["LLM_MODEL"], i["LLM_API_KEY_ENV"]) == (q["LLM_BASE_URL"], q["LLM_MODEL"],
                                                                         q["LLM_API_KEY_ENV"])   # one answer model
    assert so.trial_valves("qdrant_manager", "generic", {}) == {}


def test_models_grant_read_to_contributors_and_qdrant_manager_stays_admin_only(monkeypatch):
    sent = []

    class R:
        status_code = 200
        def json(self): return {"ok": True}
    monkeypatch.setattr(so.requests, "post", lambda url, json=None, **kw: sent.append(json) or R())
    for m in so.MODELS:
        so.create_model("http://x", {}, m, {"contributors": "gid-1", "members": "gid-2"})
    grants = {p["id"]: p["access_grants"] for p in sent}
    want = [{"principal_type": "group", "principal_id": g, "permission": "read"} for g in ("gid-1", "gid-2")]
    assert grants["memento-query"] == want and grants["memento-ingestion-prod"] == want
    assert grants["qdrant-manager"] == []
    # the raw pipes under the models: same groups (Open WebUI 0.11 checks the whole base-model chain), hidden; the query
    # pipe is a manifold, one raw model per variant (memento_query.<id>)
    raw_ids = ["memento_ingestion"] + [f"memento_query.{v}" for v in ("local", "qwen35", "qwen36")]
    assert grants["memento-document-chat"] == grants["memento_query.document-chat"] == want
    assert all(grants[i] == want for i in raw_ids) and all(grants[f"memento-query-{v}"] == want
                                                            for v in ("qwen35", "qwen36"))
    raw = {p["id"]: p for p in sent if p["id"] in raw_ids}
    assert all(p["meta"]["hidden"] is True and p["base_model_id"] is None for p in raw.values())
    assert all(p["meta"]["capabilities"]["file_context"] is False for p in sent)


def test_document_list_is_saved_then_emptied(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from memento.library import documents as db
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    for src in ("paper_a", "paper_b"):
        db.add_document(e, collection="col", entry={"source": src, "title": src}, content_hash="h", uploaded_by="u",
                        output_root=None)
    db.add_document(e, collection="other", entry={"source": "z", "title": "z"}, content_hash="h", uploaded_by="u",
                    output_root=None)
    monkeypatch.setattr(so, "PROJECT_ROOT", tmp_path)
    env = {"DATABASE_URL": "postgresql://x", "OUTPUT_ROOT": str(tmp_path / "out")}
    so.reset_document_list(env, True, "col", engine=e)
    assert len(db.list_documents(e, "col")) == 2 and not (tmp_path / "data").exists()
    so.reset_document_list(env, False, "col", engine=e)
    assert db.list_documents(e, "col") == [] and [r["source"] for r in db.list_documents(e, "other")] == ["z"]
    (backup,) = (tmp_path / "data" / "rq4").glob("memento_documents_backup_*.csv")
    lines = backup.read_text().splitlines()
    assert lines[0].startswith("collection,source,title") and [l.split(",")[1] for l in lines[1:]] == ["paper_a", "paper_b"]


def test_trial_ui_changes_only_the_decided_settings(monkeypatch):
    # the deployment's settings; every other setting read back unchanged
    state = {
        "/api/v1/users/default/permissions": {"chat": {"controls": True, "file_upload": True, "stt": True, "call": True},
                                              "features": {"notes": True, "folders": True}, "workspace": {"models": False}},
        "/api/v1/auths/admin/config": {"ENABLE_SIGNUP": False, "ENABLE_NOTES": True, "ENABLE_FOLDERS": True,
                                       "ENABLE_MESSAGE_RATING": True},
        "/api/v1/configs/models": {"DEFAULT_MODELS": None, "MODEL_ORDER_LIST": []},
        "/api/v1/tasks/config": {"ENABLE_FOLLOW_UP_GENERATION": True, "ENABLE_TAGS_GENERATION": True,
                                 "ENABLE_TITLE_GENERATION": True},
        "/api/v1/configs/banners": [{"id": "other", "type": "info", "content": "x", "dismissible": True, "timestamp": 1}],
        "/api/v1/groups/": [{"id": "g1"}, {"id": "g2"}],
        "/api/v1/groups/id/g1": {"name": "Users", "description": "d", "data": None,
                                 "permissions": {"chat": {"controls": True, "edit": True}, "features": {"notes": True}}},
        "/api/v1/groups/id/g2": {"name": "members", "description": "", "data": None, "permissions": None},
    }
    for m in so.MODELS:
        state[f"/api/v1/models/model?id={m['id']}"] = {"id": m["id"], "name": m["name"], "meta": {"description": "old", "x": 1},
                                                      "params": {}, "access_grants": []}
    posted = {}

    class R:
        status_code = 200
        def __init__(self, body): self.body = body
        def json(self): return self.body
        def raise_for_status(self): pass
    monkeypatch.setattr(so.requests, "get", lambda url, **kw: R(state[url.replace("http://x", "")]))
    monkeypatch.setattr(so.requests, "post", lambda url, json=None, **kw: posted.__setitem__(url.replace("http://x", ""), json) or R({}))
    so.apply_trial_ui("http://x", {}, dry=False)
    perms = posted["/api/v1/users/default/permissions"]
    # dictation off (plain HTTP: browsers refuse the microphone)
    assert perms["chat"] == {**perms["chat"], "controls": False, "file_upload": True, "stt": False, "call": False}
    assert perms["features"]["folders"] is True and perms["workspace"] == {"models": False}
    cfg = posted["/api/v1/auths/admin/config"]
    assert cfg["ENABLE_NOTES"] is False and cfg["ENABLE_FOLDERS"] is True and cfg["ENABLE_MESSAGE_RATING"] is True
    assert posted["/api/v1/configs/models"]["DEFAULT_MODELS"] == "memento-query"
    tasks = posted["/api/v1/tasks/config/update"]
    assert tasks["ENABLE_FOLLOW_UP_GENERATION"] is False and tasks["ENABLE_TAGS_GENERATION"] is False
    assert tasks["ENABLE_TITLE_GENERATION"] is True
    assert [b["id"] for b in posted["/api/v1/configs/banners"]["banners"]] == ["other", "memento-trial"]
    g1 = posted["/api/v1/groups/id/g1/update"]["permissions"]          # a group's own permissions add up with the defaults
    assert g1["chat"]["controls"] is False and g1["chat"]["edit"] is True and g1["features"]["notes"] is False
    assert "/api/v1/groups/id/g2/update" not in posted                    # no own permissions: defaults apply
    q = posted["/api/v1/models/model/update?id=memento-query"]["meta"]
    assert q["description"].startswith("Ask about the shared library") and q["x"] == 1


# ── Memento Query model variants ──
def test_query_variants_become_workspace_models():
    ids = {m["id"]: m for m in so.MODELS}
    local = ids["memento-query"]
    assert local["base_model_id"] == "memento_query.local" and local["name"] == "Library Chat (Qwen 3.8 Flash-Next)"
    assert "suggestion_prompts" in local["meta"] and local["meta"]["filterIds"] == ["memento_thorough"]
    for vid in ("qwen35", "qwen36"):
        m = ids[f"memento-query-{vid}"]
        assert m["base_model_id"] == f"memento_query.{vid}" and m["meta"]["filterIds"] == ["memento_thorough"]
        assert m["meta"]["description"].endswith("Runs on a cloud service.") and "suggestion_prompts" not in m["meta"]
        assert m["groups"] == so.UPLOAD_GROUPS and m["meta"]["capabilities"]["file_context"] is False
        raw = ids[f"memento_query.{vid}"]
        assert raw["meta"]["hidden"] is True and raw["groups"] == so.UPLOAD_GROUPS
    assert ids["memento_query.local"]["meta"]["hidden"] is True
    assert "memento_query" not in ids and ids["memento_ingestion"]["meta"]["hidden"] is True
    # names users see: Library Chat / Document Chat / Library Upload; ids unchanged
    assert ids["memento-ingestion-prod"]["name"] == "Library Upload"
    # local entries first, then the cloud Query models
    assert so.MODEL_ORDER == ["memento-query", "memento-document-chat", "memento-ingestion-prod",
                              "memento-query-qwen35", "memento-query-qwen36"]


def test_document_chat_is_a_model_of_its_own():
    # Memento Document Chat: its workspace model wraps the Query manifold's document-chat entry; the raw entry is hidden
    # and grants the same groups
    ids = {m["id"]: m for m in so.MODELS}
    dc, raw = ids["memento-document-chat"], ids["memento_query.document-chat"]
    assert dc["name"] == "Document Chat" and dc["base_model_id"] == "memento_query.document-chat"
    assert "deleted when you delete the chat" in dc["meta"]["description"]
    assert "only the local model" in dc["meta"]["description"]
    assert dc["meta"]["capabilities"] == {"vision": False, "file_context": False}
    # its own start-screen suggestions (else Open WebUI shows its defaults)
    prompts = dc["meta"]["suggestion_prompts"]
    assert len(prompts) == 3 and all("attached file" in x["content"] for x in prompts)
    # the upload groups
    assert so.DOC_CHAT_GROUPS == so.UPLOAD_GROUPS and dc["groups"] == raw["groups"] == so.UPLOAD_GROUPS
    assert raw["meta"]["hidden"] is True and raw["base_model_id"] is None


def test_trial_ui_dry_run_before_the_new_models_exist(monkeypatch, capsys):
    """A dry run on a live OpenWebUI where the cloud variants are not created yet: no crash, they show as new."""
    import requests
    ok = {"/api/v1/users/default/permissions": {}, "/api/v1/auths/admin/config": {}, "/api/v1/tasks/config": {},
          "/api/v1/configs/models": {"DEFAULT_MODELS": "memento-query", "MODEL_ORDER_LIST": []},
          "/api/v1/configs/banners": [], "/api/v1/groups/": [],
          "/api/v1/models/model?id=memento-query": {"id": "memento-query", "meta": {"description": "old"}},
          "/api/v1/models/model?id=memento-ingestion-prod": {"id": "memento-ingestion-prod", "meta": {"description": "old"}}}

    class R:
        def __init__(self, url):
            self.body, self.status_code = ok.get(url), 200 if url in ok else 404
        def json(self): return self.body
        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError(f"{self.status_code}")
    monkeypatch.setattr(so.requests, "get", lambda url, **kw: R(url.replace("http://x", "")))
    monkeypatch.setattr(so.requests, "post", lambda *a, **kw: pytest.fail("a dry run posts nothing"))
    so.apply_trial_ui("http://x", {}, dry=True)
    assert "Library Chat (Qwen 3.5 397B): new, created with its description" in capsys.readouterr().out


def test_sync_models_writes_only_models_and_their_order(monkeypatch):
    """Run at every Open WebUI start (services.sh): config/query_models.yaml → the picker. Models with group access,
    default model and order; permissions, banner, sign-up and the document list are left alone."""
    configs = {"DEFAULT_MODELS": "memento-query", "MODEL_ORDER_LIST": ["memento-query", "memento-ingestion-prod"], "x": 1}
    gets = {"/api/v1/groups/": [{"id": "gid-1", "name": "contributors"}, {"id": "gid-2", "name": "members"}],
            "/api/v1/configs/models": configs}
    posted = []

    class R:
        status_code = 200
        def __init__(self, body, status=200): self.body, self.status_code = body, status
        def json(self): return self.body
        def raise_for_status(self): pass
    # anything else (the second copy's functions): not installed
    monkeypatch.setattr(so.requests, "get", lambda url, **kw: R(gets[url.replace("http://x", "")])
                        if url.replace("http://x", "") in gets else R(None, 404))
    monkeypatch.setattr(so.requests, "post",
                        lambda url, json=None, **kw: posted.append((url.replace("http://x", ""), json)) or R({"ok": True}))
    so.cmd_sync_models("http://x", {}, dry=False)
    urls = {u.split("?")[0] for u, _ in posted}
    assert urls == {"/api/v1/models/create", "/api/v1/configs/models"}
    created = {b["id"] for u, b in posted if u == "/api/v1/models/create"}
    assert set(so.MODEL_ORDER) <= created
    order = next(b for u, b in posted if u == "/api/v1/configs/models")
    assert order["MODEL_ORDER_LIST"] == so.MODEL_ORDER and order["DEFAULT_MODELS"] == "memento-query" and order["x"] == 1


def test_sync_models_dry_run_posts_nothing(monkeypatch):
    gets = {"/api/v1/groups/": [], "/api/v1/configs/models": {"DEFAULT_MODELS": None, "MODEL_ORDER_LIST": []}}

    class R:
        def __init__(self, body, status_code=200): self.body, self.status_code = body, status_code
        def json(self): return self.body
        def raise_for_status(self): pass
    # anything else (the second copy's functions) does not exist: 404 (a failed read is reported, not taken as absence)
    monkeypatch.setattr(so.requests, "get", lambda url, **kw: R(gets[url.replace("http://x", "")])
                        if url.replace("http://x", "") in gets else R(None, 404))
    monkeypatch.setattr(so.requests, "post", lambda *a, **kw: pytest.fail("a dry run posts nothing"))
    so.cmd_sync_models("http://x", {}, dry=True)


def test_no_model_accepts_images_in_the_chat_box():
    # image uploads off, UI side only: Open WebUI's chat box accepts images only for models with the vision capability;
    # the answer pipe's own figure handling (config/query_models.yaml vision) is unchanged
    users = [m for m in so.MODELS if m["id"] in so.MODEL_ORDER]
    assert users and all(m["meta"]["capabilities"]["vision"] is False for m in users)
    assert all("image" not in m["meta"].get("description", "").lower() for m in users)


def test_library_chat_models_take_no_attachments():
    # Library Chat has no attach button: Open WebUI greys out "Upload Files" (and refuses drops and pastes) for a model
    # whose capabilities say file_upload: False; Document Chat and Library Upload keep theirs
    ids = {m["id"]: m for m in so.MODELS}
    for mid in ("memento-query", "memento-query-qwen35", "memento-query-qwen36"):
        assert ids[mid]["meta"]["capabilities"]["file_upload"] is False, mid
    for mid in ("memento-document-chat", "memento-ingestion-prod"):
        assert ids[mid]["meta"]["capabilities"].get("file_upload", True) is True, mid
