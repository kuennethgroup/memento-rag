# A small stateful stand-in for Open WebUI 0.11.4's admin API, as scripts/setup_openwebui.py uses it: functions (code,
# Valves, active flag; created inactive, as Open WebUI does), models, groups and the default-model config. `fail`
# injects faults by route.
import copy
import types
from urllib.parse import parse_qs, urlparse


NOT_FOUND = "We could not find what you're looking for :/"     # OpenWebUI 0.11.4's answer for a missing item


class FakeOpenWebUI:
    def __init__(self):
        self.functions: dict = {}
        self.models: dict = {}
        self.groups: dict = {}
        self.config = {"DEFAULT_MODELS": "memento-query", "MODEL_ORDER_LIST": [], "x": 1}
        self.fail: set = set()             # {"toggle"} (HTTP 500) or {"toggle-noop"} (200, nothing changes)
        self.calls: list = []

    def snapshot(self):
        return copy.deepcopy({"functions": self.functions, "models": self.models, "groups": self.groups,
                              "config": self.config})

    # ── the HTTP layer ──
    def _r(self, status, body=None):
        r = types.SimpleNamespace(status_code=status, text=str(body))
        r.json = lambda: body
        r.ok = status < 400

        def raise_for_status():
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
        r.raise_for_status = raise_for_status
        return r

    def install(self, monkeypatch, module):
        monkeypatch.setattr(module.requests, "get", lambda url, **kw: self.request("GET", url, **kw))
        monkeypatch.setattr(module.requests, "post", lambda url, **kw: self.request("POST", url, **kw))
        monkeypatch.setattr(module.requests, "delete", lambda url, **kw: self.request("DELETE", url, **kw))

    def request(self, method, url, json=None, **kw):
        u = urlparse(url)
        path, q = u.path, {k: v[0] for k, v in parse_qs(u.query).items()}
        self.calls.append((method, path, json))
        p = path.split("/")
        if path.startswith("/api/v1/functions/"):
            return self._functions(method, p[4:], json)
        if path.startswith("/api/v1/models/"):
            return self._models(method, path, q, json)
        if path.startswith("/api/v1/groups/"):
            return self._groups(method, p[4:], json)
        if path == "/api/v1/configs/models":
            if method == "GET":
                return self._r(200, copy.deepcopy(self.config))
            self.config = copy.deepcopy(json)
            return self._r(200, self.config)
        return self._r(404, {"detail": "not found"})

    def _functions(self, method, rest, body):
        if rest == ["create"] and method == "POST":
            if body["id"] in self.functions:
                return self._r(400, {"detail": "exists"})
            self.functions[body["id"]] = {"id": body["id"], "name": body["name"], "content": body["content"],
                                          "meta": body.get("meta"), "valves": None, "is_active": False}
            return self._r(200, dict(self.functions[body["id"]]))
        if len(rest) >= 2 and rest[0] == "id":
            fid, action = rest[1], (rest[2] if len(rest) > 2 else None)
            f = self.functions.get(fid)
            if f is None:
                return self._r(404, {"detail": "not found"})
            if action is None and method == "GET":
                return self._r(200, dict(f))
            if action == "valves" and len(rest) == 3 and method == "GET":
                return self._r(200, copy.deepcopy(f["valves"]) if f["valves"] is not None else None)
            if action == "valves" and len(rest) == 4 and method == "POST":
                f["valves"] = copy.deepcopy(body)
                return self._r(200, f["valves"])
            if action == "update" and method == "POST":
                f.update(name=body["name"], content=body["content"], meta=body.get("meta"))
                return self._r(200, dict(f))
            if action == "toggle" and method == "POST":
                if "toggle" in self.fail:
                    return self._r(500, {"detail": "boom"})
                if "toggle-noop" not in self.fail:
                    f["is_active"] = not f["is_active"]
                return self._r(200, dict(f))
            if action == "delete" and method == "DELETE":
                del self.functions[fid]
                return self._r(200, True)
        return self._r(404, {"detail": "not found"})

    def _models(self, method, path, q, body):
        if path == "/api/v1/models/create" and method == "POST":
            if body["id"] in self.models:
                return self._r(400, None)
            self.models[body["id"]] = copy.deepcopy(body)
            return self._r(200, body)
        if path == "/api/v1/models/model/update" and method == "POST":
            if q.get("id") not in self.models:
                return self._r(400, None)
            self.models[q["id"]] = copy.deepcopy(body)
            return self._r(200, body)
        if path == "/api/v1/models/model/delete" and method == "POST":
            if body["id"] not in self.models:             # as Open WebUI: 401 with its "not found" text
                return self._r(401, {"detail": NOT_FOUND})
            del self.models[body["id"]]
            return self._r(200, True)
        if path == "/api/v1/models/model" and method == "GET":
            if q.get("id") not in self.models:
                return self._r(401, {"detail": NOT_FOUND})
            return self._r(200, copy.deepcopy(self.models[q["id"]]))
        return self._r(404, None)

    def _groups(self, method, rest, body):
        if rest == [""] and method == "GET":
            return self._r(200, [{"id": gid, "name": g["name"]} for gid, g in self.groups.items()])
        if rest == ["create"] and method == "POST":
            gid = f"g-{len(self.groups) + 1}-{body['name']}"
            self.groups[gid] = {"name": body["name"], "users": []}
            return self._r(200, {"id": gid, "name": body["name"]})
        if len(rest) >= 3 and rest[0] == "id":
            gid, action = rest[1], "/".join(rest[2:])
            g = self.groups.get(gid)
            if g is None:
                return self._r(404, None)
            if action == "delete" and method == "DELETE":
                del self.groups[gid]
                return self._r(200, True)
            if action == "users" and method == "POST":
                return self._r(200, [{"id": u} for u in g["users"]])
            if action == "users/add":
                g["users"] = sorted(set(g["users"]) | set(body["user_ids"]))
                return self._r(200, {"id": gid})
            if action == "users/remove":
                g["users"] = [u for u in g["users"] if u not in set(body["user_ids"])]
                return self._r(200, {"id": gid})
        return self._r(404, None)
