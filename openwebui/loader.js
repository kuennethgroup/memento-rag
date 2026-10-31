/* Memento additions to Open WebUI's pages, loaded as /static/loader.js (its custom-script slot); scripts/services.sh
   copies this file into open_webui/frontend/static/ and open_webui/static/ at every start. The citation side panel
   follows Open WebUI's theme: it is a page in a sandboxed cross-origin frame that cannot read the theme, so each panel
   frame is told the theme when it loads and whenever it changes. Open WebUI marks its dark themes with class "dark" on
   <html>. */
// Open WebUI translates its labels ("Selected model: X"), so the selected model is found by its id and its name is the
// label after the first colon. Shared by the blocks below (var: a second load of this file must not fail on a repeated
// declaration).
var SEL_BTN = '#model-selector-model-button';
var selName = (b) => (b.getAttribute('aria-label') || '').replace(/^[^:]*:\s*/, '');
// current names and the earlier ones (Memento Ingestion, Memento Document Chat, Memento Query), which older chats carry
var isIngest = (b) => /^(Library Upload|Memento Ingestion)\b/.test(selName(b));
// Document Chat: its own start steps, no library picker
var isDocChat = (b) => /^(Document Chat|Memento Document Chat)\b/.test(selName(b));

/* memento:models-begin — Two copies of Memento can share one Open WebUI, each with its own library. Their models
   have the same names, so the selected model is told by its id (Open WebUI keeps it in sessionStorage.selectedModels),
   and the pipes' model map (/files/models, "copies") says which copy an id belongs to: its links (siblings) and its
   library (collection). Without the map, the default ids keep their links and any other id waits. Pure functions only
   (no page access): tests/openwebui/js/loader_models.test.mjs runs this block in Node. */
var MementoModels = (function () {
  var TODAY = { query: 'memento-query', upload: 'memento-ingestion-prod', doc: 'memento-document-chat' };
  // the default copy's ids: its Library Chat models (config/query_models.yaml), Document Chat, Library Upload
  var TODAY_IDS = ['memento-query', 'memento-query-qwen35', 'memento-query-qwen36', 'memento-query-gemma',
                   'memento-document-chat', 'memento-ingestion-prod'];
  var LINK_RE = /(\/\?model=)(memento-query|memento-ingestion-prod|memento-document-chat)(?=$|[&#"')\s])/g;
  var KEYS = { 'memento-query': 'query', 'memento-ingestion-prod': 'upload', 'memento-document-chat': 'doc' };
  function selectedId(storage) {
    try {
      var s = JSON.parse((storage && storage.getItem('selectedModels')) || 'null');
      return Array.isArray(s) && typeof s[0] === 'string' && s[0] ? s[0] : null;
    } catch (e) { return null; }
  }
  function entryFor(map, id) {
    var copies = (map && map.copies) || {};
    if (!id) return null;
    for (var fid in copies) {
      var list = Array.isArray(copies[fid]) ? copies[fid] : [];
      for (var i = 0; i < list.length; i++) if (list[i] && list[i].id === id) return list[i];
    }
    return null;
  }
  function linksFor(map, id) {
    var e = entryFor(map, id);
    if (e && e.siblings) return e.siblings;
    return !id || TODAY_IDS.indexOf(id) >= 0 ? TODAY : null;        // null: wait until the map knows the id
  }
  function relink(href, links) {
    if (!links || !href) return href;
    return String(href).replace(LINK_RE, function (m, pre, id) { return pre + (links[KEYS[id]] || id); });
  }
  function collectionFor(map, id) { var e = entryFor(map, id); return (e && e.collection) || null; }
  // Open WebUI clears its stored choice on a fresh page (the selected model is unknown until a message or a change):
  // then the page is about the only copy whose models the user may use (`mine`: Open WebUI's /api/models ids); several
  // copies (an admin) or none: the default; the map or `mine` not read yet: PENDING, which waits like an unknown id
  var PENDING = 'memento:pending';
  function ownDefault(map, mine) {
    var copies = (map && map.copies) || {}, own = [];
    for (var fid in copies) {
      var list = Array.isArray(copies[fid]) ? copies[fid] : [];
      for (var i = 0; i < list.length; i++) if (list[i] && mine.indexOf(list[i].id) >= 0) { own.push(list); break; }
    }
    if (own.length !== 1) return null;
    var q = own[0].filter(function (e) { return e && e.kind === 'query'; })[0] || own[0][0];
    return (q && q.id) || null;
  }
  function effectiveId(stored, map, mine) {
    if (stored) return stored;
    if (!map || !mine) return PENDING;
    return ownDefault(map, mine);
  }
  // the library the picker reads for the selected model: its own once the map knows the model; the default list for the
  // default ids (or none selected) until then; any other id waits and reads nothing, never another library's list
  function pickerLibrary(map, id) {
    var e = entryFor(map, id);
    if (e) return { wait: false, collection: e.collection || null };
    return { wait: !!id && TODAY_IDS.indexOf(id) < 0, collection: null };
  }
  function libraryUrl(collection) {
    return collection ? '/files/library?collection=' + encodeURIComponent(collection) : '/files/library';
  }
  function kindOf(map, id) {
    var e = entryFor(map, id);
    if (e && e.kind) return e.kind === 'upload' ? 'ingest' : e.kind;
    return id === 'memento-ingestion-prod' ? 'ingest' : id === 'memento-document-chat' ? 'doc'
      : /^memento-query/.test(id || '') ? 'query' : null;
  }
  // the picker's lists, one per library: a library is read when it was never read or its list is older than freshMs;
  // an answer only fills its own library's list, so a late answer for another library never shows
  function pickerCache(freshMs) {
    var slots = {};
    var slot = function (c) { var k = c || ''; return slots[k] || (slots[k] = { docs: null, at: 0, pending: false }); };
    return {
      docs: function (c) { return slot(c).docs; },
      shouldLoad: function (c, now) {
        var s = slot(c);
        return !s.pending && (s.docs === null || now - s.at > freshMs);
      },
      begin: function (c) { slot(c).pending = true; },
      done: function (c, docs, now) { var s = slot(c); s.pending = false; s.docs = docs; s.at = now; },
      // a failed read: an empty list only while the picker waits for it and nothing was read yet (else unknown)
      fail: function (c, pickerWaits) {
        var s = slot(c);
        s.pending = false;
        if (s.docs === null && pickerWaits) s.docs = [];
      },
    };
  }
  // an open picker belongs to one model and library ("id|collection"); another one closes it
  function pickerStale(openFor, now) { return openFor != null && openFor !== now; }
  return { TODAY: TODAY, TODAY_IDS: TODAY_IDS, selectedId: selectedId, entryFor: entryFor, linksFor: linksFor,
           relink: relink, collectionFor: collectionFor, pickerLibrary: pickerLibrary, libraryUrl: libraryUrl,
           kindOf: kindOf, PENDING: PENDING, effectiveId: effectiveId,
           pickerCache: pickerCache, pickerStale: pickerStale, state: { map: null, mine: null } };
})();
/* memento:models-end */
// the page side: the selected model's id, its entry, and the map shared by the blocks below
var mementoSelectedId = () => {
  let stored = null;
  try { stored = MementoModels.selectedId(window.sessionStorage); } catch (e) { /* no storage: nothing stored */ }
  return MementoModels.effectiveId(stored, MementoModels.state.map, MementoModels.state.mine);
};
var mementoLinks = () => MementoModels.linksFor(MementoModels.state.map, mementoSelectedId());
// the pipes' model map (/files/models), shared: read at most every 10 s (again while a selected id is not in it: a new
// copy's entries appear once it publishes them); listeners repaint
/* memento:load-begin (tests/openwebui/js/loader_load.test.mjs). A try made before signing in (the login page gets 401)
   does not hold back the first try after it, and a failed try while signed in is repeated after 10 s by a timer, not
   only when the page changes */
var mementoMapListeners = [];
var mementoToken = () => {
  try { return window.localStorage.getItem('token'); } catch (e) { return null; }   // no storage: no session
};
// a read may go 10 s after the last one, or at once when the session changed since (signed in meanwhile)
var mementoMayTry = (last, token) => Date.now() - last.at >= 10000 || last.token !== token;
var mementoRetry = (load, token) => { if (token) setTimeout(load, 10000); };
var mementoMapTry = { at: 0, token: null };
// the models this user may use (OpenWebUI's own list, with the user's session token): which library a fresh page is
// about when no model is stored yet; read once
var mementoMineTry = { at: 0, token: null };
var mementoLoadMine = () => {
  const token = mementoToken();
  if (MementoModels.state.mine !== null || !mementoMayTry(mementoMineTry, token)) return;
  mementoMineTry.at = Date.now();
  mementoMineTry.token = token;
  fetch('/api/models', { credentials: 'same-origin', cache: 'no-store',
                         headers: token ? { Authorization: 'Bearer ' + token } : {} })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || !Array.isArray(d.data)) { mementoRetry(mementoLoadMine, token); return; }
      MementoModels.state.mine = d.data.map((m) => m && m.id).filter(Boolean);
      const map = MementoModels.state.map;
      if (map) mementoMapListeners.forEach((f) => { try { f(map); } catch (e) { /* one painter's error stops no other */ } });
    })
    .catch(() => mementoRetry(mementoLoadMine, token));
};
var mementoLoadMap = () => {
  mementoLoadMine();
  const token = mementoToken();
  if (!mementoMayTry(mementoMapTry, token)) return;
  mementoMapTry.at = Date.now();
  mementoMapTry.token = token;
  fetch('/files/models', { credentials: 'same-origin', cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || !Array.isArray(d.models)) { mementoRetry(mementoLoadMap, token); return; }
      MementoModels.state.map = d;
      mementoMapListeners.forEach((f) => { try { f(d); } catch (e) { /* one painter's error stops no other */ } });
    })
    .catch(() => mementoRetry(mementoLoadMap, token));
};
/* memento:load-end */

// One interface language: Memento's own texts are English, and Open WebUI follows the browser's language (its
// DEFAULT_LOCALE does not help: its detector stores the browser's language first). So once per browser, after Open
// WebUI has stored a language, a non-English one is switched to English and the page reloaded; a language chosen later
// in Settings is kept (the "memento_locale" mark).
(() => {
  const apply = () => {
    try {
      if (localStorage.getItem('memento_locale') === '1') return true;
      const cur = localStorage.getItem('locale');
      if (!cur) return false;
      localStorage.setItem('memento_locale', '1');
      if (!/^en([-_@]|$)/i.test(cur)) { localStorage.setItem('locale', 'en-US'); location.reload(); }
      return true;
    } catch (e) { return true; }
  };
  if (apply()) return;
  let tries = 0;
  const t = setInterval(() => { if (apply() || ++tries > 40) clearInterval(t); }, 250);
})();

// Dark by default: light only when chosen. Open WebUI stores "system" both when nothing was chosen and when "System" is
// picked, so once per browser "system" or no theme becomes "dark" (the "memento_dark" mark); a later choice is kept.
(() => {
  try {
    if (localStorage.getItem('memento_dark') === '1') return;
    localStorage.setItem('memento_dark', '1');
    const cur = localStorage.getItem('theme');
    if (cur && cur !== 'system') return;
    localStorage.setItem('theme', 'dark');
    const html = document.documentElement;
    html.classList.remove('light');
    html.classList.add('dark');
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', '#171717');
  } catch (e) { /* private mode: OpenWebUI's own default */ }
})();

// The sidebar open on a desktop's first visit (Open WebUI starts it closed until opened once, hiding "New Chat" and the
// chats). This runs before Open WebUI reads its setting; a later choice is kept. Phones keep Open WebUI's default.
(() => {
  try {
    if (localStorage.getItem('memento_sidebar') === '1' || window.innerWidth < 1024) return;
    localStorage.setItem('memento_sidebar', '1');
    if (localStorage.getItem('sidebar') === null) localStorage.setItem('sidebar', 'true');
  } catch (e) {}
})();

(() => {
  const PANEL = 'iframe[src*="/view/"]';
  const theme = () => (document.documentElement.classList.contains('dark') ? 'dark' : 'light');
  const tell = (frame) => {
    try { frame.contentWindow && frame.contentWindow.postMessage({ memento_theme: theme() }, '*'); } catch (e) {}
  };
  const tellAll = () => document.querySelectorAll(PANEL).forEach(tell);
  const watch = (frame) => {
    if (frame.dataset.mementoTheme) return;
    frame.dataset.mementoTheme = '1';
    frame.addEventListener('load', () => tell(frame));
    tell(frame);
  };
  new MutationObserver(() => document.querySelectorAll(PANEL).forEach(watch))
    .observe(document.documentElement, { childList: true, subtree: true });
  new MutationObserver(tellAll).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  document.querySelectorAll(PANEL).forEach(watch);
})();

/* Side panel title without the citation number: Open WebUI titles the panel with the citation's internal name "[3]
   Doc (p. 41)"; the number keeps citation cards apart and stays in the data. Only the text of the panel's title link
   (to Memento's /view/ page) changes: "Doc (p. 41)". */
(() => {
  const TITLE = 'a[target="_blank"][href*="/view/"]';
  const NUMBER = /^(\s*)\[\d+\]\s+/;
  const strip = () => document.querySelectorAll(TITLE).forEach((a) => {
    a.childNodes.forEach((n) => { if (n.nodeType === 3 && NUMBER.test(n.data)) n.data = n.data.replace(NUMBER, '$1'); });
  });
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; strip(); });
  }).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  strip();
})();

/* Cloud quota in the model picker: "N left this hour" under each cloud model; greyed out and unclickable while the
   hour, day or month quota is used up (the reserve of 3 counts as used up), captioned with when it is back. Reads
   /files/quota (the file server, behind the login) on load, every 60 s, and when the picker opens. */
(() => {
  let state = null;
  let loadedAt = 0;          // last successful reading
  let triedAt = 0;           // last attempt (the first one runs on the login page: 401, no cookie yet)
  // each Library Chat model's colour, from config/query_models.yaml via /files/models (written by the Query pipe):
  // [{id, name, cloud, accent, accent_dark}]. Without it cloud models are told by their names and CSS gives them a
  // default cloud colour; ours are always the theme green.
  let models = null;
  let modelsTriedAt = 0;
  // the shared map: its top-level list, and every copy's Library Chat models
  mementoMapListeners.push((d) => { models = d.models; paintSelected(); paintKinds(); });
  const loadModels = () => {
    modelsTriedAt = Date.now();
    mementoLoadMap();
  };
  const modelEntry = (id, name) => (models || []).find((m) => (id && m.id === id) || m.name === name) || null;
  const setAccent = (el, m) => {
    if (m && m.accent) el.style.setProperty('--mem-accent', m.accent); else el.style.removeProperty('--mem-accent');
    if (m && m.accent_dark) el.style.setProperty('--mem-accent-dark', m.accent_dark);
    else el.style.removeProperty('--mem-accent-dark');
  };
  // the selected model in the chat box and the start-screen title: green for ours, the model's colour for a cloud model
  const paintSelected = () => {
    document.querySelectorAll(SEL_BTN).forEach((b) => {
      const name = selName(b);
      const m = modelEntry(null, name);
      const kind = (m ? m.cloud : /,\s*cloud\)\s*$/i.test(name)) ? 'cloud' : 'ours';
      if (b.dataset.memKind !== kind) b.dataset.memKind = kind;
      setAccent(b, kind === 'cloud' ? m : null);
    });
    // the start-screen title's detail line ("Mistral Medium 3.5, cloud"), the same colours
    document.querySelectorAll('span.memento-name[data-src]').forEach((s) => {
      const m = modelEntry(null, s.dataset.src);
      const kind = (m ? m.cloud : /,\s*cloud\)\s*$/i.test(s.dataset.src)) ? 'cloud' : 'ours';
      if (s.dataset.memKind !== kind) s.dataset.memKind = kind;
      setAccent(s, kind === 'cloud' ? m : null);
    });
    // answers: marked by the model named in their header, so their citations take its colour; pinned models ("Keep in
    // Sidebar") like the dropdown. Older answers carry the old names ("…, cloud)").
    const kindOf = (m, name) => ((m ? m.cloud : /,\s*cloud\)\s*$/i.test(name || '')) ? 'cloud' : 'ours');
    // "Local" / "Cloud" after the model's name above an answer and in the sidebar's pinned models, as in the dropdown.
    // The chip is a sibling of the name, so Open WebUI's text updates never meet it
    const kindChip = (el, kind) => {
      let chip = el.nextElementSibling;
      if (!chip || !chip.classList.contains('memento-kind')) {
        chip = document.createElement('span');
        chip.className = 'memento-kind memento-kind-inline';
        el.after(chip);
      }
      if (chip.dataset.kind !== kind) { chip.dataset.kind = kind; chip.textContent = kind === 'cloud' ? 'Cloud' : 'Local'; }
    };
    document.querySelectorAll('#response-message-model-name').forEach((n) => {
      const msg = n.closest('div[id^="message-"]');
      if (!msg) return;
      const name = n.textContent.trim();
      const m = modelEntry(null, name);
      const kind = kindOf(m, name);
      if (msg.dataset.memAnswer !== kind) msg.dataset.memAnswer = kind;
      setAccent(msg, kind === 'cloud' ? m : null);
      kindChip(n, kind);
      // the "Answered by …" line at the bottom of the answer takes the same colour
      msg.querySelectorAll('em:not(.memento-answered-by)').forEach((e) => {
        if (/^Answered by /.test(e.textContent.trim())) e.classList.add('memento-answered-by');
      });
    });
    document.querySelectorAll('#pinned-models-list a[href*="model="]').forEach((a) => {
      let id = null;
      try { id = new URL(a.getAttribute('href'), location.origin).searchParams.get('model'); } catch (e) { return; }
      const label = a.querySelector('div.line-clamp-1');
      const m = modelEntry(id, label ? label.textContent.trim() : null);
      const kind = kindOf(m, label && label.textContent);
      if (a.dataset.memKind !== kind) a.dataset.memKind = kind;
      if (label) kindChip(label, kind);
      setAccent(a, kind === 'cloud' ? m : null);
    });
    // the page's drawing colour (start-screen drawing, chat watermark): the selected model's
    const root = document.documentElement;
    const sel = document.querySelector(SEL_BTN);
    const cloud = !!(sel && sel.dataset.memKind === 'cloud');
    const cm = cloud ? modelEntry(null, selName(sel)) : null;
    if (cloud !== root.hasAttribute('data-mem-cloud')) root.toggleAttribute('data-mem-cloud', cloud);
    const setVar = (k, v) => { if (v) { if (root.style.getPropertyValue(k) !== v) root.style.setProperty(k, v); }
                               else if (root.style.getPropertyValue(k)) root.style.removeProperty(k); };
    setVar('--mem-draw-l', cm && cm.accent);
    setVar('--mem-draw-d', cm && cm.accent_dark);
  };
  const load = () => {
    if (!models) loadModels();
    triedAt = Date.now();
    fetch('/files/quota', { credentials: 'same-origin', cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => {
        if (s && s.windows) loadedAt = Date.now();
        if (s) { state = s; paint(); }
      })
      .catch(() => {});
  };
  const backText = (iso) => {
    const t = new Date(iso);
    const now = new Date();
    const tomorrow = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
    if (t.toDateString() === now.toDateString())
      return 'at ' + t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
    if (t.toDateString() === tomorrow.toDateString()) return 'tomorrow';
    return 'on ' + t.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  };
  const status = () => {
    if (!state || !state.windows) return null;
    const now = Date.now();
    const reserve = state.reserve ?? 3;
    const left = {};
    for (const [w, win] of Object.entries(state.windows))
      left[w] = now >= Date.parse(win.reset_at) ? win.limit : win.remaining;
    for (const w of ['month', 'day', 'hour'])
      if (w in left && left[w] <= reserve)
        return { blocked: true, text: 'Quota used up, back ' + backText(state.windows[w].reset_at) };
    return { blocked: false, text: left.hour + ' left this hour' };
  };
  // ours vs cloud in the model dropdown: our models' names green with a "Local" label, cloud models in their colour
  // with a "Cloud" label. Cloud = listed as a cloud model in the quota file, or named "(…, cloud)" in
  // config/query_models.yaml (the quota file exists only after the first cloud answer).
  const paintKinds = () => {
    const cloud = new Set(state && Array.isArray(state.cloud_models) ? state.cloud_models : []);
    document.querySelectorAll('[role="listbox"] button[role="option"][data-value]')
      .forEach((b) => {
        const name = b.querySelector('.line-clamp-1');
        if (!name) return;
        const m = modelEntry(b.dataset.value, name.textContent.trim());
        const isCloud = cloud.has(b.dataset.value) || (m ? m.cloud : /,\s*cloud\)\s*$/i.test(name.textContent));
        const kind = isCloud ? 'cloud' : 'ours';
        setAccent(b, isCloud ? m : null);
        // hover details: no tooltip on the name (it showed the internal id); the info icon is hidden for ours
        // (custom.css) and shows a cloud model's hint; the "More" menu keeps its tooltip. Open WebUI's tooltips are
        // tippy instances, kept on the element as _tippy.
        b.querySelectorAll('*').forEach((e) => {
          const t = e._tippy;
          const more = '[aria-label="More Options"], [aria-label="Mehr Optionen"]';
          if (!t || e.querySelector(more) || e.closest(more)) return;
          if (isCloud && e.querySelector('path[d^="m11.25 11.25"]')) {
            if (m && m.hint && t.props.content !== m.hint) t.setContent(m.hint);
          } else if (t.state.isEnabled) t.disable();
        });
        // classes every time: OpenWebUI rewrites the row's class attribute while searching (custom.css keys the
        // colours on the label below, so they hold even between a rewrite and this repaint)
        b.classList.toggle('memento-ours', !isCloud);
        b.classList.toggle('memento-cloud', isCloud);
        let chip = b.querySelector('.memento-kind');
        if (chip && chip.dataset.kind === kind) return;
        if (!chip) {
          chip = document.createElement('span');
          chip.className = 'memento-kind';
          name.parentElement.after(chip);
        }
        chip.dataset.kind = kind;
        chip.textContent = isCloud ? 'Cloud' : 'Local';
      });
  };
  const paint = () => {
    paintSelected();
    paintKinds();
    const s = status();
    if (!s || !Array.isArray(state.cloud_models)) return;
    for (const id of state.cloud_models) {
      document.querySelectorAll(`button[role="option"][data-value="${CSS.escape(id)}"]`).forEach((b) => {
        let cap = b.querySelector('.memento-quota');
        if (!cap) {
          cap = document.createElement('div');
          cap.className = 'memento-quota';
          b.appendChild(cap);
        }
        if (cap.textContent !== s.text) cap.textContent = s.text;
        b.classList.toggle('memento-unavailable', s.blocked);
        b.setAttribute('aria-disabled', s.blocked ? 'true' : 'false');
      });
    }
  };
  // an unavailable model cannot be picked (capture phase: runs before the picker's own handler)
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('button[role="option"].memento-unavailable')) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      paintSelected();
      if (!models && Date.now() - modelsTriedAt > 3000) loadModels();
      if (!document.querySelector('button[role="option"]')) return;
      paintKinds();                              // needs no quota reading (the names say which are cloud models)
      // the picker just opened: fresh numbers (retry at most every 3 s while there is no good reading)
      if (Date.now() - loadedAt > 10000 && Date.now() - triedAt > 3000) load(); else paint();
    });
  }).observe(document.documentElement, { childList: true, subtree: true });
  load();
  setInterval(load, 60000);
})();

/* Look and feel: start-screen drawing and labels, guide cards, upload progress bar. Finds Open WebUI 0.11.4's
   elements by structure; if one is not found the addition is not shown and Open WebUI works as before. Check after
   every Open WebUI upgrade (openwebui/README.md, "Look and feel add-ons"). */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const DESC = 'div.line-clamp-2.max-w-xl.markdown';
  const HEAD = 'div.w-full.flex.flex-col.justify-center.items-center';
  const START = 'div.m-auto.w-full.py-24';
  const DRAW = {
    // Library Chat: documents filed into a database, the shared library (distinct from Document Chat at a glance); then
    // the network and the chat bubble
    query: '<path pathLength="1" d="M25 22V5h11l4 4v13"/><path pathLength="1" d="M36 5v4h4"/>'
      + '<path pathLength="1" opacity=".6" d="M31 22V10h12l4 4v8"/><ellipse class="f" cx="36" cy="24" rx="17" ry="5"/>'
      + '<path pathLength="1" d="M19 24v20c0 2.8 7.6 5 17 5s17-2.2 17-5V24"/><path pathLength="1" d="M19 34c0 2.8 7.6 5 17 5s17-2.2 17-5"/>'
      + '<path pathLength="1" d="M56 26 C 72 26, 72 16, 88 16 M56 30 C 74 30, 74 44, 88 44"/>'
      + '<circle class="d" cx="92" cy="16" r="3.5"/><circle class="d" cx="92" cy="44" r="3.5"/><circle class="d" cx="110" cy="30" r="4.5"/>'
      + '<circle class="d" cx="128" cy="14" r="3"/><circle class="d" cx="128" cy="46" r="3"/>'
      + '<path pathLength="1" d="M92 16 L110 30 L92 44 M110 30 L128 14 M110 30 L128 46"/><path pathLength="1" d="M136 30 H150"/>'
      + '<path pathLength="1" d="M156 12h46a6 6 0 0 1 6 6v18a6 6 0 0 1-6 6h-34l-7 7v-7h-5a6 6 0 0 1-6-6V18a6 6 0 0 1 6-6z"/>'
      + '<path pathLength="1" d="M164 23h24M164 31h16"/><path pathLength="1" d="M186 28h8v7h-8z"/>',
    // Document Chat: one document with a paperclip, the file attached to this chat
    doc: '<path pathLength="1" d="M22 6h20l7 7v33H22z M42 6v7h7"/><path pathLength="1" d="M27 22h16M27 29h16M27 36h11"/>'
      + '<path pathLength="1" d="M30 15 V3.5 a4 4 0 0 1 8 0 V17 a2.5 2.5 0 0 1 -5 0 V7"/>'
      + '<path pathLength="1" d="M56 26 C 72 26, 72 16, 88 16 M56 30 C 74 30, 74 44, 88 44"/>'
      + '<circle class="d" cx="92" cy="16" r="3.5"/><circle class="d" cx="92" cy="44" r="3.5"/><circle class="d" cx="110" cy="30" r="4.5"/>'
      + '<circle class="d" cx="128" cy="14" r="3"/><circle class="d" cx="128" cy="46" r="3"/>'
      + '<path pathLength="1" d="M92 16 L110 30 L92 44 M110 30 L128 14 M110 30 L128 46"/><path pathLength="1" d="M136 30 H150"/>'
      + '<path pathLength="1" d="M156 12h46a6 6 0 0 1 6 6v18a6 6 0 0 1-6 6h-34l-7 7v-7h-5a6 6 0 0 1-6-6V18a6 6 0 0 1 6-6z"/>'
      + '<path pathLength="1" d="M164 23h24M164 31h16"/><path pathLength="1" d="M186 28h8v7h-8z"/>',
    ingest: '<path pathLength="1" d="M18 14h17l6 6v28H18z M35 14v6h6"/><path pathLength="1" opacity=".55" d="M24 8h17l6 6"/>'
      + '<path pathLength="1" opacity=".3" d="M30 2h17l6 6"/><path pathLength="1" d="M60 30 H96 M88 24 l8 6 -8 6"/>'
      + '<ellipse class="f" cx="128" cy="14" rx="20" ry="6"/><path pathLength="1" d="M108 14v30c0 3.3 9 6 20 6s20-2.7 20-6V14"/>'
      + '<path pathLength="1" d="M108 29c0 3.3 9 6 20 6s20-2.7 20-6"/><path pathLength="1" d="M166 30 H176"/>'
      + '<circle pathLength="1" cx="196" cy="30" r="13"/><path pathLength="1" d="M190 30l4.5 4.5L203 26"/>',
  };
  // the same three labels and clips on both start screens; only the drawing follows the model. A switch button is shown
  // only when it leads to the other model.
  const PLAY = '<svg class="memento-play" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.6"/>'
    + '<path d="M6.6 5.4v5.2L10.8 8z"/></svg>';
  const GUIDE_TIP = 'Watch a short video guide.';
  const STEPS = [
    { key: 'add', title: 'Add documents', sub: 'with Library Upload', clip: 'memento-ingestion',
      text: 'Use <b>Library Upload</b>: attach your files (Ctrl+A selects all files in a folder) and send. When the reply ends with ✅ Done, ask about them in Library Chat.',
      link: { query: { href: '/?model=memento-ingestion-prod', text: 'Switch to Library Upload' } } },
    { key: 'ask', title: 'Ask', sub: 'in English or German', clip: 'memento-query',
      text: 'Type your question in English or German. Follow-up questions in the same chat build on the earlier answers.',
      link: { ingest: { href: '/?model=memento-query', text: 'Switch to Library Chat' } } },
    { key: 'cite', title: 'Verify with citations', sub: 'open the cited page', clip: 'memento-cite',
      text: 'Every answer cites its sources as [1], [2] and so on. To check a statement, click its citation, or pick a source from its list: the page it comes from opens with the passage marked.' },
  ];
  const GUIDES = {
    query: STEPS.map((s) => ({ ...s, link: s.link && s.link.query })),
    ingest: STEPS.map((s) => ({ ...s, link: s.link && s.link.ingest })),
    // Document Chat: the file stays in the chat; no guide clip yet (the video hides itself)
    doc: [
      { key: 'attach', title: 'Attach a file', sub: 'it stays private', clip: 'memento-document-chat',
        text: 'Click <b>+</b> next to the message box, then <b>Upload Files</b>, and send it with your question. Reading it takes longer. The file is used only in this chat, is not added to the library, and is deleted when you delete the chat.' },
      { ...STEPS[1], link: undefined },
      { ...STEPS[2], link: undefined },
    ],
  };
  /* memento:doc-tip-begin — the dimmed Document Chat buttons' tip (tests/openwebui/js/loader_doc_tip.test.mjs). While a file
     still uploads it says so, not "attach a file" */
  const DOC_TIP = 'Attach a file first: + next to the message box, then Upload Files.';
  const docTip = (attached, uploading) => (uploading > 1 ? 'Your files are still uploading'
    : uploading ? 'Your file is still uploading' : attached ? null : DOC_TIP);
  // A suggestion clicked while its file still uploads starts once the upload is done. One waits at a time; it is
  // forgotten when the file is removed, after 2 minutes, or when the start screen goes
  // (tests/openwebui/js/loader_doc_wait.test.mjs)
  const DOC_WAIT_TIP = 'Starts once your file is uploaded';
  const docWait = { label: null, at: 0 };
  // the label of the suggestion to click now (once), or null
  const docWaitReady = (w, attached, uploading, now) => {
    if (!w.label) return null;
    if (!attached || now - w.at > 120000) { w.label = null; return null; }
    if (uploading) return null;
    const label = w.label;
    w.label = null;
    return label;
  };
  /* memento:doc-tip-end */
  let drawnOnce = false;

  const variant = () => {
    const b = document.querySelector(SEL_BTN);
    return b && isIngest(b) ? 'ingest' : b && isDocChat(b) ? 'doc' : 'query';
  };

  // guide window: opens on a label click only, centred over a dimmed page; Esc, a click on the dimmed page or ✕ closes
  // it; one at a time
  const reduced = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let current = null;                // { backdrop, btn }
  const closeGuide = () => {
    if (!current) return;
    current.btn.setAttribute('aria-expanded', 'false');
    current.backdrop.remove();
    current = null;
  };
  const openGuide = (btn, g) => {
    closeGuide();
    const backdrop = document.createElement('div');
    backdrop.className = 'memento-guide-backdrop';
    const card = document.createElement('div');
    card.className = 'memento-guide';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-label', g.title);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'memento-guide-close';
    close.setAttribute('aria-label', 'Close');
    close.textContent = '✕';
    close.addEventListener('click', closeGuide);
    card.appendChild(close);
    const video = document.createElement('video');
    video.src = `/static/memento/guides/${g.clip}.webm`;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    if (reduced()) { video.autoplay = false; video.controls = true; } else { video.autoplay = true; }
    video.addEventListener('error', () => { video.style.display = 'none'; });
    card.appendChild(video);
    card.insertAdjacentHTML('beforeend', `<h4>${g.title}</h4><p>${g.text}</p>`);
    const foot = document.createElement('div');
    foot.className = 'memento-guide-foot';
    foot.innerHTML = '<span>Esc to close</span>';
    if (g.link) {
      const a = document.createElement('a');
      a.className = 'memento-guide-link';
      // the selected copy's own model; dimmed while its copy is not known yet
      const links = mementoLinks();
      if (links) a.href = MementoModels.relink(g.link.href, links);
      else { a.setAttribute('aria-disabled', 'true'); a.style.opacity = '0.5'; mementoLoadMap(); }
      a.textContent = g.link.text + ' →';
      foot.appendChild(a);
    }
    card.appendChild(foot);
    backdrop.appendChild(card);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) closeGuide(); });
    document.body.appendChild(backdrop);
    btn.setAttribute('aria-expanded', 'true');
    current = { backdrop, btn };
    close.focus({ preventScroll: true });
  };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeGuide(); });

  // Open WebUI's tooltips on the start screen off: one repeated the model name, the other covered it. Its tooltip
  // library opens on mouseenter of the described element; stopping that event at the document (capture phase) keeps
  // them closed. Links in the description still work.
  document.addEventListener('mouseenter', (e) => {
    const el = e.target;
    if (!(el instanceof Element) || el.closest('a')) return;
    if (el.closest(HEAD) && el.closest(HEAD).querySelector(DESC)) e.stopPropagation();
  }, true);

  // start-screen title: "Library Chat" in the sidebar's serif, the model detail on a small line under it (custom.css
  // adds " · local" / " · cloud"). Open WebUI keeps writing the name into its own span (model switches), so that span
  // is hidden, not changed, and a styled copy beside it follows its text.
  const TITLE = 'div.text-2xl.line-clamp-1';
  const paintTitle = (head) => {
    const box = head && head.querySelector(TITLE);
    const src = box && [...box.querySelectorAll('span')].find((s) => !s.closest('.memento-name'));
    if (!src) return;
    const name = src.textContent.trim();
    let shown = box.querySelector('span.memento-name');
    if (shown && shown.dataset.src === name) return;
    if (!shown) {
      shown = document.createElement('span');
      shown.className = 'memento-name';
      src.after(shown);
      src.classList.add('memento-name-src');
      box.classList.add('memento-title');
    }
    shown.dataset.src = name;
    const m = name.match(/^(.+?)\s*\((.+)\)$/);
    shown.replaceChildren(m ? m[1] : name);
    if (m) {
      const detail = document.createElement('small');
      detail.textContent = m[2];
      shown.appendChild(detail);
    }
  };

  const paintStart = () => {
    const desc = document.querySelector(DESC);
    const v = variant();
    if (!desc || v !== 'doc') docWait.label = null;
    if (desc) paintTitle(desc.closest(HEAD));
    document.querySelectorAll('.memento-drawing, .memento-steps, .memento-brand').forEach((e) => {
      if (!desc || e.dataset.v !== v) { closeGuide(); e.remove(); }
    });
    if (!desc) return;
    const head = desc.closest(HEAD);
    if (head && !head.querySelector(':scope > .memento-drawing')) {
      const s = document.createElementNS(NS, 'svg');
      s.setAttribute('viewBox', '0 0 220 60');
      s.setAttribute('class', 'memento-drawing' + (drawnOnce ? ' memento-drawn' : ''));
      s.setAttribute('aria-hidden', 'true');
      s.dataset.v = v;
      s.innerHTML = DRAW[v] || DRAW.query;
      head.prepend(s);
      drawnOnce = true;
    }
    const start = desc.closest(START);
    const box = document.getElementById('message-input-container');
    const block = start && box && [...start.children].find((c) => c.contains(box));
    if (block && !start.querySelector(':scope > .memento-steps')) {
      const wrap = document.createElement('div');
      wrap.className = 'memento-steps';
      wrap.dataset.v = v;
      wrap.setAttribute('role', 'group');
      wrap.setAttribute('aria-label', 'How Memento works');
      GUIDES[v].forEach((g) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'memento-step';
        btn.dataset.key = g.key;
        btn.setAttribute('aria-expanded', 'false');
        // a play icon and a hover text mark the labels as video guides
        btn.innerHTML = `<b>${PLAY}${g.title}</b><span>${g.sub}</span>`;
        btn.dataset.tip = GUIDE_TIP;
        btn.setAttribute('aria-description', GUIDE_TIP);
        btn.addEventListener('click', (e) => { e.preventDefault(); openGuide(btn, g); });
        wrap.appendChild(btn);
      });
      block.after(wrap);
    }
    // the group's logo, large and centred as the last thing on the start screen; kept last when Open WebUI re-renders
    // the suggestions
    if (start) {
      let brand = start.querySelector(':scope > .memento-brand');
      if (!brand) {
        brand = document.createElement('div');
        brand.className = 'memento-brand';
        brand.dataset.v = v;
        // two images: light mode a recoloured copy, dark mode the original, lifted by custom.css; the theme toggle only
        // switches which one is shown
        brand.innerHTML = '<span>Developed by</span>'
          + '<img class="memento-logo-light" src="/static/memento/kuennethgroup-light.png" alt="Kuenneth Group">'
          + '<img class="memento-logo-dark" src="/static/memento/kuennethgroup.png" alt="Kuenneth Group">';
      }
      if (start.lastElementChild !== brand) start.appendChild(brand);
    }
    // Document Chat: its suggestions right under the chat box and the guide labels after them, like Library Chat's
    // buttons before its labels; Open WebUI renders the suggestions later, so they are kept in place here
    const steps = start && start.querySelector(':scope > .memento-steps');
    if (v === 'doc' && steps) {
      const sugg = [...start.children].find((c) => c !== steps && c.querySelector('button.waterfall'));
      if (sugg && sugg.nextElementSibling !== steps) sugg.after(steps);
    }
    // Document Chat: its suggestions only once every attached file has uploaded. A file chip in the chat box shows a
    // spinner while it uploads; matched without the translated "Remove File" label
    if (v === 'doc' && start) {
      const box = document.getElementById('message-input-container');
      const chips = box ? [...box.querySelectorAll('div.flex-wrap > button.group')] : [];
      const uploading = chips.filter((c) => c.querySelector('[class*="spinner"]')).length;
      const tip = docTip(chips.length, uploading);
      start.querySelectorAll('button.waterfall').forEach((b) => {
        const dim = tip !== null;
        if ((b.getAttribute('aria-disabled') === 'true') !== dim) b.setAttribute('aria-disabled', dim ? 'true' : 'false');
        const t = uploading && docWait.label === b.innerText.trim() ? DOC_WAIT_TIP : tip;
        if (dim && b.dataset.tip !== t) b.dataset.tip = t;
        if (!dim && b.dataset.tip) delete b.dataset.tip;
        b.dataset.memUploading = uploading ? '1' : '';
      });
      const go = docWaitReady(docWait, chips.length, uploading, Date.now());
      const btn = go && [...start.querySelectorAll('button.waterfall')].find((b) => b.innerText.trim() === go);
      if (btn) setTimeout(() => btn.click(), 0);
    }
  };
  // a dimmed suggestion does nothing (OpenWebUI would send it): stopped before OpenWebUI's own click handler; dimmed
  // only because its file still uploads, it waits and goes once the upload is done
  document.addEventListener('click', (e) => {
    const b = e.target instanceof Element && e.target.closest('button.waterfall[aria-disabled="true"]');
    if (!b) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (b.dataset.memUploading === '1') {
      docWait.label = b.innerText.trim();
      docWait.at = Date.now();
      b.dataset.tip = DOC_WAIT_TIP;
    }
  }, true);

  // upload progress: the Ingestion pipe's status line drawn as a bar (keep in step with ingestion_pipe._batch_status).
  // "Uploading... 3 of 25 ready (8 parsing, 2 storing)" with stored / unchanged / failed counts (a parse-only upload
  // says "read"; a store-only batch has no parsing part) and "· 45% done" (ingestion_pipe._batch_percent), which moves
  // the bar inside long documents. Older lines ("Processing papers…", no counts, no percent) still match: they fill by
  // finished documents and say "processed", never "stored".
  const RUN = /^(?:Uploading|Processing papers)\.\.\. (?<c>\d+) of (?<n>\d+) ready \((?:\d+ (?:parsing|reading), )?\d+ storing\)(?: · (?<stored>\d+) (?<verb>stored|read) · (?<unchanged>\d+) unchanged · (?<failed>\d+) failed)? · (?<mins>\d+)m elapsed(?: · (?<pct>\d+)% done)?$/;
  const DONE = /^Done · (?:(?<stored>\d+) of (?<n>\d+) (?<verb>stored|read) · (?<unchanged>\d+) unchanged · (?<failed>\d+) failed|(?<c>\d+) of (?<n0>\d+) processed) · (?<mins>\d+)m elapsed$/;
  const ICON = '<g class="memento-page"><path d="M14 1h9l4 4v10H14z"/></g>'
    + '<path d="M6 17v9c0 2 6.3 4 14 4s14-2 14-4v-9"/><ellipse cx="20" cy="17" rx="14" ry="4"/>'
    + '<path class="memento-tick" d="M15 8l3.5 3.5L26 4"/>';
  // the text under the bar: the combined percent on the left, documents stored and time on the right. Older chats
  // without "% done": the percent by finished documents.
  const uploadLabels = (run, done) => {
    const g = (run || done).groups;
    const n = Number(g.n !== undefined ? g.n : g.n0);
    const mins = g.mins === '0' ? 'under 1 min' : `${g.mins} min`;
    const docs = `${n === 1 ? 'document' : 'documents'}`;
    let right, errors = false;
    if (g.verb) {
      const f = Number(g.failed), u = Number(g.unchanged);
      right = [`${g.stored} of ${n} ${docs} ${g.verb}`, f ? `${f} failed` : '', u ? `${u} unchanged` : '', mins]
        .filter(Boolean).join(' · ');
      errors = f > 0;
    } else {
      right = `${g.c} of ${n} ${docs} processed · ${mins}`;
    }
    if (done) return [errors ? 'Done with errors' : 'Done', right, errors];
    const c = Number(g.c);
    const pct = g.pct !== undefined ? Number(g.pct) : (n ? Math.round((100 * c) / n) : 0);
    return [`${pct}%`, right, errors];
  };
  const paintProgress = () => {
    document.querySelectorAll('.status-description div.line-clamp-1').forEach((textEl) => {
      const t = (textEl.textContent || '').trim();
      const run = RUN.exec(t), done = DONE.exec(t);
      let bar = textEl.parentElement.querySelector(':scope > .memento-upload');
      if (!run && !done) {
        if (bar) bar.remove();
        textEl.classList.remove('memento-hidden');
        return;
      }
      if (!bar) {
        bar = document.createElement('div');
        bar.className = 'memento-upload';
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', '100');
        bar.innerHTML = `<svg class="memento-upload-icon" viewBox="0 0 40 32" aria-hidden="true">${ICON}</svg>`
          + '<div class="memento-upload-main"><div class="memento-upload-bar"><i></i></div>'
          + '<div class="memento-upload-meta"><span></span><span></span></div></div>';
        textEl.after(bar);
      }
      textEl.classList.add('memento-hidden');
      const [l, r, errors] = uploadLabels(run, done);
      const pct = done ? 100 : Number(l.replace('%', ''));
      const [left, right] = bar.querySelectorAll('.memento-upload-meta span');
      const fill = bar.querySelector('.memento-upload-bar i');
      if (fill.style.width !== `${pct}%`) fill.style.width = `${pct}%`;
      if (left.textContent !== l) left.innerHTML = `<b>${l}</b>`;
      if (right.textContent !== r) right.textContent = r;
      bar.classList.toggle('memento-upload-done', !!done);
      bar.classList.toggle('memento-upload-errors', !!done && errors);
      // attributes only when they change: an aria-label write is itself a mutation this observer listens to
      if (bar.getAttribute('aria-valuenow') !== String(pct)) bar.setAttribute('aria-valuenow', String(pct));
      const label = `${l}, ${r}`;
      if (bar.getAttribute('aria-label') !== label) bar.setAttribute('aria-label', label);
    });
  };

  // which model the chat box belongs to, for custom.css (the Library Upload chat box hint)
  const markModel = () => {
    const b = document.querySelector(SEL_BTN);
    let want = b ? variant() : null;
    if (!want) {    // a read-only chat has no model selector: the model of its last answer
      const n = [...document.querySelectorAll('#response-message-model-name')].pop();
      const t = n ? n.textContent || '' : '';
      want = !t ? null : /^(Library Upload|Memento Ingestion)\b/.test(t) ? 'ingest' : /^(Document Chat|Memento Document Chat)\b/.test(t) ? 'doc' : 'query';
    }
    const root = document.documentElement;
    if ((root.dataset.memModel || null) !== want) {
      if (want) root.dataset.memModel = want; else delete root.dataset.memModel;
    }
  };
  // An upload's reply: "Uploading N files:" reads "Uploaded N files:" once the upload is done (a streamed reply cannot
  // be rewritten; older chats say "Upload of"), and only the latest 20 files are listed, with "Show N earlier files"
  // above them. Display only; the text stays as written.
  const FILE_LINE = /^(?:\S+\s+)?\[\d+\/\d+\]/;      // "[3/25] ✅ name …", "⏭ [3/25] name: skipped …"
  const shownAll = new Set();                           // message ids whose whole list is open
  const paintUploadList = () => {
    document.querySelectorAll('div[id^="message-"]').forEach((m) => {
      if (!m.querySelector('.memento-upload')) return;
      const prose = m.querySelector('.markdown-prose');
      if (!prose) return;
      if (m.querySelector('.memento-upload-done')) {
        const head = prose.querySelector(':scope > p');
        const tn = head && [...head.childNodes].find((n) => n.nodeType === 3 && n.nodeValue.trim());
        if (tn && /^\s*(Uploading|Upload of)\s*$/.test(tn.nodeValue)) {
          tn.nodeValue = tn.nodeValue.replace(/(Uploading|Upload of)/, 'Uploaded');
        }
      }
      const groups = [];                                // one per file: its line, and its error details if any
      [...prose.children].forEach((el) => {
        if (FILE_LINE.test((el.textContent || '').trim())) groups.push([el]);
        else if (groups.length && el.tagName === 'DIV' && !el.hasAttribute('data-mem-end')) groups[groups.length - 1].push(el);
      });
      const extra = groups.length - 20;
      const open = shownAll.has(m.id);
      groups.forEach((g, i) => g.forEach((el) => {
        const old = !open && i < extra;
        if (el.hasAttribute('data-mem-old') !== old) el.toggleAttribute('data-mem-old', old);
      }));
      const box = m.querySelector('#response-content-container');
      let btn = m.querySelector(':scope .memento-earlier');
      if (extra <= 0 || !box) { if (btn) btn.remove(); return; }
      if (!btn) {
        btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'memento-earlier';
        btn.addEventListener('click', () => {
          if (shownAll.has(m.id)) shownAll.delete(m.id); else shownAll.add(m.id);
          paintUploadList();
        });
        box.before(btn);
      }
      const label = open ? 'Show only the latest 20 files' : `Show ${extra} earlier file${extra === 1 ? '' : 's'}`;
      if (btn.textContent !== label) btn.textContent = label;
    });
  };
  // "Capture" in the + menu hidden: every model's vision capability is off, so a photo taken here would only be
  // refused. Matched by its text (the interface is English, block above; CSS cannot select by text)
  const hideCapture = () => {
    document.querySelectorAll('div.max-h-72 button, [role="menu"] button, [role="menuitem"]').forEach((b) => {
      if (b.style.display !== 'none' && b.textContent.trim() === 'Capture') b.style.display = 'none';
    });
  };
  // the folder dialog's "Knowledge" row (Select Knowledge / Upload) hidden: files attached to a folder stay Open
  // WebUI-only copies, never enter the Memento library and are not read by our models; uploads go through Library
  // Upload only
  const hideFolderKnowledge = () => {
    document.querySelectorAll('span.truncate').forEach((sp) => {
      if (sp.textContent.trim() !== 'Select Knowledge') return;
      const row = sp.closest('div.flex.w-full.items-center');
      if (row && row.style.display !== 'none') row.style.display = 'none';
    });
    // and Open WebUI's hint under it, which points to the Workspace (switched off)
    document.querySelectorAll('div, p, span').forEach((el) => {
      if (el.children.length === 0 && el.style.display !== 'none'
          && el.textContent.trim().startsWith('To attach knowledge base here')) el.style.display = 'none';
    });
  };
  const paint = () => { paintStart(); paintProgress(); paintUploadList(); markModel(); hideCapture(); hideFolderKnowledge(); };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; paint(); });
  }).observe(document.documentElement, { childList: true, subtree: true, characterData: true, attributes: true,
                                          attributeFilter: ['aria-label'] });
  paint();
})();

/* Footer, bottom right: the group's name, the copyright year and the credit to Open WebUI (the " (Open WebUI)"
   suffix of the app name is off: openwebui/README.md, "Licence and branding"). */
(() => {
  const add = () => {
    if (!document.body || document.querySelector('div.memento-footer')) return;
    const f = document.createElement('div');
    f.className = 'memento-footer';
    // text only: a logo here ran into the message box in chats (the large logo is on the start screens)
    const year = document.createElement('span');
    year.textContent = '© 2026 Kuenneth Group ·';
    const a = document.createElement('a');
    a.className = 'memento-credit';
    a.href = 'https://openwebui.com';
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = 'Built with Open WebUI';
    f.append(year, a);
    document.body.appendChild(f);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  new MutationObserver(add).observe(document.documentElement, { childList: true });   // body replaced: add again
})();

/* Quick actions on the Library Chat start screen: buttons in place of the suggestions. "Summarize a document" and
   "Limitations & future work" open a searchable document picker (A–Z by title, the file name in grey when it differs);
   picking sends a fixed question. The list comes from /files/library (written by the pipes); a missing list counts as
   an empty library. */
(() => {
  const DESC = 'div.line-clamp-2.max-w-xl.markdown';
  const START = 'div.m-auto.w-full.py-24';
  // order and place: the library first, then summary and gaps; the row sits right under the chat box, above the guide
  // labels. tip: the hover / focus text saying why each exists; beta: the newer ones, on their own row
  const ACTIONS = [
    { key: 'list', label: "What's in the library?", pick: false, send: () => 'Which documents are in the library?',
      tip: 'Lists every document in the shared library.' },
    { key: 'summary', label: 'Summarize a document', pick: true, send: (t) => `Summarize the document "${t}"`,
      tip: 'A summary of the whole selected document.' },
    // one document's own statements, as opposed to the checked research gaps; the sent question is unchanged (stored
    // answers and the warm-up are keyed to it)
    { key: 'gaps', label: 'Limitations & future work', pick: true, send: (t) => `What research gaps does the document "${t}" name?`,
      tip: 'Detailed information on the limits, open points and next steps of the selected document.' },
    // across the library (gap ledger report): an optional topic, or one picked document's own questions; the pipe
    // recognises the exact phrase. A proposal is drafted from the button next to each question of the report (proposal
    // buttons block below).
    { key: 'proposal', label: 'Draft a proposal', topic: true, beta: true, go: 'Show',
      placeholder: 'A topic, e.g. dopants in hafnia (empty: the whole library)',
      hint: 'First the open research questions, checked against the whole library and published papers; then a draft proposal on the one you pick, with a PDF. Answers take longer.',
      send: (t) => (t ? `Research gaps on "${t.replace(/"/g, "'")}"` : 'Research gaps across the library'),
      docs: (name) => `Research gaps in the document "${name}"`,
      tip: 'Experimental. Find open research questions in the library, on a topic or in one document, then draft a proposal on one. Answers take longer.' },
  ];
  const EMPTY_TIP = 'The library is empty. Add documents with Library Upload first.';
  const WAIT_TIP = 'Loading this library...';
  let lastKey = null;
  /* memento:picker-begin — the picker's list and the buttons' action (tests/openwebui/js/loader_picker_wiring.test.mjs) */
  // one list per library: the selected copy's library, from its own file; each list [{source, title}] once read, [] =
  // empty library; a late answer for another library never shows
  const cache = MementoModels.pickerCache(10000);
  const lib = () => MementoModels.pickerLibrary(MementoModels.state.map, mementoSelectedId());
  const coll = () => lib().collection;
  const selKey = () => (mementoSelectedId() || '') + '|' + (coll() || '');
  const docsNow = () => cache.docs(coll());
  const stale = () => cache.shouldLoad(coll(), Date.now());
  const pending = {};
  const load = () => {
    if (lib().wait) { mementoLoadMap(); return Promise.resolve(); }   // its library is not known yet: read nothing
    const c = coll(), k = c || '';
    if (pending[k]) return pending[k];
    cache.begin(c);
    pending[k] = fetch(MementoModels.libraryUrl(c), { credentials: 'same-origin', cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) { cache.fail(c, !!picker); return; }          // no answer while the picker waits: say empty
        cache.done(c, (Array.isArray(d.documents) ? d.documents : [])
          .filter((e) => e && e.source)
          .map((e) => ({ source: String(e.source), title: String(e.title || e.source) }))
          .sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })), Date.now());
        if (c === coll()) paint();
      })
      .catch(() => cache.fail(c, !!picker))                    // never a loop of retries from the picker
      .finally(() => { delete pending[k]; });
    return pending[k];
  };
  // a start-screen button pressed: its picker, its topic box, or its question
  const onAction = (a, b) => {
    if (lib().wait) { mementoLoadMap(); return; }    // dimmed while the selected model's library is not known yet
    if (picker && picker.btn === b) { closePicker(true); return; }
    if (a.topic) { openTopic(b, a); if (a.docs && stale()) load(); return; }
    if (!a.pick) { closePicker(false); send(a.send()); return; }
    openPicker(b, a);
    if (stale()) load();                             // fresh list on the next open
  };
  /* memento:picker-end */

  const isQuery = () => {
    const b = document.querySelector(SEL_BTN);
    return !(b && (isIngest(b) || isDocChat(b)));
  };

  // the question goes through Open WebUI's own input and send button, like a typed one. The input is a Tiptap editor
  // (its object on the element; insertText is ignored there); else a paste, else insertText.
  const send = (text) => {
    const input = document.getElementById('chat-input');
    if (!input) return;
    const has = () => input.innerText.trim() === text;
    if (input.editor && input.editor.commands) {
      input.editor.commands.setContent(text);
      input.editor.commands.focus('end');
    }
    if (!has()) {
      input.focus();
      const sel = window.getSelection();
      sel.removeAllRanges();
      const range = document.createRange();
      range.selectNodeContents(input);
      sel.addRange(range);
      try {
        const dt = new DataTransfer();
        dt.setData('text/plain', text);
        input.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
      } catch (e) { /* no DataTransfer constructor */ }
      if (!has()) document.execCommand('insertText', false, text);
    }
    let tries = 0;
    const click = () => {
      const b = document.querySelector('#send-message-button:not([disabled])');
      if (b) b.click();
      else if (++tries < 40) setTimeout(click, 50);   // up to 2 s; else the text stays in the box for Enter
    };
    setTimeout(click, 50);
  };
  window.mementoSend = send;          // also used by the proposal buttons of a report

  // ── picker ──
  let picker = null;           // { box, btn, action }
  const closePicker = (refocus) => {
    if (!picker) return;
    picker.btn.setAttribute('aria-expanded', 'false');
    picker.box.remove();
    if (refocus) picker.btn.focus({ preventScroll: true });
    picker = null;
  };
  const mark = (el, text, terms) => {
    // text with the search terms marked; built from text nodes (titles come from documents, never as HTML)
    const low = text.toLowerCase();
    const hits = [];
    for (const t of terms) {
      let i = low.indexOf(t);
      while (t && i !== -1) { hits.push([i, i + t.length]); i = low.indexOf(t, i + 1); }
    }
    hits.sort((a, b) => a[0] - b[0]);
    let pos = 0;
    for (const [s, e] of hits) {
      if (e <= pos) continue;
      const from = Math.max(s, pos);
      if (from > pos) el.append(text.slice(pos, from));
      const m = document.createElement('mark');
      m.textContent = text.slice(from, e);
      el.append(m);
      pos = e;
    }
    if (pos < text.length) el.append(text.slice(pos));
  };
  const openPicker = (btn, action) => {
    closePicker(false);
    const box = document.createElement('div');
    box.className = 'memento-picker';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', action.label);
    picker = { box, btn, action, key: selKey() };
    btn.setAttribute('aria-expanded', 'true');
    if (docsNow() === null) {                  // not read yet (or the file server did not answer): read, then show
      const p = document.createElement('p');
      p.className = 'memento-picker-empty';
      p.textContent = 'Loading the document list...';
      box.appendChild(p);
      btn.closest('.memento-actions').appendChild(box);
      place(box, btn);
      load().then(() => { if (picker && picker.box === box) openPicker(btn, action); });
      return;
    }
    const all = docsNow();
    if (!all.length) {
      const p = document.createElement('p');
      p.className = 'memento-picker-empty';
      p.textContent = 'The library is empty. Add documents with Library Upload first.';
      const a = document.createElement('a');
      a.className = 'memento-guide-link';
      a.href = MementoModels.relink('/?model=memento-ingestion-prod', mementoLinks() || MementoModels.TODAY);
      a.textContent = 'Switch to Library Upload →';
      box.append(p, a);
      btn.closest('.memento-actions').appendChild(box);
      place(box, btn);
      a.focus({ preventScroll: true });
      return;
    }
    const { search } = docList(box, all, (name) => { closePicker(false); send(action.send(name)); });
    btn.closest('.memento-actions').appendChild(box);
    place(box, btn);
    search.focus({ preventScroll: true });
  };
  // the searchable document list (A–Z by title, the file name in grey when it differs), appended to `box`; `onPick`
  // gets the name the pipe looks up exactly: the title, or the file name when another document has the same title
  // (quotes sent as '). Shared by the picker and the "Draft a proposal" box.
  const docList = (box, all, onPick) => {
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'memento-picker-search';
    search.placeholder = 'Search by title or file name';
    search.setAttribute('aria-label', 'Search documents');
    search.setAttribute('autocomplete', 'off');
    const list = document.createElement('ul');
    list.className = 'memento-picker-list';
    list.setAttribute('role', 'listbox');
    const foot = document.createElement('div');
    foot.className = 'memento-picker-foot';
    const count = document.createElement('span');
    const keys = document.createElement('span');
    keys.textContent = '↑↓ to move · Enter to pick · Esc to close';
    foot.append(count, keys);
    box.append(search, list, foot);
    let shown = [];
    let active = -1;
    const setActive = (i) => {
      active = i;
      [...list.children].forEach((li, j) => li.setAttribute('aria-selected', j === i ? 'true' : 'false'));
      const li = list.children[i];
      if (li && li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });
    };
    const shared = (d) => all.filter((x) => x.title === d.title).length > 1;
    const pick = (d) => onPick((shared(d) ? d.source : d.title).replace(/"/g, "'"));
    const render = () => {
      const q = search.value.trim().toLowerCase();
      const terms = q.split(/\s+/).filter(Boolean);
      shown = all.filter((d) => {
        const hay = (d.title + ' ' + d.source).toLowerCase();
        return terms.every((t) => hay.includes(t));
      });
      list.replaceChildren();
      if (!shown.length) {
        const li = document.createElement('li');
        li.className = 'memento-picker-none';
        li.textContent = `No document matches "${search.value.trim()}".`;
        list.appendChild(li);
      }
      shown.forEach((d, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option');
        const t = document.createElement('div');
        t.className = 'memento-picker-title';
        mark(t, d.title, terms);
        li.appendChild(t);
        if (d.title !== d.source) {
          const f = document.createElement('div');
          f.className = 'memento-picker-file';
          mark(f, d.source, terms);
          li.appendChild(f);
        }
        li.addEventListener('mousemove', () => { if (active !== i) setActive(i); });
        li.addEventListener('mousedown', (e) => e.preventDefault());      // keep focus in the search field
        li.addEventListener('click', () => pick(d));
        list.appendChild(li);
      });
      const n = all.length;
      count.textContent = terms.length ? `${shown.length} of ${n}` : `${n} ${n === 1 ? 'document' : 'documents'}`;
      setActive(shown.length ? 0 : -1);
    };
    search.addEventListener('input', render);
    search.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' && shown.length) { e.preventDefault(); setActive((active + 1) % shown.length); }
      else if (e.key === 'ArrowUp' && shown.length) { e.preventDefault(); setActive((active - 1 + shown.length) % shown.length); }
      else if (e.key === 'Enter') { e.preventDefault(); if (shown[active]) pick(shown[active]); }
    });
    render();
    return { search, list };
  };
  // "Draft a proposal": one optional topic field (empty = the whole library); Enter or Show sends. With `docs`: "or
  // pick a document" and the document list below it; picking one sends action.docs(name)
  const openTopic = (btn, action, typed) => {
    closePicker(false);
    const box = document.createElement('div');
    box.className = 'memento-picker memento-topic';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', action.label);
    picker = { box, btn, action, key: selKey() };
    btn.setAttribute('aria-expanded', 'true');
    const input = document.createElement('input');
    input.type = 'search';
    input.className = 'memento-picker-search';
    input.placeholder = action.placeholder || 'A topic';
    input.setAttribute('aria-label', 'Topic of the open research questions');
    input.setAttribute('autocomplete', 'off');
    const go = document.createElement('button');
    go.type = 'button';
    go.className = 'memento-topic-go';
    go.textContent = action.go || 'Show';
    const hint = document.createElement('p');
    hint.className = 'memento-picker-empty memento-topic-hint';
    hint.textContent = action.hint || '';
    const row = document.createElement('div');
    row.className = 'memento-topic-row';
    row.append(input, go);
    box.append(row, hint);
    if (typed) input.value = typed;
    if (action.docs && docsNow() === null) {    // the list not read yet: say so, then open again with the typed topic
      const wait = document.createElement('p');
      wait.className = 'memento-picker-empty';
      wait.textContent = 'Loading the document list...';
      box.appendChild(wait);
      load().then(() => { if (picker && picker.box === box) openTopic(btn, action, input.value); });
    } else if (action.docs && docsNow().length) {
      const or = document.createElement('p');
      or.className = 'memento-topic-or';
      or.textContent = 'Or pick a document';
      box.appendChild(or);
      docList(box, docsNow(), (name) => { closePicker(false); send(action.docs(name)); });
    }
    const submit = () => {
      const t = input.value.trim();
      if (action.required && !t) { input.focus(); input.setAttribute('aria-invalid', 'true'); return; }
      closePicker(false); send(action.send(t));
    };
    go.addEventListener('mousedown', (e) => e.preventDefault());
    go.addEventListener('click', submit);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } });
    btn.closest('.memento-actions').appendChild(box);
    place(box, btn);
    input.focus({ preventScroll: true });
  };
  // the bottom of Open WebUI's top bar, with its banner while one is shown: a box opened upwards must stay below it, or
  // the bar (drawn on top) covers the box's first row and takes its clicks
  const topClear = () => {
    const nav = document.querySelector('nav.sticky');
    if (!nav) return 0;
    let bottom = nav.getBoundingClientRect().bottom;
    for (const c of nav.children) {
      const r = c.getBoundingClientRect();
      if (r.height > 0) bottom = Math.max(bottom, r.bottom);
    }
    return Math.max(0, bottom);
  };
  // below the buttons, or above them when the window has more room there; a short window gets a shorter list, and the
  // "Draft a proposal" box drops its hint (also the button's tip) before the list gets shorter than a few rows
  const place = (box, btn) => {
    const wrap = btn.closest('.memento-actions').getBoundingClientRect();
    const below = window.innerHeight - wrap.bottom - 16;
    const above = wrap.top - 16 - topClear();
    const list = box.querySelector('ul.memento-picker-list');
    const hint = box.querySelector('p.memento-topic-hint');
    if (list) list.style.maxHeight = '';
    if (hint) hint.hidden = false;
    const up = box.offsetHeight > below && above > below;
    box.classList.toggle('memento-picker-up', up);
    const room = (up ? above : below) - 8;
    const fit = () => {
      if (list && box.offsetHeight > room)
        list.style.maxHeight = Math.max(90, list.clientHeight - (box.offsetHeight - room)) + 'px';
    };
    fit();
    if (hint && list && box.offsetHeight > room) { list.style.maxHeight = ''; hint.hidden = true; fit(); }
  };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && picker) { e.stopPropagation(); closePicker(true); } }, true);
  document.addEventListener('mousedown', (e) => {
    if (picker && !picker.box.contains(e.target) && !picker.btn.contains(e.target)) closePicker(false);
  }, true);

  // ── buttons ──
  const paint = () => {
    const desc = document.querySelector(DESC);
    const start = desc && desc.closest(START);
    const here = start && isQuery() ? start : null;
    document.querySelectorAll('div.memento-actions').forEach((w) => {
      if (!here || !here.contains(w)) { if (picker && w.contains(picker.box)) closePicker(false); w.remove(); }
    });
    if (!here) return;
    let wrap = here.querySelector('div.memento-actions');
    if (!wrap) {
      const steps = here.querySelector(':scope > .memento-steps');
      if (!steps) return;                      // placed before the step labels; they come first
      wrap = document.createElement('div');
      wrap.className = 'memento-actions';
      wrap.setAttribute('role', 'group');
      wrap.setAttribute('aria-label', 'Quick actions');
      ACTIONS.forEach((a, i) => {
        if (a.beta && !(ACTIONS[i - 1] || {}).beta) {           // the newer ones start their own row
          const br = document.createElement('div');
          br.className = 'memento-actions-break';
          wrap.appendChild(br);
        }
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'memento-action';
        b.dataset.action = a.key;
        b.textContent = a.label;
        if (a.beta) {
          const tag = document.createElement('span');
          tag.className = 'memento-beta';
          tag.textContent = 'Beta';
          b.appendChild(tag);
        }
        if (a.tip) {
          b.dataset.tip = a.tip;
          b.setAttribute('aria-description', a.tip);
        }
        if (a.pick || a.topic) b.setAttribute('aria-haspopup', 'dialog');
        b.addEventListener('click', (e) => { e.preventDefault(); onAction(a, b); });
        wrap.appendChild(b);
      });
      steps.before(wrap);
      if (stale()) load();
    }
    // another model or library selected: an open picker closes, the new library is read
    const key = selKey();
    if (picker && MementoModels.pickerStale(picker.key, key)) closePicker(false);
    if (key !== lastKey) { lastKey = key; if (stale()) load(); }
    // a model of a copy the map does not know yet: the buttons wait (dimmed) until it does
    const waiting = lib().wait;
    if (waiting) mementoLoadMap();
    const d = docsNow();
    const empty = d !== null && !d.length;           // dimmed only once the list is known to be empty
    wrap.querySelectorAll('button.memento-action').forEach((b) => {
      const dim = waiting || (empty && b.dataset.action !== 'list');
      if ((b.getAttribute('aria-disabled') === 'true') !== dim) b.setAttribute('aria-disabled', dim ? 'true' : 'false');
      // a dimmed button says why; its own tip returns once the library has documents
      if (b.dataset.tipOwn === undefined) b.dataset.tipOwn = b.dataset.tip || '';
      const tip = waiting ? WAIT_TIP : dim ? EMPTY_TIP : b.dataset.tipOwn;
      if (tip && b.dataset.tip !== tip) { b.dataset.tip = tip; b.setAttribute('aria-description', tip); }
    });
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; paint(); });
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-label'] });
  mementoMapListeners.push(() => paint());          // the map arrived: links, library and waiting state
  paint();
})();

/* Light/dark toggle at the top right: a round sun/moon button, the same switch as Settings → Theme (the theme lives
   in localStorage "theme" and a class on <html>). It sits at the right end of the chat's top bar, which ends where an
   open citation panel begins, so it never covers the panel's close button; pages without that bar get the window
   corner. Not on the login page. */
(() => {
  const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
    + '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
    + '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  const isDark = () => document.documentElement.classList.contains('dark');
  const apply = (dark) => {
    const html = document.documentElement;
    html.classList.remove('dark', 'light', 'oled-dark', 'her');
    html.classList.add(dark ? 'dark' : 'light');
    try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) { /* private mode: this page only */ }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', dark ? '#171717' : '#ffffff');
    paint();
  };
  const paint = () => {
    const inApp = document.getElementById('chat-input') || document.getElementById('sidebar');
    let btn = document.getElementById('memento-theme');
    if (!inApp) { if (btn) btn.remove(); return; }
    if (!btn) {
      btn = document.createElement('button');
      btn.id = 'memento-theme';
      btn.type = 'button';
      btn.addEventListener('click', () => apply(!isDark()));
    }
    const home = document.querySelector('nav.drag-region') || document.body;
    if (btn.parentElement !== home) home.appendChild(btn);
    btn.classList.toggle('memento-theme-bar', home !== document.body);
    // left of Open WebUI's own buttons at the right end of the bar (so it never covers New Chat or Controls at phone
    // width), else the CSS place 16 px from the end
    let right = '';
    if (home !== document.body) {
      const nb = home.getBoundingClientRect();
      let left = Infinity;
      home.querySelectorAll('button, a').forEach((e) => {
        if (e === btn || btn.contains(e) || e.offsetParent === null) return;
        const b = e.getBoundingClientRect();
        if (b.width && b.left > nb.left + nb.width / 2 && b.right > nb.right - 240) left = Math.min(left, b.left);
      });
      if (left !== Infinity) right = `${Math.round(nb.right - left + 8)}px`;
    }
    if (btn.style.right !== right) btn.style.right = right;
    const dark = isDark();
    const label = dark ? 'Switch to light mode' : 'Switch to dark mode';
    if (btn.getAttribute('aria-label') !== label) {
      btn.setAttribute('aria-label', label);
      btn.title = label;
      btn.innerHTML = dark ? SUN : MOON;
    }
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; paint(); });
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
  window.addEventListener('resize', () => requestAnimationFrame(paint));
  paint();
})();

/* Many attached files: a sent message lists every attached file as a chip, which can grow taller than the screen.
   More than 3: the first 3 and "+ N more files"; a click shows all, "Show fewer" folds them again (per message, while
   the page is open). */
(() => {
  const SHOWN = 3;
  const open = new WeakSet();
  const paint = () => {
    document.querySelectorAll('.chat-user > div.flex-col.flex-wrap').forEach((box) => {
      const chips = [...box.children].filter((c) => c.classList.contains('self-end'));
      let more = box.querySelector(':scope > .memento-files-more');
      if (chips.length <= SHOWN) {
        if (more) more.remove();
        box.classList.remove('memento-files-folded');
        return;
      }
      if (!more) {
        more = document.createElement('button');
        more.type = 'button';
        more.className = 'memento-files-more';
        more.addEventListener('click', () => {
          if (open.has(box)) open.delete(box); else open.add(box);
          paint();
        });
      }
      if (box.lastElementChild !== more) box.appendChild(more);
      const shown = open.has(box);
      box.classList.toggle('memento-files-folded', !shown);
      const text = shown ? 'Show fewer' : `+ ${chips.length - SHOWN} more files`;
      if (more.textContent !== text) more.textContent = text;
      more.setAttribute('aria-expanded', shown ? 'true' : 'false');
    });
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; paint(); });
  }).observe(document.documentElement, { childList: true, subtree: true });
  paint();
})();

// "Thorough check" as one labelled switch in the chat box. The toggle filter lives in Open WebUI's Integrations menu
// and, when on, shows only a small icon chip. This switch shows the state and flips the same toggle: on = open the menu
// and click the item; off = click Open WebUI's chip (hidden by custom.css). Only on Library Chat and Document Chat
// (data-mem-model query / doc).
(() => {
  const NAME = 'Thorough check';
  const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
               'stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21"/>' +
               '<path d="m7.8 10.6 1.9 1.9 3.6-3.8"/></svg>';
  const chipOn = () => {
    const img = document.querySelector(`#message-input-container img[alt="${NAME}"]`);
    return img ? img.closest('button') : null;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // what the switch does, on hover or keyboard focus: a box on the page body, fixed above the switch (the chat box
  // would clip anything inside it). Live answers think only with the switch on, so On names both steps
  const TIP = '<b>Thorough check</b>' +
    '<p><b>Off:</b> a quick answer, shown while it is written.</p>' +
    '<p><b>On:</b> Memento takes time to think the question through, then checks each statement ' +
    'against its sources. More reliable, but it takes longer.</p>';
  const tipEl = () => {
    let t = document.getElementById('memento-thorough-tip');
    if (!t) {
      t = document.createElement('div');
      t.id = 'memento-thorough-tip';
      t.setAttribute('role', 'tooltip');
      t.innerHTML = TIP;
      document.body.appendChild(t);
    }
    return t;
  };
  const showTip = (sw) => {
    const t = tipEl();
    const r = sw.getBoundingClientRect();
    t.style.left = Math.max(8, Math.min(r.left, window.innerWidth - t.offsetWidth - 8)) + 'px';
    t.style.bottom = (window.innerHeight - r.top + 8) + 'px';
    t.classList.add('show');
  };
  const hideTip = () => { const t = document.getElementById('memento-thorough-tip'); if (t) t.classList.remove('show'); };
  // hover is watched on the window: the chat box stops mouse events before they reach the switch
  window.addEventListener('pointerover', (e) => {
    const sw = e.target instanceof Element ? e.target.closest('button.memento-thorough') : null;
    // a mouse click also focuses the switch; only keyboard focus (:focus-visible) keeps the tip after the pointer
    // leaves
    const kbd = document.activeElement?.classList?.contains('memento-thorough') && document.activeElement.matches(':focus-visible');
    if (sw) showTip(sw); else if (!kbd) hideTip();
  }, true);
  let busy = false;
  const flip = async () => {
    if (busy) return;
    busy = true;
    try {
      const on = chipOn();
      if (on) { on.click(); return; }
      const menuBtn = document.getElementById('integration-menu-button');
      if (!menuBtn) return;
      menuBtn.click();
      let item = null;
      for (let i = 0; i < 30 && !item; i++) {
        await sleep(50);
        item = [...document.querySelectorAll('[role="menu"] *')]
          .find((e) => e.children.length === 0 && e.textContent.trim() === NAME);
      }
      if (item) item.click();
      await sleep(80);
      const esc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true });
      (document.activeElement || document).dispatchEvent(esc);
      document.dispatchEvent(esc);
    } finally {
      setTimeout(() => { busy = false; paint(); }, 120);
    }
  };
  const paint = () => {
    const menuBtn = document.getElementById('integration-menu-button');
    let sw = document.querySelector('button.memento-thorough');
    // only the models that have the filter (Library Chat, Document Chat): admins also get the Integrations button in
    // Library Upload, where there is nothing to switch
    const kind = document.documentElement.getAttribute('data-mem-model');
    if (!menuBtn || (kind !== 'query' && kind !== 'doc')) { if (sw) { sw.remove(); hideTip(); } return; }
    const wrap = menuBtn.closest('span[role="button"]') || menuBtn.parentElement;
    if (!sw) {
      sw = document.createElement('button');
      sw.type = 'button';
      sw.className = 'memento-thorough';
      sw.innerHTML = ICON + `<span>${NAME}</span>`;
      sw.setAttribute('aria-describedby', 'memento-thorough-tip');
      sw.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); flip(); });
      sw.addEventListener('focus', () => { if (sw.matches(':focus-visible')) showTip(sw); });
      ['blur', 'click'].forEach((ev) => sw.addEventListener(ev, hideTip));
    }
    if (wrap.nextElementSibling !== sw) wrap.after(sw);
    const on = chipOn() ? 'true' : 'false';
    if (sw.getAttribute('aria-pressed') !== on) sw.setAttribute('aria-pressed', on);
    // the label says it while on, besides the highlight
    const label = sw.querySelector('span');
    const text = on === 'true' ? `${NAME} · on` : NAME;
    if (label && label.textContent !== text) label.textContent = text;
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; paint(); });
  }).observe(document.documentElement, { childList: true, subtree: true });
  paint();
})();

// Citation pills "[N] p. X": Open WebUI shows the document name cut in the middle, which says little. The number
// matches the "Cited in this answer" list and the page says where; the full name is in that list and in the side
// panel's header. Only the pill's own text changes.
(() => {
  const PILLS = '#messages-container a[data-link-preview-trigger] > button span.line-clamp-1, ' +
                '#messages-container button[aria-label*=": ["]:not([id^="source-"]) span.line-clamp-1';
  const relabel = () => {
    document.querySelectorAll(PILLS).forEach((sp) => {
      const tn = [...sp.childNodes].find((n) => n.nodeType === 3 && n.nodeValue.trim());
      if (!tn) return;
      if (/^\s*\[\d+\](?:\s+pp?\.\s*[^()\s]+)?\s*$/.test(tn.nodeValue)) return;     // already "[N] p. X"
      const num = tn.nodeValue.match(/^\s*(\[\d+\])/);
      if (!num) return;
      const page = tn.nodeValue.match(/\((pp?\.\s*[^()]*)\)\s*$/);
      const want = num[1] + (page ? ' ' + page[1] : '') + (/\s$/.test(tn.nodeValue) ? ' ' : '');
      if (tn.nodeValue !== want) tn.nodeValue = want;
    });
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; relabel(); });
  }).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  relabel();
})();

/* Sign-in / sign-up page: the Memento logo large above the title, one line on what Memento is, the group's logo
   under the form (as on the start screens); Open WebUI's tiny corner logo is hidden by custom.css. Re-added when Open
   WebUI re-renders the card (switching sign in / sign up). */
(() => {
  const paint = () => {
    const card = document.getElementById('auth-login-card');
    const form = card && card.querySelector('form');
    if (!form) return;
    if (!card.querySelector('.memento-auth-head')) {
      const head = document.createElement('div');
      head.className = 'memento-auth-head';
      head.innerHTML = '<img class="memento-auth-logo" src="/static/favicon.png" alt="Memento">';
      form.before(head);
    }
    const title = form.querySelector(':scope > div.mb-1');
    // "Sign in to Memento" / "Sign up to Memento" → "Memento" (the fields say what the page is); only the text node
    // changes, so Open WebUI's own updates still find it (and this runs again after them)
    const heading = title && title.querySelector('div.text-2xl');
    if (heading) {
      for (const n of heading.childNodes) {
        if (n.nodeType === 3 && /^\s*Sign (in|up) to\s+/i.test(n.data)) n.data = n.data.replace(/^\s*Sign (in|up) to\s+/i, '');
      }
      heading.classList.add('memento-auth-title');
    }
    if (title && !form.querySelector('.memento-auth-tagline')) {
      // one line on what Memento is
      const p = document.createElement('p');
      p.className = 'memento-auth-tagline';
      p.innerHTML = '<b>When researchers move on, their insights stay.</b><br>Ask the group\'s memory. Every answer shows its source.';
      title.after(p);
    }
    if (!card.querySelector('.memento-auth-brand')) {
      const brand = document.createElement('div');
      brand.className = 'memento-brand memento-auth-brand';
      brand.innerHTML = '<span>Developed by</span>'
        + '<img class="memento-logo-light" src="/static/memento/kuennethgroup-light.png" alt="Kuenneth Group">'
        + '<img class="memento-logo-dark" src="/static/memento/kuennethgroup.png" alt="Kuenneth Group">';
      card.appendChild(brand);
    }
  };
  let queued = false;
  const start = () => {
    paint();
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; paint(); });
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();

/* memento:propose-begin — the rows of a research-gaps report's question table that get "Draft a proposal"
   (tests/openwebui/js/loader_propose.test.mjs). Each question shown in full has a closed "Evidence for question N" block, whose
   content Open WebUI mounts only when opened, so the buttons sit in the table instead: rows of the table headed
   "# | Question" whose number has such a block. null: no such table (an older report). */
var mementoProposeRows = (toggles, tables) => {
  const shown = new Set(toggles.map((t) => (t.replace(/\s+/g, ' ').trim().match(/^Evidence for question (\d+)$/) || [])[1])
    .filter(Boolean));
  const k = tables.findIndex((t) => t.head[0] === '#' && /^question$/i.test(t.head[1] || ''));
  if (!shown.size || k < 0) return null;
  return { table: k, rows: tables[k].rows.map((c, i) => [i, c.trim()]).filter(([, n]) => shown.has(n)) };
};
/* memento:propose-end */

/* "Draft a proposal on this question" for each question of the latest research-gaps report: one click sends "Draft
   a proposal on question N" (the pipe reads the numbers from the report just above, so only the latest report gets
   buttons). In the question table's rows; under each question's heading in older reports. */
(() => {
  const BTN = 'memento-propose';
  const paint = () => {
    const answers = [...document.querySelectorAll('div[id^="message-"]')].filter((m) => m.querySelector('#response-message-model-name'));
    const last = answers[answers.length - 1];
    document.querySelectorAll(`button.${BTN}`).forEach((b) => { if (!last || !last.contains(b)) b.remove(); });
    if (!last) return;
    const h2 = [...last.querySelectorAll('h2')].find((h) => /^Open research questions\b/.test(h.textContent.trim()));
    if (!h2 || !window.mementoSend) return;
    const tables = [...last.querySelectorAll('table')];
    const at = mementoProposeRows([...last.querySelectorAll('button[aria-expanded]')].map((t) => t.textContent || ''),
      tables.map((t) => ({ head: [...t.querySelectorAll('th')].map((th) => (th.textContent || '').trim()),
                           rows: [...t.querySelectorAll('tbody tr')].map((tr) => (tr.querySelector('td') || {}).textContent || '') })));
    if (at) {
      const trs = tables[at.table].querySelectorAll('tbody tr');
      at.rows.forEach(([i, n]) => {
        const cells = trs[i].querySelectorAll('td');
        const cell = cells[cells.length - 1];
        if (!cell || cell.querySelector(`button.${BTN}`)) return;
        const b = document.createElement('button');
        b.type = 'button';
        b.className = `${BTN} memento-inrow`;
        b.textContent = 'Draft a proposal';
        b.addEventListener('click', (e) => { e.preventDefault(); window.mementoSend(`Draft a proposal on question ${n}`); });
        cell.appendChild(b);
      });
    }
    if (!at) last.querySelectorAll('h3').forEach((h3) => {
      const m = h3.textContent.trim().match(/^(\d+)\.\s/);
      if (!m) return;
      let after = h3.nextElementSibling;
      if (after && after.tagName === 'UL') { /* the facts list */ } else { after = h3; }
      if (after.nextElementSibling && after.nextElementSibling.classList.contains(BTN)) return;
      const b = document.createElement('button');
      b.type = 'button';
      b.className = BTN;
      b.textContent = 'Draft a proposal on this question';
      b.addEventListener('click', (e) => { e.preventDefault(); window.mementoSend(`Draft a proposal on question ${m[1]}`); });
      after.after(b);
    });
    // a library report's "Open questions by document": one button per row opens that document's report (the first cell
    // is the name as the picker sends it); the next page of the report after its hint
    const NEXT = 'memento-next';
    document.querySelectorAll(`button.${NEXT}, button.memento-showdoc`).forEach((x) => { if (!last.contains(x)) x.remove(); });
    last.querySelectorAll('button[aria-expanded]').forEach((t) => {
      if (!/^Open questions by document\b/.test((t.textContent || '').trim())) return;
      (t.parentElement || t).querySelectorAll('tbody tr').forEach((tr) => {
        const cells = tr.querySelectorAll('td');
        if (!cells.length || tr.querySelector('button.memento-showdoc')) return;
        const name = (cells[0].textContent || '').trim().replace(/"/g, "'");
        if (!name) return;
        const sb = document.createElement('button');
        sb.type = 'button';
        sb.className = `${BTN} memento-showdoc`;
        sb.textContent = 'Show its questions';
        sb.addEventListener('click', (e) => { e.preventDefault(); window.mementoSend(`Research gaps in the document "${name}"`); });
        cells[cells.length - 1].appendChild(sb);
      });
    });
    const hint = [...last.querySelectorAll('p')].find((p) => /reply more research gaps\b/.test((p.textContent || '').replace(/\s+/g, ' ')));
    if (hint && !(hint.nextElementSibling && hint.nextElementSibling.classList.contains(NEXT))) {
      const nb = document.createElement('button');
      nb.type = 'button';
      nb.className = `${BTN} ${NEXT}`;
      nb.textContent = 'Show the next questions';
      nb.addEventListener('click', (e) => { e.preventDefault(); window.mementoSend('more research gaps'); });
      hint.after(nb);
    }
  };
  let queued = false;
  const start = () => {
    paint();
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; paint(); });
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();


/* One kind of model per chat: picking a model of another kind (Library Chat, Document Chat, Library Upload) in a
   chat that already has an answer opens a new chat with that model, with a short note; switching among the Library Chat
   models stays in the chat. Switching kinds within a chat would re-send earlier attachments to the new model. Open
   WebUI saves a chat's model only when a message is sent, so the old chat stays as it was. */
(() => {
  const NOTE_KEY = 'memento-newchat-note';
  // the kind of a picked model: from the pipes' model map, else the default ids
  const kindOfId = (id) => MementoModels.kindOf(MementoModels.state.map, id);
  const kindOfName = (t) => !t ? null : /^(Library Upload|Memento Ingestion)\b/.test(t) ? 'ingest'
    : /^(Document Chat|Memento Document Chat)\b/.test(t) ? 'doc' : 'query';
  const chatKind = () => {
    if (!/^\/c\//.test(location.pathname)) return null;
    const n = [...document.querySelectorAll('#response-message-model-name')].pop();
    return n ? kindOfName((n.textContent || '').trim()) : null;
  };
  // capture: decided before OpenWebUI switches the chat's model
  document.addEventListener('click', (e) => {
    const opt = e.target && e.target.closest && e.target.closest('[role="listbox"] button[role="option"][data-value]');
    if (!opt) return;
    const want = kindOfId(opt.dataset.value), have = chatKind();
    if (!want || !have || want === have) return;
    e.preventDefault();
    e.stopPropagation();
    const name = ((opt.querySelector('.line-clamp-1') || opt).textContent || '').trim();
    try { sessionStorage.setItem(NOTE_KEY, name); } catch (err) { /* private mode: no note */ }
    location.assign('/?model=' + encodeURIComponent(opt.dataset.value));
  }, true);
  const showNote = () => {
    let name = null;
    try { name = sessionStorage.getItem(NOTE_KEY); sessionStorage.removeItem(NOTE_KEY); } catch (err) { return; }
    if (!name) return;
    const n = document.createElement('div');
    n.id = 'memento-note';
    n.setAttribute('role', 'status');
    n.textContent = `New chat for ${name}. Your earlier chat is in the sidebar.`;
    document.body.appendChild(n);
    // centred over the chat column (not the window, which includes the sidebar), below the top bar and its banner
    const box = document.getElementById('chat-input');
    const r = box && box.getBoundingClientRect();
    if (r && r.width) n.style.left = `${Math.round(r.left + r.width / 2)}px`;
    const nav = document.querySelector('nav.sticky');
    if (nav) {
      let bottom = nav.getBoundingClientRect().bottom;
      for (const c of nav.children) { const cr = c.getBoundingClientRect(); if (cr.height > 0) bottom = Math.max(bottom, cr.bottom); }
      n.style.top = `${Math.round(bottom + 10)}px`;
    }
    setTimeout(() => { n.classList.add('memento-note-out'); setTimeout(() => n.remove(), 600); }, 6000);
  };
  const wait = () => (document.getElementById('chat-input') ? showNote() : setTimeout(wait, 300));
  if (document.body) wait(); else document.addEventListener('DOMContentLoaded', wait);
})();

/* Memento's own collapsible blocks: Open WebUI 0.11.4 renders a Markdown <details type="memento_…"> as its
   Collapsible (a <div> with a <button aria-expanded>, the content mounted only while open) and drops the type, so
   custom.css cannot tell them apart. The wrapper of each block is marked by its title: data-memento-block = ideas (the
   model's suggestions, a report's suggested directions) | proposed (a proposal's proposed work) | evidence (a report
   question's evidence). Idempotent; repeated as answers stream in and blocks are opened. */
(() => {
  const KINDS = [
    [/^(Possible extensions \(suggested by the AI\)|Mögliche Erweiterungen \(von der KI vorgeschlagen\))$/, 'ideas'],
    [/^Directions suggested by the AI \(\d+\)$/, 'ideas'],          // a report's suggested directions
    [/^Proposed work \(suggested by the AI\)$/, 'proposed'],
    [/^Evidence for question \d+$/, 'evidence'],
  ];
  const paint = () => {
    // a proposal's "Download as PDF" link: green, with a red PDF icon (custom.css)
    document.querySelectorAll('#messages-container a').forEach((a) => {
      if (!a.classList.contains('memento-pdf-link') && (a.textContent || '').trim() === 'Download as PDF')
        a.classList.add('memento-pdf-link');
    });
    document.querySelectorAll('#messages-container button[aria-expanded]').forEach((b) => {
      const t = (b.textContent || '').replace(/\s+/g, ' ').trim();
      const k = KINDS.find(([re]) => re.test(t));
      const w = b.parentElement;
      if (k && w && w.dataset.mementoBlock !== k[1]) w.dataset.mementoBlock = k[1];
    });
  };
  let queued = false;
  const start = () => {
    paint();
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; paint(); });
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();

/* memento:file-limit-begin — Document Chat reads at most MAX_FILES files per chat (query_pipe DOC_CHAT_MAX_FILES,
   keep the two in step). With more in the message box, sending is blocked (send button, Enter) and a line above the box
   says how many to remove. Files of earlier messages are not counted here: the pipe still reports a file over the
   limit. Tests: tests/openwebui/js/loader_file_limit.test.mjs. */
(() => {
  const MAX_FILES = 5;
  // OpenWebUI 0.11.4: one remove button per attached file or image in the message box
  const form = () => { const i = document.getElementById('chat-input'); return i && i.closest('form'); };
  const over = () => {
    const f = form();
    if (document.documentElement.dataset.memModel !== 'doc' || !f) return 0;
    return Math.max(0, f.querySelectorAll('div.absolute.-top-1.-right-1 > button').length - MAX_FILES);
  };
  const paint = () => {
    const extra = over(), f = form();
    const btn = document.getElementById('send-message-button');
    if (btn && extra && !btn.dataset.memLimit) { btn.dataset.memLimit = '1'; btn.setAttribute('aria-disabled', 'true'); }
    if (btn && !extra && btn.dataset.memLimit) { delete btn.dataset.memLimit; btn.removeAttribute('aria-disabled'); }
    let note = document.getElementById('memento-file-limit');
    if (!extra) { if (note) note.remove(); return; }
    if (!note) {
      note = document.createElement('div');
      note.id = 'memento-file-limit';
      note.setAttribute('role', 'status');
    }
    const text = `Document Chat reads up to ${MAX_FILES} files per chat. Please remove ${extra} ${extra === 1 ? 'file' : 'files'}.`;
    if (note.textContent !== text) note.textContent = text;
    if (note.parentNode !== f) f.prepend(note);
  };
  const stop = (e) => { if (over()) { e.preventDefault(); e.stopImmediatePropagation(); paint(); } };
  // capture: before OpenWebUI's own handlers send the message
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.target && e.target.closest && e.target.closest('#chat-input')) stop(e);
  }, true);
  document.addEventListener('click', (e) => {
    if (e.target && e.target.closest && e.target.closest('#send-message-button')) stop(e);
  }, true);
  document.addEventListener('submit', (e) => { if (e.target === form()) stop(e); }, true);
  let queued = false;
  const start = () => {
    paint();
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; paint(); });
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();
/* memento:file-limit-end */
